// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "cuda_util.h"
#include "error.h"
#include "sort.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cassert>
#include <unordered_map>
#if WP_ENABLE_HIP
#include "temp_buffer.h"

#include <type_traits>
#endif

#include <cub/cub.cuh>

// temporary buffer for radix sort
struct RadixSortTemp {
    void* mem = NULL;
    size_t size = 0;
};

// use unique temp buffers per CUDA stream to avoid race conditions
static std::unordered_map<void*, RadixSortTemp> g_radix_sort_temp_map;

template <int Size> struct SortPayload {
    uint8_t data[Size];
};


template <typename KeyType, typename ValueType>
void radix_sort_reserve_internal(void* context, int n, void** mem_out, size_t* size_out, int begin_bit, int end_bit)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<KeyType> d_keys;
    cub::DoubleBuffer<ValueType> d_values;

    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());

    // compute temporary memory required
    size_t sort_temp_size;
    check_cuda(cub::DeviceRadixSort::SortPairs(NULL, sort_temp_size, d_keys, d_values, n, begin_bit, end_bit, stream));

    RadixSortTemp& temp = g_radix_sort_temp_map[stream];

    if (sort_temp_size > temp.size) {
        wp_free_device(WP_CURRENT_CONTEXT, temp.mem);
        temp.mem = wp_alloc_device(WP_CURRENT_CONTEXT, sort_temp_size, "(native:sort)");
        temp.size = sort_temp_size;
    }

    if (mem_out)
        *mem_out = temp.mem;
    if (size_out)
        *size_out = temp.size;
}

void radix_sort_reserve(void* context, int n, void** mem_out, size_t* size_out, int begin_bit, int end_bit)
{
    radix_sort_reserve_internal<int, int>(context, n, mem_out, size_out, begin_bit, end_bit);
}

void radix_sort_release(void* context, void* stream)
{
    // release temporary buffer for the given stream, if it exists
    auto it = g_radix_sort_temp_map.find(stream);
    if (it != g_radix_sort_temp_map.end()) {
        wp_free_device(context, it->second.mem);
        g_radix_sort_temp_map.erase(it);
    }
}

#if WP_ENABLE_HIP
// rocPRIM sorts floating-point keys over their full width only. For a partial bit range, the keys are sorted by the
// bit pattern a radix sort orders them by (negative values with every bit flipped, others with the sign bit set), as
// unsigned integers over the same range, and decoded afterwards. As in CUB, -0.0 sorts as +0.0 and every key keeps its
// bits: zeros are encoded alike, so the stable sort keeps them in input order, and the sign of each is restored from
// the signs of the input zeros in that order.
template <typename Bits>
__global__ void radix_sort_float_encode(Bits* keys, int n, int* zero_flags, uint8_t* zero_negative)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        const Bits sign = Bits(1) << (sizeof(Bits) * 8 - 1);
        const Bits key = keys[i];
        const bool is_zero = Bits(key << 1) == 0;
        zero_flags[i] = is_zero;
        zero_negative[i] = is_zero && (key & sign);
        keys[i] = is_zero ? sign : (key & sign) ? Bits(~key) : Bits(key | sign);
    }
}

template <typename Bits> __global__ void radix_sort_float_mark_zeros(const Bits* keys, int n, int* zero_flags)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        zero_flags[i] = keys[i] == Bits(Bits(1) << (sizeof(Bits) * 8 - 1));
}

template <typename Bits>
__global__ void radix_sort_float_decode(Bits* keys, int n, const int* zero_ranks, const uint8_t* zero_negative)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        const Bits sign = Bits(1) << (sizeof(Bits) * 8 - 1);
        const Bits key = keys[i];
        if (key == sign)
            keys[i] = zero_negative[zero_ranks[i]] ? sign : Bits(0);
        else
            keys[i] = (key & sign) ? Bits(key ^ sign) : Bits(~key);
    }
}

template <typename Bits, typename ValueType>
void radix_sort_pairs_float_range_device(
    void* context, Bits* keys, ValueType* values, int n, int begin_bit, int end_bit
);
#endif

template <typename KeyType, typename ValueType>
void radix_sort_pairs_device(void* context, KeyType* keys, ValueType* values, int n, int begin_bit, int end_bit)
{
    ContextGuard guard(context);

#if WP_ENABLE_HIP
    if constexpr (std::is_floating_point<KeyType>::value) {
        if (n > 0 && (begin_bit != 0 || end_bit != int(sizeof(KeyType) * 8))) {
            using Bits = std::conditional_t<sizeof(KeyType) == 4, uint32_t, uint64_t>;
            radix_sort_pairs_float_range_device<Bits, ValueType>(
                context, reinterpret_cast<Bits*>(keys), values, n, begin_bit, end_bit
            );
            return;
        }
    }
#endif

    cub::DoubleBuffer<KeyType> d_keys(keys, keys + n);
    cub::DoubleBuffer<ValueType> d_values(values, values + n);

    RadixSortTemp temp;
    radix_sort_reserve_internal<KeyType, ValueType>(WP_CURRENT_CONTEXT, n, &temp.mem, &temp.size, begin_bit, end_bit);

    // sort
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            temp.mem, temp.size, d_keys, d_values, n, begin_bit, end_bit, (cudaStream_t)wp_cuda_stream_get_current()
        )
    );

    if (d_keys.Current() != keys)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(KeyType) * n);

    if (d_values.Current() != values)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(ValueType) * n);
}

#if WP_ENABLE_HIP
template <typename Bits, typename ValueType>
void radix_sort_pairs_float_range_device(
    void* context, Bits* keys, ValueType* values, int n, int begin_bit, int end_bit
)
{
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    const int blocks = (n + 255) / 256;

    ScopedTemporary<int> zero_flags(WP_CURRENT_CONTEXT, n);
    ScopedTemporary<int> zero_ranks(WP_CURRENT_CONTEXT, n);
    ScopedTemporary<uint8_t> zero_negative(WP_CURRENT_CONTEXT, n);
    ScopedTemporary<uint8_t> input_zero_negative(WP_CURRENT_CONTEXT, n);
    ScopedTemporary<int> num_zeros(WP_CURRENT_CONTEXT, 1);

    radix_sort_float_encode<Bits><<<blocks, 256, 0, stream>>>(keys, n, zero_flags.buffer(), zero_negative.buffer());
    check_cuda(cudaGetLastError());

    // the signs of the input zeros, in input order
    size_t select_bytes = 0;
    size_t scan_bytes = 0;
    check_cuda(
        cub::DeviceSelect::Flagged(
            nullptr, select_bytes, zero_negative.buffer(), zero_flags.buffer(), input_zero_negative.buffer(),
            num_zeros.buffer(), n, stream
        )
    );
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, zero_flags.buffer(), zero_ranks.buffer(), n, stream));
    ScopedTemporary<> scratch(WP_CURRENT_CONTEXT, select_bytes > scan_bytes ? select_bytes : scan_bytes);
    check_cuda(
        cub::DeviceSelect::Flagged(
            scratch.buffer(), select_bytes, zero_negative.buffer(), zero_flags.buffer(), input_zero_negative.buffer(),
            num_zeros.buffer(), n, stream
        )
    );

    radix_sort_pairs_device<Bits, ValueType>(context, keys, values, n, begin_bit, end_bit);

    // each output zero's rank among the output zeros, which are the input zeros in input order
    radix_sort_float_mark_zeros<Bits><<<blocks, 256, 0, stream>>>(keys, n, zero_flags.buffer());
    check_cuda(cudaGetLastError());
    check_cuda(
        cub::DeviceScan::ExclusiveSum(scratch.buffer(), scan_bytes, zero_flags.buffer(), zero_ranks.buffer(), n, stream)
    );

    radix_sort_float_decode<Bits>
        <<<blocks, 256, 0, stream>>>(keys, n, zero_ranks.buffer(), input_zero_negative.buffer());
    check_cuda(cudaGetLastError());
}
#endif

template <typename KeyType>
void radix_sort_pairs_device_dispatch_value(
    void* context, KeyType* keys, void* values, int n, int begin_bit, int end_bit, int value_size
)
{
    if (value_size == 4) {
        radix_sort_pairs_device<KeyType, SortPayload<4>>(
            context, keys, reinterpret_cast<SortPayload<4>*>(values), n, begin_bit, end_bit
        );
    } else if (value_size == 8) {
        radix_sort_pairs_device<KeyType, SortPayload<8>>(
            context, keys, reinterpret_cast<SortPayload<8>*>(values), n, begin_bit, end_bit
        );
    } else {
        wp::set_error_string("Warp sort error: Unsupported radix sort value size %d", value_size);
        assert(false && "Unsupported radix sort value size");
    }
}

void radix_sort_pairs_device(void* context, int* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, uint32_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, float* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, double* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, int64_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, uint64_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

// Record-and-execute a radix sort under CUDA APIC capture: record params into
// the byte stream, then fall through so the live sort issues onto the captured
// stream. Mirror of apic_capture_radix_sort in sort.cpp, but device-scoped and
// non-skipping (the CUDA op must execute so the driver captures it into the
// native graph; the byte stream carries it for persistent .wrp save/load).
// No-op outside a CUDA APIC capture and during graph rebuild.
static void apic_capture_radix_sort_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size, uint8_t dtype, uint64_t key_size
)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || n <= 0)
        return;
    if (value_size != 4 && value_size != 8)
        return;
    uint64_t keys_bytes = static_cast<uint64_t>(2) * static_cast<uint64_t>(n) * key_size;
    uint64_t values_bytes = static_cast<uint64_t>(2) * static_cast<uint64_t>(n) * static_cast<uint64_t>(value_size);
    APICAddress keys_addr = apic_resolve_live_ptr(state, keys, keys_bytes);
    APICAddress values_addr = apic_resolve_live_ptr(state, values, values_bytes);
    apic_record_radix_sort(
        state, keys_addr.region_id, keys_addr.offset, values_addr.region_id, values_addr.offset,
        static_cast<uint32_t>(n), begin_bit, end_bit, value_size, dtype
    );
}

// Record-and-execute a segmented sort under CUDA APIC capture. Mirror of
// apic_capture_segmented_sort in sort.cpp, device-scoped and non-skipping.
static void apic_capture_segmented_sort_device(
    uint64_t keys, uint64_t values, int n, uint64_t segment_start, uint64_t segment_end, int num_segments, uint8_t dtype
)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || n <= 0)
        return;
    // keys/values span 2*n elements (sort scratch). Keys are int32 or float32
    // (both 4 bytes); values are always int32.
    uint64_t kv_bytes = static_cast<uint64_t>(2) * static_cast<uint64_t>(n) * sizeof(uint32_t);
    // Inferred-end captures alias segment_end into the start array one element
    // in, so the start array spans num_segments+1 entries; explicit-end captures
    // use two separate num_segments-entry arrays. Match the recorded span.
    bool inferred_end = (segment_end == segment_start + sizeof(int32_t));
    uint64_t segstart_count = static_cast<uint64_t>(num_segments) + (inferred_end ? 1u : 0u);
    uint64_t segstart_bytes = segstart_count * sizeof(int32_t);
    uint64_t segend_bytes = static_cast<uint64_t>(num_segments) * sizeof(int32_t);
    APICAddress keys_addr = apic_resolve_live_ptr(state, keys, kv_bytes);
    APICAddress values_addr = apic_resolve_live_ptr(state, values, kv_bytes);
    APICAddress segstart_addr = apic_resolve_live_ptr(state, segment_start, segstart_bytes);
    APICAddress segend_addr = apic_resolve_live_ptr(state, segment_end, segend_bytes);
    apic_record_segmented_sort(
        state, keys_addr.region_id, keys_addr.offset, values_addr.region_id, values_addr.offset,
        segstart_addr.region_id, segstart_addr.offset, segend_addr.region_id, segend_addr.offset,
        static_cast<uint32_t>(n), static_cast<uint32_t>(num_segments), dtype
    );
}

void wp_radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_INT32, sizeof(int32_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<int*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_uint_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_UINT32, sizeof(uint32_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<uint32_t*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_float_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_FLOAT32, sizeof(float));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<float*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_double_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size
)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_FLOAT64, sizeof(double));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<double*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_int64_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_INT64, sizeof(int64_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<int64_t*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_uint64_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size
)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_UINT64, sizeof(uint64_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<uint64_t*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void segmented_sort_reserve(void* context, int n, int num_segments, void** mem_out, size_t* size_out)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys;
    cub::DoubleBuffer<int> d_values;

    int* start_indices = NULL;
    int* end_indices = NULL;

    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());

    // compute temporary memory required
    size_t sort_temp_size;
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortPairs(
            NULL, sort_temp_size, d_keys, d_values, n, num_segments, start_indices, end_indices, 0, 32, stream
        )
    );

    RadixSortTemp& temp = g_radix_sort_temp_map[stream];

    if (sort_temp_size > temp.size) {
        wp_free_device(WP_CURRENT_CONTEXT, temp.mem);
        temp.mem = wp_alloc_device(WP_CURRENT_CONTEXT, sort_temp_size, "(native:sort)");
        temp.size = sort_temp_size;
    }

    if (mem_out)
        *mem_out = temp.mem;
    if (size_out)
        *size_out = temp.size;
}

// segment_start_indices and segment_end_indices are arrays of length num_segments, where segment_start_indices[i] is
// the index of the first element in the i-th segment and segment_end_indices[i] is the index after the last element in
// the i-th segment https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
void segmented_sort_pairs_device(
    void* context,
    float* keys,
    int* values,
    int n,
    int* segment_start_indices,
    int* segment_end_indices,
    int num_segments
)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<float> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    RadixSortTemp temp;
    segmented_sort_reserve(WP_CURRENT_CONTEXT, n, num_segments, &temp.mem, &temp.size);

    // sort
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp.mem, temp.size, d_keys, d_values, n, num_segments, segment_start_indices, segment_end_indices, 0, 32,
            (cudaStream_t)wp_cuda_stream_get_current()
        )
    );

    if (d_keys.Current() != keys)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(float) * n);

    if (d_values.Current() != values)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int) * n);
}

void wp_segmented_sort_pairs_float_device(
    uint64_t keys,
    uint64_t values,
    int n,
    uint64_t segment_start_indices,
    uint64_t segment_end_indices,
    int num_segments
)
{
    apic_capture_segmented_sort_device(
        keys, values, n, segment_start_indices, segment_end_indices, num_segments, APIC_TYPE_FLOAT32
    );
    segmented_sort_pairs_device(
        WP_CURRENT_CONTEXT, reinterpret_cast<float*>(keys), reinterpret_cast<int*>(values), n,
        reinterpret_cast<int*>(segment_start_indices), reinterpret_cast<int*>(segment_end_indices), num_segments
    );
}

// segment_indices is an array of length num_segments + 1, where segment_indices[i] is the index of the first element in
// the i-th segment The end of a segment is given by segment_indices[i+1]
// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedSort.html#a-simple-example
void segmented_sort_pairs_device(
    void* context, int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments
)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    RadixSortTemp temp;
    segmented_sort_reserve(WP_CURRENT_CONTEXT, n, num_segments, &temp.mem, &temp.size);

    // sort
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp.mem, temp.size, d_keys, d_values, n, num_segments, segment_start_indices, segment_end_indices, 0, 32,
            (cudaStream_t)wp_cuda_stream_get_current()
        )
    );

    if (d_keys.Current() != keys)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(int) * n);

    if (d_values.Current() != values)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int) * n);
}

void wp_segmented_sort_pairs_int_device(
    uint64_t keys,
    uint64_t values,
    int n,
    uint64_t segment_start_indices,
    uint64_t segment_end_indices,
    int num_segments
)
{
    apic_capture_segmented_sort_device(
        keys, values, n, segment_start_indices, segment_end_indices, num_segments, APIC_TYPE_INT32
    );
    segmented_sort_pairs_device(
        WP_CURRENT_CONTEXT, reinterpret_cast<int*>(keys), reinterpret_cast<int*>(values), n,
        reinterpret_cast<int*>(segment_start_indices), reinterpret_cast<int*>(segment_end_indices), num_segments
    );
}
