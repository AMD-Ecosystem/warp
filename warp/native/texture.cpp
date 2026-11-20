// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "error.h"
#include "texture.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// ============================================================================
// CPU Host Texture Implementation
// ============================================================================

using wp::get_texture_bytes_per_channel;
using wp::Texture;

uint64_t wp_texture_create_host(
    int ndim,
    int* shape,
    int num_channels,
    int dtype,
    int filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    void** data_ptr_out
)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture dimensions must be positive integers, got %d at dimension %d", shape[i], i
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    Texture* tex = nullptr;
    if (ndim == 1) {
        tex = new (wp_alloc_host(sizeof(Texture), "(native:texture)"))
            Texture(shape[0], num_channels, dtype, filter_mode, address_modes[0], use_normalized_coords);
    } else if (ndim == 2) {
        tex = new (wp_alloc_host(sizeof(Texture), "(native:texture)")) Texture(
            shape[0], shape[1], num_channels, dtype, filter_mode, address_modes[0], address_modes[1],
            use_normalized_coords
        );
    } else {
        tex = new (wp_alloc_host(sizeof(Texture), "(native:texture)")) Texture(
            shape[0], shape[1], shape[2], num_channels, dtype, filter_mode, address_modes[0], address_modes[1],
            address_modes[2], use_normalized_coords
        );
    }

    if (data_ptr_out) {
        *data_ptr_out = tex->data;
    }

    return reinterpret_cast<uint64_t>(tex);
}

void wp_texture_destroy_host(uint64_t tex_handle)
{
    Texture* tex = (Texture*)tex_handle;
    tex->~Texture();
    wp_free_host(tex);
}

#if WP_ENABLE_CUDA

// ============================================================================
// CUDA Device Texture Implementation
// ============================================================================


#include "cuda_util.h"

#if defined(__HIP_PLATFORM_AMD__)
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

// hipArrayCreate / hipArray3DCreate return hipErrorNotSupported (801) on current ROCm,
// so a HIP texture "array" is a tightly packed device buffer, and each texture object
// is a device-side TextureDesc that points at it. Sampling reuses the CPU sampler.
// The array handle is the pixel buffer address; the texture handle is the descriptor address.

static_assert(std::is_trivially_copyable<wp::TextureDesc>::value, "TextureDesc is memcpy'd to the device");

struct HipLinearArray {
    void* pixels = nullptr;
    int ndim = 0;
    int width = 0;
    int height = 0;
    int depth = 0;
    int num_channels = 0;
    int dtype = 0;
    size_t nbytes = 0;
};

// Guards both registries. Held while a copy is enqueued so a concurrent destroy cannot free
// the buffer between lookup and cudaMemcpyAsync.
static std::mutex g_hip_tex_mutex;
static std::unordered_map<uint64_t, HipLinearArray> g_hip_arrays;
static std::unordered_set<uint64_t> g_hip_tex_objects;  // descriptor handles

static const HipLinearArray* hip_array_find_locked(uint64_t handle)
{
    auto it = g_hip_arrays.find(handle);
    return it == g_hip_arrays.end() ? nullptr : &it->second;
}

static uint64_t hip_array_create(int ndim, int* shape, int num_channels, int dtype, bool surface_access)
{
    if (surface_access) {
        wp::set_error_string(
            "Warp error: surface textures are not supported on HIP/ROCm (hipArray creation is not supported)"
        );
        return 0;
    }

    int bytes_per_channel = wp::get_texture_bytes_per_channel(dtype);
    if (bytes_per_channel == 0) {
        wp::set_error_string("Warp error: Unsupported texture dtype %d", dtype);
        return 0;
    }

    HipLinearArray rec;
    rec.ndim = ndim;
    rec.width = shape[0];
    rec.height = ndim > 1 ? shape[1] : 0;
    rec.depth = ndim > 2 ? shape[2] : 0;
    rec.num_channels = num_channels;
    rec.dtype = dtype;
    rec.nbytes = size_t(rec.width) * size_t(ndim > 1 ? rec.height : 1) * size_t(ndim > 2 ? rec.depth : 1)
        * size_t(num_channels) * size_t(bytes_per_channel);

    cudaError_t err = cudaMalloc(&rec.pixels, rec.nbytes);
    if (err != cudaSuccess) {
        wp::set_error_string("Warp error: failed to allocate HIP texture buffer (%s)", cudaGetErrorString(err));
        return 0;
    }
    // The legacy null stream does not order against non-blocking user streams, so finish
    // the clear before a later copy_from on such a stream can run.
    err = cudaMemset(rec.pixels, 0, rec.nbytes);
    if (err == cudaSuccess)
        err = cudaStreamSynchronize(nullptr);
    if (err != cudaSuccess) {
        (void)cudaFree(rec.pixels);
        wp::set_error_string("Warp error: failed to clear HIP texture buffer (%s)", cudaGetErrorString(err));
        return 0;
    }

    uint64_t handle = reinterpret_cast<uint64_t>(rec.pixels);
    std::lock_guard<std::mutex> lock(g_hip_tex_mutex);
    g_hip_arrays.emplace(handle, rec);
    return handle;
}

// Returns false if the handle is not a HIP linear array.
static bool hip_array_destroy(uint64_t handle)
{
    std::lock_guard<std::mutex> lock(g_hip_tex_mutex);
    auto it = g_hip_arrays.find(handle);
    if (it == g_hip_arrays.end())
        return false;
    (void)cudaFree(it->second.pixels);
    g_hip_arrays.erase(it);
    return true;
}

// Returns the descriptor handle, 0 on error, or leaves *handled false if not ours.
static uint64_t hip_texture_object_create(
    uint64_t array_handle, int filter_mode, int* address_modes, bool use_normalized_coords, bool* handled
)
{
    std::lock_guard<std::mutex> lock(g_hip_tex_mutex);
    const HipLinearArray* arr = hip_array_find_locked(array_handle);
    *handled = arr != nullptr;
    if (!arr)
        return 0;

    wp::TextureDesc desc = {};
    desc.data = arr->pixels;
    desc.width = arr->width;
    desc.height = arr->height;
    desc.depth = arr->depth;
    desc.num_channels = arr->num_channels;
    desc.dtype = arr->dtype;
    desc.filter_mode = filter_mode;
    desc.address_mode_u = address_modes[0];
    desc.address_mode_v = arr->ndim > 1 ? address_modes[1] : WP_TEXTURE_ADDRESS_CLAMP;
    desc.address_mode_w = arr->ndim > 2 ? address_modes[2] : WP_TEXTURE_ADDRESS_CLAMP;
    desc.use_normalized_coords = use_normalized_coords ? 1 : 0;

    void* device_desc = nullptr;
    cudaError_t err = cudaMalloc(&device_desc, sizeof(desc));
    if (err != cudaSuccess) {
        wp::set_error_string("Warp error: failed to allocate HIP texture descriptor (%s)", cudaGetErrorString(err));
        return 0;
    }
    err = cudaMemcpy(device_desc, &desc, sizeof(desc), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        (void)cudaFree(device_desc);
        wp::set_error_string("Warp error: failed to upload HIP texture descriptor (%s)", cudaGetErrorString(err));
        return 0;
    }

    uint64_t handle = reinterpret_cast<uint64_t>(device_desc);
    g_hip_tex_objects.insert(handle);
    return handle;
}

// Returns false if the handle is not a HIP texture object.
static bool hip_texture_object_destroy(uint64_t tex_handle)
{
    std::lock_guard<std::mutex> lock(g_hip_tex_mutex);
    auto it = g_hip_tex_objects.find(tex_handle);
    if (it == g_hip_tex_objects.end())
        return false;
    (void)cudaFree(reinterpret_cast<void*>(tex_handle));
    g_hip_tex_objects.erase(it);
    return true;
}

// 1 = copied, 0 = neither side is a HIP linear array, -1 = error
static int hip_texture_copy_linear(
    unsigned width_bytes,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    void* stream
)
{
    // Python passes CUDA memory-type values (array = 3). HIP's hipMemoryTypeArray is 10,
    // and 3 is hipMemoryTypeManaged, so compare against the CUDA numbers explicitly.
    constexpr int kMemHost = 0x01;
    constexpr int kMemDevice = 0x02;
    constexpr int kMemArray = 0x03;

    std::lock_guard<std::mutex> lock(g_hip_tex_mutex);
    const HipLinearArray* dst_arr = dst_memory_type == kMemArray ? hip_array_find_locked(dst_handle) : nullptr;
    const HipLinearArray* src_arr = src_memory_type == kMemArray ? hip_array_find_locked(src_handle) : nullptr;
    if (!dst_arr && !src_arr)
        return 0;

    if ((dst_memory_type == kMemArray && !dst_arr) || (src_memory_type == kMemArray && !src_arr)) {
        wp::set_error_string("Warp error: HIP texture copies to or from a hipArray are not supported");
        return -1;
    }

    // Linear storage is tightly packed, so the linear side of the copy must be too.
    if ((!dst_arr && dst_pitch != 0 && dst_pitch != width_bytes)
        || (!src_arr && src_pitch != 0 && src_pitch != width_bytes)) {
        wp::set_error_string("Warp error: pitched copies are not supported for HIP textures");
        return -1;
    }

    size_t nbytes = size_t(width_bytes) * size_t(height == 0 ? 1 : height) * size_t(depth == 0 ? 1 : depth);
    if ((dst_arr && nbytes > dst_arr->nbytes) || (src_arr && nbytes > src_arr->nbytes)) {
        wp::set_error_string(
            "Warp error: HIP texture copy of %zu bytes exceeds the texture size (%zu bytes)", nbytes,
            dst_arr ? dst_arr->nbytes : src_arr->nbytes
        );
        return -1;
    }

    void* dst_ptr;
    if (dst_arr)
        dst_ptr = dst_arr->pixels;
    else if (dst_memory_type == kMemDevice || dst_memory_type == kMemHost)
        dst_ptr = reinterpret_cast<void*>(dst_handle);
    else {
        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
        return -1;
    }

    void* src_ptr;
    if (src_arr)
        src_ptr = src_arr->pixels;
    else if (src_memory_type == kMemDevice || src_memory_type == kMemHost)
        src_ptr = reinterpret_cast<void*>(src_handle);
    else {
        wp::set_error_string("Invalid source memory type %d", src_memory_type);
        return -1;
    }

    bool src_host = src_memory_type == kMemHost;
    bool dst_host = dst_memory_type == kMemHost;
    cudaMemcpyKind kind = src_host ? cudaMemcpyHostToDevice
        : dst_host                 ? cudaMemcpyDeviceToHost
                                   : cudaMemcpyDeviceToDevice;

    cudaError_t err = cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, kind, static_cast<cudaStream_t>(stream));
    if (err != cudaSuccess) {
        wp::set_error_string("Warp error: HIP texture copy failed (%s)", cudaGetErrorString(err));
        return -1;
    }
    return 1;
}
#endif

// Helper function to get CUDA array format from dtype
static CUarray_format get_cuda_format(int dtype)
{
    switch (dtype) {
    case WP_TEXTURE_DTYPE_UINT8:
        return CU_AD_FORMAT_UNSIGNED_INT8;
    case WP_TEXTURE_DTYPE_UINT16:
        return CU_AD_FORMAT_UNSIGNED_INT16;
    case WP_TEXTURE_DTYPE_UINT32:
        return CU_AD_FORMAT_UNSIGNED_INT32;
    case WP_TEXTURE_DTYPE_INT8:
        return CU_AD_FORMAT_SIGNED_INT8;
    case WP_TEXTURE_DTYPE_INT16:
        return CU_AD_FORMAT_SIGNED_INT16;
    case WP_TEXTURE_DTYPE_INT32:
        return CU_AD_FORMAT_SIGNED_INT32;
    case WP_TEXTURE_DTYPE_FLOAT16:
        return CU_AD_FORMAT_HALF;
    case WP_TEXTURE_DTYPE_FLOAT32:
    default:
        return CU_AD_FORMAT_FLOAT;
    }
}

// Helper function to convert address mode int to CUDA address mode
static CUaddress_mode get_cuda_address_mode(int address_mode)
{
    switch (address_mode) {
    case 0:
        return CU_TR_ADDRESS_MODE_WRAP;
    case 1:
        return CU_TR_ADDRESS_MODE_CLAMP;
    case 2:
        return CU_TR_ADDRESS_MODE_MIRROR;
    case 3:
        return CU_TR_ADDRESS_MODE_BORDER;
    default:
        return CU_TR_ADDRESS_MODE_CLAMP;
    }
}

uint64_t wp_texture_create_device(void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access)
{
    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Warp error: Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            wp::set_error_string(
                "Warp error: Texture dimensions must be positive integers, got %d at dimension %d", shape[i], i
            );
            return 0;
        }
    }

    if (num_channels != 1 && num_channels != 2 && num_channels != 4) {
        wp::set_error_string("Warp error: Textures support 1, 2, or 4 channels, got (%d)", num_channels);
        return 0;
    }

    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__)
    return hip_array_create(ndim, shape, num_channels, dtype, surface_access);
#endif

    size_t width = shape[0];
    size_t height = ndim > 1 ? shape[1] : 0;
    size_t depth = ndim > 2 ? shape[2] : 0;

    // determine the CUDA array format
    CUarray_format format = get_cuda_format(dtype);

    // create array descriptor
    CUDA_ARRAY3D_DESCRIPTOR arr_desc = {};
    arr_desc.Width = width;
    arr_desc.Height = height;
    arr_desc.Depth = depth;
    arr_desc.Format = format;
    arr_desc.NumChannels = num_channels;
    arr_desc.Flags = surface_access ? CUDA_ARRAY3D_SURFACE_LDST : 0;

    // create the CUDA array
    CUarray cuda_array = NULL;
    check_cu(cuArray3DCreate_f(&cuda_array, &arr_desc));

    return reinterpret_cast<uint64_t>(cuda_array);
}

void wp_texture_destroy_device(void* context, uint64_t array_handle)
{
    ContextGuard guard(context);
#if defined(__HIP_PLATFORM_AMD__)
    if (hip_array_destroy(array_handle))
        return;
#endif
    if (array_handle != 0) {
        check_cu(cuArrayDestroy_f((CUarray)array_handle));
    }
}

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__)
    {
        int hip_copy = hip_texture_copy_linear(
            width_bytes, height, depth, dst_memory_type, dst_handle, dst_pitch, src_memory_type, src_handle,
            src_pitch, stream
        );
        if (hip_copy != 0)
            return hip_copy > 0;
    }
#endif

    CUstream cuda_stream = static_cast<CUstream>(stream);

    CUDA_MEMCPY3D copy_params = {};
    copy_params.WidthInBytes = width_bytes;
    copy_params.Height = height;
    copy_params.Depth = depth;

    copy_params.dstMemoryType = static_cast<CUmemorytype>(dst_memory_type);
    if (dst_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.dstHost = reinterpret_cast<void*>(dst_handle);
    } else if (dst_memory_type == CU_MEMORYTYPE_DEVICE) {
        // CUDA: CUdeviceptr is `unsigned long long` (static_cast OK).
        // HIP:  CUdeviceptr is `void*` (only reinterpret_cast is valid).
#if defined(__HIP_PLATFORM_AMD__)
        copy_params.dstDevice = reinterpret_cast<CUdeviceptr>(dst_handle);
#else
        copy_params.dstDevice = static_cast<CUdeviceptr>(dst_handle);
#endif
    } else if (dst_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.dstArray = reinterpret_cast<CUarray>(dst_handle);
    } else {
        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
        return false;
    }
    copy_params.dstPitch = dst_pitch;
    copy_params.dstHeight = dst_height;

    copy_params.srcMemoryType = static_cast<CUmemorytype>(src_memory_type);
    if (src_memory_type == CU_MEMORYTYPE_HOST) {
        copy_params.srcHost = reinterpret_cast<void*>(src_handle);
    } else if (src_memory_type == CU_MEMORYTYPE_DEVICE) {
#if defined(__HIP_PLATFORM_AMD__)
        copy_params.srcDevice = reinterpret_cast<CUdeviceptr>(src_handle);
#else
        copy_params.srcDevice = static_cast<CUdeviceptr>(src_handle);
#endif
    } else if (src_memory_type == CU_MEMORYTYPE_ARRAY) {
        copy_params.srcArray = reinterpret_cast<CUarray>(src_handle);
    } else {
        wp::set_error_string("Invalid source memory type %d", src_memory_type);
        return false;
    }
    copy_params.srcPitch = src_pitch;
    copy_params.srcHeight = src_height;

    return check_cu(cuMemcpy3DAsync_f(&copy_params, cuda_stream));
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    if (!array_handle || !desc_out) {
        wp::set_error_string("Warp error: NULL array handle");
        return false;
    }

    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__)
    {
        std::lock_guard<std::mutex> lock(g_hip_tex_mutex);
        if (const HipLinearArray* arr = hip_array_find_locked(array_handle)) {
            desc_out->dtype = arr->dtype;
            desc_out->shape[0] = arr->width;
            desc_out->shape[1] = arr->height;
            desc_out->shape[2] = arr->depth;
            desc_out->ndim = arr->ndim;
            desc_out->num_channels = arr->num_channels;
            return true;
        }
    }
#endif

    CUarray cuda_array = reinterpret_cast<CUarray>(array_handle);

    CUDA_ARRAY3D_DESCRIPTOR desc;
    if (!check_cu(cuArray3DGetDescriptor_f(&desc, cuda_array)))
        return false;

    switch (desc.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT8;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT16;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_UINT32;
        break;
    case CU_AD_FORMAT_SIGNED_INT8:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT8;
        break;
    case CU_AD_FORMAT_SIGNED_INT16:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT16;
        break;
    case CU_AD_FORMAT_SIGNED_INT32:
        desc_out->dtype = WP_TEXTURE_DTYPE_INT32;
        break;
    case CU_AD_FORMAT_HALF:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT16;
        break;
    case CU_AD_FORMAT_FLOAT:
        desc_out->dtype = WP_TEXTURE_DTYPE_FLOAT32;
        break;
    default:
        wp::set_error_string("Warp error: Unsupported texture format");
        return false;
    }

    desc_out->shape[0] = int32_t(desc.Width);
    desc_out->shape[1] = int32_t(desc.Height);
    desc_out->shape[2] = int32_t(desc.Depth);

    if (desc.Depth == 0) {
        if (desc.Height == 0)
            desc_out->ndim = 1;
        else
            desc_out->ndim = 2;
    } else {
        desc_out->ndim = 3;
    }

    // TODO: desc.Flags?

    desc_out->num_channels = int32_t(desc.NumChannels);

    return true;
}

uint64_t wp_texture_object_create_device(
    void* context, uint64_t array_handle, int ndim, int filter_mode, int* address_modes, bool use_normalized_coords
)
{
    if (!array_handle) {
        wp::set_error_string("Null texture array handle");
        return 0;
    }

    if (ndim < 1 || ndim > 3) {
        wp::set_error_string("Number of texture dimensions must be 1, 2, or 3, got %d", ndim);
        return 0;
    }

    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__)
    {
        bool handled = false;
        uint64_t tex = hip_texture_object_create(array_handle, filter_mode, address_modes, use_normalized_coords, &handled);
        if (handled)
            return tex;
    }
#endif

    // Create resource descriptor
    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = reinterpret_cast<CUarray>(array_handle);
    res_desc.flags = 0;

    // Create texture descriptor
    CUDA_TEXTURE_DESC tex_desc = {};

    // Per-axis address modes
    for (int i = 0; i < 3; i++) {
        if (i < ndim)
            tex_desc.addressMode[i] = get_cuda_address_mode(address_modes[i]);
        else
            tex_desc.addressMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
    }

    // Filter mode: 0=nearest, 1=linear
    tex_desc.filterMode = (filter_mode == 0) ? CU_TR_FILTER_MODE_POINT : CU_TR_FILTER_MODE_LINEAR;

    // Coordinate mode: normalized [0,1] or texel space [0,width/height]
    // For uint8/uint16 textures, CUDA automatically normalizes values to [0,1] when sampled
    // (since CU_TRSF_READ_AS_INTEGER is NOT set). Float32 textures return values as-is.
    tex_desc.flags = use_normalized_coords ? CU_TRSF_NORMALIZED_COORDINATES : 0;

    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = 0;

    // Create texture object
    CUtexObject tex_object = 0;
    check_cu(cuTexObjectCreate_f(&tex_object, &res_desc, &tex_desc, nullptr));

    // CUDA: CUtexObject is `unsigned long long`. HIP: CUtexObject is `__hip_texture*`.
#if defined(__HIP_PLATFORM_AMD__)
    return reinterpret_cast<uint64_t>(tex_object);
#else
    return static_cast<uint64_t>(tex_object);
#endif
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle)
{
    ContextGuard guard(context);

#if defined(__HIP_PLATFORM_AMD__)
    if (hip_texture_object_destroy(tex_handle))
        return;
#endif

    if (tex_handle != 0) {
        check_cu(cuTexObjectDestroy_f((CUtexObject)tex_handle));
    }
}

uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    ContextGuard guard(context);

    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = reinterpret_cast<cudaArray_t>(array_handle);

    cudaSurfaceObject_t surface = 0;
    check_cuda(cudaCreateSurfaceObject(&surface, &desc));

    // CUDA: cudaSurfaceObject_t is `unsigned long long`. HIP: it is `__hip_surface*`.
#if defined(__HIP_PLATFORM_AMD__)
    return reinterpret_cast<uint64_t>(surface);
#else
    return static_cast<uint64_t>(surface);
#endif
}

void wp_surface_object_destroy_device(void* context, uint64_t surface_handle)
{
    if (!surface_handle)
        return;

    ContextGuard guard(context);
#if defined(__HIP_PLATFORM_AMD__)
    check_cuda(cudaDestroySurfaceObject(reinterpret_cast<cudaSurfaceObject_t>(surface_handle)));
#else
    check_cuda(cudaDestroySurfaceObject(static_cast<cudaSurfaceObject_t>(surface_handle)));
#endif
}

#else  // WP_ENABLE_CUDA

// Stub implementations for non-CUDA builds

uint64_t wp_texture_create_device(void* context, int ndim, int* shape, int num_channels, int dtype, bool surface_access)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_destroy_device(void* context, uint64_t array_handle) { }

bool wp_texture_copy_device(
    void* context,
    unsigned width_bytes,
    unsigned height,
    unsigned depth,
    int dst_memory_type,
    uint64_t dst_handle,
    unsigned dst_pitch,
    unsigned dst_height,
    int src_memory_type,
    uint64_t src_handle,
    unsigned src_pitch,
    unsigned src_height,
    void* stream
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

bool wp_texture_descriptor_from_cuda_array(void* context, uint64_t array_handle, wp::cuda_array_desc_t* desc_out)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return false;
}

uint64_t wp_texture_object_create_device(
    void* context, uint64_t array_handle, int ndim, int filter_mode, int* address_modes, bool use_normalized_coords
)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

void wp_texture_object_destroy_device(void* context, uint64_t tex_handle) { }

WP_API uint64_t wp_surface_object_create_device(void* context, uint64_t array_handle)
{
    wp::set_error_string("Warp error: CUDA not enabled");
    return 0;
}

WP_API void wp_surface_object_destroy_device(void* context, uint64_t surface_handle) { }

#endif  // WP_ENABLE_CUDA
