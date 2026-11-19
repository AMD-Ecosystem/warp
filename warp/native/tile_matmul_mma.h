// SPDX-License-Identifier: Apache-2.0

// Matrix-core GEMM for tile_matmul on AMD GPUs: MFMA on 64-lane wavefronts, WMMA on 32-lane
// wavefronts. partitioned_gemm::scalar_matmul hands a call over when tile_mma::supported holds and
// keeps everything else. The output tile is cut into 16x16 fragments; each fragment is computed by
// one whole wavefront, so every lane executes every matrix instruction and wavefronts write
// disjoint parts of C. Operands are read through the layouts' compile-time strides, so the
// transposed views of the adjoint need nothing extra. The instructions are selected per target
// feature through __has_builtin.

#pragma once

#if defined(__HIP_DEVICE_COMPILE__)

namespace wp {
namespace tile_mma {

// one 16x16 fragment instruction: Elem is the tile's element type
template <typename Elem> struct op {
    static constexpr bool available = false;
};

#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x4f32)
#define WP_TILE_MMA 1
constexpr int kWave = 64;

// lane l = 16 g + lo: operand v of lane l is A[lo][kpl * g + v] / B[kpl * g + v][lo];
// accumulator r of lane l is C[row(g, r)][lo] (measured on gfx950)
template <> struct op<float32> {
    static constexpr bool available = true;
    static constexpr int kpl = 1;
    typedef float Acc __attribute__((ext_vector_type(4)));
    typedef float In;
    static inline CUDA_CALLABLE In in(float32 x) { return x; }
    static inline CUDA_CALLABLE Acc mma(const In* a, const In* b, Acc c)
    {
        return __builtin_amdgcn_mfma_f32_16x16x4f32(a[0], b[0], c, 0, 0, 0);
    }
    static inline CUDA_CALLABLE int row(int g, int r) { return 4 * g + r; }
};

template <> struct op<float64> {
    static constexpr bool available = true;
    static constexpr int kpl = 1;
    typedef double Acc __attribute__((ext_vector_type(4)));
    typedef double In;
    static inline CUDA_CALLABLE In in(float64 x) { return x; }
    static inline CUDA_CALLABLE Acc mma(const In* a, const In* b, Acc c)
    {
        return __builtin_amdgcn_mfma_f64_16x16x4f64(a[0], b[0], c, 0, 0, 0);
    }
    static inline CUDA_CALLABLE int row(int g, int r) { return g + 4 * r; }
};

template <> struct op<half> {
    static constexpr bool available = true;
    static constexpr int kpl = 4;
    typedef float Acc __attribute__((ext_vector_type(4)));
    typedef _Float16 In;
    static inline CUDA_CALLABLE In in(half x) { return __builtin_bit_cast(_Float16, x.u); }
    static inline CUDA_CALLABLE Acc mma(const In* a, const In* b, Acc c)
    {
        typedef _Float16 V __attribute__((ext_vector_type(4)));
        V va = { a[0], a[1], a[2], a[3] }, vb = { b[0], b[1], b[2], b[3] };
        return __builtin_amdgcn_mfma_f32_16x16x16f16(va, vb, c, 0, 0, 0);
    }
    static inline CUDA_CALLABLE int row(int g, int r) { return 4 * g + r; }
};


#ifndef WP_NO_BFLOAT16
// bf16 inputs; tile_matmul accepts them only with a float32 output tile
template <> struct op<bfloat16> {
    static constexpr bool available = true;
    static constexpr int kpl = 4;
    typedef float Acc __attribute__((ext_vector_type(4)));
    typedef short In;
    static inline CUDA_CALLABLE In in(bfloat16 x) { return short(x.u); }
    static inline CUDA_CALLABLE Acc mma(const In* a, const In* b, Acc c)
    {
        typedef short V __attribute__((ext_vector_type(4)));
        V va = { a[0], a[1], a[2], a[3] }, vb = { b[0], b[1], b[2], b[3] };
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(va, vb, c, 0, 0, 0);
    }
    static inline CUDA_CALLABLE int row(int g, int r) { return 4 * g + r; }
};
#endif

#elif (defined(WP_TILE_WMMA) && WP_TILE_WMMA) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
// 32-lane WMMA. The gfx12 fp16 layout is measured and on; the gfx11 and gfx1250 paths stay off
// unless WP_TILE_WMMA is defined to 1 (each WMMA builtin exists on exactly one target family).
#define WP_TILE_MMA 1
constexpr int kWave = 32;

#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
// gfx11 fp16: both half-waves hold all 16 K values of their row (A) or column (B);
// accumulator r of lane l is C[2 r + l / 16][l % 16]
template <> struct op<half> {
    static constexpr bool available = true;
    static constexpr int kpl = 16;
    static constexpr bool k_split = false;
    typedef float Acc __attribute__((ext_vector_type(8)));
    typedef _Float16 In;
    static inline CUDA_CALLABLE In in(half x) { return __builtin_bit_cast(_Float16, x.u); }
    static inline CUDA_CALLABLE Acc mma(const In* a, const In* b, Acc c)
    {
        typedef _Float16 V __attribute__((ext_vector_type(16)));
        V va, vb;
        for (int v = 0; v < 16; ++v) {
            va[v] = a[v];
            vb[v] = b[v];
        }
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, vb, c);
    }
    static inline CUDA_CALLABLE int row(int g, int r) { return 2 * r + g; }
};
#endif

#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
// gfx12 fp16: lanes 0-15 hold K 0-7 and lanes 16-31 K 8-15 of their row (A) or column (B);
// accumulator r of lane l is C[8 (l / 16) + r][l % 16]
template <> struct op<half> {
    static constexpr bool available = true;
    static constexpr int kpl = 8;
    static constexpr bool k_split = true;
    typedef float Acc __attribute__((ext_vector_type(8)));
    typedef _Float16 In;
    static inline CUDA_CALLABLE In in(half x) { return __builtin_bit_cast(_Float16, x.u); }
    static inline CUDA_CALLABLE Acc mma(const In* a, const In* b, Acc c)
    {
        typedef _Float16 V __attribute__((ext_vector_type(8)));
        V va, vb;
        for (int v = 0; v < 8; ++v) {
            va[v] = a[v];
            vb[v] = b[v];
        }
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, vb, c);
    }
    static inline CUDA_CALLABLE int row(int g, int r) { return 8 * g + r; }
};
#endif

#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x4_f32)
// gfx1250 fp32: lanes 0-15 hold K 0-1 and lanes 16-31 K 2-3; accumulator as gfx12
template <> struct op<float32> {
    static constexpr bool available = true;
    static constexpr int kpl = 2;
    static constexpr bool k_split = true;
    typedef float Acc __attribute__((ext_vector_type(8)));
    typedef float In;
    static inline CUDA_CALLABLE In in(float32 x) { return x; }
    static inline CUDA_CALLABLE Acc mma(const In* a, const In* b, Acc c)
    {
        typedef float V __attribute__((ext_vector_type(2)));
        V va = { a[0], a[1] }, vb = { b[0], b[1] };
        return __builtin_amdgcn_wmma_f32_16x16x4_f32(false, va, false, vb, (short)0, c, false, false);
    }
    static inline CUDA_CALLABLE int row(int g, int r) { return 8 * g + r; }
};
#endif
#endif  // instruction families

#if defined(WP_TILE_MMA)

// lanes with the same lo cooperate on one row of A and one column of B; g selects their K slice
template <typename Op> inline CUDA_CALLABLE int k_offset(int g)
{
    if constexpr (kWave == 64)
        return Op::kpl * g;
    else if constexpr (Op::k_split)
        return Op::kpl * g;
    else
        return 0;  // gfx11 WMMA: both half-waves hold the whole K step
}

// K covered by one instruction: the lane groups that share a lo each hold kpl values of it
template <typename Op> constexpr int k_step()
{
    if constexpr (kWave == 64)
        return 4 * Op::kpl;
    else if constexpr (Op::k_split)
        return 2 * Op::kpl;
    else
        return Op::kpl;
}

template <
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename ElemA,
    typename ElemB,
    typename ElemC,
    typename T>
constexpr bool supported()
{
    // the instruction is chosen by the input type; inputs match the output type, except bf16
    // inputs, which tile_matmul pairs with a float32 output
    if constexpr (!is_same<ElemA, ElemB>::value || !is_same<ElemC, T>::value || !op<ElemA>::available)
        return false;
    else if constexpr (
        !is_same<ElemA, T>::value
        && !(is_same<T, float32>::value && !is_same<ElemA, half>::value && !is_same<ElemA, float64>::value)
    )
        return false;
    else
        return LayoutC::Shape::dim(0) % 16 == 0 && LayoutC::Shape::dim(1) % 16 == 0
            && LayoutA::Shape::dim(1) % k_step<op<ElemA>>() == 0 && WP_TILE_BLOCK_DIM % kWave == 0;
}

template <
    bool Accumulate,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename StorageA,
    typename StorageB,
    typename StorageC,
    typename T>
inline CUDA_CALLABLE void matmul(const StorageA& A, const StorageB& B, StorageC& C, T& alpha, T& beta)
{
    using ElemA = typename remove_reference<decltype(A.ptr[0])>::type;
    using Op = op<ElemA>;
    constexpr int M = LayoutC::Shape::dim(0);
    constexpr int N = LayoutC::Shape::dim(1);
    constexpr int K = LayoutA::Shape::dim(1);
    constexpr int sa0 = LayoutA::Stride::dim(0), sa1 = LayoutA::Stride::dim(1);
    constexpr int sb0 = LayoutB::Stride::dim(0), sb1 = LayoutB::Stride::dim(1);
    constexpr int sc0 = LayoutC::Stride::dim(0), sc1 = LayoutC::Stride::dim(1);
    constexpr int fragments_n = N / 16;
    constexpr int fragments = (M / 16) * fragments_n;
    constexpr int waves = WP_TILE_BLOCK_DIM / kWave;
    constexpr int acc_size = kWave == 64 ? 4 : 8;
    // each wavefront computes a row of `group` fragments that share the A operand of every K step;
    // the widest group that divides the fragment row and still leaves work for every wavefront
    constexpr int group = (fragments_n % 4 == 0 && fragments / 4 >= waves) ? 4
        : (fragments_n % 2 == 0 && fragments / 2 >= waves)                 ? 2
                                                                           : 1;
    constexpr int groups = fragments / group;

    const int lane = WP_TILE_THREAD_IDX % kWave;
    const int lo = lane % 16;
    const int g = lane / 16;
    const int kofs = k_offset<Op>(g);

    for (int f = WP_TILE_THREAD_IDX / kWave; f < groups; f += waves) {
        const int fi = (f * group / fragments_n) * 16;
        const int fj = (f * group % fragments_n) * 16;
        const ElemA* a_row = &A.ptr[(fi + lo) * sa0];
        const ElemA* b_col = &B.ptr[(fj + lo) * sb1];

        typename Op::Acc acc[group] = {};
        for (int k0 = 0; k0 < K; k0 += k_step<Op>()) {
            typename Op::In a[Op::kpl];
            WP_PRAGMA_UNROLL
            for (int v = 0; v < Op::kpl; ++v)
                a[v] = Op::in(a_row[(k0 + kofs + v) * sa1]);
            WP_PRAGMA_UNROLL
            for (int h = 0; h < group; ++h) {
                typename Op::In b[Op::kpl];
                WP_PRAGMA_UNROLL
                for (int v = 0; v < Op::kpl; ++v)
                    b[v] = Op::in(b_col[16 * h * sb1 + (k0 + kofs + v) * sb0]);
                acc[h] = Op::mma(a, b, acc[h]);
            }
        }

        WP_PRAGMA_UNROLL
        for (int h = 0; h < group; ++h) {
            WP_PRAGMA_UNROLL
            for (int r = 0; r < acc_size; ++r) {
                const int idx = (fi + Op::row(g, r)) * sc0 + (fj + 16 * h + lo) * sc1;
                if constexpr (Accumulate)
                    C.ptr[idx] = alpha * T(acc[h][r]) + beta * C.ptr[idx];
                else
                    C.ptr[idx] = alpha * T(acc[h][r]);
            }
        }
    }
}

#endif  // WP_TILE_MMA

}  // namespace tile_mma
}  // namespace wp

#endif  // __HIP_DEVICE_COMPILE__
