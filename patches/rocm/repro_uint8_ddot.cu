// SPDX-License-Identifier: Apache-2.0
#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#if !defined(__HIPCC__)
#define WP_NO_CRT
#endif
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so the debugger attributes the stop to the generated source line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#elif defined(__HIPCC__) && !defined(_MSC_VER)
#define __debugbreak() __builtin_trap()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif



extern "C" __global__ void check_mat_dot_uint8_8a5cc2b4_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::mat_t<2, 2, wp::uint8>> var_s2,
    wp::array_t<wp::mat_t<4, 4, wp::uint8>> var_s4,
    wp::array_t<wp::mat_t<2, 2, wp::uint8>> var_v2,
    wp::array_t<wp::mat_t<4, 4, wp::uint8>> var_v4,
    wp::array_t<wp::uint8> var_dot2,
    wp::array_t<wp::uint8> var_dot4)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        const wp::int32 var_0 = 2;
        wp::uint8 var_1;
        const wp::int32 var_2 = 0;
        wp::mat_t<2, 2, wp::uint8>* var_3;
        const wp::int32 var_4 = 0;
        wp::mat_t<2, 2, wp::uint8>* var_5;
        wp::uint8 var_6;
        wp::mat_t<2, 2, wp::uint8> var_7;
        wp::mat_t<2, 2, wp::uint8> var_8;
        wp::uint8 var_9;
        const wp::int32 var_10 = 0;
        const wp::int32 var_11 = 2;
        wp::uint8 var_12;
        const wp::int32 var_13 = 0;
        wp::mat_t<4, 4, wp::uint8>* var_14;
        const wp::int32 var_15 = 0;
        wp::mat_t<4, 4, wp::uint8>* var_16;
        wp::uint8 var_17;
        wp::mat_t<4, 4, wp::uint8> var_18;
        wp::mat_t<4, 4, wp::uint8> var_19;
        wp::uint8 var_20;
        const wp::int32 var_21 = 0;
        //---------
        // forward
        // def check_mat_dot(                                                                     <L 545>
        // dot2[0] = wptype(2) * wp.ddot(v2[0], s2[0])                                            <L 554>
        var_1 = wp::uint8(var_0);
        var_3 = wp::address(var_v2, var_2);
        var_5 = wp::address(var_s2, var_4);
        var_7 = wp::load(var_3);
        var_8 = wp::load(var_5);
        var_6 = wp::ddot(var_7, var_8);
        var_9 = wp::mul(var_1, var_6);
        wp::array_store(var_dot2, var_10, var_9);
        // dot4[0] = wptype(2) * wp.ddot(v4[0], s4[0])                                            <L 555>
        var_12 = wp::uint8(var_11);
        var_14 = wp::address(var_v4, var_13);
        var_16 = wp::address(var_s4, var_15);
        var_18 = wp::load(var_14);
        var_19 = wp::load(var_16);
        var_17 = wp::ddot(var_18, var_19);
        var_20 = wp::mul(var_12, var_17);
        wp::array_store(var_dot4, var_21, var_20);
    }
}
