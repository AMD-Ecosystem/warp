// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1

#include "warp.h"

#include "bvh.h"
#include "cuda_util.h"
#include "sort.h"

#include <algorithm>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#include "hip_util.h"

#include <hipcub/hipcub.hpp>
// cuBQL's builder headers (builder/cuda/builder_common.h) alias cub onto hipcub
// with `namespace cub { using namespace hipcub; }`. A namespace *alias*
// (`namespace cub = hipcub;`) cannot coexist with that namespace *definition* in
// the same translation unit, so use the reopen-namespace form here too. Both
// forms make Warp's own `cub::` uses (BlockReduce, DeviceRadixSort) resolve to
// hipcub, and reopening the namespace with duplicate using-directives is legal.
namespace cub { using namespace hipcub; }
#else
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#define THRUST_IGNORE_CUB_VERSION_CHECK
#define REORDER_HOST_TREE

// CUB must be included before cuBQL. cuBQL's math/common.h includes <stdexcept>,
// which causes CCCL's _CCCL_HAS_EXCEPTIONS() to be true when typeid.h is later
// pulled in by CUB. This makes __throw_out_of_range non-constexpr, breaking a
// static_assert in typeid.h on GCC < 12 (which lacks P2448R2 relaxed constexpr).
// On HIP hipcub is already included above; cuBQL now has a HIP port so it is
// pulled in on both toolchains unless explicitly disabled with WP_DISABLE_CUBQL.
#if !defined(__HIP_PLATFORM_AMD__)
#include <cub/cub.cuh>
#endif

#ifndef WP_DISABLE_CUBQL
#include "cuBQL/bvh.h"
#endif


extern CUcontext get_current_context();

namespace wp {
void bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, BVH& bvh, int leaf_size);
void bvh_destroy_host(BVH& bvh);

__global__ void memset_kernel(int* dest, int value, size_t n)
{
    const size_t tid
        = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);

    if (tid < n) {
        dest[tid] = value;
    }
}

// for LBVH: this will start with some muted leaf nodes, but that is okay, we can still trace up because there parents
// information is still valid the only thing worth mentioning is that when the parent leaf node is also a leaf node, we
// need to recompute its bounds, since their child information are lost for a compact tree such as those from SAH or
// Median constructor, there is no muted leaf nodes
__global__ void bvh_refit_kernel(
    int n,
    const int* __restrict__ parents,
    int* __restrict__ child_count,
    const int* __restrict__ primitive_indices,
    BVHPackedNodeHalf* __restrict__ node_lowers,
    BVHPackedNodeHalf* __restrict__ node_uppers,
    const vec3* __restrict__ item_lowers,
    const vec3* __restrict__ item_uppers
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        bool leaf = node_lowers[index].b;
        int parent = parents[index];

        if (leaf) {
            BVHPackedNodeHalf& lower = node_lowers[index];
            BVHPackedNodeHalf& upper = node_uppers[index];
            // update the leaf node

            // only need to compute bound when this is a valid leaf node
            if (!node_lowers[parent].b) {
                const int start = lower.i;
                const int end = upper.i;

                bounds3 bound;
                for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                    const int primitive = primitive_indices[primitive_counter];
                    bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                }
                (vec3&)lower = bound.lower;
                (vec3&)upper = bound.upper;
            }
        } else {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;) {
            parent = parents[index];
            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();

            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (finished == 1) {
                BVHPackedNodeHalf& parent_lower = node_lowers[parent];
                BVHPackedNodeHalf& parent_upper = node_uppers[parent];
                if (parent_lower.b)
                // a packed leaf node can still be a parent in LBVH, we need to recompute its bounds
                // since we've lost its left and right child node index in the muting process
                {
                    // update the leaf node
                    int parent_parent = parents[parent];
                    ;

                    // only need to compute bound when this is a valid leaf node
                    if (!node_lowers[parent_parent].b) {
                        const int start = parent_lower.i;
                        const int end = parent_upper.i;
                        bounds3 bound;
                        for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                            const int primitive = primitive_indices[primitive_counter];
                            bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                        }

                        (vec3&)parent_lower = bound.lower;
                        (vec3&)parent_upper = bound.upper;
                    }
                } else {
                    const int left_child = parent_lower.i;
                    const int right_child = parent_upper.i;

                    vec3 left_lower = (vec3&)(node_lowers[left_child]);
                    vec3 left_upper = (vec3&)(node_uppers[left_child]);
                    vec3 right_lower = (vec3&)(node_lowers[right_child]);
                    vec3 right_upper = (vec3&)(node_uppers[right_child]);

                    // union of child bounds
                    vec3 lower = min(left_lower, right_lower);
                    vec3 upper = max(left_upper, right_upper);

                    // write new BVH nodes
                    (vec3&)parent_lower = lower;
                    (vec3&)parent_upper = upper;
                }
                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf
//
// All scratch memory is now pool-allocated inside build() – no persistent
// members, constructor or destructor are needed.  This eliminates several
// individual hipMalloc / hipFree calls and the
// mid-build device synchronisation that was required to free the
// intermediate block-bounds buffers.
#if defined(__HIP_PLATFORM_AMD__)
class LinearBVHBuilderGPU {
public:
    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each triangle
    void build(
        BVH& bvh,
        const vec3* item_lowers,
        const vec3* item_uppers,
        int num_items,
        bounds3* total_bounds,
        int* item_groups
    );
};

#else
class LinearBVHBuilderGPU {
public:
    LinearBVHBuilderGPU();
    ~LinearBVHBuilderGPU();

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each triangle
    void build(
        BVH& bvh,
        const vec3* item_lowers,
        const vec3* item_uppers,
        int num_items,
        bounds3* total_bounds,
        int* item_groups
    );
}
#endif

__global__ void compute_morton_codes(
    const vec3* __restrict__ item_lowers,
    const vec3* __restrict__ item_uppers,
    int n,
    const vec3* grid_lower,
    const vec3* grid_inv_edges,
    int* __restrict__ indices,
    uint64_t* __restrict__ keys,
    const int* __restrict__ item_groups
)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        vec3 lower = item_lowers[index];
        vec3 upper = item_uppers[index];

        vec3 center = 0.5f * (lower + upper);

        vec3 local = cw_mul((center - grid_lower[0]), grid_inv_edges[0]);

        // 10-bit Morton codes stored in lower 30bits (1024^3 effective resolution)
        // Group stored in upper 32 bits
        uint64_t morton_code = static_cast<uint64_t>(morton3<1024>(local[0], local[1], local[2]));
        const uint64_t group = item_groups ? static_cast<uint32_t>(item_groups[index]) : 0u;
        const uint64_t key = (group << 32) | morton_code;

        indices[index] = index;
        keys[index] = key;
    }
}

// compute a distance metric between adjacent keys; larger across groups
// Using raw XOR magnitude preserves the original builder's assumptions
__global__ void compute_key_deltas(const uint64_t* __restrict__ keys, int* __restrict__ deltas, int n)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        const uint64_t diff = keys[index] ^ keys[index + 1];
        deltas[index] = (diff == 0) ? 64 : __clzll(diff);
    }
}

// Deterministic Karras-style LBVH topology build (avoids atomics/threadfence issues on HIP).
// Builds internal nodes [n..2n-2] from sorted keys[0..n-1] and writes:
// - node child pointers into bvh node_lowers/node_uppers (as indices)
// - node_parents for all nodes
// - range_lefts/range_rights for all nodes (inclusive range in sorted order)
// - root index
static __device__ __forceinline__ int delta_prefix(const uint64_t* __restrict__ keys, int n, int i, int j)
{
    if (j < 0 || j >= n)
        return -1;
    const uint64_t diff = keys[i] ^ keys[j];
    if (diff == 0ull) {
        const unsigned int idiff = (unsigned int)(i ^ j);
        return 64 + __clz(idiff);
    }
    return __clzll(diff);
}

// Variant that accepts a pre-loaded key to avoid redundant global loads
// when the same index is queried repeatedly (common in Karras construction).
static __device__ __forceinline__ int delta_prefix_cached(
    const uint64_t* __restrict__ keys, int n, uint64_t key_i, int i, int j)
{
    if (j < 0 || j >= n)
        return -1;
    const uint64_t diff = key_i ^ keys[j];
    if (diff == 0ull) {
        const unsigned int idiff = (unsigned int)(i ^ j);
        return 64 + __clz(idiff);
    }
    return __clzll(diff);
}

__global__ void build_karras_topology(
    int n,
    int* root,
    const uint64_t* __restrict__ keys,
    volatile int* __restrict__ range_lefts,
    volatile int* __restrict__ range_rights,
    volatile int* __restrict__ parents,
    volatile BVHPackedNodeHalf* __restrict__ lowers,
    volatile BVHPackedNodeHalf* __restrict__ uppers
)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n - 1)
        return;  // n-1 internal nodes

    const int internal_offset = n;

    const uint64_t key_i = keys[i];

    // Determine direction of the range (+1 or -1)
    const int delta_next = delta_prefix_cached(keys, n, key_i, i, i + 1);
    const int delta_prev = delta_prefix_cached(keys, n, key_i, i, i - 1);
    const int d = (delta_next - delta_prev) >= 0 ? 1 : -1;

    // Compute upper bound for the length of the range
    const int delta_min = delta_prefix_cached(keys, n, key_i, i, i - d);

    int l_max = 2;
    while (delta_prefix_cached(keys, n, key_i, i, i + l_max * d) > delta_min)
        l_max *= 2;

    // Find the other end using binary search
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (delta_prefix_cached(keys, n, key_i, i, i + (l + t) * d) > delta_min)
            l += t;
    }
    const int j = i + l * d;

    const int first = min(i, j);
    const int last = max(i, j);

    // Find split position (canonical Karras) — cache keys[first] for binary search
    const uint64_t key_first = keys[first];
    const int delta_node = delta_prefix_cached(keys, n, key_first, first, last);

    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        const int new_split = split + step;
        if (new_split < last) {
            const int delta_split = delta_prefix_cached(keys, n, key_first, first, new_split);
            if (delta_split > delta_node)
                split = new_split;
        }
    } while (step > 1);

    const int internal_index = internal_offset + i;

    const int left_child = (split == first) ? first : (internal_offset + split);
    const int right_child = (split + 1 == last) ? last : (internal_offset + split + 1);

    // Write internal node children
    lowers[internal_index].i = left_child;
    lowers[internal_index].b = 0;
    uppers[internal_index].i = right_child;
    uppers[internal_index].b = 0;

    // Parent pointers
    parents[left_child] = internal_index;
    parents[right_child] = internal_index;

    // Range (inclusive)
    range_lefts[internal_index] = first;
    range_rights[internal_index] = last;

    // Identify root: the node whose range covers [0, n-1]
    if (first == 0 && last == n - 1) {
        *root = internal_index;
        parents[internal_index] = -1;
    }
}

// Fused kernel: compute depth-from-root for every node AND find the global
// maximum depth in a single launch (avoids a second kernel + D2H copy for the
// max-reduction that was previously required).  The per-block max is folded
// into the depth-walk itself using shared-memory reduction + atomicMax.
__global__ void compute_node_depths_and_max(
    int n_nodes, const int* __restrict__ parents, int* __restrict__ depths, int* __restrict__ out_max, int max_steps
)
{
    __shared__ int sdata[WP_BVH_BLOCK_DIM];

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int depth = 0;  // 0 for out-of-range threads – won't affect max
    if (idx < n_nodes) {
        depth = 1;
        int p = parents[idx];
        int steps = 0;
        while (p != -1 && steps < max_steps) {
            p = parents[p];
            ++depth;
            ++steps;
        }
        depths[idx] = depth;
    }

    // Block-level max reduction
    sdata[threadIdx.x] = depth;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicMax(out_max, sdata[0]);
}

__global__ void refit_nodes_at_depth(
    int n_internal,
    int internal_offset,
    int target_depth,
    const int* __restrict__ depths,
    BVHPackedNodeHalf* __restrict__ node_lowers,
    BVHPackedNodeHalf* __restrict__ node_uppers)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_internal)
        return;

    int idx = internal_offset + tid;

    if (depths[idx] != target_depth)
        return;

    const int left = node_lowers[idx].i;
    const int right = node_uppers[idx].i;

    BVHPackedNodeHalf ll = bvh_load_node(node_lowers, left);
    BVHPackedNodeHalf lu = bvh_load_node(node_uppers, left);
    BVHPackedNodeHalf rl = bvh_load_node(node_lowers, right);
    BVHPackedNodeHalf ru = bvh_load_node(node_uppers, right);

    const vec3 out_lower = min(vec3(ll.x, ll.y, ll.z), vec3(rl.x, rl.y, rl.z));
    const vec3 out_upper = max(vec3(lu.x, lu.y, lu.z), vec3(ru.x, ru.y, ru.z));

    make_node(node_lowers + idx, out_lower, left, false);
    make_node(node_uppers + idx, out_upper, right, false);
}

__global__ void build_leaves(
    const vec3* __restrict__ item_lowers,
    const vec3* __restrict__ item_uppers,
    int n,
    const int* __restrict__ indices,
    int* __restrict__ range_lefts,
    int* __restrict__ range_rights,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers
)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        const int item = indices[index];

        vec3 lower = item_lowers[item];
        vec3 upper = item_uppers[item];

        // write leaf nodes using position indices and [start, end) range
        lowers[index] = make_node(lower, index, true);
        uppers[index] = make_node(upper, index, false);

        // write leaf key ranges
        range_lefts[index] = index;
        range_rights[index] = index;
    }
}

// this bottom-up process assigns left and right children and combines bounds to form internal nodes
// there is one thread launched per-leaf node, each thread calculates it's parent node and assigns
// itself to either the left or right parent slot, the last child to complete the parent and moves
// up the hierarchy
__global__ void build_hierarchy(
    int n,
    int* root,
    const int* __restrict__ deltas,
    const uint64_t* __restrict__ keys,
    int* __restrict__ num_children,
    const int* __restrict__ primitive_indices,
    volatile int* __restrict__ range_lefts,
    volatile int* __restrict__ range_rights,
    volatile int* __restrict__ parents,
    volatile BVHPackedNodeHalf* __restrict__ lowers,
    volatile BVHPackedNodeHalf* __restrict__ uppers
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        const int internal_offset = n;

        for (;;) {
            int left = range_lefts[index];
            int right = range_rights[index];

            // check if we are the root node, if so then store out our index and terminate
            if (left == 0 && right == n - 1) {
                *root = index;
                parents[index] = -1;

                break;
            }

            int childCount = 0;

            int parent;

            // group away selection of parent. We need to make sure that:
            // 1) If node spans a single group and exactly one neighbor keeps us in-group, choose that side
            // 2) Otherwise, use the original delta rule with parity tie-break to avoid ladders
            // This is important to make sure that our tree does not mix groups until a single group root is formed.
            bool parent_right = false;
            if (left == 0) {
                parent_right = true;
            } else {
                bool decided = false;
                const uint32_t group_left = (uint32_t)(keys[left] >> 32);
                const uint32_t group_right = (uint32_t)(keys[right] >> 32);

                // Check if either of the neighbors keep us in the same group
                if (group_left == group_right) {
                    const uint32_t group = group_left;
                    const bool can_go_right_same = (right < n - 1) && ((uint32_t)(keys[right + 1] >> 32) == group);
                    const bool can_go_left_same = ((uint32_t)(keys[left - 1] >> 32) == group);

                    // If exactly one neighbor keeps us in the same group, choose that side
                    if (can_go_right_same ^ can_go_left_same) {
                        parent_right = can_go_right_same;
                        decided = true;
                    }

                    // If both or neither keep us in the same group, fall through to delta rule
                }

                if (!decided) {
                    // prefer side with greater common prefix
                    if (right != n - 1 && deltas[right] >= deltas[left - 1]) {
                        if (deltas[right] == deltas[left - 1])
                            parent_right = (primitive_indices[left - 1] % 2) ^ (primitive_indices[right] % 2);
                        else
                            parent_right = true;
                    } else {
                        parent_right = false;
                    }
                }
            }

            if (parent_right) {
                parent = right + internal_offset;

                // set parent left child
                parents[index] = parent;
                lowers[parent].i = index;
                range_lefts[parent] = left;

                // ensure above writes are visible to all threads
                __threadfence();

                childCount = atomicAdd(&num_children[parent], 1);
            } else {
                parent = left + internal_offset - 1;

                // set parent right child
                parents[index] = parent;
                uppers[parent].i = index;
                range_rights[parent] = right;

                // ensure above writes are visible to all threads
                __threadfence();

                childCount = atomicAdd(&num_children[parent], 1);
            }

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (childCount == 1) {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x, lowers[left_child].y, lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x, uppers[left_child].y, uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x, lowers[right_child].y, lowers[right_child].z);


                vec3 right_upper = vec3(uppers[right_child].x, uppers[right_child].y, uppers[right_child].z);

                // bounds_union of child bounds; ensure min on lowers and max on uppers
                vec3 lower = min(left_lower, right_lower);
                vec3 upper = max(left_upper, right_upper);

                // write new BVH nodes
                make_node(lowers + parent, lower, left_child, false);
                make_node(uppers + parent, upper, right_child, false);

                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

/*
 * LBVH uses a bottom-up constructor which makes variable-sized leaf nodes more challenging to achieve.
 * Simply splitting the ordered primitives into uniform groups of size leaf_size will result in poor
 * quality. Instead, after the hierarchy is built, we convert any intermediate node whose size is
 * <= leaf_size into a new leaf node. This process is done using the new kernel function called
 * mark_packed_leaf_nodes .
 */
__global__ void mark_packed_leaf_nodes(
    int n,
    const int* __restrict__ range_lefts,
    const int* __restrict__ range_rights,
    const int* __restrict__ parents,
    const uint64_t* __restrict__ keys,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers,
    const int leaf_size,
    const int* __restrict__ precomputed_depths
)
{
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index < n) {
        int depth;
        if (precomputed_depths) {
            depth = precomputed_depths[node_index];
        } else {
            depth = 1;
            int parent = parents[node_index];
            while (parent != -1) {
                parent = parents[parent];
                depth++;
            }
        }

        int left = range_lefts[node_index];
        int right = range_rights[node_index] + 1;

        bool single_group = true;
        const uint64_t group_left = keys[left] >> 32;
        const uint64_t group_right = keys[right - 1] >> 32;
        single_group = (group_left == group_right);

        if (single_group && (right - left <= leaf_size || depth >= BVH_QUERY_STACK_SIZE)) {
            lowers[node_index].b = 1;
            lowers[node_index].i = left;
            uppers[node_index].i = right;
        }
    }
}


CUDA_CALLABLE inline vec3 Vec3Max(const vec3& a, const vec3& b) { return wp::max(a, b); }
CUDA_CALLABLE inline vec3 Vec3Min(const vec3& a, const vec3& b) { return wp::min(a, b); }

#if defined(__HIP_PLATFORM_AMD__)
// Deterministic bounds reduction for HIP (avoids vec3 atomics which are not safe/portable on HIP).
__global__ void compute_block_bounds(
    const vec3* item_lowers, const vec3* item_uppers, vec3* block_lowers, vec3* block_uppers, int num_items
)
{
    typedef cub::BlockReduce<vec3, WP_BVH_BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int blockStart = blockDim.x * blockIdx.x;
    const int numValid = ::min(num_items - blockStart, (int)blockDim.x);
    const int tid = blockStart + threadIdx.x;

    vec3 lower = vec3(FLT_MAX);
    vec3 upper = vec3(-FLT_MAX);
    if (tid < num_items) {
        lower = item_lowers[tid];
        upper = item_uppers[tid];
    }

    vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);
    __syncthreads();
    vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);

    if (threadIdx.x == 0) {
        block_lowers[blockIdx.x] = block_lower;
        block_uppers[blockIdx.x] = block_upper;
    }
}

// Block size for the final reduction kernel
#define REDUCE_BLOCK_DIM 256

// Fused final reduction: block bounds -> total bounds + inverse edge lengths.
// Eliminates the separate compute_total_inv_edges kernel launch.
__global__ void reduce_block_bounds(
    const vec3* block_lowers, const vec3* block_uppers, int num_blocks,
    vec3* total_lower, vec3* total_upper, vec3* total_inv_edges
)
{
    typedef cub::BlockReduce<vec3, REDUCE_BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    vec3 lo = vec3(FLT_MAX);
    vec3 hi = vec3(-FLT_MAX);

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        lo = min(lo, block_lowers[i]);
        hi = max(hi, block_uppers[i]);
    }

    vec3 block_upper = BlockReduce(temp_storage).Reduce(hi, Vec3Max);
    __syncthreads();
    vec3 block_lower = BlockReduce(temp_storage).Reduce(lo, Vec3Min);

    if (threadIdx.x == 0) {
        total_lower[0] = block_lower;
        total_upper[0] = block_upper;
        if (total_inv_edges) {
            vec3 edges = block_upper - block_lower + vec3(0.0001f);
            total_inv_edges[0] = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
        }
    }
}
#else
    // Original CUDA version with atomic vec3 operations
    __global__ void compute_total_bounds(
        const vec3* item_lowers, const vec3* item_uppers, vec3* total_lower, vec3* total_upper, int num_items
    )
    {
        typedef cub::BlockReduce<vec3, 256> BlockReduce;

        __shared__ typename BlockReduce::TempStorage temp_storage;

        const int blockStart = blockDim.x * blockIdx.x;
        const int numValid = ::min(num_items - blockStart, blockDim.x);

        const int tid = blockStart + threadIdx.x;

        if (tid < num_items) {
            vec3 lower = item_lowers[tid];
            vec3 upper = item_uppers[tid];

            vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);

            // sync threads because second reduce uses same temp storage as first
            __syncthreads();

            vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);

            if (threadIdx.x == 0) {
                // write out block results, expanded by the radius
                atomic_max(total_upper, block_upper);
                atomic_min(total_lower, block_lower);
            }
        }
    }
#endif

// compute inverse edge length, this is just done on the GPU to avoid a CPU->GPU sync point
__global__ void compute_total_inv_edges(const vec3* total_lower, const vec3* total_upper, vec3* total_inv_edges)
{
    vec3 edges = (total_upper[0] - total_lower[0]);
    edges += vec3(0.0001f);

    total_inv_edges[0] = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
}


// On HIP, constructor / destructor are intentionally empty -- all scratch
// (including total_lower/upper/inv_edges) is pool-allocated in build().
#if !defined(__HIP_PLATFORM_AMD__)
LinearBVHBuilderGPU::LinearBVHBuilderGPU()
    : indices(NULL)
    , keys(NULL)
    , deltas(NULL)
    , range_lefts(NULL)
    , range_rights(NULL)
    , num_children(NULL)
    , total_lower(NULL)
    , total_upper(NULL)
    , total_inv_edges(NULL)
{
    total_lower = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
    total_upper = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
    total_inv_edges = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
}

LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
{
    wp_free_device(WP_CURRENT_CONTEXT, total_lower);
    wp_free_device(WP_CURRENT_CONTEXT, total_upper);
    wp_free_device(WP_CURRENT_CONTEXT, total_inv_edges);
}
#endif  // !__HIP_PLATFORM_AMD__


#if defined(__HIP_PLATFORM_AMD__)
void LinearBVHBuilderGPU::build(
    BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, int num_items, bounds3* total_bounds, int* item_groups
)
{
    constexpr int tree_threads = WP_BVH_BLOCK_DIM;

    auto align_up = [](size_t x, size_t a) -> size_t { return (x + a - 1) & ~(a - 1); };
    constexpr size_t ALIGN = 256;

    const int nb_bounds = (num_items + WP_BVH_BLOCK_DIM - 1) / WP_BVH_BLOCK_DIM;

    // 30-bit Morton codes in lower bits; group id in upper 32 bits (if present)
    const int sort_end_bit = item_groups ? 64 : 30;

    // Query radix-sort temp storage requirement so we can include it in the pool
    size_t sort_temp_bytes = 0;
    (void)cub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes,
        (const uint64_t*)nullptr, (uint64_t*)nullptr,
        (const int*)nullptr, (int*)nullptr,
        num_items, 0, sort_end_bit
    );

    const size_t sz_indices     = align_up(sizeof(int) * num_items, ALIGN);
    const size_t sz_keys        = align_up(sizeof(uint64_t) * num_items * 2, ALIGN);  // *2 for cub internal double-buffer
    const size_t sz_range_lefts = align_up(sizeof(int) * bvh.max_nodes, ALIGN);
    const size_t sz_range_rights= align_up(sizeof(int) * bvh.max_nodes, ALIGN);
    const size_t sz_num_children= align_up(sizeof(int) * bvh.max_nodes, ALIGN);  // reused as node_depths
    const size_t sz_total_lower = align_up(sizeof(vec3), ALIGN);
    const size_t sz_total_upper = align_up(sizeof(vec3), ALIGN);
    const size_t sz_inv_edges   = align_up(sizeof(vec3), ALIGN);
    const size_t sz_blk_lowers  = align_up(sizeof(vec3) * nb_bounds, ALIGN);
    const size_t sz_blk_uppers  = align_up(sizeof(vec3) * nb_bounds, ALIGN);
    const size_t sz_max_depth   = align_up(sizeof(int), ALIGN);
    const size_t sz_sort_temp   = align_up(sort_temp_bytes, ALIGN);

    const size_t pool_bytes = sz_indices + sz_keys + sz_range_lefts + sz_range_rights + sz_num_children
        + sz_total_lower + sz_total_upper + sz_inv_edges + sz_blk_lowers + sz_blk_uppers + sz_max_depth + sz_sort_temp;

    char* pool = (char*)wp_alloc_device(WP_CURRENT_CONTEXT, pool_bytes);

    char* ptr = pool;
    int*      indices       = (int*)ptr;      ptr += sz_indices;
    uint64_t* keys          = (uint64_t*)ptr; ptr += sz_keys;
    int*      range_lefts   = (int*)ptr;      ptr += sz_range_lefts;
    int*      range_rights  = (int*)ptr;      ptr += sz_range_rights;
    int*      num_children  = (int*)ptr;      ptr += sz_num_children;
    vec3*     total_lower   = (vec3*)ptr;     ptr += sz_total_lower;
    vec3*     total_upper   = (vec3*)ptr;     ptr += sz_total_upper;
    vec3*     total_inv_edges = (vec3*)ptr;   ptr += sz_inv_edges;
    vec3*     block_lowers  = (vec3*)ptr;     ptr += sz_blk_lowers;
    vec3*     block_uppers  = (vec3*)ptr;     ptr += sz_blk_uppers;
    int*      max_depth_dev = (int*)ptr;      ptr += sz_max_depth;
    void*     sort_temp     = (void*)ptr;     // ptr += sz_sort_temp;

    // COMPUTE TOTAL BOUNDS
    if (total_bounds) {
        vec3 edges = (*total_bounds).edges();
        edges += vec3(0.0001f);
        vec3 inv_edges = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);

        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &total_bounds->lower[0], sizeof(vec3));
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &total_bounds->upper[0], sizeof(vec3));
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_inv_edges, &inv_edges[0], sizeof(vec3));
    } else {
        wp_launch_device(
            WP_CURRENT_CONTEXT, compute_block_bounds, num_items,
            (item_lowers, item_uppers, block_lowers, block_uppers, num_items)
        );
        wp_launch_device(
            WP_CURRENT_CONTEXT, reduce_block_bounds, 1,
            (block_lowers, block_uppers, nb_bounds, total_lower, total_upper, total_inv_edges)
        );
    }

    // MORTON CODES
    wp_launch_device(
        WP_CURRENT_CONTEXT, compute_morton_codes, num_items,
        (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys, item_groups)
    );

    // SORT: keys in-place, values from indices -> bvh.primitive_indices (eliminates D2D copy)
    hipStream_t stream = (hipStream_t)wp_cuda_stream_get_current();
    (void)cub::DeviceRadixSort::SortPairs(
        sort_temp, sort_temp_bytes,
        keys, keys,
        indices, bvh.primitive_indices,
        num_items, 0, sort_end_bit, stream
    );

    // BUILD TREE TOPOLOGY (Karras-style, deterministic)
    wp_launch_device(
        WP_CURRENT_CONTEXT, build_leaves, num_items,
        (item_lowers, item_uppers, num_items, bvh.primitive_indices, range_lefts, range_rights,
         bvh.node_lowers, bvh.node_uppers)
    );

    wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_parents, 0xFF, sizeof(int) * bvh.max_nodes);

    wp_launch_device(
        WP_CURRENT_CONTEXT, build_karras_topology, num_items,
        (num_items, bvh.root, keys, range_lefts, range_rights, bvh.node_parents, bvh.node_lowers, bvh.node_uppers)
    );

    // DEPTH-LEVEL REFIT: compute per-node depths, then refit from deepest to shallowest
    int* node_depths = num_children;
    wp_memset_device(WP_CURRENT_CONTEXT, max_depth_dev, 0, sizeof(int));
    wp_launch_device(
        WP_CURRENT_CONTEXT, compute_node_depths_and_max, bvh.max_nodes,
        (bvh.max_nodes, bvh.node_parents, node_depths, max_depth_dev, 1024)
    );

    {
        int max_depth = 0;
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, &max_depth, max_depth_dev, sizeof(int));

        const int num_internal = num_items - 1;
        for (int d = max_depth; d >= 1; --d)
        {
            wp_launch_device(
                WP_CURRENT_CONTEXT, refit_nodes_at_depth, num_internal,
                (num_internal, num_items, d, node_depths, bvh.node_lowers, bvh.node_uppers)
            );
        }
    }

    // PACK SMALL SUB-TREES INTO LEAF NODES (use pre-computed depths to avoid parent-chain walk)
    int* precomputed_depths = node_depths;
    wp_launch_device(
        WP_CURRENT_CONTEXT, mark_packed_leaf_nodes, bvh.max_nodes,
        (bvh.max_nodes, range_lefts, range_rights, bvh.node_parents, keys, bvh.node_lowers, bvh.node_uppers,
         bvh.leaf_size, precomputed_depths)
    );

    // CLEANUP – single sync + single free
    (void)cuCtxSynchronize_f();
    wp_free_device(WP_CURRENT_CONTEXT, pool);
}
#else
    void LinearBVHBuilderGPU::build(
        BVH& bvh,
        const vec3* item_lowers,
        const vec3* item_uppers,
        int num_items,
        bounds3* total_bounds,
        int* item_groups
    )
    {
        // allocate temporary memory used during building
        indices = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items * 2, "(native:bvh)");  // *2 for radix sort
        keys = (uint64_t*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(uint64_t) * num_items * 2, "(native:bvh)");  // *2 for radix sort
        deltas = (int*)wp_alloc_device(
            WP_CURRENT_CONTEXT, sizeof(int) * num_items, "(native:bvh)"
        );  // highest differentiating bit between keys for item i and i+1
        range_lefts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
        range_rights = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
        num_children = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");

        // if total bounds supplied by the host then we just
        // compute our edge length and upload it to the GPU directly
        if (total_bounds) {
            // calculate Morton codes
            vec3 edges = (*total_bounds).edges();
            edges += vec3(0.0001f);

            vec3 inv_edges = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);

            wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &total_bounds->lower[0], sizeof(vec3));
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &total_bounds->upper[0], sizeof(vec3));
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_inv_edges, &inv_edges[0], sizeof(vec3));
        } else {
            // IEEE-754 bit patterns for ± FLT_MAX
            constexpr int FLT_MAX_BITS = 0x7f7fffff;
            constexpr int NEG_FLT_MAX_BITS = 0xff7fffff;

            // total_lower := ( +FLT_MAX, +FLT_MAX, +FLT_MAX )
            wp_launch_device(
                WP_CURRENT_CONTEXT, memset_kernel, sizeof(vec3) / 4, ((int*)total_lower, FLT_MAX_BITS, sizeof(vec3) / 4)
            );

            // total_upper := ( -FLT_MAX, -FLT_MAX, -FLT_MAX )
            wp_launch_device(
                WP_CURRENT_CONTEXT, memset_kernel, sizeof(vec3) / 4,
                ((int*)total_upper, NEG_FLT_MAX_BITS, sizeof(vec3) / 4)
            );

            // compute the total bounds on the GPU (CUDA version with atomics)
            wp_launch_device(
                WP_CURRENT_CONTEXT, compute_total_bounds, num_items,
                (item_lowers, item_uppers, total_lower, total_upper, num_items)
            );

            // compute the total edge length
            wp_launch_device(
                WP_CURRENT_CONTEXT, compute_total_inv_edges, 1, (total_lower, total_upper, total_inv_edges)
            );
        }

        // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
        wp_launch_device(
            WP_CURRENT_CONTEXT, compute_morton_codes, num_items,
            (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys, item_groups)
        );

        // sort items based on Morton key (note the 64-bit sort key includes group in upper 32 bits and morton code in
        // lower 32 bits)
        radix_sort_pairs_device(WP_CURRENT_CONTEXT, keys, indices, num_items);
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, bvh.primitive_indices, indices, sizeof(int) * num_items);

        // calculate deltas between adjacent keys (kept for debugging, not required by Karras builder)
        wp_launch_device(WP_CURRENT_CONTEXT, compute_key_deltas, num_items, (keys, deltas, num_items - 1));

        // initialize leaf nodes
        wp_launch_device(
            WP_CURRENT_CONTEXT, build_leaves, num_items,
            (item_lowers, item_uppers, num_items, indices, range_lefts, range_rights, bvh.node_lowers, bvh.node_uppers)
        );

        // Original CUDA version with atomics-based hierarchy builder
        // reset children count, this is our atomic counter so we know when an internal node is complete
        wp_memset_device(WP_CURRENT_CONTEXT, num_children, 0, sizeof(int) * bvh.max_nodes);

        // build the tree and internal node bounds
        wp_launch_device(
            WP_CURRENT_CONTEXT, build_hierarchy, num_items,
            (num_items, bvh.root, deltas, keys, num_children, bvh.primitive_indices, range_lefts, range_rights,
             bvh.node_parents, bvh.node_lowers, bvh.node_uppers)
        );

        wp_launch_device(
            WP_CURRENT_CONTEXT, mark_packed_leaf_nodes, bvh.max_nodes,
            (bvh.max_nodes, range_lefts, range_rights, bvh.node_parents, keys, bvh.node_lowers, bvh.node_uppers,
             bvh.leaf_size, (const int*)nullptr)
        );


        // free temporary memory
        wp_free_device(WP_CURRENT_CONTEXT, indices);
        wp_free_device(WP_CURRENT_CONTEXT, keys);
        wp_free_device(WP_CURRENT_CONTEXT, deltas);

        wp_free_device(WP_CURRENT_CONTEXT, range_lefts);
        wp_free_device(WP_CURRENT_CONTEXT, range_rights);
        wp_free_device(WP_CURRENT_CONTEXT, num_children);
    }
#endif
// buffer_size is the number of T, not the number of bytes
template <typename T> T* make_device_buffer_of(void* context, T* host_buffer, size_t buffer_size)
{
    T* device_buffer = (T*)wp_alloc_device(context, sizeof(T) * buffer_size, "(native:bvh)");
    ;
    wp_memcpy_h2d(context, device_buffer, host_buffer, sizeof(T) * buffer_size);

    return device_buffer;
}

void copy_host_tree_to_device(void* context, BVH& bvh_host, BVH& bvh_device_on_host)
{
    if (!bvh_host.item_groups)
    // if it's grouped bvh it's already reordered
    {
        reorder_top_down_bvh(bvh_host);
    }

    bvh_device_on_host.num_nodes = bvh_host.num_nodes;
    bvh_device_on_host.num_leaf_nodes = bvh_host.num_leaf_nodes;
    bvh_device_on_host.max_nodes = bvh_host.max_nodes;
    bvh_device_on_host.num_items = bvh_host.num_items;
    bvh_device_on_host.max_depth = bvh_host.max_depth;
    bvh_device_on_host.leaf_size = bvh_host.leaf_size;

    bvh_device_on_host.root = (int*)wp_alloc_device(context, sizeof(int), "(native:bvh)");
    wp_memcpy_h2d(context, bvh_device_on_host.root, bvh_host.root, sizeof(int));
    bvh_device_on_host.context = context;

    bvh_device_on_host.node_lowers = make_device_buffer_of(context, bvh_host.node_lowers, bvh_host.max_nodes);
    bvh_device_on_host.node_uppers = make_device_buffer_of(context, bvh_host.node_uppers, bvh_host.max_nodes);
    bvh_device_on_host.node_parents = make_device_buffer_of(context, bvh_host.node_parents, bvh_host.max_nodes);
    bvh_device_on_host.primitive_indices
        = make_device_buffer_of(context, bvh_host.primitive_indices, bvh_host.num_items);
}

// create in-place given existing descriptor
void bvh_create_device(
    void* context,
    vec3* lowers,
    vec3* uppers,
    int num_items,
    int constructor_type,
    int* groups,
    int leaf_size,
    BVH& bvh_device_on_host
)
{
    ContextGuard guard(context);
    if (constructor_type == BVH_CONSTRUCTOR_SAH || constructor_type == BVH_CONSTRUCTOR_MEDIAN)
    // CPU based constructors
    {
        // copy bounds back to CPU
        std::vector<vec3> lowers_host(num_items);
        std::vector<vec3> uppers_host(num_items);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, lowers_host.data(), lowers, sizeof(vec3) * num_items);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, uppers_host.data(), uppers, sizeof(vec3) * num_items);

        // copy groups back to CPU (if exists)
        std::vector<int> groups_host(num_items);
        int* groups_host_ptr = nullptr;
        if (groups) {
            wp_memcpy_d2h(WP_CURRENT_CONTEXT, groups_host.data(), groups, sizeof(int) * num_items);
            groups_host_ptr = groups_host.data();
        }

        // run CPU based constructor
        wp::BVH bvh_host;
        wp::bvh_create_host(
            lowers_host.data(), uppers_host.data(), num_items, constructor_type, groups_host_ptr, leaf_size, bvh_host
        );

        // copy host tree to device
        wp::copy_host_tree_to_device(WP_CURRENT_CONTEXT, bvh_host, bvh_device_on_host);
        // replace host bounds with device bounds
        bvh_device_on_host.item_lowers = lowers;
        bvh_device_on_host.item_uppers = uppers;
        bvh_device_on_host.item_groups = groups;
        // node_counts is not allocated for host tree
        bvh_device_on_host.node_counts
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes, "(native:bvh)");
        wp::bvh_destroy_host(bvh_host);
    } else if (constructor_type == BVH_CONSTRUCTOR_LBVH) {
        bvh_device_on_host.leaf_size = leaf_size;
        bvh_device_on_host.num_items = num_items;
        bvh_device_on_host.max_nodes = 2 * num_items - 1;
        bvh_device_on_host.num_leaf_nodes = num_items;
        bvh_device_on_host.node_lowers = (BVHPackedNodeHalf*)wp_alloc_device(
            WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes, "(native:bvh)"
        );
        wp_memset_device(
            WP_CURRENT_CONTEXT, bvh_device_on_host.node_lowers, 0,
            sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes
        );
        bvh_device_on_host.node_uppers = (BVHPackedNodeHalf*)wp_alloc_device(
            WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes, "(native:bvh)"
        );
        wp_memset_device(
            WP_CURRENT_CONTEXT, bvh_device_on_host.node_uppers, 0,
            sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes
        );
        bvh_device_on_host.node_parents
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes, "(native:bvh)");
        bvh_device_on_host.node_counts
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes, "(native:bvh)");
        bvh_device_on_host.root = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int), "(native:bvh)");
        bvh_device_on_host.primitive_indices
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items, "(native:bvh)");
        bvh_device_on_host.item_lowers = lowers;
        bvh_device_on_host.item_uppers = uppers;
        bvh_device_on_host.item_groups = groups;
        bvh_device_on_host.context = context ? context : wp_cuda_context_get_current();

        LinearBVHBuilderGPU builder;
        builder.build(bvh_device_on_host, lowers, uppers, num_items, NULL, groups);
    } else {
        printf(
            "Unrecognized Constructor type: %d! For GPU constructor it should be SAH (0), Median (1), or LBVH (2)!\n",
            constructor_type
        );
    }
}

void bvh_destroy_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_lowers);
    bvh.node_lowers = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_uppers);
    bvh.node_uppers = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_parents);
    bvh.node_parents = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_counts);
    bvh.node_counts = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.primitive_indices);
    bvh.primitive_indices = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.root);
    bvh.root = NULL;
}

void bvh_refit_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    // clear child counters
    wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);
    wp_launch_device(
        WP_CURRENT_CONTEXT, bvh_refit_kernel, bvh.num_leaf_nodes,
        (bvh.num_leaf_nodes, bvh.node_parents, bvh.node_counts, bvh.primitive_indices, bvh.node_lowers, bvh.node_uppers,
         bvh.item_lowers, bvh.item_uppers)
    );
}

void bvh_rebuild_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    LinearBVHBuilderGPU builder;
    builder.build(bvh, bvh.item_lowers, bvh.item_uppers, bvh.num_items, NULL, bvh.item_groups);
}

#ifndef WP_DISABLE_CUBQL

__global__ void cubql_make_boxes(const vec3* lowers, const vec3* uppers, cuBQL::box3f* boxes, int n)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        const vec3 lower = lowers[tid];
        const vec3 upper = uppers[tid];
        boxes[tid]
            = cuBQL::box3f(cuBQL::vec3f(lower[0], lower[1], lower[2]), cuBQL::vec3f(upper[0], upper[1], upper[2]));
    }
}

static inline cuBQL::bvh3f cubql_native_view(const CuBQLBVH& bvh)
{
    cuBQL::bvh3f native;
    native.nodes = reinterpret_cast<cuBQL::bvh3f::node_t*>(bvh.nodes);
    native.numNodes = uint32_t(bvh.num_nodes);
    native.primIDs = reinterpret_cast<uint32_t*>(bvh.primitive_indices);
    native.numPrims = uint32_t(bvh.num_prims);
    return native;
}

static inline void cubql_assign(CuBQLBVH& bvh, const cuBQL::bvh3f& native)
{
    bvh.nodes = reinterpret_cast<CuBQLNode*>(native.nodes);
    bvh.num_nodes = int(native.numNodes);
    bvh.primitive_indices = native.primIDs;
    bvh.num_prims = int(native.numPrims);
    if (bvh.root) {
        int root_index = native.numNodes > 0 ? 0 : -1;
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh.root, &root_index, sizeof(int));
    }
}

static inline void cubql_update_device_boxes(CuBQLBVH& bvh)
{
    if (!bvh.boxes || bvh.num_items <= 0) {
        return;
    }
    wp_launch_device(
        WP_CURRENT_CONTEXT, cubql_make_boxes, bvh.num_items,
        (bvh.item_lowers, bvh.item_uppers, reinterpret_cast<cuBQL::box3f*>(bvh.boxes), bvh.num_items)
    );
}

void cubql_bvh_create_device(
    void* context, vec3* lowers, vec3* uppers, int num_items, int leaf_size, CuBQLBVH& bvh_device_on_host
)
{
    ContextGuard guard(context);
    memset(&bvh_device_on_host, 0, sizeof(CuBQLBVH));

    bvh_device_on_host.context = context ? context : wp_cuda_context_get_current();
    bvh_device_on_host.item_lowers = lowers;
    bvh_device_on_host.item_uppers = uppers;
    bvh_device_on_host.num_items = num_items;
    bvh_device_on_host.leaf_size = leaf_size;
    bvh_device_on_host.root = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int));

    int root_index = -1;
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh_device_on_host.root, &root_index, sizeof(int));

    if (num_items <= 0) {
        return;
    }

    bvh_device_on_host.boxes = wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(cuBQL::box3f) * num_items);
    cubql_update_device_boxes(bvh_device_on_host);

    cuBQL::bvh3f native;
    cuBQL::BuildConfig build_config;
    build_config.enableSAH();
    build_config.makeLeafThreshold = leaf_size;
    cuBQL::gpuBuilder(
        native, reinterpret_cast<cuBQL::box3f*>(bvh_device_on_host.boxes), uint32_t(num_items), build_config
    );
    cubql_assign(bvh_device_on_host, native);
}

void cubql_bvh_destroy_device(CuBQLBVH& bvh)
{
    ContextGuard guard(bvh.context);

    if (bvh.nodes || bvh.primitive_indices) {
        cuBQL::bvh3f native = cubql_native_view(bvh);
        cuBQL::cuda::free(native);
    }

    if (bvh.boxes) {
        wp_free_device(WP_CURRENT_CONTEXT, bvh.boxes);
        bvh.boxes = nullptr;
    }
    if (bvh.root) {
        wp_free_device(WP_CURRENT_CONTEXT, bvh.root);
        bvh.root = nullptr;
    }
    bvh.nodes = nullptr;
    bvh.num_nodes = 0;
    bvh.primitive_indices = nullptr;
    bvh.num_prims = 0;
    bvh.leaf_size = 0;
}

void cubql_bvh_refit_device(CuBQLBVH& bvh)
{
    ContextGuard guard(bvh.context);
    if (!bvh.nodes || !bvh.boxes || bvh.num_items <= 0) {
        return;
    }

    cubql_update_device_boxes(bvh);
    cuBQL::bvh3f native = cubql_native_view(bvh);
    cuBQL::cuda::refit(native, reinterpret_cast<cuBQL::box3f*>(bvh.boxes));
}

void cubql_bvh_rebuild_device(CuBQLBVH& bvh)
{
    ContextGuard guard(bvh.context);

    if (bvh.nodes || bvh.primitive_indices) {
        cuBQL::bvh3f old_native = cubql_native_view(bvh);
        cuBQL::cuda::free(old_native);
    }

    if (bvh.num_items <= 0) {
        bvh.nodes = nullptr;
        bvh.primitive_indices = nullptr;
        bvh.num_nodes = 0;
        bvh.num_prims = 0;
        if (bvh.root) {
            int root_index = -1;
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh.root, &root_index, sizeof(int));
        }
        return;
    }

    if (!bvh.boxes) {
        bvh.boxes = wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(cuBQL::box3f) * bvh.num_items);
    }
    cubql_update_device_boxes(bvh);

    cuBQL::bvh3f native;
    cuBQL::BuildConfig build_config;
    build_config.enableSAH();
    build_config.makeLeafThreshold = bvh.leaf_size;
    cuBQL::gpuBuilder(native, reinterpret_cast<cuBQL::box3f*>(bvh.boxes), uint32_t(bvh.num_items), build_config);
    cubql_assign(bvh, native);
}

#else  // WP_DISABLE_CUBQL

void cubql_bvh_create_device(
    void* context, vec3* lowers, vec3* uppers, int num_items, int leaf_size, CuBQLBVH& bvh_device_on_host
)
{
    fprintf(stderr, "Warp error: cuBQL support disabled (WP_DISABLE_CUBQL)\n");
    memset(&bvh_device_on_host, 0, sizeof(CuBQLBVH));
}

void cubql_bvh_destroy_device(CuBQLBVH&) { }
void cubql_bvh_refit_device(CuBQLBVH&) { }
void cubql_bvh_rebuild_device(CuBQLBVH&) { }

#endif  // WP_DISABLE_CUBQL


}  // namespace wp


void wp_bvh_refit_device(uint64_t id)
{
    wp::BVH bvh;
    if (bvh_get_descriptor(id, bvh)) {
        ContextGuard guard(bvh.context);

        wp::bvh_refit_device(bvh);
    }
}

void wp_bvh_rebuild_device(uint64_t id)
{
    wp::BVH bvh;
    if (bvh_get_descriptor(id, bvh)) {
        ContextGuard guard(bvh.context);

        wp::bvh_rebuild_device(bvh);
    }
}

/*
 * Since we don't even know the number of true leaf nodes, never mention where they are, we will launch
 * the num_items threads, which are identical to the number of leaf nodes in the original tree. The
 * refitting threads will start from the nodes corresponding to the original leaf nodes, which might be
 * muted. However, the muted leaf nodes will still have the pointer to their parents, thus the up-tracing
 * can still work. We will only compute the bounding box of a leaf node if its parent is not a leaf node.
 */
uint64_t wp_bvh_create_device(
    void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size
)
{
    ContextGuard guard(context);
    wp::BVH bvh_device_on_host;
    wp::BVH* bvh_device_ptr = nullptr;

    wp::bvh_create_device(
        WP_CURRENT_CONTEXT, lowers, uppers, num_items, constructor_type, groups, leaf_size, bvh_device_on_host
    );

    // create device-side BVH descriptor
    bvh_device_ptr = (wp::BVH*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BVH), "(native:bvh)");
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh_device_ptr, &bvh_device_on_host, sizeof(wp::BVH));

    uint64_t bvh_id = (uint64_t)bvh_device_ptr;
    wp::bvh_add_descriptor(bvh_id, bvh_device_on_host);
    return bvh_id;
}

uint64_t wp_cubql_bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int leaf_size)
{
    ContextGuard guard(context);
    wp::CuBQLBVH bvh_device_on_host;
    memset(&bvh_device_on_host, 0, sizeof(wp::CuBQLBVH));

    wp::cubql_bvh_create_device(WP_CURRENT_CONTEXT, lowers, uppers, num_items, leaf_size, bvh_device_on_host);

    if (!bvh_device_on_host.nodes && num_items > 0) {
        return 0;
    }

    wp::CuBQLBVH* bvh_device_ptr = (wp::CuBQLBVH*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::CuBQLBVH));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh_device_ptr, &bvh_device_on_host, sizeof(wp::CuBQLBVH));

    uint64_t bvh_id = (uint64_t)bvh_device_ptr;
    wp::cubql_bvh_add_descriptor(bvh_id, bvh_device_on_host);
    return bvh_id;
}


void wp_bvh_destroy_device(uint64_t id)
{
    wp::BVH bvh;
    if (wp::bvh_get_descriptor(id, bvh)) {
        wp::bvh_destroy_device(bvh);
        wp::bvh_rem_descriptor(id);

        // free descriptor
        wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
}

void wp_cubql_bvh_refit_device(uint64_t id)
{
    wp::CuBQLBVH bvh;
    if (wp::cubql_bvh_get_descriptor(id, bvh)) {
        ContextGuard guard(bvh.context);
        wp::cubql_bvh_refit_device(bvh);
        wp::cubql_bvh_add_descriptor(id, bvh);
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, (void*)id, &bvh, sizeof(wp::CuBQLBVH));
    }
}

void wp_cubql_bvh_rebuild_device(uint64_t id)
{
    wp::CuBQLBVH bvh;
    if (wp::cubql_bvh_get_descriptor(id, bvh)) {
        ContextGuard guard(bvh.context);
        wp::cubql_bvh_rebuild_device(bvh);
        wp::cubql_bvh_add_descriptor(id, bvh);
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, (void*)id, &bvh, sizeof(wp::CuBQLBVH));
    }
}

void wp_cubql_bvh_destroy_device(uint64_t id)
{
    wp::CuBQLBVH bvh;
    if (wp::cubql_bvh_get_descriptor(id, bvh)) {
        wp::cubql_bvh_destroy_device(bvh);
        wp::cubql_bvh_rem_descriptor(id);
        wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
}
