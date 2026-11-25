// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "bvh.h"
#include "cuda_util.h"
#include "error.h"
#include "sort.h"

#include <algorithm>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define THRUST_IGNORE_CUB_VERSION_CHECK
#define REORDER_HOST_TREE

#include <cub/cub.cuh>

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

            // Single-node trees store the root leaf without a parent.
            if (parent == -1 || !node_lowers[parent].b) {
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

                    // only need to compute bound when this is a valid leaf node
                    if (parent_parent == -1 || !node_lowers[parent_parent].b) {
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

/////////////////////////////////////////////////////////////////////////////////////////////

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf
//
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

private:
    // temporary data used during building
    int* indices;
    uint64_t* keys;
    int* deltas;
    int* range_lefts;
    int* range_rights;
    int* num_children;

    // bounds data when total item bounds built on GPU
    vec3* total_lower;
    vec3* total_upper;
    vec3* total_inv_edges;
};

////////////////////////////////////////////////////////


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
    const int leaf_size
#if defined(__HIP_PLATFORM_AMD__)
    ,
    const int* __restrict__ depths
#endif
)
{
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index < n) {
        // mark the node as leaf if its range is less than leaf_size or it is deeper than BVH_QUERY_STACK_SIZE
        // this will forever mute its child nodes so that they will never be accessed

#if defined(__HIP_PLATFORM_AMD__)
        const int depth = depths[node_index];
#else
        // calculate depth
        int depth = 1;
        int parent = parents[node_index];
        while (parent != -1) {
            parent = parents[parent];
            depth++;
        }
#endif

        int left = range_lefts[node_index];
        // the LBVH constructor's range is defined as left <= i <= right
        // we need to convert it to our convention: left <= i < right
        int right = range_rights[node_index] + 1;

        // avoid creating packed leaves that straddle group boundaries
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

// compute inverse edge length, this is just done on the GPU to avoid a CPU->GPU sync point
__global__ void compute_total_inv_edges(const vec3* total_lower, const vec3* total_upper, vec3* total_inv_edges)
{
    vec3 edges = (total_upper[0] - total_lower[0]);
    edges += vec3(0.0001f);

    total_inv_edges[0] = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
}

#if defined(__HIP_PLATFORM_AMD__)
// Length of the common prefix of sorted keys i and j, with equal keys told apart by their positions.
static __device__ __forceinline__ int key_prefix(const uint64_t* __restrict__ keys, int n, uint64_t key_i, int i, int j)
{
    if (j < 0 || j >= n)
        return -1;
    const uint64_t diff = key_i ^ keys[j];
    if (diff == 0)
        return 64 + __clz((unsigned int)(i ^ j));
    return __clzll(diff);
}

// Internal node i of the binary radix tree over the sorted keys (Karras 2012), one thread per internal node.
__global__ void build_karras_topology(
    int n,
    int* root,
    const uint64_t* __restrict__ keys,
    int* __restrict__ range_lefts,
    int* __restrict__ range_rights,
    int* __restrict__ parents,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers
)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0 && n == 1)
        *root = 0;
    if (i >= n - 1)
        return;

    const uint64_t key_i = keys[i];

    // direction of the node's range, and its other end
    const int d = key_prefix(keys, n, key_i, i, i + 1) >= key_prefix(keys, n, key_i, i, i - 1) ? 1 : -1;
    const int prefix_min = key_prefix(keys, n, key_i, i, i - d);
    int l_max = 2;
    while (key_prefix(keys, n, key_i, i, i + l_max * d) > prefix_min)
        l_max *= 2;
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (key_prefix(keys, n, key_i, i, i + (l + t) * d) > prefix_min)
            l += t;
    }
    const int first = min(i, i + l * d);
    const int last = max(i, i + l * d);

    // split position
    const uint64_t key_first = keys[first];
    const int prefix_node = key_prefix(keys, n, key_first, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        if (split + step < last && key_prefix(keys, n, key_first, first, split + step) > prefix_node)
            split += step;
    } while (step > 1);

    const int node = n + i;
    const int left_child = (split == first) ? first : n + split;
    const int right_child = (split + 1 == last) ? last : n + split + 1;

    lowers[node].i = left_child;
    lowers[node].b = 0;
    uppers[node].i = right_child;
    uppers[node].b = 0;
    parents[left_child] = node;
    parents[right_child] = node;
    range_lefts[node] = first;
    range_rights[node] = last;

    if (first == 0 && last == n - 1) {
        *root = node;
        parents[node] = -1;
    }
}

// Depth of every node (the root is 1) and the largest depth.
__global__ void
compute_node_depths(int num_nodes, const int* __restrict__ parents, int* __restrict__ depths, int* max_depth)
{
    __shared__ int block_max[256];

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    int depth = 0;
    if (index < num_nodes) {
        depth = 1;
        for (int parent = parents[index]; parent != -1; parent = parents[parent])
            depth++;
        depths[index] = depth;
    }

    block_max[threadIdx.x] = depth;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_max[threadIdx.x] = max(block_max[threadIdx.x], block_max[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0 && max_depth)
        atomicMax(max_depth, block_max[0]);
}

// Bounds of the internal nodes at one depth from their children's.
__global__ void refit_nodes_at_depth(
    int n,
    int depth,
    const int* __restrict__ max_depth,
    const int* __restrict__ depths,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers
)
{
    const int node = n + blockDim.x * blockIdx.x + threadIdx.x;
    if (depth > *max_depth || node >= 2 * n - 1 || depths[node] != depth)
        return;

    const int left = lowers[node].i;
    const int right = uppers[node].i;
    const vec3 lower = min(
        vec3(lowers[left].x, lowers[left].y, lowers[left].z), vec3(lowers[right].x, lowers[right].y, lowers[right].z)
    );
    const vec3 upper = max(
        vec3(uppers[left].x, uppers[left].y, uppers[left].z), vec3(uppers[right].x, uppers[right].y, uppers[right].z)
    );
    make_node(lowers + node, lower, left, false);
    make_node(uppers + node, upper, right, false);
}

// Refit of the nodes at one depth, deepest level first, giving the bottom-up bvh_refit_kernel's results: a leaf or
// packed leaf takes its primitives' bounds unless its parent is a packed leaf; an internal node takes its children's.
__global__ void refit_tree_at_depth(
    int num_nodes,
    int depth,
    const int* __restrict__ depths,
    const int* __restrict__ parents,
    const int* __restrict__ primitive_indices,
    const vec3* __restrict__ item_lowers,
    const vec3* __restrict__ item_uppers,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers
)
{
    const int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= num_nodes || depths[node] != depth)
        return;

    if (lowers[node].b) {
        const int parent = parents[node];
        if (parent != -1 && lowers[parent].b)
            return;
        bounds3 bound;
        for (int primitive_counter = lowers[node].i; primitive_counter < uppers[node].i; primitive_counter++) {
            const int primitive = primitive_indices[primitive_counter];
            bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
        }
        (vec3&)lowers[node] = bound.lower;
        (vec3&)uppers[node] = bound.upper;
    } else {
        const int left_child = lowers[node].i;
        const int right_child = uppers[node].i;
        (vec3&)lowers[node] = min((vec3&)lowers[left_child], (vec3&)lowers[right_child]);
        (vec3&)uppers[node] = max((vec3&)uppers[left_child], (vec3&)uppers[right_child]);
    }
}

// Total bounds in two passes (per-block bounds, then one block over those) instead of vec3 atomics on one address.
__global__ void compute_block_bounds(
    const vec3* item_lowers, const vec3* item_uppers, vec3* block_lowers, vec3* block_uppers, int num_items
)
{
    typedef cub::BlockReduce<vec3, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    vec3 lower = vec3(FLT_MAX);
    vec3 upper = vec3(-FLT_MAX);
    if (tid < num_items) {
        lower = item_lowers[tid];
        upper = item_uppers[tid];
    }

    vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max);
    __syncthreads();
    vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min);

    if (threadIdx.x == 0) {
        block_lowers[blockIdx.x] = block_lower;
        block_uppers[blockIdx.x] = block_upper;
    }
}

__global__ void reduce_block_bounds(
    const vec3* block_lowers,
    const vec3* block_uppers,
    int num_blocks,
    vec3* total_lower,
    vec3* total_upper,
    vec3* total_inv_edges
)
{
    typedef cub::BlockReduce<vec3, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    vec3 lower = vec3(FLT_MAX);
    vec3 upper = vec3(-FLT_MAX);
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        lower = min(lower, block_lowers[i]);
        upper = max(upper, block_uppers[i]);
    }

    vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max);
    __syncthreads();
    vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min);

    if (threadIdx.x == 0) {
        total_lower[0] = block_lower;
        total_upper[0] = block_upper;
        vec3 edges = block_upper - block_lower + vec3(0.0001f);
        total_inv_edges[0] = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
    }
}
#endif


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
#if !defined(__HIP_PLATFORM_AMD__)
    total_lower = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
    total_upper = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
    total_inv_edges = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
#endif
}

LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
{
#if !defined(__HIP_PLATFORM_AMD__)
    wp_free_device(WP_CURRENT_CONTEXT, total_lower);
    wp_free_device(WP_CURRENT_CONTEXT, total_upper);
    wp_free_device(WP_CURRENT_CONTEXT, total_inv_edges);
#endif
}


#if defined(__HIP_PLATFORM_AMD__)
// On AMD GPUs the bottom-up build_hierarchy pass, with a __threadfence() and an atomic per node it climbs, takes most
// of the build. HIP builds the topology top-down from the sorted keys instead, then fills the internal bounds one depth
// level at a time; scratch memory is one allocation and the sort writes straight to the primitive indices.
void LinearBVHBuilderGPU::build(
    BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, int num_items, bounds3* total_bounds, int* item_groups
)
{
    // keys hold a 30-bit Morton code, and the group in the upper 32 bits when there are groups
    const int key_bits = item_groups ? 64 : 30;
    const int num_bounds_blocks = (num_items + 255) / 256;

    size_t sort_temp_bytes = 0;
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_bytes, (uint64_t*)nullptr, (uint64_t*)nullptr, (int*)nullptr, (int*)nullptr, num_items,
            0, key_bits
        )
    );

    size_t pool_bytes = 0;
    auto reserve = [&pool_bytes](size_t bytes) {
        const size_t offset = pool_bytes;
        pool_bytes += (bytes + 255) & ~size_t(255);
        return offset;
    };
    const size_t indices_at = reserve(sizeof(int) * num_items);
    const size_t keys_at = reserve(sizeof(uint64_t) * num_items * 2);  // keys, then the sorted keys
    const size_t range_lefts_at = reserve(sizeof(int) * bvh.max_nodes);
    const size_t range_rights_at = reserve(sizeof(int) * bvh.max_nodes);
    const size_t max_depth_at = reserve(sizeof(int));
    const size_t total_at = reserve(sizeof(vec3) * 3);
    const size_t block_bounds_at = reserve(sizeof(vec3) * num_bounds_blocks * 2);
    const size_t sort_temp_at = reserve(sort_temp_bytes);

    char* pool = (char*)wp_alloc_device(WP_CURRENT_CONTEXT, pool_bytes, "(native:bvh)");
    indices = (int*)(pool + indices_at);
    keys = (uint64_t*)(pool + keys_at);
    range_lefts = (int*)(pool + range_lefts_at);
    range_rights = (int*)(pool + range_rights_at);
    int* depths = bvh.node_counts;  // kept for refits
    int* max_depth = (int*)(pool + max_depth_at);
    total_lower = (vec3*)(pool + total_at);
    total_upper = total_lower + 1;
    total_inv_edges = total_lower + 2;
    vec3* block_lowers = (vec3*)(pool + block_bounds_at);
    vec3* block_uppers = block_lowers + num_bounds_blocks;

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
            (block_lowers, block_uppers, num_bounds_blocks, total_lower, total_upper, total_inv_edges)
        );
    }

    wp_launch_device(
        WP_CURRENT_CONTEXT, compute_morton_codes, num_items,
        (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys, item_groups)
    );

    uint64_t* sorted_keys = keys + num_items;
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            pool + sort_temp_at, sort_temp_bytes, keys, sorted_keys, indices, bvh.primitive_indices, num_items, 0,
            key_bits, (cudaStream_t)wp_cuda_stream_get_current()
        )
    );

    wp_launch_device(
        WP_CURRENT_CONTEXT, build_leaves, num_items,
        (item_lowers, item_uppers, num_items, bvh.primitive_indices, range_lefts, range_rights, bvh.node_lowers,
         bvh.node_uppers)
    );

    wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_parents, 0xFF, sizeof(int) * bvh.max_nodes);
    wp_launch_device(
        WP_CURRENT_CONTEXT, build_karras_topology, num_items,
        (num_items, bvh.root, sorted_keys, range_lefts, range_rights, bvh.node_parents, bvh.node_lowers,
         bvh.node_uppers)
    );

    wp_memset_device(WP_CURRENT_CONTEXT, max_depth, 0, sizeof(int));
    wp_launch_device(
        WP_CURRENT_CONTEXT, compute_node_depths, bvh.max_nodes, (bvh.max_nodes, bvh.node_parents, depths, max_depth)
    );

    // A child's common key prefix is longer than its parent's, so no internal node is deeper than the key bits plus
    // the bits that tell equal keys apart. Launching that many levels keeps the build free of host synchronization
    // (and capturable); levels below the deepest node return at once.
    int index_bits = 0;
    while ((1ll << index_bits) < num_items)
        index_bits++;
    bvh.max_depth = key_bits + index_bits;  // a bound on the internal depth, for bvh_refit_device
    for (int depth = key_bits + index_bits; depth >= 1; depth--) {
        wp_launch_device(
            WP_CURRENT_CONTEXT, refit_nodes_at_depth, num_items - 1,
            (num_items, depth, max_depth, depths, bvh.node_lowers, bvh.node_uppers)
        );
    }

    wp_launch_device(
        WP_CURRENT_CONTEXT, mark_packed_leaf_nodes, bvh.max_nodes,
        (bvh.max_nodes, range_lefts, range_rights, bvh.node_parents, sorted_keys, bvh.node_lowers, bvh.node_uppers,
         bvh.leaf_size, depths)
    );

    wp_free_device(WP_CURRENT_CONTEXT, pool);
}
#else
void LinearBVHBuilderGPU::build(
    BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, int num_items, bounds3* total_bounds, int* item_groups
)
{
    // allocate temporary memory used during  building
    indices
        = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items * 2, "(native:bvh)");  // *2 for radix sort
    keys = (uint64_t*)wp_alloc_device(
        WP_CURRENT_CONTEXT, sizeof(uint64_t) * num_items * 2, "(native:bvh)"
    );  // *2 for radix sort
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
            WP_CURRENT_CONTEXT, memset_kernel, sizeof(vec3) / 4, ((int*)total_upper, NEG_FLT_MAX_BITS, sizeof(vec3) / 4)
        );

        // compute the total bounds on the GPU
        wp_launch_device(
            WP_CURRENT_CONTEXT, compute_total_bounds, num_items,
            (item_lowers, item_uppers, total_lower, total_upper, num_items)
        );

        // compute the total edge length
        wp_launch_device(WP_CURRENT_CONTEXT, compute_total_inv_edges, 1, (total_lower, total_upper, total_inv_edges));
    }

    // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
    wp_launch_device(
        WP_CURRENT_CONTEXT, compute_morton_codes, num_items,
        (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys, item_groups)
    );

    // sort items based on Morton key (note the 64-bit sort key includes group in upper 32 bits and morton code in lower
    // 32 bits)
    radix_sort_pairs_device(WP_CURRENT_CONTEXT, keys, indices, num_items);
    wp_memcpy_d2d(WP_CURRENT_CONTEXT, bvh.primitive_indices, indices, sizeof(int) * num_items);

    // calculate deltas between adjacent keys
    wp_launch_device(WP_CURRENT_CONTEXT, compute_key_deltas, num_items, (keys, deltas, num_items - 1));

    // initialize leaf nodes
    wp_launch_device(
        WP_CURRENT_CONTEXT, build_leaves, num_items,
        (item_lowers, item_uppers, num_items, indices, range_lefts, range_rights, bvh.node_lowers, bvh.node_uppers)
    );

    // reset children count, this is our atomic counter so we know when an internal node is complete, only used during
    // building
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
         bvh.leaf_size)
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
    // Skip the reorder for cuBQL trees: it scrambles cuBQL's natural
    // contiguous child-pair layout (root at 0, children at offset/offset+1)
    // and hurts ray traversal cache behaviour. Grouped BVHs are already
    // reordered by the builder so they don't need it either.
    if (!bvh_host.item_groups && bvh_host.constructor_type != BVH_CONSTRUCTOR_CUBQL) {
        reorder_top_down_bvh(bvh_host);
    }

    bvh_device_on_host.num_nodes = bvh_host.num_nodes;
    bvh_device_on_host.num_leaf_nodes = bvh_host.num_leaf_nodes;
    bvh_device_on_host.max_nodes = bvh_host.max_nodes;
    bvh_device_on_host.num_items = bvh_host.num_items;
    bvh_device_on_host.max_depth = bvh_host.max_depth;
    bvh_device_on_host.leaf_size = bvh_host.leaf_size;
    bvh_device_on_host.constructor_type = bvh_host.constructor_type;

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
    if (constructor_type == BVH_CONSTRUCTOR_CUBQL) {
        if (groups) {
            wp::set_error_string("Warp error: grouped BVHs are not supported with cuBQL construction");
            memset(&bvh_device_on_host, 0, sizeof(BVH));
            return;
        }
        cubql_bvh_create_device(context, lowers, uppers, num_items, leaf_size, bvh_device_on_host);
        bvh_device_on_host.constructor_type = constructor_type;
        return;
    }

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
        bvh_device_on_host.constructor_type = constructor_type;
        // node_counts is not allocated for host tree
        bvh_device_on_host.node_counts
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes, "(native:bvh)");
#if defined(__HIP_PLATFORM_AMD__)
        // node depths for the refit; nodes past num_nodes are unused and get depth 0
        wp_memset_device(
            WP_CURRENT_CONTEXT, bvh_device_on_host.node_counts, 0, sizeof(int) * bvh_device_on_host.max_nodes
        );
        wp_launch_device(
            WP_CURRENT_CONTEXT, compute_node_depths, bvh_device_on_host.num_nodes,
            (bvh_device_on_host.num_nodes, bvh_device_on_host.node_parents, bvh_device_on_host.node_counts, nullptr)
        );
#endif
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
        bvh_device_on_host.constructor_type = constructor_type;

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

#if defined(__HIP_PLATFORM_AMD__)
int bvh_refit_max_depth(const BVH& bvh)
{
    int index_bits = 0;
    while ((1ll << index_bits) < bvh.num_items)
        index_bits++;
    // the depth of a host-built tree, or the bound of the tree an in-place rebuild may have made of it; leaves are one
    // level deeper than the internal nodes
    return std::max(bvh.max_depth, (bvh.item_groups ? 64 : 30) + index_bits) + 1;
}
#endif

void bvh_refit_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    if (!bvh.node_lowers || !bvh.node_uppers || !bvh.node_parents || !bvh.node_counts || !bvh.primitive_indices
        || bvh.num_leaf_nodes <= 0) {
        return;
    }

#if defined(__HIP_PLATFORM_AMD__)
    // On AMD GPUs the bottom-up walk's per-node __threadfence() dominates on large trees; refit one depth level at a
    // time there instead. Its launch count (fixed, so the refit stays capturable) costs more than the walk on small
    // trees.
    if (bvh.num_leaf_nodes >= 32768) {
        for (int depth = bvh_refit_max_depth(bvh); depth >= 1; depth--) {
            wp_launch_device(
                WP_CURRENT_CONTEXT, refit_tree_at_depth, bvh.max_nodes,
                (bvh.max_nodes, depth, bvh.node_counts, bvh.node_parents, bvh.primitive_indices, bvh.item_lowers,
                 bvh.item_uppers, bvh.node_lowers, bvh.node_uppers)
            );
        }
        return;
    }
#endif
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

    if (bvh.constructor_type == BVH_CONSTRUCTOR_CUBQL) {
        cubql_bvh_rebuild_device(bvh);
        bvh.constructor_type = BVH_CONSTRUCTOR_CUBQL;
        return;
    }

    LinearBVHBuilderGPU builder;
    builder.build(bvh, bvh.item_lowers, bvh.item_uppers, bvh.num_items, NULL, bvh.item_groups);
    bvh.constructor_type = BVH_CONSTRUCTOR_LBVH;
}


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

        // A cuBQL rebuild reallocates the device buffers, so the BVH struct (pointers
        // and sizes) changes; an in-place LBVH rebuild reuses its buffers, leaving the
        // struct unchanged.
        const bool was_cubql = (bvh.constructor_type == BVH_CONSTRUCTOR_CUBQL);

        wp::bvh_rebuild_device(bvh);

        // Refresh the host-side descriptor in case any fields changed.
        wp::bvh_add_descriptor(id, bvh);

        // Only the cuBQL path needs the device-side descriptor refreshed. The host->device
        // copy of pageable memory is not CUDA-graph-capture safe, so we must skip it for the
        // in-place LBVH rebuild to keep grouped/LBVH rebuilds capture-safe (its device struct
        // is unchanged anyway).
        if (was_cubql) {
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, (void*)id, &bvh, sizeof(wp::BVH));
        }
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

    if (!bvh_device_on_host.node_lowers && num_items > 0) {
        return 0;
    }

    // create device-side BVH descriptor
    bvh_device_ptr = (wp::BVH*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BVH), "(native:bvh)");
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh_device_ptr, &bvh_device_on_host, sizeof(wp::BVH));

    uint64_t bvh_id = (uint64_t)bvh_device_ptr;
    wp::bvh_add_descriptor(bvh_id, bvh_device_on_host);
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
