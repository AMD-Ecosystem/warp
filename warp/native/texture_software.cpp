// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "error.h"
#include "texture.h"
#include "texture_software.h"

#include <algorithm>
#include <mutex>
#include <unordered_set>

#if WP_ENABLE_CUDA

namespace {

// Texel storage that stands in for a CUDA array, or for a mipmapped array and its levels. The levels of a
// mipmapped array are stored one after another in a single allocation, in the layout of the CPU Texture.
struct SoftwareTextureArray {
    void* texels;
    int width;
    int height;
    int depth;
    int num_channels;
    int dtype;
    int num_mip_levels;
    SoftwareTextureArray* levels[WP_TEXTURE_MAX_MIP_LEVELS];  // views of each level (mipmapped arrays only)
    bool is_level;  // a view of one level, owned by its mipmapped array
};

// Memory types as texture.py passes them (the CUDA driver's values)
enum { kHost = 1, kDevice = 2, kArray = 3 };

// software array and texture handles; a device may hold hardware textures of other dimensions
std::mutex g_handles_mutex;
std::unordered_set<uint64_t> g_handles;

void track(uint64_t handle, bool add)
{
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    if (add)
        g_handles.insert(handle);
    else
        g_handles.erase(handle);
}

// shape of a mip level: each sampled dimension halves per level, down to 1
int level_dim(int dim, int level) { return dim > 0 ? std::max(1, dim >> level) : 0; }

size_t level_bytes(const SoftwareTextureArray* array, int level)
{
    return size_t(level_dim(array->width, level)) * std::max(1, level_dim(array->height, level))
        * std::max(1, level_dim(array->depth, level)) * array->num_channels
        * wp::get_texture_bytes_per_channel(array->dtype);
}

}  // namespace

bool wp_texture_software_enabled(void* context, int ndim)
{
    return !wp_cuda_device_is_texture_hardware_supported(wp_cuda_context_get_device_ordinal(context), ndim);
}

bool wp_texture_software_owns(uint64_t handle)
{
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return g_handles.count(handle) != 0;
}

bool wp_texture_software_owns_copy(int dst_memory_type, uint64_t dst_handle, int src_memory_type, uint64_t src_handle)
{
    return (dst_memory_type == kArray && wp_texture_software_owns(dst_handle))
        || (src_memory_type == kArray && wp_texture_software_owns(src_handle));
}

uint64_t
wp_texture_software_create(void* context, int ndim, int* shape, int num_channels, int dtype, int num_mip_levels)
{
    SoftwareTextureArray* array = new SoftwareTextureArray();
    array->width = shape[0];
    array->height = ndim > 1 ? shape[1] : 0;
    array->depth = ndim > 2 ? shape[2] : 0;
    array->num_channels = num_channels;
    array->dtype = dtype;
    array->num_mip_levels = num_mip_levels;

    size_t bytes = 0;
    for (int level = 0; level < num_mip_levels; ++level)
        bytes += level_bytes(array, level);

    array->texels = wp_alloc_device(context, bytes, "(native:texture)");
    if (!array->texels) {
        delete array;
        return 0;
    }

    if (num_mip_levels > 1) {
        size_t offset = 0;
        for (int level = 0; level < num_mip_levels; ++level) {
            SoftwareTextureArray* view = new SoftwareTextureArray(*array);
            view->texels = static_cast<char*>(array->texels) + offset;
            view->width = level_dim(array->width, level);
            view->height = level_dim(array->height, level);
            view->depth = level_dim(array->depth, level);
            view->num_mip_levels = 1;
            std::fill(std::begin(view->levels), std::end(view->levels), nullptr);
            view->is_level = true;
            array->levels[level] = view;
            track(reinterpret_cast<uint64_t>(view), true);
            offset += level_bytes(array, level);
        }
    }

    track(reinterpret_cast<uint64_t>(array), true);
    return reinterpret_cast<uint64_t>(array);
}

void wp_texture_software_destroy(void* context, uint64_t array_handle)
{
    SoftwareTextureArray* array = reinterpret_cast<SoftwareTextureArray*>(array_handle);
    track(array_handle, false);
    for (int level = 0; level < WP_TEXTURE_MAX_MIP_LEVELS; ++level) {
        if (array->levels[level]) {
            track(reinterpret_cast<uint64_t>(array->levels[level]), false);
            delete array->levels[level];
        }
    }
    wp_free_device(context, array->texels);
    delete array;
}

uint64_t wp_texture_software_get_mip_level(uint64_t array_handle, int level)
{
    const SoftwareTextureArray* array = reinterpret_cast<const SoftwareTextureArray*>(array_handle);
    if (level < 0 || level >= array->num_mip_levels || !array->levels[level]) {
        wp::set_error_string("Warp error: Invalid mip level %d", level);
        return 0;
    }
    return reinterpret_cast<uint64_t>(array->levels[level]);
}

bool wp_texture_software_copy(
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
    size_t rows = height ? height : 1;
    size_t slices = depth ? depth : 1;

    // texels are stored densely; Warp's linear sides are dense too, so every copy is one block
    if ((src_memory_type != kArray && ((src_pitch && src_pitch != width_bytes) || (depth && src_height != rows)))
        || (dst_memory_type != kArray && ((dst_pitch && dst_pitch != width_bytes) || (depth && dst_height != rows)))) {
        wp::set_error_string("Warp error: Pitched texture copies are not supported on this device");
        return false;
    }

    void* dst = dst_memory_type == kArray ? reinterpret_cast<SoftwareTextureArray*>(dst_handle)->texels
                                          : reinterpret_cast<void*>(dst_handle);
    void* src = src_memory_type == kArray ? reinterpret_cast<SoftwareTextureArray*>(src_handle)->texels
                                          : reinterpret_cast<void*>(src_handle);
    size_t bytes = size_t(width_bytes) * rows * slices;

    if (src_memory_type == kHost && dst_memory_type != kHost)
        return wp_memcpy_h2d(context, dst, src, bytes, stream);
    if (dst_memory_type == kHost && src_memory_type != kHost)
        return wp_memcpy_d2h(context, dst, src, bytes, stream);
    if (src_memory_type != kHost && dst_memory_type != kHost)
        return wp_memcpy_d2d(context, dst, src, bytes, stream);

    wp::set_error_string("Warp error: Invalid texture copy memory types %d -> %d", src_memory_type, dst_memory_type);
    return false;
}

uint64_t wp_texture_software_object_create(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords
)
{
    const SoftwareTextureArray* array = reinterpret_cast<const SoftwareTextureArray*>(array_handle);
    const int num_mip_levels = array->num_mip_levels;

    int widths[WP_TEXTURE_MAX_MIP_LEVELS];
    int heights[WP_TEXTURE_MAX_MIP_LEVELS];
    int depths[WP_TEXTURE_MAX_MIP_LEVELS];
    for (int level = 0; level < num_mip_levels; ++level) {
        widths[level] = level_dim(array->width, level);
        heights[level] = level_dim(array->height, level);
        depths[level] = level_dim(array->depth, level);
    }

    // the sampler's view of the texture, with the level pointers at the device texels (the CPU Texture lays the
    // levels out the same way, so its offsets locate them)
    wp::Texture desc(
        ndim, num_mip_levels, widths, heights, depths, array->num_channels, array->dtype, filter_mode, mip_filter_mode,
        address_modes[0], ndim > 1 ? address_modes[1] : 0, ndim > 2 ? address_modes[2] : 0, use_normalized_coords
    );
    void* host_texels = desc.data;
    desc.data = array->texels;
    for (int level = 0; level < num_mip_levels; ++level)
        desc.mip_data[level] = static_cast<char*>(array->texels) + desc.mip_offsets[level];

    wp::Texture* desc_device = (wp::Texture*)wp_alloc_device(context, sizeof(wp::Texture), "(native:texture)");
    uint64_t tex_handle = 0;
    if (desc_device) {
        if (wp_memcpy_h2d(context, desc_device, &desc, sizeof(wp::Texture))) {
            tex_handle = reinterpret_cast<uint64_t>(desc_device);
            // on a GPU that samples this kind of texture in hardware, kernels tell a software texture by the lowest
            // bit of its handle (see WP_TEXTURE_SOFTWARE_FALLBACK in texture.h)
            if (!wp_texture_software_enabled(context, ndim))
                tex_handle |= 1;
            track(tex_handle, true);
        } else {
            wp_free_device(context, desc_device);
        }
    }

    desc.data = host_texels;
    return tex_handle;
}

void wp_texture_software_object_destroy(void* context, uint64_t tex_handle)
{
    track(tex_handle, false);
    wp_free_device(context, reinterpret_cast<void*>(tex_handle & ~uint64_t(1)));
}

#endif  // WP_ENABLE_CUDA
