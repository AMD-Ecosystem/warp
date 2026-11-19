// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "error.h"
#include "texture.h"
#include "texture_software.h"

#include <mutex>
#include <unordered_set>

#if WP_ENABLE_CUDA

namespace {

// Texel storage that stands in for a CUDA array
struct SoftwareTextureArray {
    void* texels;
    int width;
    int height;
    int depth;
    int num_channels;
    int dtype;
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

uint64_t wp_texture_software_create(void* context, int ndim, int* shape, int num_channels, int dtype)
{
    SoftwareTextureArray* array = new SoftwareTextureArray();
    array->width = shape[0];
    array->height = ndim > 1 ? shape[1] : 0;
    array->depth = ndim > 2 ? shape[2] : 0;
    array->num_channels = num_channels;
    array->dtype = dtype;

    size_t bytes = size_t(array->width) * (array->height ? array->height : 1) * (array->depth ? array->depth : 1)
        * num_channels * wp::get_texture_bytes_per_channel(dtype);
    array->texels = wp_alloc_device(context, bytes, "(native:texture)");
    if (!array->texels) {
        delete array;
        return 0;
    }
    track(reinterpret_cast<uint64_t>(array), true);
    return reinterpret_cast<uint64_t>(array);
}

void wp_texture_software_destroy(void* context, uint64_t array_handle)
{
    SoftwareTextureArray* array = reinterpret_cast<SoftwareTextureArray*>(array_handle);
    track(array_handle, false);
    wp_free_device(context, array->texels);
    delete array;
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
    void* context, uint64_t array_handle, int ndim, int filter_mode, int* address_modes, bool use_normalized_coords
)
{
    const SoftwareTextureArray* array = reinterpret_cast<const SoftwareTextureArray*>(array_handle);

    // the sampler's view of the texture, pointing at the device texels
    wp::Texture desc(1, array->num_channels, array->dtype, filter_mode, address_modes[0], use_normalized_coords);
    void* host_texels = desc.data;
    desc.data = array->texels;
    desc.width = array->width;
    desc.height = array->height;
    desc.depth = array->depth;
    desc.address_mode_v = ndim > 1 ? address_modes[1] : 0;
    desc.address_mode_w = ndim > 2 ? address_modes[2] : 0;

    wp::Texture* desc_device = (wp::Texture*)wp_alloc_device(context, sizeof(wp::Texture), "(native:texture)");
    if (desc_device) {
        wp_memcpy_h2d(context, desc_device, &desc, sizeof(wp::Texture));
        track(reinterpret_cast<uint64_t>(desc_device), true);
    }

    desc.data = host_texels;
    return reinterpret_cast<uint64_t>(desc_device);
}

void wp_texture_software_object_destroy(void* context, uint64_t tex_handle)
{
    track(tex_handle, false);
    wp_free_device(context, reinterpret_cast<void*>(tex_handle));
}

#endif  // WP_ENABLE_CUDA
