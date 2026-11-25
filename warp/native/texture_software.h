// SPDX-License-Identifier: Apache-2.0

// Device textures sampled in software, for GPUs without image hardware (AMD GPUs whose HIP
// device reports no image support) and for 3D textures on GPUs without 3D linear filtering (gfx90a). The texels live in
// ordinary device memory and the texture handle points to a device copy of the same Texture struct the CPU sampler
// reads, so kernels built for those GPUs use the CPU sampling code unchanged.

#pragma once

#include <cstdint>

// True when ndim-dimensional textures on the context's device are kept by these functions
bool wp_texture_software_enabled(void* context, int ndim);
// True for array and texture handles these functions created
bool wp_texture_software_owns(uint64_t handle);
bool wp_texture_software_owns_copy(int dst_memory_type, uint64_t dst_handle, int src_memory_type, uint64_t src_handle);

uint64_t wp_texture_software_create(void* context, int ndim, int* shape, int num_channels, int dtype);
void wp_texture_software_destroy(void* context, uint64_t array_handle);
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
);
uint64_t wp_texture_software_object_create(
    void* context,
    uint64_t array_handle,
    int ndim,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords
);
void wp_texture_software_object_destroy(void* context, uint64_t tex_handle);
