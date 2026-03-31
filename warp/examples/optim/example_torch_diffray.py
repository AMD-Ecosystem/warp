# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#############################################################################
# Example Differentiable Ray Caster implemented with torch module
#
# Shows how to use the built-in wp.Mesh data structure and wp.mesh_query_ray()
# function to implement a basic differentiable ray caster within a torch model.
#
##############################################################################

import itertools
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples

# Maps registry id -> (scene_bufs, render_mesh). Ids are passed as a tensor into custom ops so
# torch.compile can trace the handle; mutating a global list before forward does not.
_RAY_CAST_REGISTRY: dict[int, tuple] = {}
_next_ray_cast_registry_id = itertools.count(1)


def _register_ray_cast_buffers(scene_bufs, render_mesh) -> int:
    rid = next(_next_ray_cast_registry_id)
    _RAY_CAST_REGISTRY[rid] = (scene_bufs, render_mesh)
    return rid


class MeshRotationModule(torch.nn.Module):
    """PyTorch module with `mesh_rot` (quaternion [4]) and `mesh_pos` (translation [3]).
    Forward takes target pixels; scene/render_mesh are set via context before call.
    """

    def __init__(self, scene_bufs, render_mesh, init_rot=None, init_pos=None):
        super().__init__()
        if init_rot is None:
            init_rot = [0.0, 0.0, 0.0, 1.0]
        if init_pos is None:
            init_pos = [0.0, 0.0, 0.0]
        self._scene_bufs = scene_bufs
        self._render_mesh = render_mesh
        device = wp.device_to_torch(wp.get_device())
        self.mesh_rot = torch.nn.Parameter(torch.tensor(init_rot, dtype=torch.float32, device=device))
        self.mesh_pos = torch.nn.Parameter(torch.tensor(np.array(init_pos, dtype=np.float32), device=device))
        buf_id = _register_ray_cast_buffers(scene_bufs, render_mesh)
        self.register_buffer(
            "_ray_cast_handle",
            torch.tensor([buf_id], dtype=torch.int64, device=device),
            persistent=False,
        )

    def forward(self, target_pixels: torch.Tensor) -> torch.Tensor:
        """Run draw + loss. target_pixels: torch.Tensor [N, 3]. Returns scalar loss."""
        target = wp.to_torch(target_pixels) if hasattr(target_pixels, "ptr") else target_pixels
        return ray_cast_forward(self.mesh_rot, self.mesh_pos, target, self._ray_cast_handle)


class RenderMode:
    """Rendering modes
    grayscale: Lambertian shading from multiple directional lights
    texture: 2D texture map
    normal_map: mesh normal computed from interpolated vertex normals
    """

    grayscale = 0
    texture = 1
    normal_map = 2


@wp.struct
class RenderMesh:
    """Mesh to be ray casted.
    Assumes a triangle mesh as input.
    Per-vertex normals are computed with compute_vertex_normals()
    """

    id: wp.uint64
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    tex_coords: wp.array(dtype=wp.vec2)
    tex_indices: wp.array(dtype=int)
    vertex_normals: wp.array(dtype=wp.vec3)


@wp.struct
class Camera:
    """Basic camera for ray casting"""

    horizontal: float
    vertical: float
    aspect: float
    e: float
    tan: float
    pos: wp.vec3
    rot: wp.quat


@wp.struct
class DirectionalLights:
    """Stores arrays of directional light directions and intensities."""

    dirs: wp.array(dtype=wp.vec3)
    intensities: wp.array(dtype=float)
    num_lights: int


@wp.kernel
def vertex_normal_sum_kernel(
    verts: wp.array(dtype=wp.vec3), indices: wp.array(dtype=int), normal_sums: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()

    i = indices[tid * 3]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    a = verts[i]
    b = verts[j]
    c = verts[k]

    ab = b - a
    ac = c - a

    area_normal = wp.cross(ab, ac)
    wp.atomic_add(normal_sums, i, area_normal)
    wp.atomic_add(normal_sums, j, area_normal)
    wp.atomic_add(normal_sums, k, area_normal)


@wp.kernel
def normalize_kernel(
    normal_sums: wp.array(dtype=wp.vec3),
    vertex_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    vertex_normals[tid] = wp.normalize(normal_sums[tid])


@wp.func
def texture_interpolation(tex_interp: wp.vec2, texture: wp.array2d(dtype=wp.vec3)):
    tex_width = texture.shape[1]
    tex_height = texture.shape[0]
    tex = wp.vec2(tex_interp[0] * float(tex_width - 1), (1.0 - tex_interp[1]) * float(tex_height - 1))

    x0 = int(tex[0])
    x1 = x0 + 1
    alpha_x = tex[0] - float(x0)
    y0 = int(tex[1])
    y1 = y0 + 1
    alpha_y = tex[1] - float(y0)
    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]
    lower = (1.0 - alpha_x) * c00 + alpha_x * c10
    upper = (1.0 - alpha_x) * c01 + alpha_x * c11
    color = (1.0 - alpha_y) * lower + alpha_y * upper

    return color


@wp.kernel
def draw_kernel(
    mesh: RenderMesh,
    rot: wp.array(dtype=wp.quat),
    pos: wp.array(dtype=wp.vec3),
    camera: Camera,
    texture: wp.array2d(dtype=wp.vec3),
    rays_width: int,
    rays_height: int,
    rays: wp.array(dtype=wp.vec3),
    lights: DirectionalLights,
    mode: int,
):
    tid = wp.tid()

    x = tid % rays_width
    y = rays_height - tid // rays_width

    sx = 2.0 * float(x) / float(rays_width) - 1.0
    sy = 2.0 * float(y) / float(rays_height) - 1.0

    # compute view ray in world space
    ro_world = camera.pos
    rd_world = wp.normalize(wp.quat_rotate(camera.rot, wp.vec3(sx * camera.tan * camera.aspect, sy * camera.tan, -1.0)))

    # compute view ray in mesh space
    inv = wp.transform_inverse(wp.transform(pos[0], rot[0]))
    ro = wp.transform_point(inv, ro_world)
    rd = wp.transform_vector(inv, rd_world)

    color = wp.vec3(0.0, 0.0, 0.0)

    query = wp.mesh_query_ray(mesh.id, ro, rd, 1.0e6)
    if query.result:
        i = mesh.indices[query.face * 3]
        j = mesh.indices[query.face * 3 + 1]
        k = mesh.indices[query.face * 3 + 2]

        a_n = mesh.vertex_normals[i]
        b_n = mesh.vertex_normals[j]
        c_n = mesh.vertex_normals[k]

        # vertex normal interpolation
        normal = query.u * a_n + query.v * b_n + (1.0 - query.u - query.v) * c_n

        if mode == 0 or mode == 1:
            if mode == 0:  # grayscale
                color = wp.vec3(1.0)

            elif mode == 1:  # texture interpolation
                tex_a = mesh.tex_coords[mesh.tex_indices[query.face * 3]]
                tex_b = mesh.tex_coords[mesh.tex_indices[query.face * 3 + 1]]
                tex_c = mesh.tex_coords[mesh.tex_indices[query.face * 3 + 2]]

                tex = query.u * tex_a + query.v * tex_b + (1.0 - query.u - query.v) * tex_c

                color = texture_interpolation(tex, texture)

            # lambertian directional lighting
            lambert = float(0.0)
            for i in range(lights.num_lights):
                dir = wp.transform_vector(inv, lights.dirs[i])
                val = lights.intensities[i] * wp.dot(normal, dir)
                if val < 0.0:
                    val = 0.0
                lambert = lambert + val

            color = lambert * color

        elif mode == 2:  # normal map
            color = normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

        if color[0] > 1.0:
            color = wp.vec3(1.0, color[1], color[2])
        if color[1] > 1.0:
            color = wp.vec3(color[0], 1.0, color[2])
        if color[2] > 1.0:
            color = wp.vec3(color[0], color[1], 1.0)

    rays[tid] = color


@wp.kernel
def downsample_kernel(
    rays: wp.array(dtype=wp.vec3), pixels: wp.array(dtype=wp.vec3), rays_width: int, num_samples: int
):
    tid = wp.tid()

    pixels_width = rays_width / num_samples
    px = tid % pixels_width
    py = tid // pixels_width
    start_idx = py * num_samples * rays_width + px * num_samples

    color = wp.vec3(0.0, 0.0, 0.0)

    for i in range(0, num_samples):
        for j in range(0, num_samples):
            ray = rays[start_idx + i * rays_width + j]
            color = wp.vec3(color[0] + ray[0], color[1] + ray[1], color[2] + ray[2])

    num_samples_sq = float(num_samples * num_samples)
    color = wp.vec3(color[0] / num_samples_sq, color[1] / num_samples_sq, color[2] / num_samples_sq)
    pixels[tid] = color


@wp.kernel
def loss_kernel(pixels: wp.array(dtype=wp.vec3), target_pixels: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()

    pixel = pixels[tid]
    target_pixel = target_pixels[tid]

    diff = target_pixel - pixel

    # pseudo Huber loss
    delta = 1.0
    x = delta * delta * (wp.sqrt(1.0 + (diff[0] / delta) * (diff[0] / delta)) - 1.0)
    y = delta * delta * (wp.sqrt(1.0 + (diff[1] / delta) * (diff[1] / delta)) - 1.0)
    z = delta * delta * (wp.sqrt(1.0 + (diff[2] / delta) * (diff[2] / delta)) - 1.0)
    sum = x + y + z

    wp.atomic_add(loss, 0, sum)


@wp.kernel
def normalize(x: wp.array(dtype=wp.quat)):
    tid = wp.tid()

    x[tid] = wp.normalize(x[tid])


def _ray_cast(scene_bufs, render_mesh):
    """Flat ray-cast: draw + downsample. Used internally."""
    wp.launch(
        kernel=draw_kernel,
        dim=scene_bufs.num_rays,
        inputs=[
            render_mesh,
            scene_bufs.rot,
            scene_bufs.pos,
            scene_bufs.camera,
            scene_bufs.texture,
            scene_bufs.rays_width,
            scene_bufs.rays_height,
            scene_bufs.rays,
            scene_bufs.lights,
            scene_bufs.render_mode,
        ],
    )
    wp.launch(
        kernel=downsample_kernel,
        dim=scene_bufs.num_pixels,
        inputs=[scene_bufs.rays, scene_bufs.pixels, scene_bufs.rays_width, pow(2, scene_bufs.num_samples)],
    )


def _torch_pos_to_wp_vec3_array(mesh_pos: torch.Tensor):
    """Torch [3] or [1,3] -> wp array length 1 of vec3 (for wp.copy into scene_bufs.pos)."""
    t = mesh_pos.detach().contiguous()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return wp.from_torch(t, dtype=wp.vec3)


@torch.library.custom_op("wp::ray_cast_forward", mutates_args=())
def ray_cast_forward(
    rot: torch.Tensor,
    mesh_pos: torch.Tensor,
    target_pixels: torch.Tensor,
    buffer_key: torch.Tensor,
) -> torch.Tensor:
    """Forward: copy rot / mesh pos to Warp buffers, run draw + downsample + loss, return loss."""
    rid = int(buffer_key.reshape(-1)[0].item())
    scene_bufs, render_mesh = _RAY_CAST_REGISTRY[rid]
    wp.copy(
        scene_bufs.rot,
        wp.from_torch(rot.detach().contiguous(), dtype=wp.quat),
    )
    wp.copy(scene_bufs.pos, _torch_pos_to_wp_vec3_array(mesh_pos))
    target_wp = wp.from_torch(target_pixels.contiguous(), dtype=wp.vec3)
    _ray_cast(scene_bufs, render_mesh)
    wp.launch(loss_kernel, dim=scene_bufs.num_pixels, inputs=[scene_bufs.pixels, target_wp, scene_bufs.loss])
    return wp.to_torch(scene_bufs.loss)


@torch.library.custom_op("wp::ray_cast_backward", mutates_args=())
def ray_cast_backward(
    rot: torch.Tensor,
    mesh_pos: torch.Tensor,
    loss: torch.Tensor,
    adj_loss: torch.Tensor,
    target_pixels: torch.Tensor,
    buffer_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward: run adjoint of loss, downsample, draw kernels (no Tape)."""
    rid = int(buffer_key.reshape(-1)[0].item())
    scene_bufs, render_mesh = _RAY_CAST_REGISTRY[rid]
    device = wp.get_device()
    num_samples = pow(2, scene_bufs.num_samples)

    target_wp = wp.from_torch(target_pixels.contiguous(), dtype=wp.vec3)
    adj_loss_wp = wp.from_torch(adj_loss.contiguous(), dtype=wp.float32)
    adj_pixels = wp.zeros(scene_bufs.num_pixels, dtype=wp.vec3, device=device, requires_grad=False)
    loss_wp = wp.from_torch(loss.contiguous(), dtype=wp.float32, requires_grad=False)
    wp.launch(
        kernel=loss_kernel,
        dim=scene_bufs.num_pixels,
        inputs=[scene_bufs.pixels, target_wp],
        outputs=[loss_wp],
        adj_inputs=[adj_pixels, None],
        adj_outputs=[adj_loss_wp],
        adjoint=True,
    )

    adj_rays = wp.zeros(scene_bufs.num_rays, dtype=wp.vec3, device=device, requires_grad=False)
    wp.launch(
        kernel=downsample_kernel,
        dim=scene_bufs.num_pixels,
        inputs=[scene_bufs.rays, scene_bufs.pixels, scene_bufs.rays_width, num_samples],
        outputs=[],
        adj_inputs=[adj_rays, adj_pixels, None, None],
        adj_outputs=[],
        adjoint=True,
    )

    adj_rot = wp.zeros(1, dtype=wp.quat, device=device, requires_grad=False)
    adj_pos = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=False)
    adj_mesh = RenderMesh()
    adj_mesh.id = render_mesh.id
    adj_mesh.vertices = None
    adj_mesh.indices = None
    adj_mesh.tex_coords = None
    adj_mesh.tex_indices = None
    adj_mesh.vertex_normals = None

    # Struct args (Camera, DirectionalLights) cannot be None in adj_inputs
    adj_camera = scene_bufs.camera
    adj_lights = scene_bufs.lights
    wp.launch(
        kernel=draw_kernel,
        dim=scene_bufs.num_rays,
        inputs=[
            render_mesh,
            scene_bufs.rot,
            scene_bufs.pos,
            scene_bufs.camera,
            scene_bufs.texture,
            scene_bufs.rays_width,
            scene_bufs.rays_height,
            scene_bufs.rays,
            scene_bufs.lights,
            scene_bufs.render_mode,
        ],
        outputs=[],
        adj_inputs=[adj_mesh, adj_rot, adj_pos, adj_camera, None, None, None, adj_rays, adj_lights, None],
        adj_outputs=[],
        adjoint=True,
    )
    grad_rot = wp.to_torch(adj_rot).reshape(4)
    grad_pos = wp.to_torch(adj_pos).reshape(3)
    return grad_rot, grad_pos


@ray_cast_forward.register_fake
def ray_cast_forward_fake(rot, mesh_pos, target_pixels, buffer_key):
    return torch.empty(1, dtype=torch.float32, device=rot.device)


@ray_cast_backward.register_fake
def ray_cast_backward_fake(rot, mesh_pos, loss, adj_loss, target_pixels, buffer_key):
    return (
        torch.empty(4, dtype=torch.float32, device=rot.device),
        torch.empty(3, dtype=torch.float32, device=rot.device),
    )


def ray_cast_backward_impl(ctx, adj_loss):
    grad_rot, grad_pos = ray_cast_backward(ctx.rot, ctx.mesh_pos, ctx.loss, adj_loss, ctx.target_pixels, ctx.buffer_key)
    return (grad_rot, grad_pos, None, None)


def ray_cast_setup_context(ctx, inputs, output):
    ctx.rot, ctx.mesh_pos, ctx.target_pixels, ctx.buffer_key = inputs
    ctx.loss = output


ray_cast_forward.register_autograd(ray_cast_backward_impl, setup_context=ray_cast_setup_context)


class Example:
    """
    Non-differentiable variables:
    camera.horizontal: camera horizontal aperture size
    camera.vertical: camera vertical aperture size
    camera.aspect: camera aspect ratio
    camera.e: focal length
    camera.pos: camera displacement
    camera.rot: camera rotation (quaternion)
    pix_width: final image width in pixels
    pix_height: final image height in pixels
    num_samples: anti-aliasing. calculated as pow(2, num_samples)
    directional_lights: characterized by intensity (scalar) and direction (vec3)
    render_mesh.indices: mesh vertex indices
    render_mesh.tex_indices: texture indices

    Differentiable variables:
    pos: parent transform translation (vec3), passed separately to draw_kernel
    rot: parent transform rotation (quaternion), passed separately
    render_mesh.vertices: mesh vertex positions
    render_mesh.vertex_normals: mesh vertex normals
    render_mesh.tex_coords: 2D texture coordinates
    """

    def __init__(self, height=1024, train_iters=150, rot_array=None, pos_array=None):
        cam_pos = wp.vec3(0.0, 0.75, 7.0)
        cam_rot = wp.quat(0.0, 0.0, 0.0, 1.0)
        horizontal_aperture = 36.0
        vertical_aperture = 20.25
        aspect = horizontal_aperture / vertical_aperture
        focal_length = 50.0
        self.height = height
        self.width = int(aspect * self.height)
        self.num_pixels = self.width * self.height

        if rot_array is None:
            rot_array = [0.0, 0.0, 0.0, 1.0]

        if pos_array is None:
            pos_array = [0.0, 0.0, 0.0]

        asset_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bunny"))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())
        num_points = points.shape[0]
        num_faces = int(indices.shape[0] / 3)

        # manufacture texture coordinates + indices for this asset
        distance = np.linalg.norm(points, axis=1)
        radius = np.max(distance)
        distance = distance / radius
        tex_coords = np.stack((distance, distance), axis=1)
        tex_indices = indices

        # manufacture texture for this asset
        x = np.arange(256.0)
        xx, yy = np.meshgrid(x, x)
        zz = np.zeros_like(xx)
        texture_host = np.stack((xx, yy, zz), axis=2) / 255.0

        # set anti-aliasing
        self.num_samples = 1

        # set render mode
        self.render_mode = RenderMode.texture

        # set training iterations
        self.train_rate = 5.00e-8
        self.momentum = 0.5
        self.dampening = 0.1
        self.weight_decay = 0.0
        self.train_iters = train_iters
        self.period = 10  # Training iterations between render() calls
        self.iter = 0

        # storage for training animation
        self.images = np.zeros((self.height, self.width, 3, max(int(self.train_iters / self.period), 1)))
        self.image_counter = 0

        # construct RenderMesh
        self.render_mesh = RenderMesh()
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, requires_grad=True),
            indices=wp.array(indices, dtype=int),
        )
        self.render_mesh.id = self.mesh.id
        self.render_mesh.vertices = self.mesh.points
        self.render_mesh.indices = self.mesh.indices
        self.render_mesh.tex_coords = wp.array(tex_coords, dtype=wp.vec2, requires_grad=True)
        self.render_mesh.tex_indices = wp.array(tex_indices, dtype=int)
        self.normal_sums = wp.zeros(num_points, dtype=wp.vec3, requires_grad=True)
        self.render_mesh.vertex_normals = wp.zeros(num_points, dtype=wp.vec3, requires_grad=True)
        self.pos = wp.array(np.asarray(pos_array, dtype=np.float32).reshape(1, 3), dtype=wp.vec3, requires_grad=True)
        self.rot = wp.array(np.array(rot_array), dtype=wp.quat, requires_grad=True)

        # compute vertex normals
        wp.launch(
            kernel=vertex_normal_sum_kernel,
            dim=num_faces,
            inputs=[self.render_mesh.vertices, self.render_mesh.indices, self.normal_sums],
        )
        wp.launch(
            kernel=normalize_kernel,
            dim=num_points,
            inputs=[self.normal_sums, self.render_mesh.vertex_normals],
        )

        # construct camera
        self.camera = Camera()
        self.camera.horizontal = horizontal_aperture
        self.camera.vertical = vertical_aperture
        self.camera.aspect = aspect
        self.camera.e = focal_length
        self.camera.tan = vertical_aperture / (2.0 * focal_length)
        self.camera.pos = cam_pos
        self.camera.rot = cam_rot

        # construct texture
        self.texture = wp.array2d(texture_host, dtype=wp.vec3, requires_grad=True)

        # construct lights
        self.lights = DirectionalLights()
        self.lights.dirs = wp.array(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), dtype=wp.vec3, requires_grad=True)
        self.lights.intensities = wp.array(np.array([2.0, 0.2]), dtype=float, requires_grad=True)
        self.lights.num_lights = 2

        # construct rays
        self.rays_width = self.width * pow(2, self.num_samples)
        self.rays_height = self.height * pow(2, self.num_samples)
        self.num_rays = self.rays_width * self.rays_height
        self.rays = wp.zeros(self.num_rays, dtype=wp.vec3, requires_grad=True)

        # construct pixels
        self.pixels = wp.zeros(self.num_pixels, dtype=wp.vec3, requires_grad=True)
        self.target_pixels = wp.zeros(self.num_pixels, dtype=wp.vec3)

        # loss array
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # scene_bufs: flat buffer container for custom op context
        self.scene_bufs = SimpleNamespace(
            camera=self.camera,
            texture=self.texture,
            lights=self.lights,
            rays=self.rays,
            pixels=self.pixels,
            loss=self.loss,
            rot=self.rot,
            pos=self.pos,
            rays_width=self.rays_width,
            rays_height=self.rays_height,
            num_rays=self.num_rays,
            num_pixels=self.num_pixels,
            num_samples=self.num_samples,
            render_mode=self.render_mode,
        )
        self.model = MeshRotationModule(self.scene_bufs, self.render_mesh, init_rot=rot_array, init_pos=pos_array)

        self.optimizer = torch.optim.SGD(
            [
                self.model.mesh_rot,
            ],
            lr=self.train_rate,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
        )

        # Tape is created per-step in RayCastAndLoss (no cuda graph with torch optimizer)
        self.use_cuda_graph = False

    def ray_cast(self, render_mesh=None, rot=None):
        """Ray cast using render_mesh (or self.render_mesh if None) and rot (or self.rot if None)."""
        mesh = render_mesh if render_mesh is not None else self.render_mesh
        rot_arr = rot if rot is not None else self.rot
        wp.launch(
            kernel=draw_kernel,
            dim=self.num_rays,
            inputs=[
                mesh,
                rot_arr,
                self.pos,
                self.camera,
                self.texture,
                self.rays_width,
                self.rays_height,
                self.rays,
                self.lights,
                self.render_mode,
            ],
        )

        # downsample
        wp.launch(
            kernel=downsample_kernel,
            dim=self.num_pixels,
            inputs=[self.rays, self.pixels, self.rays_width, pow(2, self.num_samples)],
        )

    def _warp_forward_with(self, render_mesh, target_pixels):
        """Warp forward with given render_mesh and target_pixels. Used by MeshRotationModule."""
        self.ray_cast(render_mesh=render_mesh)
        target_wp = (
            wp.from_torch(target_pixels.contiguous(), dtype=wp.vec3)
            if isinstance(target_pixels, torch.Tensor)
            else target_pixels
        )
        wp.launch(loss_kernel, dim=self.num_pixels, inputs=[self.pixels, target_wp, self.loss])

    def _warp_forward(self):
        """Warp-only forward: ray cast + loss. Used for backward compatibility."""
        self._warp_forward_with(self.render_mesh, self.target_pixels)

    def forward(self):
        """Full forward for backward compatibility (e.g. CUDA graph capture)."""
        self._warp_forward()

    def step(self):
        with wp.ScopedTimer("step"):
            self.optimizer.zero_grad()
            loss = self.model(wp.to_torch(self.target_pixels))
            loss.backward()
            self.optimizer.step()

            # Normalize quaternion after optimizer step
            with torch.no_grad():
                self.model.mesh_rot.data /= self.model.mesh_rot.data.norm()
            wp.copy(self.rot, wp.from_torch(self.model.mesh_rot.detach(), dtype=wp.quat))
            wp.copy(self.pos, _torch_pos_to_wp_vec3_array(self.model.mesh_pos.detach()))

            if self.iter % self.period == 0:
                print(f"Iter: {self.iter} Loss: {loss.item():.6f}")

            self.loss.zero_()
            self.iter = self.iter + 1

    def render(self):
        with wp.ScopedTimer("render"):
            self.images[:, :, :, self.image_counter] = self.get_image()
            self.image_counter += 1

    def get_image(self):
        return self.pixels.numpy().reshape((self.height, self.width, 3))

    def get_animation(self):
        fig, ax = plt.subplots()
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        frames = []
        for i in range(self.images.shape[3]):
            frame = ax.imshow(self.images[:, :, :, i], animated=True)
            frames.append([frame])

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
        return ani


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--train_iters", type=int, default=150, help="Total number of training iterations.")
    parser.add_argument("--height", type=int, default=1024, help="Height of rendered image in pixels.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        reference_example = Example(height=args.height)

        # render target rotation
        reference_example.ray_cast()

        # offset mesh rotation
        example = Example(
            train_iters=args.train_iters,
            height=args.height,
            rot_array=[
                0.0,
                (math.sqrt(3) - 1) / (2.0 * math.sqrt(2.0)),
                0.0,
                (math.sqrt(3) + 1) / (2.0 * math.sqrt(2.0)),
            ],
        )

        wp.copy(example.target_pixels, reference_example.pixels)

        # recover target rotation
        for i in range(example.train_iters):
            example.step()

            if i % example.period == 0:
                example.render()

        if not args.headless:
            import matplotlib.animation as animation
            import matplotlib.image as img
            import matplotlib.pyplot as plt

            target_image = reference_example.get_image()
            target_image_filename = "example_diffray_target_image.png"
            img.imsave(target_image_filename, target_image)
            print(f"Saved the target image at `{target_image_filename}`")

            final_image = example.get_image()
            final_image_filename = "example_diffray_final_image.png"
            img.imsave(final_image_filename, final_image)
            print(f"Saved the final image at `{final_image_filename}`")

            anim = example.get_animation()
            anim_filename = "example_diffray_animation.gif"
            anim.save(anim_filename, dpi=300, writer=animation.PillowWriter(fps=5))
            print(f"Saved the animation at `{anim_filename}`")
