.. meta::
  :description: What is Warp?
  :keywords: Warp, documentation, simulation, robotics, GPU, AMD, ROCm, overview, introduction

.. _what-is-warp:

********************************************************************
What is Warp?
********************************************************************

`Warp <https://github.com/NVIDIA/warp>`__ is a Python framework for writing
high-performance simulation and graphics code. Warp takes regular Python
functions decorated with ``@wp.kernel`` and JIT compiles them to efficient
kernel code that can run on the CPU or GPU.

Warp is designed for `spatial computing <https://en.wikipedia.org/wiki/Spatial_computing>`__
and comes with a rich set of primitives that make it easy to write programs for
physics simulation, perception, robotics, and geometry processing. In addition,
Warp kernels are differentiable and can be used as part of machine-learning
pipelines with frameworks such as PyTorch.

Warp on ROCm is a port of Warp that adds a HIP backend for AMD Instinct GPUs.
HIP is detected automatically at build time, and AMD GPUs are exposed to Warp
programs through the same ``cuda:N`` device names used upstream, so existing
Warp code runs on AMD Instinct GPUs without modification.

Why Warp?
====================================================================

Warp is well suited for GPU-accelerated simulation for the following reasons:

- Kernels are written in Python and compiled to native GPU code, which keeps
  research code readable while delivering high performance.

- Built-in support for automatic differentiation makes it possible to
  optimize through simulations and embed them in training loops.

- Zero-copy interoperability with PyTorch and NumPy allows Warp to be dropped
  into existing machine-learning and robotics pipelines.

- The ROCm build targets AMD Instinct hardware, so large batched simulations
  (for example, thousands of parallel robot environments) can run on AMD GPUs.

Features and use cases
====================================================================

Warp provides the following key features:

- **Python-based kernel programming:** Write SIMT kernels in Python using
  ``@wp.kernel`` and ``@wp.func``, with a rich set of vector, matrix, and
  quaternion types.

- **Differentiable programming:** Record kernel launches with ``wp.Tape`` and
  compute gradients through entire simulations.

- **Spatial data structures:** Use built-in meshes, hash grids, bounding volume
  hierarchies (BVH), and sparse volumes for ray casting, collision detection, and
  neighbor queries.

- **Tile programming:** Use cooperative tile primitives for operations such as
  matrix multiplication, reductions, and small neural networks inside kernels.

- **Framework interoperability:** Exchange arrays with PyTorch and NumPy without
  copying data.

Warp is commonly used in the following scenarios:

- **Robotics and reinforcement learning:** Run massively parallel physics
  environments to train robot control policies.

- **Physics simulation:** Implement particle, rigid-body, cloth, and fluid
  solvers that run entirely on the GPU.

- **Geometry processing and perception:** Accelerate ray casting, mesh
  queries, and point cloud processing.

- **Differentiable simulation research:** Optimize physical parameters and
  controllers by back-propagating through simulations.

Supported GPUs
====================================================================

Warp on ROCm compiles its native library ahead of time for a specific AMD GPU
architecture. The following architectures are supported:

.. list-table::
   :header-rows: 1
   :align: left

   * - GPU architecture
     - AMD Instinct GPUs
     - ROCm version
   * - ``gfx942``
     - MI325X
     - 7.2.0
   * - ``gfx950``
     - MI350X, MI355X
     - 10.1.0

For the complete list of software requirements, see :doc:`Install Warp <install/warp-install>`.
