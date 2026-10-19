.. meta::
  :description: Warp documentation
  :keywords: Warp, ROCm, documentation, simulation, robotics, spatial computing, GPU

.. _warp-documentation-index:

********************************************************************
Warp on ROCm documentation
********************************************************************

Accelerate physics simulation, robotics, and spatial computing workloads with
Warp on ROCm for AMD Instinct GPUs, writing GPU kernels in regular Python that
are JIT-compiled to efficient HIP code.

`Warp <https://github.com/AMD-Ecosystem/warp>`__ is a Python framework for writing
high-performance simulation and graphics code. Warp takes regular Python
functions and JIT compiles them to efficient kernel code that can run on the
CPU or GPU. It comes with a rich set of primitives that make it easy to write
programs for physics simulation, perception, robotics, and geometry processing.

Warp on ROCm adds HIP support so that Warp kernels run on AMD Instinct MI325X,
MI350X, and MI355X GPUs. Warp kernels are differentiable and can be used
as part of machine-learning pipelines with frameworks such as PyTorch.

Warp is part of the `AMD Simulation toolkit
<https://rocm.docs.amd.com/projects/rocm-simulation/en/docs-26.10/>`__.

The Warp public repository is located at `https://github.com/AMD-Ecosystem/warp <https://github.com/AMD-Ecosystem/warp>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install Warp <install/warp-install>`

  .. grid-item-card:: Examples

    * :doc:`Run a Warp example <examples/warp-examples>`

To contribute to the documentation, refer to
`Contributing to Warp <https://github.com/AMD-Ecosystem/warp/blob/amd-integration/CONTRIBUTING.md>`__.

You can find licensing information on the :doc:`Licensing <about/license>` page.
