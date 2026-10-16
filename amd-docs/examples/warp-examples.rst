.. meta::
  :description: Warp examples
  :keywords: Warp, programming, ROCm, example, sample, tutorial, simulation

.. _run-a-warp-example:

********************************************************************
Run a Warp example
********************************************************************

These examples assume that Warp is installed as described in
:doc:`Install Warp <../install/warp-install>`. Run the examples in the same
Docker container where Warp is installed.

Write and run a Warp kernel
====================================================================

Warp kernels are Python functions decorated with ``@wp.kernel``. They are
compiled for your AMD GPU the first time they are launched. AMD GPUs are
addressed through the ``cuda:N`` device names.

1. Save the following code snippet to a Python script named ``warp_example.py``.

   .. code-block:: python

      import numpy as np
      import warp as wp


      @wp.kernel
      def saxpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
          i = wp.tid()
          y[i] = a * x[i] + y[i]


      n = 1_000_000
      device = wp.get_device("cuda:0")

      x = wp.array(np.arange(n, dtype=np.float32), device=device)
      y = wp.ones(n, dtype=float, device=device)

      wp.launch(saxpy, dim=n, inputs=[2.0, x, y], device=device)

      result = y.numpy()
      print(f"Ran on {device.name} ({device.arch_str})")
      print("y[:5] =", result[:5])
      assert np.allclose(result, 2.0 * np.arange(n, dtype=np.float32) + 1.0)
      print("SAXPY result is correct")

2. Run the script.

   .. code-block:: bash

      python warp_example.py

   After Warp prints its initialization banner, the output ends with lines
   similar to the following:

   .. code-block:: text

      Ran on AMD Instinct MI355X (gfx950:sramecc+:xnack-)
      y[:5] = [1. 3. 5. 7. 9.]
      SAXPY result is correct

   Compiled kernels are cached in ``~/.cache/warp``, so later runs start faster.

Run the built-in examples
====================================================================

The `warp/examples <https://github.com/AMD-Ecosystem/warp/tree/amd-integration/warp/examples>`__
directory contains scripts, grouped into subdirectories, that show how to
implement various simulation methods using the Warp API. Most examples write
USD files with time-sampled animations to the current working directory.

1. Install the packages used by the examples.

   .. code-block:: bash

      pip install usd-core matplotlib

   If ``pip`` reports an ``externally-managed-environment`` error, add
   ``--break-system-packages`` to the ``pip install`` command.

2. Run an example from the command line. Pass ``--headless`` so that examples
   that can open a plot window run without a display.

   .. code-block:: bash

      python -m warp.examples.<example_subdir>.<example> --headless

   For example, to run the discrete element method (DEM) simulation:

   .. code-block:: bash

      python -m warp.examples.core.example_dem --headless

   This writes the animation to ``example_dem.usd``, which you can open in any
   USD viewer.

The following examples run on AMD Instinct GPUs:

.. list-table::
   :header-rows: 1
   :align: left

   * - Module
     - Description
   * - ``warp.examples.core.example_dem``
     - Discrete element method (DEM) particle simulation with cohesion, using a hash grid for neighbor search
   * - ``warp.examples.core.example_marching_cubes``
     - Iso-surface extraction from a density field with marching cubes
   * - ``warp.examples.core.example_mesh``
     - Position-based dynamics particles colliding with a deforming triangle mesh
   * - ``warp.examples.core.example_raycast``
     - Basic ray tracer using the built-in ``wp.Mesh`` data structure
   * - ``warp.examples.core.example_raymarch``
     - Signed distance field (SDF) ray marching renderer
   * - ``warp.examples.core.example_sample_mesh``
     - Sampling points on a mesh surface using a cumulative distribution function
   * - ``warp.examples.core.example_sph``
     - Smoothed particle hydrodynamics (SPH) fluid simulation
   * - ``warp.examples.core.example_torch``
     - Optimizing the Rosenbrock function with the PyTorch Adam optimizer (requires PyTorch)
   * - ``warp.examples.core.example_wave``
     - 2D wave equation solver with collision against a moving sphere
   * - ``warp.examples.optim.example_diffray``
     - Differentiable ray caster
   * - ``warp.examples.tile.example_tile_mlp``
     - Coordinate-based multilayer perceptron (MLP) trained on an image with tile primitives

Examples that use the OpenGL renderer, such as ``example_render_opengl``,
might not work because AMD Instinct GPUs run headless.

Learn more
====================================================================

For more information about the Warp API, interoperability with PyTorch, and
differentiable simulation, see the
`upstream Warp documentation <https://nvidia.github.io/warp/>`__.
