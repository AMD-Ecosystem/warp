.. meta::
  :description: installing Warp for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, Warp, MI325X, MI350X/MI355X

.. _warp-on-rocm-installation:

********************************************************************
Warp on ROCm installation
********************************************************************

System requirements
====================================================================

To use Warp `1.13.0+rocm.0 <https://github.com/AMD-Ecosystem/warp>`__, you need the following prerequisites:

- **ROCm version:** `7.2.0 <https://rocm.docs.amd.com/en/docs-7.2.0/>`__ (MI325X), `10.1.0 <https://rocm.docs.amd.com/en/docs-10.1.0/>`__ (MI350X, MI355X)
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI325X, MI350X, MI355X
- **PyTorch:** `2.9.1 <https://github.com/ROCm/pytorch/tree/release/2.9>`__ (MI325X), `2.14.0 <https://github.com/ROCm/pytorch/tree/release/2.14>`__ (MI350X, MI355X)
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31213/>`__

Install Warp
====================================================================

To run Warp with ROCm enabled, build from source using the provided Dockerfile.

Build Warp from source
--------------------------------------------------------------------

1. Clone the Warp repository containing the Dockerfile:

   .. code-block:: bash

      git clone -b amd-integration https://github.com/AMD-Ecosystem/warp.git
      cd warp

2. Build the Docker image for your GPU architecture:

   .. tab-set::

      .. tab-item:: MI325X (ROCm 7.2.0)

         .. code-block:: bash

            docker build --network=host -f docker/rocm/Dockerfile --target warp_gfx942 -t warp:rocm_7.2_ubuntu24.04_py3.12_torch2.9.1 .

      .. tab-item:: MI350X/MI355X (ROCm 10.1.0)

         .. code-block:: bash

            docker build --network=host -f docker/rocm/Dockerfile --target warp_gfx950 -t warp:rocm10.1.0_ubuntu24.04_py3.12_torch2.14.0 .

   This builds an image with Warp 1.13.0+rocm.0 and the required dependencies, including
   PyTorch 2.9.1 (MI325X) or PyTorch 2.14.0 (MI350X, MI355X).

3. Run the Docker container:

   .. tab-set::

      .. tab-item:: MI325X (ROCm 7.2.0)

         .. code-block:: bash

            docker run -it --rm \
               --name my_warp \
               --device /dev/kfd \
               --device /dev/dri \
               --privileged \
               --network=host \
               --group-add video \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --ipc=host \
               --shm-size=16g \
               warp:rocm_7.2_ubuntu24.04_py3.12_torch2.9.1

      .. tab-item:: MI350X/MI355X (ROCm 10.1.0)

         .. code-block:: bash

            docker run -it --rm \
               --name my_warp \
               --device /dev/kfd \
               --device /dev/dri \
               --privileged \
               --network=host \
               --group-add video \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --ipc=host \
               --shm-size=16g \
               warp:rocm10.1.0_ubuntu24.04_py3.12_torch2.14.0

   .. note::

      To check which architecture your GPUs use, run ``rocm_agent_enumerator | sort -u``.
      ``gfx942`` corresponds to MI325X, and ``gfx950`` corresponds to MI350X and MI355X.

Test the Warp installation
====================================================================

After starting the container, verify that Warp and ROCm are working correctly.

1. Confirm that Warp is installed and detects your GPUs:

   .. code-block:: bash

      python -c "import warp as wp; wp.init(); print(wp.get_device('cuda:0').arch_str)"

   Expected output on an AMD Instinct MI355X:

   .. code-block:: text

      Warp 1.13.0+rocm.0 initialized:
         ...
         Devices:
           "cpu"      : "x86_64"
           "cuda:0"   : "AMD Instinct MI355X" (288 GiB, MI350X/MI355X:sramecc+:xnack-, mempool enabled)
           ...
      MI350X/MI355X:sramecc+:xnack-

   On ROCm, Warp reports AMD GPUs through the ``cuda:N`` device names.

2. Optionally, run the Warp unit tests:

   .. code-block:: bash

      python -m warp.tests -j 8

   Running the full test suite takes a while. Use ``-j`` to set the number of
   parallel test processes.

If all checks pass, proceed to :doc:`run a Warp example <../examples/warp-examples>`.
