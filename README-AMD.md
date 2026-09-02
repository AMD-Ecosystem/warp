# Warp on AMD HIP/ROCm

AMD HIP/ROCm build of NVIDIA Warp 1.17.0.dev4 (upstream main).

## Quickstart

Use a clean, **isolated environment** (installing into the system Python often
fails because build/runtime dependencies are missing). Conda below is the
turnkey option; if you already have Python 3.11/3.12 a plain `venv` works too —
see [Environment setup](#environment-setup). The short version:

```bash
# 0. Clone this branch. The repository is private: you need access, and the
#    clone must go over SSH (or an authenticated HTTPS remote). git-lfs must
#    be installed FIRST — the USD/NVDB test and example assets are LFS
#    objects, and without it the clone silently contains pointer files.
git lfs install
git clone -b amd-integration-halo https://github.com/AMD-Ecosystem/warp
cd warp

# 1. Create and activate an environment (Python 3.12; 3.11 also works)
conda create -n warp-amd python=3.12 -y
conda activate warp-amd

# 2. Install build dependencies
pip install numpy setuptools packaging wheel
# Optional, only for Warp's torch interop — NOT needed for Warp or Newton.
# PyTorch for gfx1151 comes from AMD's gfx1151 index (the stock
# download.pytorch.org ROCm wheels do not carry gfx1151 kernels; rolling
# nightly index):
#   pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# 3. Build Warp for HIP. The compile takes about 1-2 minutes at the default
#    8 jobs (measured 72 s); the first-ever build additionally downloads the
#    LLVM/Clang toolchain dependency. Must run before step 4:
#    `pip install -e .` expects warp/bin/warp.so to already exist.
#    build_amd.sh auto-detects python/python3; override with
#    PYTHON=/path/to/python
./build_amd.sh

# 4. Install into the active environment, then sanity-check the build.
#    On ROCm 7.14, export LD_LIBRARY_PATH first (see Prerequisites) or
#    warp.so will not load.
pip install -e .
python tools/run_gfx1151_smoke.py
```

Recommended ROCm version: **7.14.0** — the version all current recorded
numbers were produced on. See [Choosing a ROCm version](#choosing-a-rocm-version),
including a validated container route that needs no host ROCm upgrade.

A passing smoke test means the build loads and runs kernels. It does **not**
mean the build is sound: it runs two elementwise kernels on 1024 floats and
exercises no graph capture, no sustained execution and no BVH or mesh query, so
it cannot detect any of the defects in `KNOWN_ISSUES-AMD.md`. Sustained
throughput in particular must be measured over minutes, because the ROCm 7.2.x
graph-capture leak only becomes visible after a few hundred frames.

To run Newton physics examples (ANYmal, cloth, softbody, etc.) see
[Newton integration](#newton-integration).
(`torch` in step 2 is optional; it is not required for Warp or Newton.)

## Supported hardware

This port is **validated only on `gfx1151`** (AMD Radeon 8060S / Strix Halo iGPU). Other RDNA architectures are buildable but untested by the port maintainers.

| Architecture | Status | Notes |
|---|---|---|
| `gfx1151` | Validated | Test matrix and per-example throughput in `KNOWN_ISSUES-AMD.md` |
| `gfx1100`, `gfx1101`, `gfx1102` | Buildable | RDNA 3 dGPU; untested |
| `gfx1030` | Buildable | RDNA 2; untested |

If you run on something other than gfx1151 — pass or fail — please open an issue.

Confirm your GPU:

```bash
rocminfo | grep -E 'Name:|gfx'
# or, if rocminfo is not on PATH:
find /opt/rocm -name rocminfo -executable 2>/dev/null
```

## Choosing a ROCm version

**Recommended: ROCm 7.14.0.** All current recorded numbers — the five-suite
validation matrix (Warp, Newton, MuJoCo-Warp, benchmarks, CDNA build), the
per-example Newton FPS figures, and the unit-test results — were produced on
ROCm 7.14.0, run against an NVIDIA reference machine at the same commits.
ROCm 7.2.x is also validated and remains necessary for one workload class
below.

The two releases fail in opposite directions:

**ROCm 7.2.x does not reclaim memory allocated inside a captured graph.** Every
graph replay leaks, so sustained simulation degrades badly: the ANYmal C
walking example loses most of its throughput within a few hundred frames and
keeps degrading. On ROCm 7.14.0 the same example holds a steady rate.

**ROCm 7.14.0 fails small device allocations under capture-heavy load.**
`hipMalloc` returns NULL for 56-byte requests with 97 per cent of device memory
free, which costs several Newton test modules outright.

For sustained simulation and reinforcement-learning rollouts, use 7.14.0. For
workloads that trip the allocation failure, 7.2.x remains necessary. Detail and
evidence for both are in `KNOWN_ISSUES-AMD.md`.

Two 7.14.0 specifics:

- The runtime libraries live under `/opt/rocm/core-7.14/lib` behind an
  `/etc/alternatives` symlink chain, and no `ld.so.conf.d` entry is installed,
  so `warp.so` does not load until that directory is on `LD_LIBRARY_PATH`
  (command under [Prerequisites](#prerequisites)).
- A native 7.14 install is not required: the validated configuration runs the
  7.14.0 userspace in a container over the host's existing kernel driver
  (7.2.1-era driver confirmed working underneath). See
  [ROCm 7.14 in a container](#rocm-714-in-a-container).

### ROCm 7.14 in a container

The validated container is `rocm/dev-ubuntu-24.04:7.14.0-full` under rootless
podman with GPU passthrough. Build from the repository checkout:

```bash
podman run --rm \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -v "$PWD":/work rocm/dev-ubuntu-24.04:7.14.0-full \
  bash -c 'export PIP_BREAK_SYSTEM_PACKAGES=1 && \
           pip install numpy setuptools packaging wheel && \
           cd /work && python3 build_lib.py --no-cuda --rocm-path=/opt/rocm --hip-arch=gfx1151 --quick'
```

The image ships `pip` but not `python3.12-venv`; with `--rm`, packages
installed into the container filesystem are gone on exit, so for repeated use
install them onto the mounted volume with `pip install --target` and put that
directory on `PYTHONPATH`. To run the resulting `warp.so` (in or out of the
container) set `LD_LIBRARY_PATH` as under [Prerequisites](#prerequisites).

The container's ROCm tree can also be installed natively, side by side with an
existing apt-installed ROCm: copy `/opt/rocm/core-7.14` out of the image to
`/opt/rocm-7.14.0`, register it with
`update-alternatives --install /opt/rocm rocm /opt/rocm-7.14.0 <priority>`,
and point one `ld.so.conf.d` entry at `/opt/rocm/lib` (replacing any entry
pinned to another version — two registered versions in the linker path let a
process mix libraries). This layout is validated: the host build, smoke test,
and Newton examples run against it with no environment variables, and versions
switch with `update-alternatives --set rocm /opt/rocm-<version>` plus
`ldconfig`.

## Tested configuration

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.3 |
| Kernel | 7.0.0-28-generic, in-tree `amdgpu` (6.17.0-1017-oem also validated; some other kernels are known bad — see [Troubleshooting](#troubleshooting)) |
| GPU | AMD Radeon 8060S (gfx1151), 96 GiB unified memory |
| ROCm | 7.14.0 (recommended; all current recorded numbers). Validated both in a `rocm/dev-ubuntu-24.04:7.14.0-full` container and as a native install of the same tree at `/opt/rocm-7.14.0`, selected via `update-alternatives`, with 7.2.1 kept alongside. **See [Choosing a ROCm version](#choosing-a-rocm-version).** |
| Newton | 1.4.0, branch `tomas/gfx1151-fixes` of `github.com/tomasthoresen/newton` (two commits on top of the 1.4.0 release — see [Newton integration](#newton-integration)) |
| MuJoCo | 3.10.0 |
| MuJoCo-Warp | 3.10.0.2 (Newton 1.4.0 is incompatible with 3.10.0.3, which removed `Model.qLD_dof_simple`) |
| usd-core | 26.3 (Newton 1.4.0 requires `>=25.5,<26.5`; 26.8 is unusable with it) |
| JAX | 0.11.0 |
| PyTorch | 2.12/2.13 nightlies from `https://rocm.nightlies.amd.com/v2/gfx1151/` on the ROCm 7.2.1 host. Not usable inside the 7.14.0 container: the wheel bundles a rocm-sdk whose `libamd_comgr` conflicts with the container's LLVM. |
| Python | 3.12 (3.11 also tested) |
| GCC | 13 (13.3.0 tested) |

### Validated environment (full pinned baseline)

Historical record of the original 1.12.0.dev0 port, retained for provenance.
These pins were recorded in May 2026 against Newton 1.0.0 and do **not**
describe the current tree, which is Warp 1.17.0.dev4 with Newton 1.4.0 and 104
examples.

```
warp-lang             1.12.0.dev0   # editable install of an AMD port branch
mujoco                3.5.0
mujoco-warp           3.5.0.2
newton                1.0.0
torch                 2.11.0+rocm7.2
torchvision           0.26.0+rocm7.2
```

Two known-working warp ports exist for gfx1151. Either one can be the
editable install behind `warp-lang 1.12.0.dev0` / `1.12.1`:

| Tag / branch | warp version | Notes |
|---|---|---|
| `gfx1151-anymal-working` (branch `amd-integration`, commit `5e9aef4c`) | 1.12.0.dev0 | Original 10-patch port; confirmed working ANYmal baseline. |
| `v1.12.1-amd-gfx1151-uma` (branch `amd-port-v1.12.1`, this repo) | 1.12.1 | UMA hybrid allocator + HIP `graphInstantiate` worker-thread fix + capture-aware allocator, tile-reduce, int64-atomic and pow fixes. |

To check which build is active in a given conda env:

```bash
pip show warp-lang | grep -E '^(Version|Editable project location)'
```

### Kernel selection

The in-tree `amdgpu` module must work correctly on gfx1151; not every
kernel's does.

| Kernel | Status |
|---|---|
| `7.0.0-28-generic` | Validated (current test configuration) |
| `6.17.0-1017-oem` (srcversion `FC7DA320ED9D733CA6A3F1E`) | Validated |
| `6.17.0-1020-oem` | Known bad: page-faults on the first GPU dispatch (CPF MAPPING_ERROR / PERMISSION_FAULTS), hanging the process at 100% CPU |
| `amdgpu-dkms` 30.30 (6.16.x) and 31.20 (6.19.x) series | Known bad: same fault class |

See [Troubleshooting](#troubleshooting) for symptoms and recovery. If your
distribution's update mechanism can replace a working kernel (observed with
Ubuntu OEM kernels via `unattended-upgrade`), hold the kernel and its
meta-package:

```bash
sudo apt-mark hold \
    linux-image-6.17.0-1017-oem \
    linux-modules-6.17.0-1017-oem \
    linux-headers-6.17.0-1017-oem \
    linux-image-oem-24.04d \
    linux-oem-24.04c \
    linux-oem-24.04d \
    linux-headers-oem-24.04d
```

The meta-package (`linux-image-oem-24.04d` or whichever your system
tracks) is the critical one — that is what `unattended-upgrade`
follows. Confirm the holds with `apt-mark showhold | grep linux`.

### Verifying the environment matches the baseline

This subsection applies to the historical 1.12 baseline above, not to the
current 1.17.0.dev4 stack: on the current configuration the script exits
nonzero by design (it checks the 1.12-era kernel and package pins). Run it only to compare
against that baseline:

```bash
bash tools/check_gfx1151_baseline.sh           # exit 0 = matches baseline
bash tools/check_gfx1151_baseline.sh --strict  # also fail on missing kernel pin
```

It validates kernel + amdgpu srcversion, `rocminfo` reports gfx1151,
mujoco/mujoco-warp/newton/torch pins, the editable warp install path
and git state, and that the ROCm/kernel apt holds are in place.

Manual equivalent if you want to spot-check individual pieces:

```bash
# Kernel + driver
uname -r                            # expect: 6.17.0-1017-oem
modinfo amdgpu | awk '/srcversion/' # expect: srcversion: FC7DA320ED9D733CA6A3F1E

# Python stack (don't import warp/newton — it can hang on a broken kernel)
pip list | grep -iE '^(warp|mujoco|newton|torch)'

# Apt holds
apt-mark showhold | grep -E 'linux-image-6\.17|linux-image-oem|amdgpu-dkms|hsa-'
```

If `mujoco` or `mujoco-warp` have drifted (commonly upgraded by a
transitive dep), restore with:

```bash
pip install mujoco==3.5.0 mujoco-warp==3.5.0.2 --force-reinstall --no-deps
```

## Prerequisites

- ROCm at `/opt/rocm`. Recommended: 7.14.0 (the version all current recorded
  numbers were produced on — natively or via
  [the container](#rocm-714-in-a-container)); 7.2.1 and 7.2.4 also validated.
  See [Choosing a ROCm version](#choosing-a-rocm-version).
  **On 7.14 the runtime libraries moved** to `/opt/rocm/core-<version>/lib`
  with no `ld.so.conf.d` entry installed, so `warp.so` will not load until that
  directory is on `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH="$(dirname "$(find -L /opt/rocm -maxdepth 3 -name 'libamdhip64.so.7' | head -1)")":$LD_LIBRARY_PATH
  ```
- Linux kernel with the **in-tree** `amdgpu` driver. Do **not** install `amdgpu-dkms` — see [Troubleshooting](#troubleshooting)
- GCC 13
- Python 3.11 or 3.12 with `numpy`, `setuptools`, `packaging`, `wheel`
- `git-lfs`, installed before cloning (`git lfs install`) — the USD/NVDB test
  and example assets are LFS objects; without it the clone contains pointer
  files and anything referencing those assets fails.
- Optional: PyTorch built with ROCm (`torch>=2.11.0+rocm7.2`), only for Warp's
  torch interop. Not required for Warp or Newton.

## Environment setup

You need **Python 3.11 or 3.12** in an isolated environment. Conda is not
required: if you already have one of those Python versions, a plain `venv` is
enough — skip conda entirely:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy setuptools packaging wheel
```

Conda is recommended mainly when you **don't** have the right Python, since it
provides both a pinned interpreter and the environment in one step (`pyenv` or
`uv python install 3.11` work too). Example with conda:

```bash
conda create -n warp-amd python=3.11 -y
conda activate warp-amd

# PyTorch built for gfx1151, from AMD's gfx1151 index (the stock
# download.pytorch.org ROCm wheels do not carry gfx1151). Rolling nightly
# index: you get the current build; the validated baseline was 2.11.0+rocm7.2.
pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# Build dependencies
pip install numpy setuptools packaging wheel

# After Warp is built, install it (see Build below):
# pip install -e .

# For Newton physics examples, see the pinned stack under
# [Newton integration](#newton-integration).
```

PyTorch is optional (Warp's torch interop and nothing else requires it). If it
is installed, import `torch` before `warp` in any process so the correct ROCm
runtime is loaded first.

> **This PyTorch build is GPU-accelerated on the iGPU** — it is not a CPU-only
> build. ROCm PyTorch reuses PyTorch's **CUDA** namespace for the AMD GPU, so
> `torch.device("cuda")`, `torch.cuda.is_available()`, and `tensor.to("cuda")`
> all target the gfx1151 iGPU via HIP (this is *not* NVIDIA CUDA). Verify:
>
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.hip)"
> # -> True AMD Radeon 8060S Graphics 7.2.xxxxx
> ```
>
> If `is_available()` is `False` or `torch.version.hip` is `None`, the wrong
> (CPU-only or non-gfx1151) wheel is installed — reinstall from the gfx1151
> index above.

## Build

```bash
./build_amd.sh
```

Environment variables:

- `HIP_ARCH` — comma-separated target architectures (default: `gfx1151`)
- `ROCM_PATH` — ROCm install path (default: `/opt/rocm`)

Equivalent explicit invocation:

```bash
python build_lib.py --no-cuda --rocm-path=/opt/rocm --hip-arch=gfx1151 --quick
```

Output: `warp/bin/warp.so` and `warp/bin/warp-clang.so` (the CPU kernel
compiler). The compile takes about 1-2 minutes per architecture at the default
8 jobs on a modern desktop CPU (measured 72 s for gfx1151); the first-ever
build additionally downloads the LLVM/Clang toolchain dependency.

## Install

```bash
pip install -e .
```

## Smoke test

```bash
python tools/run_gfx1151_smoke.py
```

Exercises kernel compilation, device memory allocation, and two elementwise
kernel launches. On success the output includes `✅ GFX1151 smoke test PASSED`.
It does not exercise graph capture, BVH/mesh queries, or sustained execution;
for those classes see `KNOWN_ISSUES-AMD.md`.

## Newton integration

Install the pinned MuJoCo stack and Newton 1.4.0 from the gfx1151 branch into
the same environment:

```bash
pip install mujoco==3.10.0 mujoco-warp==3.10.0.2 "usd-core>=25.5,<26.5"
git clone -b tomas/gfx1151-fixes git@github.com:tomasthoresen/newton
pip install -e ./newton --no-deps
```

Pin rationale: Newton 1.4.0 requires `usd-core>=25.5,<26.5` and is
incompatible with mujoco-warp 3.10.0.3, which removed `Model.qLD_dof_simple`.
The `tomas/gfx1151-fixes` branch is the 1.4.0 release plus two commits:
disable mujoco-warp conditional graphs where the platform does not support
them, and the matching model-comparison test change.

**Conditional graph nodes are not supported on HIP** (the API does not exist
in ROCm's headers). On the branch above, Newton's MuJoCo solver detects this
and sets `graph_conditional = False` itself, so stock examples run directly:

```bash
python -m newton.examples robot_anymal_c_walk --viewer null --benchmark --num-frames 1200
```

With **stock** Newton 1.4.0 instead of the branch, the MuJoCo solver takes the
conditional-graph path and raises at graph capture
(`Conditional graph nodes are not supported on HIP/ROCm`). Workaround wrapper
for that case:

```bash
python -c "
import newton.solvers
_o = newton.solvers.SolverMuJoCo.__init__
def _p(self, *a, **kw):
    _o(self, *a, **kw)
    if hasattr(self, 'mjw_model') and self.mjw_model is not None:
        self.mjw_model.opt.graph_conditional = False
newton.solvers.SolverMuJoCo.__init__ = _p

import sys, runpy
sys.argv = ['newton.examples', 'robot_anymal_c_walk', '--viewer', 'null', '--benchmark', '--num-frames', '1200']
runpy.run_module('newton.examples', run_name='__main__')
"
```

`graph_conditional = False` is also a performance win in eager (non-captured)
stepping on this port: with the default `True`, mujoco-warp's solver does a
device-to-host sync per solver iteration to check convergence; with `False` it
launches a fixed iteration count with no intermediate syncs, which measures
as a substantial per-step win on gfx1151.

The `patches/newton/` directory targets Newton 1.0.0 and does not apply to
1.4.0: the branch above replaces patch 04, and the code patch 03 modified was
refactored away upstream after 1.0.0.

### Verifying Newton works

Current per-example status on gfx1151 at this branch head: 96 of 104 Newton
examples run to completion in benchmark mode. The non-runners are all in
documented platform classes (the graph-replay fault class, several of them
intermittent; `cloth_franka`; one diffsim timeout; one benchmark-output
harness quirk) — see `KNOWN_ISSUES-AMD.md`. Quick
checks with Newton's own benchmark mode, expected sustained rates from the
recorded sweep:

Newton's benchmark mode prints its own sustained rate. Run a long frame
count: on ROCm 7.2.x, graph-replaying examples decay with runtime instead of
holding a steady rate — see
[Choosing a ROCm version](#choosing-a-rocm-version).

```bash
# ANYmal C walk (MuJoCo solver)
python -m newton.examples robot_anymal_c_walk --viewer null --benchmark --num-frames 1200

# cable_twist (no MuJoCo)
python -m newton.examples cable_twist --viewer null --benchmark --num-frames 1200
```

## Troubleshooting

### `Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes`

You ran a Newton MuJoCo example without the `graph_conditional = False`
patch. Use the wrapper in [Newton integration](#newton-integration).

### `hsa_queue_create` page fault / hang at first kernel launch

Two causes share the same symptom: first GPU op hangs at 100% CPU, GPU
idle on `rocm-smi`, kernel log shows a CPF (Command Processor Fetch)
page fault — typically `GCVM_L2_PROTECTION_FAULT_STATUS` with
`MAPPING_ERROR: 0x1`, `PERMISSION_FAULTS: 0x3`.

**Cause A — `amdgpu-dkms` overriding the in-tree module.**
The DKMS packages in the 30.30 (6.16.x) and 31.20 (6.19.x) series
trigger this on gfx1151 (ROCm issue #6118). Fix:

```bash
sudo apt remove amdgpu-dkms
sudo reboot
```

Keep `amdgpu-dkms-firmware` — it only provides firmware blobs and is safe.

**Cause B — newer in-tree kernel (Ubuntu OEM kernel updates).**
The Ubuntu OEM kernel `6.17.0-1020-oem` (and likely later) ships an
in-tree `amdgpu` with a different srcversion (`0AA0C630...` vs the
known-good `FC7DA320...` in 1017). It hits the same fault class on the
first GPU dispatch even though no DKMS is involved.
`unattended-upgrade` can replace 1017 with 1020.

Fix: reboot into the known-good kernel via GRUB, then hold it.

```bash
# Find the entry
sudo awk -F"'" '/menuentry / {print $2}' /boot/grub/grub.cfg | grep 1017

# Set next-boot only (not permanent)
sudo grub-reboot 'Advanced options for Ubuntu>Ubuntu, with Linux 6.17.0-1017-oem'
sudo reboot

# After login, pin so the meta-package can't move it again
sudo apt-mark hold \
    linux-image-6.17.0-1017-oem linux-modules-6.17.0-1017-oem linux-headers-6.17.0-1017-oem \
    linux-image-oem-24.04d linux-oem-24.04c linux-oem-24.04d linux-headers-oem-24.04d
```

The meta-package name (`linux-image-oem-24.04d` above) varies; check
`apt-cache rdepends --installed linux-image-6.17.0-<current>-oem` to
find which meta is pulling new kernels in.

### Confirming which `amdgpu` module is loaded

```bash
modinfo amdgpu | grep -E 'filename|srcversion'
```

A working gfx1151 setup loads the in-tree module from your distribution kernel — the `filename` should be under `/lib/modules/$(uname -r)/kernel/...`, not `.../updates/dkms/`. The known-good srcversion for `6.17.0-1017-oem` is `FC7DA320ED9D733CA6A3F1E`.

### `HSA_STATUS_ERROR_EXCEPTION code 0x1016` in cloth/softbody examples

Historically an illegal-opcode trap from sub-warp (`block_dim < 32`)
`wp.tile_reduce` in Newton VBD/Style3D kernels. It does **not** occur on this
build with ROCm 7.2 — verified that ROCm 7.2 runs one workgroup per wave, so the
reduction is correctly scoped and no patch is needed. If you hit it on an older
ROCm, pad those launches to a full wave (32) with passenger-thread gating.

### Segfault during `wp.capture_launch` on long differentiable simulations

This was an HIP runtime stack overflow during `hipGraphInstantiate` on long
graph dependency chains. It is fixed in this build via a worker-thread
workaround inside `wp_cuda_graph_create_exec`. If you still see segfaults,
verify you are running the patched `warp.so` (not a stale install).

### Diffsim examples appear to hang (`diffsim_bear`, etc.)

They are not hung — differentiable-simulation examples JIT-compile very
large forward+adjoint tile kernels, and HIPRTC (which cannot use
precompiled headers) takes **~15 minutes on the first run** on gfx1151.
The kernel cache makes subsequent runs skip this entirely. `diffsim_bear`
then takes ~10 minutes of actual training on the iGPU. If you need to
demo one, prewarm the cache first (run it once with `--viewer null`).
Watch progress by listing new module directories in the kernel cache
(`ls -lat ~/.cache/warp/<version>/ | head`) — if new entries keep
appearing, compilation is progressing.

### `libmathdx` errors

`libmathdx` is unsupported on HIP builds. There is no workaround on AMD currently.

### `texture_sample_*` returns zero

Fixed. The HIP device branches of ``texture_sample_helper`` were stubs;
they now call the ``tex1D/tex2D/tex3D`` intrinsics (HIPRTC compiles them
fine on ROCm 7.2). Validated bit-exact against the CPU sampler on gfx1151.
If you still see zeros, the kernel using ``texture_sample`` may have a
stale cached cubin compiled against the old stub header — Warp's module
hash does not include native headers, so touch the kernel source (or use
a fresh ``WARP_CACHE_PATH``) to force a recompile.

## Optional settings

All are off by default; the defaults are the validated configuration.

| Setting | Effect |
|---|---|
| `WARP_HIP_FAST_FP_ATOMICS=1` | Lower float32 atomic accumulation to the hardware atomic instead of the compare-and-swap retry loop HIP emits. Also available as `warp.config.hip_fast_fp_atomics` or the `"hip_fast_fp_atomics"` module option. |
| `WARP_ENABLE_UMA_HYBRID=1` | Serve allocations of 64 KB or more from unified (managed) memory for zero-copy CPU access. |
| `WARP_HIP_GRAPH_FREE_NODES=1` | Enable graph memory-free nodes. |
| `WARP_HIP_USE_ASYNC_POOL=1` | Use the stream-ordered pool for all allocations rather than only during graph capture. |

`WARP_HIP_FAST_FP_ATOMICS` is worth knowing about if a kernel funnels a reduction
through one atomic address, where the default lowering is slow — see
[Single-address atomic contention](KNOWN_ISSUES-AMD.md#single-address-atomic-contention).
It is safe on Warp's own allocations, including the UMA hybrid allocator and
`ReadbackBuffer`, which are marked coarse-grained so the hardware instruction
works on them. It is **not** safe on managed memory Warp did not allocate that
way, such as `CudaManagedAllocator` arrays: the hardware atomic silently discards
updates to host-coherent memory.

The other three change allocation or graph behaviour that the validated baseline
does not use. `WARP_HIP_GRAPH_FREE_NODES=1` in particular fixes a small number of
examples and breaks a larger number — see
[KNOWN_ISSUES-AMD.md](KNOWN_ISSUES-AMD.md).

## Limitations

- `libmathdx` is unsupported on HIP builds.
- Single-architecture compile time: about 1-2 minutes, plus a one-time LLVM/Clang toolchain download on the first build.
- Newton MuJoCo examples need the `tomas/gfx1151-fixes` Newton branch (or the
  `graph_conditional = False` wrapper) — see [Newton integration](#newton-integration).
- Validated only on gfx1151. Other RDNA architectures are buildable but untested.

## License

Apache-2.0. See `LICENSE.md`.
