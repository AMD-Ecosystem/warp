# Known issues — Warp on AMD HIP/ROCm (gfx1151)

Status of the gfx1151 (Strix Halo APU, RDNA 3.5, wave32) port, now on upstream
main at Warp 1.17.0.dev4. Current test counts are in the 1.17.0.dev4 section
below; counts in the dated sections further down were produced on the stack
recorded there (the oldest from a clean, orphan-reaped, serial run of the full
suite on ROCm 7.2).

The items below are the remaining gaps. Each is characterized so a reader can
tell a genuine hardware/driver limitation from an open bug.

## Warp 1.17.0.dev4 — current status

The port now applies to upstream main at Warp 1.17.0.dev4. The sections below
were written against the 1.15.0 and 1.12.1 bases and are retained as recorded;
resolution notes are added in place where the current base changes a verdict.

Validation at this base:

- **Warp unit suite, gfx1151 / ROCm 7.14**: full `warp.tests` suite
  **8313 ok / 226 skip / 6 fail** (~8547 tests). The six failures, each in a
  documented class:
  - `cuda/test_streams` `test_event_elapsed_time_graph` and
    `test_event_external` — ROCm external-event semantics across replayed
    graphs (section below).
  - `matrix/test_mat` `test_inverse_float16` — the 4-ULP fp16 rounding
    difference documented below; deliberately not widened.
  - `test_apic` `test_save_load_padded_bsr_transpose_cuda_rebuild` and
    `test_save_load_padded_bsr_transpose_too_small` — the
    graph-capture-allocation class.
  - `test_optim` `example_fluid_checkpoint` — the platform firmware wedge
    (lost HSA completion signal; the process spins in system time).
  - `geometry/test_bvh` `test_bvh_aabb` — intermittent wrong query result (the
    device AABB query misses an intersection the host finds), about 7 % of
    single-process runs on 2026-09-07; the "Rare wrong BVH query result" open bug
    below, with the evidence. Open.
- **CUDA reference at the same commit** (RTX A4000, CUDA 12.9): builds with
  the standard CUDA build; full suite **8590 ok / 62 skip / 0 fail**. No AMD
  failure reproduces on CUDA.
- **Newton** (fork `tomas/gfx1151-fixes`) on gfx1151: unit suite
  **4851 ok / 107 fail / 164 skip**, against 4838 ok / 123 fail / 164 skip at
  the 1.15 base — a net improvement. The implicit-MPM rebuildable-Volume test
  blocks now pass because Warp >= 1.16 provides the rebuildable Volume APIs.
  Examples: 96 of 104 run in-sweep (see the Newton examples section below).
- Profiler start/stop reports unsupported on ROCm — `hipProfilerStart` is
  deprecated and returns `hipErrorNotSupported` — so the smoke test skips
  that check.

### ROCm external event semantics on replayed graphs

An external event-record node in a replayed graph updates the event only when
the node executes. Unlike CUDA, the host-visible event state is not marked
pending at graph launch, so a host `synchronize_event` racing the replay
returns before the replayed record has run and reads stale data. A minimal
reproducer is about ten lines: capture a busy kernel, a device-to-host copy
and an external event record; replay the graph and synchronize the event
immediately — the read is stale.

Warp works around the **host** side: an event recorded with the external flag
during capture falls back to a device synchronize on host sync. Conservative —
a stronger wait than CUDA's per-event wait — but correct.

The **device** side is not covered: a captured wait gating on another graph's
replayed external record has the same underlying semantics, and no workaround
exists for it. The two `cuda/test_streams` failures above exercise exactly
that.

### Fixed on this base (previously open below)

- **Tile solve backward numerics** (the `test_tile_cholesky` backward value
  mismatches below) — the adjoint's scratch memory was declared `__shared__`
  only under `__CUDA_ARCH__`, so the HIP build ran the adjoint on non-shared
  scratch and produced wrong numerics. Fixed.
- **BSR/sparse nnz reads zero after a captured compress** (the intermittent
  `test_capturability` failures below) — the cause is the external event
  semantics above, not a device-primitive race: `nnz_sync()` waits on an event
  recorded with the external flag during capture, and the host wait raced the
  replayed record node. The host-side workaround resolves it.
- **Over-budget dynamic shared memory launches faulted with error 700 and
  poisoned the context** — `hipFuncSetAttribute` validates nothing on
  `hipFunction_t` handles and `hipModuleLaunchKernel` accepts the over-budget
  request. Such launches are now rejected natively with a diagnostic naming
  the shortfall, matching CUDA's launch-time error. The 64 KB LDS capacity
  itself is unchanged.

## Warp 1.15.0 — new gaps (in addition to the 1.12.1 items below)

This section was written while the port tracked upstream Warp 1.15.0; the
current base is above. Cross-vendor numerics were validated at that base
against **stock NVIDIA Warp 1.15.0 on an RTX A4000**: gfx1151 (HIP) matched it
**73/73** on a representative operation set (math builtins, atomics incl.
int64, tile reductions, scan, BSR, mesh/BVH/hash-grid queries, graph-capture
replay, a composed vec/mat **backward** pass, and a mixed-dtype compile canary),
and the port built for CUDA matched stock 73/73 (no CUDA regressions). So
forward compute and simple adjoints are faithful; the gaps below are specific
subsystems.

### Driver wrappers still stubbed on HIP (audited 2026-07-28)

Every `cu*_f` wrapper in `cuda_util.cpp` that returns `CUDA_ERROR_NOT_SUPPORTED`
under `__HIP_PLATFORM_AMD__` was checked against the ROCm headers. Three are
genuine gaps and one is not a gap at all:

| wrapper | ROCm equivalent | verdict |
|---|---|---|
| `cuStreamGetCtx_f` | none | real gap; HIP has no stream-to-context query |
| `cuGraphAddNode_f` | `hipGraphAddNode` exists | stub, but only reached from conditional graph nodes, which HIP does not support anyway |
| `cuGraphicsGLRegisterBuffer_f` / `...Image_f` | `hipGraphicsGLRegisterBuffer` / `...Image` exist in `hip_gl_interop.h` | **implementable gap** |
| `cuDeviceGetUuid_f` | `hipDeviceGetUuid` | not a gap — already implemented; the error return is a null-pointer guard |

The OpenGL interop one is the only worthwhile candidate. `wp.RegisteredGLBuffer`
therefore never gets a resource on AMD and always takes its `fallback_to_copy`
path. Nothing in Warp's own tests or in Newton's viewer uses it — Newton uploads
through `glBufferSubData` from a host pointer regardless — so the practical cost
today is zero and there is no test that would prove an implementation correct.
It is left alone deliberately: enabling a code path that has never run on this
backend is exactly what caused the graph memory-free crash above.

### Genuine platform gaps (scoped off HIP)

- **Graph memory-allocation nodes** (`test_graph` in-capture alloc/free node
  insertion and dependency-topology tests) require CUDA graph MemAllocNode
  support; HIP capture-time async allocation does not surface a MemAllocNode
  (`wp_cuda_graph_insert_alloc_node` returns null). Scoped to non-HIP.
- **Conditional graph nodes** (`capture_while` / `capture_if`) are unsupported
  on HIP/ROCm (tracked by `is_conditional_graph_supported()` and a filed ROCm
  feature request). The `test_graph` `capture_while` case is skipped.
- **cuBQL BVH constructor** is CUDA-only (its GPU builder needs CUDA). HIP
  reports `is_cubql_available()` = False and the `"cubql"` Mesh/Bvh constructor
  is gated off; the SAH/median/LBVH constructors are the HIP path.
- **Deterministic atomics** (`warp.DeterministicMode.RUN_TO_RUN` /
  `GPU_TO_GPU`) are not yet validated on HIP: the binned-accumulator reduction
  in `deterministic.cu` does not currently reproduce bit-identical results
  across runs on gfx1151. The `deterministic/` GPU tests are scoped to non-HIP.

### Fixed this session

- **BVH radix-sort aliasing** (was misfiled as "flaky rocPRIM"; 9 `bvh` + 2
  `mesh_query_aabb` tests) — the HIP `LinearBVHBuilderGPU::build` (`bvh.cu`)
  called the non-DoubleBuffer `cub::DeviceRadixSort::SortPairs(temp, bytes,
  keys, keys, …)` with the **same pointer for `d_keys_in` and `d_keys_out`**.
  CUB requires these not alias; an aliased radix pass scatters into the buffer it
  is still reading, so the Morton order was intermittently corrupted → invalid
  Karras topology → CUDA error 700 (illegal memory access) at query time
  (~4/6 process runs). Fixed by switching the HIP builder to the same
  `cub::DoubleBuffer(keys, keys + n)` ping-pong used by Warp's own
  `radix_sort_pairs_device` (`sort.cu`) and the upstream CUDA build in the same
  file; distinct key/value halves, result read from `.Current()`, sorted indices
  copied into `bvh.primitive_indices`. All inside `#if __HIP_PLATFORM_AMD__`
  (CUDA untouched). Verified deterministic: `test_bvh` 20/20,
  `test_mesh_query_aabb` 10/10 (was ~2/6). This is distinct from the multi-stream
  **BVH refit** race parked on `tomas/hip-multistream-refit-race` (that is a
  bottom-up refit memory-visibility issue under concurrent streams, not the
  build-time sort).

- **BVH and mesh descriptor addresses could not be recycled** (`test_bvh` 10/12
  process runs failing on ROCm 7.14, `test_mesh` intermittently) — a BVH or mesh
  id *is* the device address of its descriptor, so destroying one object and
  creating another handed the same address to the new object. A traversal kernel
  reading a descriptor through a recycled address saw the previous contents
  rather than the ones just written; an all-zero descriptor has a null `root`,
  and `bvh_query()` dereferences it, producing `Memory Fault Error ... faulting
  addr: 0x0` in `bvh_query_aabb` and `query_ray_kernel`.

  What the fault tracks is the address being released and re-acquired, not any
  ordering or coherence step around the write. Measured on the same reproducer
  (`test_bvh -k aabb_cuda -k refit_root_leaves`, 12 process runs each):
  synchronizing the stream before the free 7/12 failing, synchronizing the whole
  device 4/12, allocating the descriptor with the blocking allocator instead of
  the pool 5/12, disabling the memory pool entirely 9/12, writing the descriptor
  from a kernel instead of a host copy 9/12 — against 9-11/12 unchanged. Not
  freeing the block at all: 0/12.

  Fixed by pooling descriptor blocks in Warp (`wp_descriptor_alloc` /
  `wp_descriptor_free` in `warp.cu`) so their addresses stay mapped for the life
  of the process. The pool is bounded by the high-water mark of concurrently
  live objects at one descriptor each. It is under `__HIP_PLATFORM_AMD__`; on
  CUDA both entry points forward to `wp_alloc_device` / `wp_free_device`, so
  that path is unchanged. Afterwards `test_bvh` 12/12, `test_mesh` 8/8,
  `test_mesh_query_aabb` 6/6 on gfx1151, and on an RTX A4000 all five geometry
  suites 0/6 failing.

  Across the whole Warp unit suite on ROCm 7.14 this fix alone moves 22 failures
  to 13 with no regressions. With the allocator, external-event and grid-cap
  fixes below it reaches **10 failures, +12 fixed and 0 regressed**, measured
  with a fresh kernel cache on both sides. Two of those ten,
  `fem.test_fem_quadrature.test_gimp_quadrature` and
  `fem.test_fem_shape.test_cube_shape_functions`, fail on the RTX A4000 as well
  and are therefore upstream, not port defects; the port-specific figure is
  **eight**. (A measured 8 included the
  graph-capture change, which was then reverted for segfaulting two Newton
  modules; two of the three tests it fixed are failing again, while
  `test_fem_integrate.test_capturability` turns out to be repaired by the
  allocator change rather than by it.)
  (The post-fix runs collect 11 fewer ids than the baseline because they ran
  outside the container, where `jax` is not installed, so the CPU-only jax and
  dlpack interop tests do not collect. They are unrelated and passed in both
  environments where they ran.)

  **Give each native build its own `WARP_CACHE_PATH` when measuring.** Warp's
  module hash does not cover `warp.so`, so a cache reused across builds mixes
  artifacts and manufactures failures that look exactly like a code regression —
  in one sweep, 39 of them, including 136 occurrences of
  `AttributeError: 'Adjoint' object has no attribute 'return_var'` that drop to
  zero with a fresh cache. Reproduce a sweep failure with the sweep's own
  invocation (`python -m unittest -v <module>`); running the test file directly
  executes its `__main__` block instead and can pass where the module form
  fails.

  The mesh half is not inferred from the BVH half: `test_mesh` failed 1 run in 6
  with `faulting addr: 0x0` in `query_ray_kernel` while only the BVH descriptor
  was pooled, and passes 8 of 8 once the mesh descriptor is pooled too.

  **`0x0` on its own is not diagnostic of this defect.** An earlier note here
  claimed the same signature in mujoco_warp `primitive_narrowphase` was
  "expected to be the same defect". That was wrong, and the correction matters
  because it would otherwise keep this fix being blamed for faults it cannot
  cause. Those kernels take no mesh or BVH id at all — `collision_primitive.py`
  contains no `uint64` parameter — and the fault arrives after hundreds of
  `wp_alloc_device_async: failed to find memory allocation node` warnings (280 in
  `test_anymal_reset`). The null is an array argument from a capture-time
  allocation that was never made, which is the separate graph-capture defect
  below.

- **Graph-capture allocations have no allocation node** (`test_sparse`,
  `test_fem_integrate`, `test_apic`, and the `0x0` faults in Newton's collision
  kernels) — `get_capture_dependencies` in `cuda_util.cpp` returns `false`
  unconditionally on HIP, so `wp_alloc_device_async` cannot find the
  `MemAllocNode` it just added during capture. Every capture-time allocation is
  recorded without one, and replaying such a graph does not perform the
  allocation, so kernels read memory that was never allocated. A single captured
  Newton step logs 280 `failed to find memory allocation node` warnings before
  faulting at `0x0`; `test_sparse` logs 119.

  **Partly fixed, and the boundary is deliberate.** ROCm provides
  `hipStreamGetCaptureInfo_v2` with the required dependency list, so the lookup
  works and the warning count is zero everywhere. The related capturability
  tests remained intermittent after this fix: over repeated module-alone runs
  on ROCm 7.14, `test_sparse.test_capturability`,
  `test_fem_integrate.test_capturability`, and
  `test_sparse.test_bsr_compress_compact_capturability` each failed in
  roughly half to two-thirds of runs, with rates that varied between sessions
  and with test composition and did not track the capture-dependency flag
  value (see the dependency-flag entry below). That intermittency is resolved
  on the current base: it was the replayed-graph external-event semantics
  (1.17.0.dev4 section above), not this allocation-node defect, and the tests
  pass with the host-side event workaround.

  What is *not* enabled is the branch that adds a `MemFreeNode`, which is what
  the free path does next once the allocation node is known. That branch has
  never executed on HIP, and with it live `newton.tests.test_cloth` and
  `newton.tests.test_collision_pipeline` **segfault in 2 of 2 runs**, against
  0 of 2 without it, measured with a fresh kernel cache on both arms.
  Implementing `get_graph_leaf_nodes` as well — the other stubbed half of that
  path — does not rescue it. Two modules, roughly 216 tests, that stop running
  are not worth the tests it would recover, so the free path keeps the fallback
  HIP has always used by default. It is reachable with
  `WARP_HIP_GRAPH_FREE_NODES=1`, and the measured trade-off in both directions
  is under the illegal-access entry below; the segfault above is one of the two
  ROCm behaviours recorded there.

  Still failing and belonging to this defect class:
  `test_apic.test_save_load_padded_bsr_transpose_cuda_rebuild` and
  `test_apic.test_save_load_padded_bsr_transpose_too_small` (the two
  `test_apic` failures on the current base; the capturability failures
  formerly listed here were the external-event semantics instead), and a `0x0`
  fault in mujoco_warp `forward_test` (`ccd_kernel`) that survives the warning
  fix, so the missing nodes were never its cause. Closing this out means
  validating HIP's graph memory-node machinery as a whole.
  `newton.tests.test_menagerie_usd_mujoco` crashes at the same rate with and
  without any of this, so it is unrelated.

  The cost of leaving the free-node path disabled is larger than these tests:
  captured graphs accumulate allocation nodes with no matching frees, which is
  the most probable source of the illegal-access-during-replay class described
  under open bugs below.

- **Capture-dependency and IPC flag mappings in `hip_util.h`** — found by
  auditing every `#define X 0` in `hip_util.h` against ROCm's real value.

  `cudaStreamSetCaptureDependencies` was defined as `0` in an early block,
  which won the `#ifndef` race against the correct mapping further down the
  same header and made it equal to `hipStreamAddCaptureDependencies`, so
  every call meaning "replace the dependency set" appended instead. Fixed:
  the mapping is `hipStreamSetCaptureDependencies`.

  The flag value does not control the intermittent `cloth_franka` non-finite
  failure (`numpy.linalg.LinAlgError: SVD did not converge`) that was at one
  point attributed to it. Measured on otherwise identical builds differing
  only in this mapping: at 200 frames one session showed replace failing
  8 of 25 and append 0 of 24; at 400 frames, arms interleaved run-for-run
  in one session, append failed 10 of 12 and replace 5 of 12; at 300 frames
  both were 0 for 26; the same example passed a 1200-frame run between those
  sessions. The failure is real, session-dependent, and independent of this
  flag — see "cloth_franka intermittent non-finite state" under open bugs.

  `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS` was `0` where ROCm defines `0x01`, so the
  flag `cuIpcOpenMemHandle` needs — the code comment says it is required — was
  never passed. Fixed. The remaining zero mappings in that header are either
  zero in CUDA too or correctly guarded fallbacks.

- **External events were recorded as ordinary graph nodes** — `hip_util.h`
  defined `CU_EVENT_RECORD_EXTERNAL` and `CU_EVENT_WAIT_EXTERNAL` as `0`, the
  default flag, while ROCm defines both as `0x01`. Two graphs meant to
  synchronize through an external event therefore did not. Corrected, which
  moves `cuda.test_streams.test_event_external` from 12 to 14 of an expected 20,
  so HIP's external-event-in-graph support looks incomplete beyond the flag and
  the test still fails. The mapping was wrong either way. The residual failure
  is now root-caused: ROCm's replayed-graph external-event semantics
  (1.17.0.dev4 section above).

### Open bugs (under investigation, not yet fixed)

- **cloth_franka intermittent non-finite state** — the Newton example fails
  with `numpy.linalg.LinAlgError: SVD did not converge` when the Jacobian
  read back from the device contains non-finite values. The rate varies
  strongly between sessions and run lengths on the same build: observed 0 of
  50 in some session/length combinations and 10 of 12 in others (measurement
  matrix under the dependency-flag entry above). Not controlled by the
  capture-dependency flag; the state variable that modulates it is not
  identified. Runs that fail crash early (~7 s) rather than degrading.

- **Rare wrong BVH query result** — `test_bvh_aabb_cuda_0` reported one missed
  intersection (`host_intersected 1 != device 0`) in one of eleven module runs
  at the current head on ROCm 7.14 (one full-sweep run plus ten consecutive
  re-runs in the same session, the re-runs all passing). A wrong result
  without a fault matches the residual lost-write class documented below;
  the rate is too low for the 12-run protocols used elsewhere in this file.
  2026-09-07 (kernel 7.0.0-31, ROCm 7.14.1): 3 misses in 40 single-process runs on the 1.17
  release line and the 1.18 development line alike (2/8, 1/16, 0/16), no memory fault; the
  BVH query tests (aabb, capsule, sphere, ray) also flip in the pooled runner. Documented as
  a known intermittent failure in the validation status above; still open.

- **Nominally passing examples can crash intermittently** — examples recorded
  as `ok` in the FPS sweep (`robot_h1`, `sensor_contact`) crashed with
  `CUDA error 700: an illegal memory access` in roughly 1 of 3 retries during
  an independent verification pass, with the same signature as the sweep's
  intermittent-replay failures. A single passing run of an example is not
  evidence of stability in this class.

- **Illegal access during graph replay, localized to graph-owned capture-time
  allocations.** The intermittent `CUDA error 700` class in Newton examples
  (`selection_*`, `sensor_imu`, `basic_plotting`, `basic_multi_solver_overlay`,
  `recording`, and the retry crashes above) requires graph replay: on
  `selection_materials` at 1200 frames the captured mode faulted in 3 runs of
  11, while eager execution was clean in 3 of 3 (2 unverified, 1 with
  `wp.config.verify_cuda`). It surfaces mid-run as a `RuntimeError` from
  `wp.capture_launch`; the `wp_cuda_graph_destroy` 700 and the
  `wp_cuda_context_pop_current` error 3 at exit are the sticky-error tail, not
  the defect.

  ROCr names the faulting kernel without any logging environment variables:
  `primitive_narrowphase` from mujoco-warp's collision pipeline, identical in
  all three faulting runs and independently confirmed by the AQL dispatch
  packet (its `group_seg_size=58368` is a unique fingerprint among the 70,660
  dispatches in a `rocprofv3 --kernel-trace` of a short captured run). In the
  captured graph it is node 32 of 7300, recurring once per MuJoCo substep at
  stride 730. `AMD_LOG_LEVEL`/`AMD_LOG_MASK` dispatch logging is inert on this
  stack; `rocprofv3` and `hipGraphDebugDotPrint` both work and do observe
  graph-replay dispatches.

  The faults are reads (`RW: 0x0`, UTCL2 client TCP) whose addresses span
  about 1 TiB across ten distinct pages, with clusters matching 4-byte and
  36-byte element strides and the 504-byte row stride of `geom_xmat`. A stale
  base pointer cannot produce that spread, so the corrupt quantity is an
  **index**, not a buffer address. One observed fault is at address `0x0`
  exactly, consistent with `pair_margin[..., pairid]` where this model's
  `pair_margin` has shape `[16, 0]` and a null pointer.

  The three index arrays the kernel consumes — `collision_pair`,
  `collision_pairid`, `collision_worldid` — are `wp.empty()` allocations made
  **inside the captured region** (10 measured, all with capture active), and
  their MEM_ALLOC nodes sit immediately before the faulting kernel (nodes
  26-28, broadphase writing them at node 31). The captured graph contains
  **360 MEM_ALLOC nodes and 0 MEM_FREE nodes**: on HIP `add_free_node` is
  hard-`false` (see the deliberate boundary above), and the fallback path then
  calls `get_graph_leaf_nodes()`, which is also stubbed to return `false`
  (`cuda_util.cpp`), so no free node is added while Warp erases its own
  allocation record. Every replay re-executes 360 allocations that are never
  released.

  **Confirmed by enabling the free path, and available as an opt-in.**
  `WARP_HIP_GRAPH_FREE_NODES=1` adds a free node for every graph allocation and
  instantiates without `AutoFreeOnLaunch`. Measured on `selection_materials` at
  1200 frames, interleaved run-for-run with checksum-verified binaries:

  | Configuration | Fault runs | `newton.tests.test_cloth` | `test_collision_pipeline` |
  |---|---|---|---|
  | default (free nodes off) | 11 of 18 | fails the same 3 tests as always | passes |
  | free nodes on, `AutoFreeOnLaunch` kept | 1 of 10 | **segfault**, 2 of 2 | not run |
  | free nodes on, `AutoFreeOnLaunch` dropped | **0 of 12** | 8 errors, no crash | 1 error |

  So the allocation nodes are the cause: freeing them removes the fault class
  outright. Two ROCm behaviours prevent making it the default, and together
  leave no configuration that serves every graph:

  - A graph containing `MemFree` nodes, instantiated with `AutoFreeOnLaunch`,
    faults inside `hipGraphLaunch` — both mechanisms reclaim the same
    allocation. CUDA tolerates the combination.
  - Without the flag, a graph whose capture allocations outlive the capture has
    nothing to reclaim them, and relaunching it fails with `invalid argument`.
    Those graphs are what the errors in the third row are.

  One test in the default suite fails on exactly that signature:
  `test_apic::test_save_load_padded_bsr_transpose_cuda_rebuild_cuda_0` raises
  `Graph launch error: ... error 1: invalid argument` from
  `wp_cuda_graph_launch`. It is this class, not a sparse-matrix defect.

  The opt-in is worthwhile where graphs are replayed heavily and the affected
  test modules are not in play; it also removes the per-replay allocation churn.
  Enabling it changes nothing on non-HIP builds. Remaining work is upstream: the
  `hipGraphLaunch` fault is reportable on its own.

  Earlier framing, still accurate as far as it goes — not yet proven — validating the context contents
  after every replay found 0 out-of-range entries across 5 clean runs, but no
  faulting run has yet been caught with that instrumentation, and it covered 1
  of 10 substeps. Post-capture address-reuse probing found no collisions, and
  `hipMemPoolAttrUsedMemCurrent` reads 0 throughout on this stack, so pool
  growth cannot be observed that way. Two unbounded index sites in mujoco-warp
  (`geom_xpos_in[worldid, g]`, `pair_margin[..., pairid]`) are what turn a bad
  index into a page fault rather than silently wrong contacts. Investigation
  record: `scratchpad/teardown_fault_localization.md`.

- **Shared-memory capacity (64 KB LDS)** — gfx1151's LDS is a hard 64 KB (sm_75+
  can opt into 100+ KB). Tile kernels sized right at 64 KB overflow once the tile
  framework's own overhead is added, so their launch faults with an HSA invalid
  allocation. Affected tests are skipped where the tile does not fit:
  `test_tile_shared_mem_large` (a deliberate 64 KB forward kernel),
  `test_tile_view::test_tile_assign_2d` (two differentiable 3D tiles keep data +
  gradient buffers in shared), and the 3D large-tile axis-reduce *backward*
  (~66 KB). These faults, under the concurrent full-suite runner, also poisoned
  unrelated files (see the concurrency note below); running affected files in
  isolation shows the true, much smaller failure set. The fault-and-poison
  behaviour is fixed on the current base: `hipFuncSetAttribute` validates
  nothing on `hipFunction_t` handles and `hipModuleLaunchKernel` accepts an
  over-budget request, so the launch faulted with error 700 and poisoned the
  context. Over-budget dynamic shared memory launches are now rejected
  natively with a diagnostic naming the shortfall, matching CUDA's launch-time
  error; the 64 KB capacity and the skips above are unchanged.
- **Tile-op adjoints — mostly fixed.** A multi-warp reduction bug (`tile_dot`
  passed its `__shared__` partials through a function boundary, dropping the
  shared address-space qualifier on HIP, corrupting the result when a trailing
  warp is fully idle at `block_dim > tile size`) broke `tile_axpy`/`tile_dot`
  gradients; it is now fixed by inlining the reduction. A missing `CUDA_CALLABLE`
  on `~tile_stack_t` (hipRTC rejects a `__host__` destructor in device code) is
  fixed. The `test_tile_cholesky` backward value mismatches (4) are fixed on
  the current base: the tile solve adjoint's scratch memory was declared
  `__shared__` only under `__CUDA_ARCH__`, so the HIP build ran the adjoint on
  non-shared scratch. Remaining: the Tier 3 block-level axis reduction
  (reduction dim > 256, skipped with a marker; Tier 1/2 and all non-axis
  reductions are correct).
- **libmathdx-dependent tile ops** — `test_tile_fft_no_mathdx` and
  `test_tile_cholesky_no_mathdx` exercise the fallback path when NVIDIA
  libmathdx is unavailable; libmathdx is CUDA-only and never present on HIP, so
  these are platform gaps.

### Full-suite concurrency caveat

The `-s autodetect` suite runner executes test files concurrently in a process
pool. On gfx1151 a genuine fault in one file's kernel (an out-of-bounds
access; formerly also an LDS overflow, which the current base rejects at
launch instead) aborts the **whole GPU queue**, which crashes the other
pool processes running at that moment — so a single real fault inflates the
reported failure count across many unrelated files (e.g. `test_atomic`,
`test_array`, `test_arithmetic`, `matrix.*`, which all pass in isolation and are
golden-clean). The honest per-file status is obtained by running each file in
its own process (one-file-per-process, matching the 1.12.1 methodology); the
pooled full-suite total is not a reliable file-pass metric on this hardware.
- **Texture — HIP host↔array copy corrupts the array's leading texels**
  (root-caused and fixed). On gfx1151/ROCm 7.2, uploading texture data from host
  memory into a CUDA/HIP array (`hipDrvMemcpy3D` / `hipMemcpy2DToArray` with a
  **host** source) leaves the first ~16 bytes (one texel block) of the array
  stale, in a size- and allocation-dependent (partly non-deterministic) pattern.
  Verified via a differential against NVIDIA: identical tests pass on CUDA-GPU and
  on the CPU software path, and the corruption tracks the copy path, not the
  sampling math. The **device**↔array path is clean. Fix (`native/texture.cpp`,
  entirely `#if __HIP_PLATFORM_AMD__`-guarded, so CUDA is unaffected): stage any
  host endpoint of a texture-array copy through a temporary device buffer
  (host→device linear, then device→array). Result on `test_texture` (3 runs,
  now deterministic): **93→125 pass, 34→2 FAIL, 6 ERROR**.
  - Remaining **2 FAIL**: `test_texture3d_cuda_array_copy_api_graph_capture`
    (and `_explicit_stream`) — 3D texture copy *under graph capture*. The
    device-staging (alloc/sync) can't be captured, and HIP's driver 3D array
    copy doesn't replay correctly under capture; the 2D equivalents pass. Edge
    case; not used by the Newton examples.
  - Remaining **6 ERROR**: mipmapped-array tests — `hipMipmappedArrayCreate`
    returns "operation not supported" (error 801) on gfx1151/ROCm 7.2. Genuine
    platform gap (like thread-block clusters), not a Warp bug.
- **clang AMDGPU backend segfault on uint8 matrix `ddot` with `-DNDEBUG`**
  (root-caused; ROCm compiler bug, not a Warp/port bug — **worked around**).
  ROCm 7.2.1's clang (v22) crashes in `SelectionDAG::FoldConstantArithmetic`
  during type legalization when compiling a **uint8 (i8) matrix double-dot**
  kernel with `-DNDEBUG` at `-O2`/`-O3`. Warp adds `-DNDEBUG` in release to strip
  device-side `assert()`s (~3× smaller/2× faster kernels); with the asserts gone
  the i8 arithmetic trips the fold bug. It is **content-specific, not size**: a
  single 106-line kernel reproduces (see `patches/rocm/`), and `float16` is fine.
  It surfaced first in `matrix/test_mat_linalg.py`, whose parametrized tests bundle
  a `uint8` `ddot` into a large per-file module — HIPRTC compiles in-process, so
  the segfault took down the whole test file. Verified the code is valid: dropping
  `-DNDEBUG` (or `-O1`) compiles cleanly, as does NVRTC.
  - **Workaround** (`warp/_src/build.py`): route HIP modules above
    `WP_HIP_HIPRTC_MAX_SRC_BYTES` (default 2 MB, env-overridable; 0 = always) to
    out-of-process `hipcc --genco` (crash-isolated). It tries `-DNDEBUG` first —
    so innocent large modules keep the smaller/faster release codegen — and only
    retries **without `-DNDEBUG`** if the compiler segfaults. `test_mat_linalg`
    now passes (154/154).
    A hypothetical *small* module hitting the same kernel would still crash under
    HIPRTC; set `WARP_HIP_HIPRTC_MAX_SRC_BYTES=0` to force AOT there. Remove once
    the upstream clang bug is fixed. Bug report: `patches/rocm/`.
- Distinct fp16 note: `test_mat`'s `test_inverse_float16` fails by 4 fp16 ULP,
  an expected cross-backend rounding difference against the test's `atol=0.05`.
  Measured 2026-08-04: actual `-31.375` against expected `-31.3125`, an absolute
  difference of 0.0625 where one ULP is 0.015625 at that magnitude — 0.2 %
  relative, against a tolerance that allows 3.2 ULP. Left alone rather than
  widening an upstream tolerance for one backend.

The 1.12.1 gaps below still apply where the subsystem is unchanged.

## Newton 1.4.0 + MuJoCo validation (2026-07-19)

Newton v1.4.0 (mujoco 3.10.0, mujoco-warp 3.10.0.2) was run on gfx1151 with a
per-file fresh-process sweep (150 test files, 5021 tests) against a CUDA
reference on the RTX A4000 (`-j 1 --disable-process-pooling --strict-warnings`,
4632 ran: 4455 pass / 2 fail / 14 error / 161 skip; the 16 non-passes are
device-independent cpu+cuda pairs plus 2 example scripts).

Two fixes landed from this validation:

- **Texture destroy during stream capture** (warp, `fix(amd)` commit
  `c4bb97e7d`): `Texture.__del__` destroyed the texture/surface objects and the
  backing array unconditionally; during an active stream capture this raises
  CUDA error 900, invalidates the capture, and poisons the memory pool (observed
  as 176 cascading tiny-allocation failures in `test_collision_pipeline`). The
  teardown is now deferred until captures end. `test_collision_pipeline` went
  from ~24 errors to 154/154; `test_mesh_backface` (SIGABRT heap corruption) and
  6 further files went to OK. CUDA re-verified: `cuda/test_texture.py` 133/133,
  newton `test_collision_pipeline` 154/154.
- **Conditional graph nodes in MuJoCoSolver** (newton branch
  `tomas/gfx1151-fixes`, commit `89dfb6df`): newton now sets
  `mjw_model.opt.graph_conditional = False` when
  `wp.is_conditional_graph_supported()` is false, selecting mujoco_warp's
  non-conditional solver loop (the JAX fallback path). `test_anymal_reset`,
  `test_actuators`, and the MuJoCo solver files went to OK.

Remaining Newton-on-gfx1151 items:

- **Graph-capture alloc class (crash variant).** The known HIP
  mempool-alloc-under-capture limitation (`wp_alloc_device_async: failed to find
  memory allocation node`, see the sparse/BSR entry below) also produces
  CUDA error 700 crash cascades when a Newton simulation loop that allocates
  is run under `wp.ScopedCapture`. Deterministic repro:
  `newton/tests/test_example_browser.py cable_pile` (981 error-700 lines);
  the identical example passes fully with capture disabled and
  `wp.config.verify_cuda=True`. Affects `test_example_browser`,
  `test_examples`, and `test_physics_verification` (which captures at line
  936 and also pays a large fixed-iteration cost from the non-conditional
  fallback). Eager (non-captured) execution is the working mitigation.
- **USD physics parse heap corruption — ROOT-CAUSED 2026-07-20: usd-core
  26.3 bug, NOT a port defect.** The `test_menagerie_usd_mujoco` glibc heap
  aborts (`double free or corruption (fasttop)`, `malloc(): unaligned tcache
  chunk`, previously described here as "cumulative corruption under repeated
  MuJoCoSolver instantiation, not localized") are a thread-safety bug inside
  the usd-core 26.3 wheel: `UsdPhysics.LoadUsdPhysicsFromRange` corrupts the
  glibc heap in its TBB-parallel collision-finalize phase
  (`_FinalizeCollisionDescs<UsdPhysicsMeshShapeDesc>` under
  `WorkParallelForN`; abort fires under a worker's temporary
  `UsdGeomXformCache` teardown). Evidence, all runs on 2026-07-20
  (newton `amd_sweep/repro_usd_parse_race.py` — pure `pxr` calls, no
  warp/newton/GPU, apptronik_apollo.usda):
  - default pxr threading, glibc-hardened
    (`GLIBC_TUNABLES=glibc.malloc.tcache_count=0:glibc.malloc.perturb=165
    MALLOC_CHECK_=3`): crash 3/3 runs;
  - default threading, UN-hardened: core dump 3/3 runs within 1-3
    iterations;
  - `PXR_WORK_THREAD_LIMIT=1` (set before any pxr work starts): clean 3/3
    hardened runs, and clean with runtime `Work.SetConcurrencyLimit(1)`
    applied before the first `Stage.Open` (3/3);
  - `Stage.Open` + full traverse without the physics parse: clean 3/3
    hardened — the race is specific to `LoadUsdPhysicsFromRange`;
  - runtime `Work.SetConcurrencyLimit(1)` scoped around the parse call only
    is NOT reliable (crashed 3/3) — TBB workers already spun up by
    `Stage.Open` still participate.
  The earlier "planted by MuJoCo dynamics classes" bisect reading was wrong:
  every menagerie class (robot and import alike) triggers the parse via
  `newton add_usd` → `import_usd.py:520`; the race is probabilistic per
  call. The prior "not reproducible on CUDA" observation is a
  concurrency/timing artifact, not a platform property — the repro contains
  no GPU code; cross-check on the NVIDIA box is pending. No existing
  upstream report found for `LoadUsdPhysicsFromRange`; filing an OpenUSD
  issue with the standalone repro is queued. Mitigation: the sweep driver
  (`scratchpad/run_amd_suite.py`) now sets `PXR_WORK_THREAD_LIMIT=1` for
  child processes.
- **Lost HSA completion signal (host boot state, gfx1151) — not reproducible
  after 2026-07-21; no workaround required.** Processes nondeterministically
  spun forever inside HIP synchronization paths (synchronous free, D2H
  readback) at ~100% CPU while the GPU was idle: during a live wedge,
  `/sys/class/drm/card*/device/gpu_busy_percent` read 0 and the process's
  KFD event threads sat in `kfd_wait_on_events` — the kernel had completed
  but the completion signal was never observed. Probability rose with the
  number of launch+sync cycles, so mesh-import-heavy Newton tests wedged
  near-deterministically (USD menagerie import: 3/3 wedged >300 s) while
  few-sync repros usually passed. `HSA_ENABLE_INTERRUPT=0` masked it, which is
  consistent with the cause below: that variable makes the runtime busy-poll
  completion signals instead of waiting on interrupts.

  **Scope correction (2026-07-21).** This is a property of one host boot, not
  of gfx1151. The affected boot (2026-07-20 13:24 to 2026-07-21 15:51) logged
  2099 `amdgpu: Fence fallback timer expired` messages, the first 3 seconds
  after boot during driver ring initialization and before any Warp workload,
  together with display `flip_done` timeouts, a `gfx_0.0.0` ring timeout, a
  ring reset and `device wedged, but recovered through reset`. GPU interrupt
  delivery was already broken at driver init. The preceding boot logged zero
  such messages, and so has every boot since. The entire original
  investigation ran inside that one boot.

  Re-tested on a later boot of the same kernel (`7.0.0-28-generic`), same MES
  firmware `0x80` and same `linux-firmware` 20240318, after a change to BIOS
  settings:

  | Arm | Result | Original record |
  |---|---|---|
  | `TestMenagerieUsdImport.test_import_ur5e`, no workaround | 3/3 OK, 6.0 s | 3/3 wedged >300 s |
  | same, with `HSA_ENABLE_INTERRUPT=0` | 3/3 OK, 6.0 s | 3/3 OK, 5.9/11.9/12.5 s |
  | full `test_menagerie_usd_mujoco.py`, no workaround | 2/2 OK, `Ran 43 tests in 14.5 s` | 43 tests in 100.7 s, with workaround |
  | `Fence fallback timer expired` during all runs | 0 | 2099 over the affected boot |

  The workaround has been removed from the sweep driver. A run that wedges
  again should first check
  `journalctl -k -b 0 | grep -c "Fence fallback timer expired"`: a nonzero and
  rising count identifies the degraded boot state rather than a port defect.
  Related but distinct-signature gfx1151 reports: ROCm/ROCm#6165 (MES ring
  freeze under sustained load), ROCm/TheRock#2684 (HSA_USE_SVM=0 lore).
- **Run-to-run determinism** — `determinism/test_solver_determinism` fails 3
  particle tests with ULP-level mismatches (26-32 of 192 elements across
  runs). This is the documented deterministic-atomics gap above (the
  `deterministic.cu` binned accumulator is not bit-reproducible on gfx1151);
  newton's tests exercise the mode unconditionally where warp's own
  deterministic tests are scoped to non-HIP.
- **`test_solver_vbd::test_edge_face_pushes_vertices_out_cuda_0`** — fp-marginal
  test geometry, not a port defect. The contact optimizer converges onto the
  box SDF's face-selection tie surface (the deepest-penetration plateau ends
  exactly where `qx == qy`): measured contact x = 0.069924 (cpu, +y face,
  passes) vs 0.070041 (gfx1151, +x face, fails) against the tie at x = 0.07 —
  a 1.2e-4 fp-path difference across a genuine normal discontinuity.

## mujoco_warp test set on gfx1151 (2026-07-21)

The mujoco_warp 3.10.0.2 unit set (29 `_test.py` files under
`site-packages/mujoco_warp/_src/`) ran per-file on both boxes with the
observed-verdict harness (`scratchpad/sweep_v2.py`, absl-parameterized id
support). gfx1151: 1205 test ids, 1127 ok / 60 fail / 18 skip. A4000
reference: 1201 ids, 1176 ok / 4 fail / 21 skip. 1116 concordant passes.
The 59 AMD-fail/NVIDIA-ok tests decompose into three causes:

- **cuBQL mesh constructor unavailable (16 tests: ray_test 12, render_test 3,
  io_test 1).** mujoco_warp's ray/render path builds `wp.Mesh` with a cuBQL
  BVH constructor (`mujoco_warp/_src/bvh.py:444`); the HIP build compiles
  with `WP_DISABLE_CUBQL` (cuBQL is a CUDA library), so mesh creation raises
  `Failed to create mesh: cuBQL support disabled`. This is a feature gap of
  the ray/render subsystem on HIP. Constructor selection by platform
  capability inside mujoco_warp would restore its non-cuBQL fallback.
- **Conditional graph nodes, no skip-guard (6 tests:
  `forward_test.test_graph_capture{0,2,4,6}` — the `graph_conditional=True`
  parameters — and `unroll_test.test_aloha_lifts_pot{0,1}`).** The tests
  exercise conditional graph nodes directly; HIP/ROCm has none
  (`hipGraphConditionalHandle` absent as of ROCm 7.2), so warp raises
  `Conditional graph nodes are not supported on HIP/ROCm`. Same capability
  gap that newton commit `89dfb6df` routes around at the solver level.
- **Graph-capture error 700 plus in-file cascade (1 + 37 tests,
  forward_test).** `test_graph_capture7 (xml='collision.xml',
  graph_conditional=False)` — the non-conditional path — fails at
  `wp_cuda_graph_launch` with CUDA error 700; the poisoned context then fails
  the remaining 37 tests of the file with `Failed to allocate 4 bytes`.
  `test_implicit0` passes standalone (`Ran 1 test ... OK`), which confirms
  the 37 are cascade, not defects. The trigger belongs to the documented
  graph-capture alloc class (Newton section above).

Reference-side finding, not counted in the port's favour or against it:
4 io_test cases (`test_get_data_into_io_test_models{20..23}`,
`flex/floppy.xml` with the ELLIPTIC cone) fail on the A4000 with CUDA error
701 (`too many resources requested for launch`) on
`_linesearch_iterative_kernel`; gfx1151 passes them.

Id-set asymmetries, each explained: `RenderTest.test_backface_cull_matches_mujoco`
expands to 6 named parameters on the AMD box but collapses to one skipped id
on the headless NVIDIA box (`skipIf` OpenGL guard; `render_test.py`
md5-identical on both boxes). One AMD id
(`collision_driver_test.test_hfield_maxconpair`) has no parseable per-test
verdict: the runner's stderr verdict token interleaves with C-level stdout
warnings (`...resolutiok`, log line 357); the file-level `Ran 55 ... OK`
summary (exit 0) records that no test in the file failed. The same
interleaving affects `newton.tests.test_multiworld_body_properties`
(`..._cuda:0_cuda_0`) identically on both boxes.

## Interop on gfx1151 — torch, jax, dlpack (2026-07-19)

Working configuration, validated on this box (Ubuntu 24.04, apt ROCm 7.2.1,
warp built against `/opt/rocm`):

- Packages, from the TheRock nightly index
  (`https://rocm.nightlies.amd.com/v2/gfx1151/`), installed into the warp venv:
  `torch==2.10.0+rocm7.13.0a20260513` (with its `rocm-sdk` runtime wheels) and
  the jax stack **pinned** to `jax==0.10.2 jaxlib==0.10.2
  jax-rocm7-plugin==0.10.2 jax-rocm7-pjrt==0.10.2`.
- Required runtime configuration: `LD_LIBRARY_PATH=<venv
  site-packages>/_rocm_sdk_core/lib` so TheRock's newer ROCr loads first.
  Without it, warp.so (built against apt ROCm 7.2) pulls the system ROCr and
  TheRock's `libamdhip64.so.7` fails with `undefined symbol:
  hsa_ext_image_create_v2`. Both stacks run correctly on the newer ROCr.
- Results with this configuration: `interop/test_jax.py` 54 ran, 50 pass / 4
  skip; `interop/test_torch.py` 32/32 from a fresh kernel cache;
  `interop/test_dlpack.py` 23/23 (after the `__dlpack_device__` kDLROCM fix);
  `tile/test_tile_mlp.py` 3 ran OK. Paddle has no gfx1151 build; its tests
  remain environmental skips.

The same holds with a native ROCm 7.14.0 install as the system default
(warp built against `/opt/rocm` → 7.14.0): torch runs standalone on its
bundled runtime, and `interop/test_torch.py` passes 32/32 with the
`LD_LIBRARY_PATH=<venv site-packages>/_rocm_sdk_core/lib` override. Without
the override, importing torch after warp in one process fails —
`libamd_comgr.so.3: undefined symbol ... LLVM_23.0` — because torch's bundled
`rocm-sdk` and the system 7.14 LLVM cannot mix. The override remains
required for any process that uses both.

Caveats:

- **jax 0.11.0 nightly aborts on gfx1151**: with
  `jax/jaxlib/jax-rocm7-* == 0.11.0` the process dies inside XLA
  (absl/Eigen threadpool stack) after ~14-18 warp interop tests, with or
  without the `LD_LIBRARY_PATH` setting. The 0.10.2 pin is the working set.
- **torch external-capture warm-cache race**: the six
  `test_torch_graph_*` tests pass from a fresh kernel cache (32/32) but fail
  with `Warp error: stream is not capturing` when every module loads from a
  warm cache (run completes in 0.07 s). The module-compile delay masks a race
  between torch's stream-capture start and warp's
  `capture_begin(external=True)` check on HIP with the torch 2.10 nightly.
  Not further localized.

## Warp unit tests (7 files with failures)

### Deep / open bugs

- **`test_fem` / `test_sparse` — `test_capturability` — resolved on the
  1.17.0.dev4 base.** The zero `nnz` after replay was the replayed-graph
  external-event semantics (1.17.0.dev4 section above): `nnz_sync()` waits on
  an event recorded with the external flag during capture, and the host wait
  returned before the replayed record node executed. The host-side workaround
  resolves it and the tests pass on the current base. The record below stands
  as written at the time; its elimination of temp-buffer aliasing remains
  valid, but the closing attribution to a rocPRIM-under-replay race was wrong.

  BSR (block-sparse)
  assembly under graph capture was **flaky** on HIP (passed ~1 run in 3; on CUDA
  it always passes). Root-caused at the time as follows: the failure is
  deterministic in shape —
  after replay, `C.nnz_sync()` returns `0` instead of `9`. Traced through
  `bsr_mm` → the native `wp_bsr_matrix_from_triplets_device` (`sparse.cu`): the
  hipCUB `DeviceRunLengthEncode` writes the unique-block count to a one-int temp,
  and the next kernel `bsr_find_row_offsets` reads it as `*d_nnz`; when it reads
  `0`, every row offset (and thus the nnz) is `0`. The read races only under
  graph replay (eager execution always passes). **Async-pool aliasing ruled
  out:** replacing every sparse temp buffer with a per-context persistent cache
  (`CachedTemporary`), so that *nothing* is alloc'd/freed inside the captured
  graph — verified sized by the warm-up run — did **not** change the failure rate
  (still ~1/3 over 10 runs). So this is *not* the temp-buffer aliasing that
  affects the multi-stream/BVH-refit path; the caching change was reverted. What
  remains is a race in the hipCUB/rocPRIM device primitives themselves
  (`DeviceRunLengthEncode` / `DeviceScan`) under HIP graph *replay* on RDNA: the
  scalar count they publish is not reliably visible/ordered before the consuming
  `bsr_find_row_offsets` kernel on replay, though it always is eagerly. This
  points to a ROCm/rocPRIM-under-graph-capture limitation rather than a Warp bug.
  Candidate Warp-side workarounds (untried): run the cub-based assembly eager
  (don't capture it), or reformulate the count hand-off to avoid depending on a
  device-published scalar under capture. Likely belongs with the multi-stream
  work as a platform-gap follow-up rather than a quick fix.

### Parked on a branch

- **`test_hash_grid` — `test_hashgrid_multiple_streams`** — building `HashGrid`s
  concurrently on multiple streams races because native `wp_alloc_device` /
  `wp_free_device` use ROCm's async pool, which aliases still-in-use memory
  across streams (the Python allocator was fixed for this; the native half was
  not). A capture-aware native allocator fixes it, but that change exposed a
  second defect in the bottom-up BVH refit, which is why both were parked on
  branch `tomas/hip-multistream-refit-race` with a design doc
  (`design/hip-multistream-refit-race.md`). That second defect was the
  descriptor recycling fixed above, so the allocator work no longer has to be
  landed together with a refit fix.

  Narrowed 2026-07-28 (`scratchpad/diag_hashgrid_lifetime.py`). It takes two
  things together, and either alone is fine:

  | grid destroyed during the loop | streams | result |
  |---|---|---|
  | no (held in a list) | 10 | pass |
  | no | 1 | pass |
  | yes (one local rebound) | 1 | pass |
  | yes | 10 | **fail, 7-9 of 10 grids wrong** |

  So the free *is* correctly ordered within a stream: rebinding on a single
  stream is safe. What breaks is a block freed on one stream being handed to an
  allocation on another before the first stream's queued kernel has read it. The
  upstream test rebinds one local per iteration, which is why it fails while a
  diagnostic that keeps every grid alive passes.

  Not reachable through the pool's reuse policy. Warp already disables
  `hipMemPoolReuseAllowOpportunistic` and
  `hipMemPoolReuseAllowInternalDependencies`; disabling
  `hipMemPoolReuseFollowEventDependencies` as well does not help, and all three
  read back with the values Warp set (`hipMemPoolGetAttribute` returns success),
  so ROCm's pool recycles across streams regardless of them.

  **Fixed 2026-07-28** by keeping native allocations off the pool outside graph
  capture, mirroring what Warp's Python allocator already did. The pool is still
  used during capture, where plain allocations are not capturable, and pointers
  served from it are tracked so each free goes back the matching way.
  `test_hash_grid` now passes 6 of 6 process runs.

- **BVH refit — intermittent hang and fault on gfx1151 (2026-07-21).**
  **Resolved 2026-07-28**; this was the descriptor-recycling fault above, not a
  defect in refit. `test_bvh_refit_root_leaves_cuda_0` was the first test in the
  file to create and destroy enough BVHs to hit a recycled descriptor address,
  which is why it looked like a refit bug; the three tests that error afterwards
  were poisoned by it rather than failing on their own.
  GPU busy at 100 percent distinguishes the hang from the host boot-state wedge
  described above, where the GPU sat idle.

  **Re-measured 2026-07-24 at head `b7408b8d0`. `refit()` is not involved, and
  the name of this entry is retained only for continuity.** The trigger was
  re-derived by removing one suspect per run rather than by reasoning from the
  parked design document. Driver `scratchpad/bracket_bvh_fault.py`, artifacts
  `scratchpad/bracket/`, 200 iterations x 3 repeats:

  | Variant | Verdict |
  |---|---|
  | build, copy new bounds, refit, query, destroy | 1 of 3 fault |
  | new bounds hoisted out of the loop | 3 of 3 fault |
  | no `refit()` | 2 of 3 fault |
  | neither copy nor `refit()` | 2 of 3 fault, 1 hang |
  | BVH never destroyed | 3 of 3 clean |
  | no query | 3 of 3 clean |
  | whole loop on `cpu` | 3 of 3 clean |

  The minimal failing loop is therefore **build, query, destroy** — no refit and
  no bounds update. The earlier control that appeared to clear the no-refit case
  ran only 40 iterations; faults here arrive between iterations 8 and 132, so it
  stopped short. The two necessary ingredients are a kernel that dereferences
  the BVH and per-iteration destruction of it.

  A second matrix (`scratchpad/bracket2_bvh_fault.py`, `scratchpad/bracket2/`,
  300 iterations x 4 repeats) separates BVH code from allocator churn:

  | Variant | Verdict |
  |---|---|
  | build, BVH query, destroy | 3 fault, 1 hang |
  | identical churn, launched kernel never touches the BVH | 4 of 4 clean |
  | no BVH at all: array allocate, kernel, free | 4 of 4 clean |
  | as the first row, memory pool disabled | 3 fault, 1 hang |
  | as the first row, device synchronize after each destroy | 4 of 4 fault |

  Rows two and three allocate and free at the same rate as row one, so this is
  not generic allocator churn: the fault needs a kernel that actually reads the
  BVH. Row four retires the parked design document's remaining theory — the
  stream-ordered pool is not the mechanism, since disabling it changes nothing.

  **Fault signature.** The kernel log records every occurrence identically:
  `[gfxhub] page fault (src_id:0 ring:24 vmid:8)`, `in page starting at address
  0x0000000000000000`, `Faulty UTCL2 client ID: TCP (0x8)`, `PERMISSION_FAULTS:
  0x3`, `RW: 0x0`. That is a shader reading through a null or near-null pointer.
  The hang is the same defect on a different value: a garbage node index makes
  the unbounded `for (;;)` parent walk in `bvh_refit_kernel` never reach -1.
  Because a device synchronize follows the build, the faulting launch is the
  query itself.

  **Five hypotheses tested and rejected, all measured on gfx1151.**

  1. *Acquire-release on the refit child counter.* 4 of 4 repro arms and 1 of 10
     `test_bvh` runs still failed. Joins `__threadfence()` as ineffective.
  2. *A memory-visibility race at all.* The defect survives
     `wp.config.verify_cuda`, which synchronizes after every launch.
  3. *The stream-ordered pool recycling a block before its free retires.*
     Disabling the pool leaves the failure rate unchanged (row four above).
  4. *Uninitialized parent pointers.* `build_karras_topology` writes nothing when
     `n == 1`, so a single-item tree never receives a parent from the builder —
     but the port already memsets `node_parents` to `0xFF` (bvh.cu) and zeroes
     `root`, and `scratchpad/diag_bvh_state.py`, which reads the live device
     descriptor and its buffers, confirms `node_parents[0] == -1`. The CUDA path
     uses `build_hierarchy`, which does write `parents[0] = -1` for `n == 1`.
  5. *The device-side descriptor published from a dying stack frame.*
     `wp_bvh_create_device` copies its descriptor from a local with
     `wp_memcpy_h2d`, which is `cudaMemcpyAsync` with no synchronization
     (warp.cu), so the source can be reclaimed before the copy reads it; CUDA
     stages pageable sources and HIP does not. Waiting for that copy before the
     local dies was built and measured: `minimal` still failed 3 of 4 with one
     hang. The change was reverted rather than carried as an unjustified delta.
     The pattern remains a latent hazard at six sites (`mesh.cu`, `bvh.cu`,
     `bvh_cubql.cu`) but is not this defect.

  **What the tree looks like when it fails.** `scratchpad/diag_bvh_state.py`
  reads the device `wp::BVH` through aliased pointers. Over 200 iterations of
  every case, `node_parents`, `node_counts`, `root`, `primitive_indices` and
  both node arrays are correct at every iteration, including immediately before
  a faulting query. Those readbacks also mask the fault completely (3 x 200
  clean), which is itself evidence that this is timing dependent rather than
  wrong data at rest.

  **The faulting pointer, identified 2026-07-26.** The traversal was
  instrumented as that next step: `bvh_query` and `bvh_query_next` were given
  null and range checks on every pointer and index they dereference, on the
  failure path only, printing what the shader saw and returning an empty query
  instead of faulting (working-tree edit to `warp/native/bvh.h`, kept at
  `scratchpad/bvh.h.wp_bvh_diag`; a JIT header, so a fresh `WARP_CACHE_PATH` is
  enough). Across 6 runs x 300 iterations of the minimal loop, every event named
  the same thing:

  ```
  WP_BVH_DIAG query null desc: lowers=(nil) uppers=(nil) parents=(nil)
    counts=(nil) prims=(nil) root=(nil) item_lo=(nil) item_up=(nil)
    items=0 nodes=0 max_nodes=0 leaves=0 ctor=0
  ```

  The query kernel reads the **entire device descriptor as zeros**. Every
  pointer in it is then null, and dereferencing one produces exactly the
  recorded signature: a shader read of address `0x0` from the TCP client. The
  hang is the same value in a parent walk. This closes the "which pointer"
  question: it is not one pointer, it is the descriptor itself.

  **The zeros are lost writes, not a stale view.** On each mismatch,
  `scratchpad/diag_zero_read.py` re-launches the identical query, reads the
  descriptor and every buffer back over DMA, and re-launches again. The zeros
  persist across all three: a later DMA readback agrees the memory is zero, and
  re-launching returns the same wrong answer. Some iterations show a partial
  version — a valid descriptor whose `primitive_indices` reads `[0, 0]` instead
  of a permutation, or whose node 0 is zeroed — which is why the query returns
  1 or 0 hits instead of 2. **This corrects the paragraph above**: the tree is
  not always correct at rest. The earlier readbacks saw only clean iterations
  because performing them masks the defect.

  **Runtime A/B, 8 repeats x 300 iterations, stock header, crash level**
  (`scratchpad/ab_fragment_allocator.py`, artifacts `scratchpad/ab_frag/`):

  | Arm | Verdict |
  |---|---|
  | default | 7 fault, 1 hang |
  | `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` | 8 of 8 clean, 0 wrong results |
  | native allocations forced to plain `hipMalloc` | no crash, but 5 to 11 wrong results per 300 iterations |
  | both together | 8 of 8 clean, 0 wrong results |

  Two conclusions. The ROCm runtime's fragment suballocator is the mechanism:
  disabling it removes both the crash and the wrong results, in either allocator
  mode. And Warp's own allocator choice is not: forcing plain `hipMalloc`
  removes the crash but leaves the lost writes, now visible as silent wrong
  answers.

  **A correction to the row above.** The earlier "memory pool disabled" row used
  `wp.set_mempool_enabled(device, False)`, which only swaps the *Python*
  allocator; `wp_alloc_device` gates on `wp_cuda_device_is_mempool_supported`,
  so the native BVH buffers stayed on `cudaMallocAsync` and the native pool path
  was never actually tested. The row-three arm above tests it (via a temporary
  `WARP_ALLOC_DEFAULT` build knob, since reverted).

  **The workaround is not recommended as a blanket setting.** With
  `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` the fault disappears but `test_bvh` then
  fails deterministically, 6 of 6, in `test_bvh_aabb_cuda_0`. That failure
  localizes (`scratchpad/diag_aabb_miss.py`, `diag_first_build.py`) to the
  **first GPU LBVH build in a process**, whose `primitive_indices` comes out as
  the identity permutation `[0, 1, 2, ...]` instead of a Morton ordering: the
  hierarchy is then built on unsorted keys, `mark_packed_leaf_nodes` converts 30
  internal nodes into leaves, and the query misses about 75 of 100 boxes.
  `scratchpad/diag_sort_first_call.py` traces it to the first
  `wp.utils.radix_sort_pairs` call in the process returning
  `hipErrorInvalidValue` from hipCUB's `SortPairs` (temp storage is valid:
  256 bytes, non-null, identical to the calls that succeed). Warp logs that
  error and proceeds, so a failed sort silently yields a corrupt BVH rather than
  an exception — worth knowing independently of this platform quirk. It does not
  occur in the shipped default configuration.

  **Not reproducible without Warp.** Two standalone HIP programs
  (`scratchpad/hip_frag_repro.cpp`, `hip_frag_repro2.cpp`), the second
  mirroring `wp::bvh_create_device`'s exact per-BVH pattern — the same twelve
  buffer sizes, the same memsets, the same device-to-device copy, the same
  descriptor copied asynchronously from a host local, the same free order —
  stay clean for 3000 iterations on both arms and across four ingredient
  variants. So a minimal repro to hand to AMD does not exist yet; the Warp
  repro (`scratchpad/bracket2_bvh_fault.py --variant minimal`) reproduces in
  under a second and is the artifact to attach.

  Still open: what makes the fragment suballocator lose writes to freshly
  suballocated small blocks under this workload. The next step is a ROCm-side
  one (an allocator trace, or a report against the runtime), not a further Warp
  change — Warp's allocator has now been eliminated as the mechanism.

  Two unrelated defects were found and fixed during this audit; neither changes
  the failure above. `wp_bvh_create_device` declared its descriptor as an
  uninitialized local and the LBVH branch never set `num_nodes`, so garbage
  reached the device descriptor (measured: `num_nodes = 5574997` on every LBVH
  tree). No device query code reads that field today, but `bvh_cubql.cu` gates
  on `num_nodes == 0`. And `wp_bvh_destroy_device` freed the descriptor with no
  `ContextGuard` active, where `wp_mesh_destroy_device` guards its whole
  teardown.

### Niche / platform-dependent (candidates for HIP skip-guards after review)

- **`test_streams`** — two event tests still fail on the current base:
  `test_event_elapsed_time_graph` and `test_event_external`, both now
  root-caused to ROCm's replayed-graph external-event semantics (1.17.0.dev4
  section above) — the device side, which the host-side workaround does not
  cover. `test_stream_priority_timings` (a timing-based assertion — inherently
  fragile, returns 0 elapsed for a tiny kernel on this iGPU) is no longer
  among the failing set on the current base.

- **`test_verify_fp` — `test_nan`** — the `wp.config.verify_fp` debug mode's
  NaN-detected warning is not present in captured stdout on HIP (detection or
  output routing differs). Debug-only feature; fixable-vs-limitation unclear.

- **`test_texture`** (2) and **`test_examples`** (3 example scripts) — not yet
  triaged to the specific failing cases.

### Resolved this port (for reference)

`test_large_launch_large_kernel` (2³³-thread launch) is **skipped** on HIP — the
driver rejects a grid that large on gfx1151 (a hardware grid-size limit), the
same reason it is already skipped on CPU. `test_cuda_arch_suffix` and `test_ipc`
are skipped on HIP as CUDA-only / non-UMA features. Graph capture of
`bvh.rebuild()`, multi-warp `tile_sum`, int64 atomics, `pow` precision, and the
mempool-aliasing cascade are fixed (see git history).

**`test_async`** (2,115 errors → all passing) was the single biggest gap. Bisection
pinned it to one test — a host→device copy into a non-contiguous array with
mempools disabled, mid-capture. That copy legitimately fails ("cannot allocate a
staging buffer during capture"), but on HIP the failed *synchronous* allocation
invalidated the capture, and `hipStreamEndCapture` then returned error 900 without
transitioning the stream out of capture mode — leaving it stuck and poisoning the
context for the file's remaining ~2,100 cases (on CUDA the same allocation does
not poison the context). Fixed by refusing the synchronous allocation up front
while a capture is active (it can never succeed then anyway), so the capture ends
cleanly; the UMA hybrid allocator falls back to the capture-safe mempool path.

## Newton examples (96 of 104 passing)

Current status at the 1.17.0.dev4 base on ROCm 7.14.0 with Newton 1.4.0
(`tomas/gfx1151-fixes` branch): 96 of 104 examples run to completion in the
sweep. The non-runners are all in documented platform classes: the
graph-replay fault class (several of them intermittent rather than
deterministic), `cloth_franka` (the intermittent non-finite state above), one
diffsim example exceeding the harness timeout, and one benchmark-output
harness quirk.

At the 1.15.0 base the sweep ran each example through
`python -m newton.examples <name> --viewer null --benchmark` at 200 and 1200
frames; 74 of 93 ran to completion. The 19 that did not:

- 6 `kamino_*` — require conditional graph nodes, which HIP does not provide
  (`RuntimeError: Conditional graph nodes are not supported on HIP/ROCm`).
- 4 `*_mpm_*` coupled-solver examples (`mpm_twoway_coupling`,
  `mujoco_mpm_coupled_solver`, `vbd_mpm_coupled_solver`,
  `xpbd_mpm_coupled_solver`) — root-caused and fixed on the Newton side
  (`tomas/gfx1151-fixes` commit `dee368ca`); all four now run to completion.
  Chain: `SolverImplicitMPM`'s convergence loop is capture-safe only with
  conditional graph nodes; on HIP it falls back to a host loop that reads
  residuals back each batch. The examples captured their step gated on
  device type alone, so the readback executed inside the capture. On CUDA
  that readback fails cleanly (error 906 from `cudaMemcpyAsync`); on ROCm it
  silently executes via the unified-memory readback path (no memcpy call —
  breakpoint-verified), corrupts the stream-capture state, and
  `hipStreamEndCapture` then dereferences a null internal object
  (`pthread_mutex_lock(0x1d0)`). Minimal reproducers:
  `scratchpad/repro_numpy_during_capture.py`,
  `scratchpad/repro_capture_variants.py`. Open Warp-side hardening: the
  device-to-host readback path taken on unified-memory devices must refuse
  to run while the stream is capturing, mirroring CUDA's error; a guard in
  `wp_memcpy_d2h` alone is insufficient because that path is bypassed.
  Captured device-to-device copies and the explicit-synchronize guard are
  verified unaffected.
- 7 intermittent process-teardown failures (`Warp CUDA error 3: initialization
  error` or `error 700` in `wp_cuda_context_pop_current`) — the affected
  example set changes from run to run; in this sweep: `basic_multi_solver_
  overlay`, `basic_plotting`, `recording`, `selection_articulations`,
  `selection_materials`, `selection_multiple`, `sensor_imu`.
- `diffsim_bear` — exceeds the 600 s harness timeout at 1200 frames.
- `contacts_rj45_plug` — `ViewerNull` has no `picking` attribute; fails
  identically on the NVIDIA reference (`recording` also fails on both
  platforms), so neither is a port difference.

An earlier count on ROCm 7.2.1 with Newton 1.0.0 reported 4 of 74 not passing
and required the `patches/newton/03-*` contact-reduction patch for
`cloth_franka`, `cloth_h1`, and `nut_bolt_sdf`. On the current stack those
three pass with no patch (Newton refactored the affected snippets after
1.0.0), and `patches/newton/` does not apply to Newton 1.4.0.

## ROCm version coverage

Primary bring-up and the full suite above are on **ROCm 7.2.1**. The port also **builds and runs on ROCm 7.2.4** (validated in a container, `HIP 7.2.53211`, host install untouched): `warp.so` compiles cleanly, the gfx1151 smoke — including graph capture — passes, and a representative test set is identical to 7.2.1 (the two failures are the same documented `hash_grid` / `fem` gaps, not 7.2.4 regressions). No behavioral difference was observed between 7.2.1 and 7.2.4.

### Small device allocations fail on ROCm 7.14 under graph-capture workloads

On ROCm 7.14.0, `hipMalloc` returns NULL for allocations of a few dozen bytes
while the device has essentially all of its memory free. The same workload on
ROCm 7.2.1 does not fail once.

Measured with `newton.tests.test_mesh_backface` at newton `220864f5`, with
identical Python packages on both sides (numpy 2.5.1, mujoco-warp 3.10.0.2,
usd-core 26.3, trimesh 4.12.2):

| | ROCm 7.2.1 | ROCm 7.14.0 |
|---|---|---|
| Allocation failures | **0** | **47** (48 at 56 bytes, 2 at 12 bytes) |
| Test result | 43 of 43 pass, 802 s | 43 of 43 fail, 15 s |
| Peak VRAM at failure | n/a | **2.49 GB of 103 GB (2.4 %)** |

Four candidate explanations were tested and each is refuted by measurement:

1. **Not the graph-capture allocation gap.** Warp appends a `reason` string to
   this error only when `device.is_capturing` is true. The reason is empty, so
   the failing call is a plain allocation outside any capture.
2. **Not memory exhaustion.** VRAM was sampled once per second for the whole
   run; the peak is 2.4 % of the device.
3. **Not leaked capture allocations.** Warp logs
   `wp_alloc_device_async: failed to find memory allocation node` when it cannot
   match a capture-time allocation to its graph node. ROCm 7.2.1 emits **more**
   of these (1440 against 960) and still fails nothing.
4. **Not library skew.** The package sets were compared field by field and are
   identical.

The failing allocation is an ordinary `wp.array` created inside
`newton builder.finalize()`.

Scope: the damage concentrates in graph-capture-heavy workloads. Across the
whole Warp test suite only one module logs this error on 7.14 (and there it is a
cascade after an unrelated fault), against none on 7.2.1; Newton, whose solver
captures graphs continuously, loses `test_mesh_backface` entirely and takes
partial losses in `test_heightfield` and `test_remesh`.

Containerization is not the variable. `scratchpad/hip_small_alloc_repro.cpp`
runs the same standalone program on three configurations, and the ROCm 7.2.4
*container* behaves like the ROCm 7.2.1 *host*, not like the 7.14 container.

**The standalone reproducer does not reproduce the failure, and what it does
show contradicts the obvious hypothesis.** Over 400 rounds of capture,
instantiate, launch, destroy, with eight `hipMallocAsync` per capture:

| Configuration | Free memory, start to end | 56-byte failures |
|---|---|---|
| ROCm 7.14, container | 102.85 GB → 102.72 GB | 0 |
| ROCm 7.2.4, container | 102.85 GB → 96.01 GB | 0 |
| ROCm 7.2.1, host | 102.85 GB → 95.98 GB | 0 |

ROCm 7.14 reclaims graph-capture memory correctly. ROCm 7.2.x leaks roughly
17 MB per round, about 6.9 GB across the run. The version that fails Warp's
small allocations is therefore the version with the *better* capture-memory
accounting, which rules out the intuitive explanation that untracked capture
allocations exhaust something.

The mechanism is consequently **not established**. What is established is the
observable: on this workload ROCm 7.14 returns NULL for allocations of a few
dozen bytes and ROCm 7.2.1 does not, with all four alternative explanations
above eliminated. The smallest known reproducer remains the Warp one
(`newton.tests.test_mesh_backface`); no Warp-free case exists yet.

**Consequence for deployment: neither version is unconditionally better.** See
the graph-capture leak below before choosing.

### Graph-capture memory leak on ROCm 7.2.x

ROCm 7.2.x does not reclaim memory allocated inside a captured graph, and
replaying such a graph leaks on every launch. Sustained simulation degrades
severely as a result.

Measured with the ANYmal C walking example (`newton.examples.robot.
example_robot_anymal_c_walk`), which captures its policy step and solver once
and then replays the graph per frame. Throughput in blocks of 60 frames,
rendering excluded:

On 7.2.1 throughput falls **8.2x across 600 frames** while VRAM climbs
monotonically by 290 MB, about 0.5 MB per frame. On 7.14.0 throughput is
flat across the same span and VRAM does not drift.

The cause is not thermal. Edge temperature holds at 44-46 C on both. The
core clock does fall on 7.2.1, from 2348 MHz to 1325 MHz, but only 1.8x
against an 8.2x throughput loss, and it falls because the device is
increasingly idle waiting on the host, not the reverse.

The same leak is reproducible without Warp: `scratchpad/hip_small_alloc_repro.cpp`
over 400 capture rounds loses 6.9 GB on ROCm 7.2.x and nothing on 7.14.0.

**Choosing a runtime.** The two defects pull in opposite directions:

| | ROCm 7.2.x | ROCm 7.14.0 |
|---|---|---|
| Sustained graph replay | degrades 8.2x, leaks | flat |
| Small allocations under capture-heavy load | correct | fails (see above) |
| BVH lost-write defect | present | present |

For sustained simulation and reinforcement-learning rollouts, which replay
captured graphs continuously, ROCm 7.14.0 is the usable runtime. For workloads
that hit the small-allocation failure, ROCm 7.2.x remains necessary until that
defect is resolved. Neither is recommended without reference to the workload.

### Single-address atomic contention

`atomic_add` to one destination from every thread is far slower on gfx1151 than
on the RTX A4000. Measured with `asv/benchmarks/deterministic.py`
`AtomicAddDeterminismOverhead`, whose `num_outputs=1` case is defined as the
worst-case contention arm — every thread targets the same output — against
`num_outputs=65536` as the lower-contention control:

| Destinations | Elements | gfx1151 | RTX A4000 | Ratio |
|---|---|---|---|---|
| 1 | 65,536 | 167,930 us | 143.7 us | 1168x slower |
| 1 | 262,144 | 393,344 us | 547.6 us | 718x slower |
| 1 | 1,048,576 | 1,316,132 us | 2,156.9 us | 610x slower |
| 65,536 | 262,144 | 43.2 us | 16.6 us | 2.6x slower |
| 65,536 | 1,048,576 | 123.5 us | 47.2 us | 2.6x slower |

The gap is confined to the maximum-contention case. At 65,536 destinations the
ratio is 2.6x, in line with the rest of the suite. Only the one-destination arm
degrades by two to three orders of magnitude, and the absolute cost reaches 1.3
seconds for a single benchmark iteration.

**Cause established, and a remedy is available behind an opt-in.** HIP compiles
`atomicAdd()` on a float to a `global_atomic_cmpswap_b32` retry loop rather than
the hardware `global_atomic_add_f32`, because it cannot prove the target is
coarse-grained memory. Confirmed by disassembly
(`hipcc -S --offload-arch=gfx1151`): the default lowering emits one
`cmpswap_b32`, the hardware form one `add_f32`. Under contention the retry loop
is what costs the two to three orders of magnitude; with the hardware
instruction the same contended benchmark improves by over 100x.

Set `warp.config.hip_fast_fp_atomics = True`, the `WARP_HIP_FAST_FP_ATOMICS=1`
environment variable, or the `"hip_fast_fp_atomics"` module option to select the
hardware instruction. It covers every float32 accumulation routed through
`atomic_add`, including `atomic_sub` and the vector, matrix, quaternion and
transform overloads. `atomic_max`, `atomic_min` and the float64 forms are
unaffected — they are compare-and-swap loops on this architecture, as is float16.

The whole-workload effect is much smaller than the microbenchmark and tracks how
much a kernel actually contends: measured on Newton, 1.44x on a self-colliding
cloth example at its default frame count, 1.30x on granular MPM, and no
measurable change on a cloth example without self-collision. Quote the frame
count with any cloth figure — per-frame cost rises as self-contact builds, so the
ratio grows with run length.

**Off by default because of a correctness boundary, not a performance one.** The
hardware instruction silently discards every update whose target is host-coherent
memory: it executes in the GPU's L2 cache, and host-coherent pages are mapped so
device accesses bypass L2, leaving the instruction with no ALU to run on. It
retires with no effect and returns zero. Integer atomics are unaffected, because
the fabric implements them.

Warp's own managed allocations — the UMA hybrid allocator and `ReadbackBuffer` —
are marked coarse-grained when created, so they are cached in L2 and this setting
is safe on them. It is **not** safe on managed memory Warp did not allocate that
way, notably `CudaManagedAllocator` arrays, whose CUDA managed-memory coherence
contract is left unchanged; a float32 atomic into such an array is discarded with
no error.

### array_sum SIMT and tile paths invert

Re-measured on ROCm 7.14.0, consistent with the earlier ROCm 7.2.1 measurement
in direction and magnitude:

| Benchmark | gfx1151 | RTX A4000 | Ratio |
|---|---|---|---|
| `tile.array_sum.ArraySumSimt` | 727.5 us | 28,692.2 us | 39.4x faster |
| `tile.array_sum.ArraySumTile` | 5,438.4 us | 420.8 us | 12.9x slower |

The same computation is an order of magnitude faster than the reference through
the SIMT path and an order of magnitude slower through the tile path. Cause
unestablished, but three plausible explanations have been eliminated by reading
the code (2026-07-29):

- **Not a wavefront-width mistake.** `WP_TILE_WARP_SIZE` keys off `__GFX9__`, so
  gfx1151 correctly takes the 32 branch; the reduction loop halves from
  `WP_TILE_WARP_SIZE / 2` accordingly.
- **Not emulated shuffles.** Warp's own `__shfl_down_sync` in `cuda_crt.h` is
  inline PTX and so is NVIDIA-only; on HIP the tile reduction binds to ROCm's
  `amd_warp_sync_functions.h`, which lowers to hardware cross-lane instructions.
- **Not shared-memory capacity.** The tile in this benchmark is far below the
  64 KB LDS limit that constrains the tile tests listed above.

What remains is the tile framework's own memory traffic and barrier pattern,
which needs a profiler rather than source reading.

### ROCm 7.14.0

Re-validated 2026-07-26 on **ROCm 7.14.0** (released 2026-07-15), again in a
container with the host install untouched. gfx1151 is visible to 7.14 userspace
through the host's 7.2.1-era kernel driver, so no host upgrade was required:
`rocminfo` reports `gfx1151` / `AMD Radeon 8060S Graphics`, the port compiles
clean, and Warp initializes as `ROCm 7.14, HIP driver 7.14 … (96 GiB, gfx1151,
mempool enabled)`.

| Check | Result on 7.14.0 |
|---|---|
| Build for gfx1151 | succeeds |
| Warp differential vs RTX A4000 | 7723 concordant passes, 19 port discrepancies, 25 AMD-better, 3 shared fail |
| CDNA gfx90a + gfx942 build | succeeds, 0 errors |
| BVH lost-write defect | **still present** — see the entry above |

Three deployment notes, each of which cost a debugging cycle and none of which
are Warp defects:

1. **7.14 relocated its libraries.** They live under `/opt/rocm/core-7.14/lib`,
   reached through an `/opt/rocm/lib → /etc/alternatives/rocm-lib` symlink
   chain, and no `ld.so.conf.d` entry is installed — `ldconfig -p` lists none of
   them. A `warp.so` linked against `libamdhip64.so.7` therefore fails to load
   until `LD_LIBRARY_PATH` includes that directory. The same restructuring traps
   in-place upgrades: with `/opt/rocm` still pointing at the old tree, a naive
   install nests `core-7.14` inside the previous version's directory.
2. **7.14 uses a different package channel.** `repo.amd.com/rocm/packages-multi-arch/`
   with packages named `amdrocm7.14`, not the `repo.radeon.com/rocm/apt/` path
   used through 7.2.4, and it requires uninstalling 7.2.x first.
3. **Newton 1.4.0 constrains `usd-core` to `>=25.5,<26.5`**, so the current
   usd-core 26.8 cannot be used when Newton is in the picture; 26.3 is the
   version both sides were validated on.

## Batched RL (MuJoCo-Warp) — constraint-buffer sizing on HIP

Massively-parallel RL data collection (MuJoCo-Warp `testspeed`, batched humanoid)
runs correctly on gfx1151 with **512 parallel humanoid environments**. It
first appeared to crash, and the crash is now root-caused —
it is **not** a batched-collision bug. The auto-sized default `njmax` (constraint
limit, ~64/world) is too small for RL exploration: random controls make the
humanoid flail, occasionally spiking a world's constraint count above `njmax`.
Under graph capture the buffer cannot resize, so the excess constraint write goes
out of bounds. **HIP faults on that OOB (CUDA error 700, illegal access) where
CUDA silently masks it**, so the run crashes on AMD with the default sizing (single
world, and eager execution, are fine — only batched-under-capture overflows).

- **Fix (verified):** pass an adequate `njmax` / `nconmax` (e.g. `--njmax 4096
  --nconmax 16384`) → 512 humanoids run clean.
- **Re-measured 2026-07-19** (mujoco-warp 3.10.0.2 on the current port, EVO-X2II
  box, eager mode — graph-captured `testspeed` now hits the capture-alloc class
  on replay, and mujoco_warp's conditional-graph solver loop must be disabled
  via `opt.graph_conditional=False`): throughput scales through 128, 512, and
  1,024 worlds, past the previously noted ~512-world ceiling in eager mode.
  Harness:
  `~/git/newton/amd_sweep/rl_eager_bench.py`.
- **End-to-end MJX/brax PPO training (2026-07-19): blocked upstream, not by the
  port.** With TheRock jax-rocm (0.10.2 pinned and 0.11.0), brax PPO on pure-jax
  MJX (no warp in the process) dies compiling the fused physics graphs:
  `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` at 2048 envs, silent segfaults at
  512/128 envs (`HSA_XNACK=1` does not help); light jax workloads (the 54-test
  warp interop suite) pass. The mjx `impl='warp'` bridge fails on GraphMode API
  skew between mujoco-mjx 3.10.0 and released mujoco-warp versions.
- **Robust upstream fix (characterized — deeper than a one-line clamp):** on
  overflow the contact kernel drops the excess rows and sets their
  `contact.efc_address` to `-1`. A *partially* dropped contact (row 0 fits, rows
  1–2 dropped) still passes the solver's `efcid == efc_address[conid,0]` check,
  then indexes `ctx_quad[.., efc_address[conid,1]]` with `-1` → out of bounds
  (HIP faults, CUDA masks). Clamping `nefc` to `njmax` is necessary but *not*
  sufficient; the real fix must make contact-row reservation all-or-nothing, or
  guard every `-1` `efc_address` read in the solver — a MuJoCo-Warp change with
  solver-correctness implications, best done upstream with review. (Attempted
  here and reverted rather than ship an unverified solver change.)
- Batches beyond ~512 humanoids hit a separate memory/scaling ceiling on this iGPU.
