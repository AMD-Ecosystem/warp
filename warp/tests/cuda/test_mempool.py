# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import sys
import threading
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def get_device_pair_without_mempool_access_support():
    devices = wp.get_cuda_devices()
    for target_device in devices:
        for peer_device in devices:
            if target_device != peer_device:
                if not wp.is_mempool_access_supported(target_device, peer_device):
                    return (target_device, peer_device)
    return None


def test_mempool_release_threshold(test, device):
    device = wp.get_device(device)

    assert device.is_mempool_supported

    test.assertEqual(wp.is_mempool_supported(device), device.is_mempool_supported)

    was_enabled = wp.is_mempool_enabled(device)

    # toggle
    wp.set_mempool_enabled(device, not was_enabled)
    test.assertEqual(wp.is_mempool_enabled(device), not was_enabled)

    # restore
    wp.set_mempool_enabled(device, was_enabled)
    test.assertEqual(wp.is_mempool_enabled(device), was_enabled)

    saved_threshold = wp.get_mempool_release_threshold(device)

    # set new absolute threshold
    wp.set_mempool_release_threshold(device, 42000)
    test.assertEqual(wp.get_mempool_release_threshold(device), 42000)

    # set new fractional threshold
    wp.set_mempool_release_threshold(device, 0.5)
    test.assertEqual(wp.get_mempool_release_threshold(device), int(0.5 * device.total_memory))

    # restore threshold
    wp.set_mempool_release_threshold(device, saved_threshold)
    test.assertEqual(wp.get_mempool_release_threshold(device), saved_threshold)


def test_mempool_release_threshold_config(test, device):
    """WARP_MEMPOOL_RELEASE_THRESHOLD sets every device's release threshold at initialization."""
    device = wp.get_device(device)
    cases = {"0.25": int(0.25 * device.total_memory), "1048576": 1048576, "0": 0, "18446744073709551615": 2**64 - 1}
    for value, expected in cases.items():
        source = f"import warp as wp\nprint(wp.get_mempool_release_threshold(wp.get_device('{device.alias}')))\n"
        result = run_python_subprocess(source, env_updates={"WARP_MEMPOOL_RELEASE_THRESHOLD": value})
        test.assertEqual(result.returncode, 0, result.stderr)
        test.assertEqual(
            int(result.stdout.strip().splitlines()[-1]), expected, f"WARP_MEMPOOL_RELEASE_THRESHOLD={value}"
        )

    # unset: nothing is applied and the driver default stays
    source = (
        "import os\n"
        "os.environ.pop('WARP_MEMPOOL_RELEASE_THRESHOLD', None)\n"
        "import warp as wp\n"
        "print(wp.config.mempool_release_threshold)\n"
    )
    result = run_python_subprocess(source, hide_gpu=True)
    test.assertEqual(result.returncode, 0, result.stderr)
    test.assertEqual(result.stdout.strip().splitlines()[-1], "None")

    # values the API would clamp or that cannot reach the driver are rejected at import, naming the variable
    for value in ("-1", "nan", "inf", "1e30", "18446744073709551616", "1.5", "abc"):
        result = run_python_subprocess(
            "import warp.config\n", hide_gpu=True, env_updates={"WARP_MEMPOOL_RELEASE_THRESHOLD": value}
        )
        test.assertNotEqual(result.returncode, 0, f"WARP_MEMPOOL_RELEASE_THRESHOLD={value} was accepted")
        test.assertIn(
            "ValueError: WARP_MEMPOOL_RELEASE_THRESHOLD", result.stderr, f"WARP_MEMPOOL_RELEASE_THRESHOLD={value}"
        )


def test_mempool_usage_queries(test, device):
    """Check API to query mempool memory usage."""

    device = wp.get_device(device)
    gc.collect()
    wp.synchronize_device(device)

    pre_alloc_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    pre_alloc_mempool_usage_high = wp.get_mempool_used_mem_high(device)
    test.assertIsInstance(pre_alloc_mempool_usage_curr, int, "before allocation: current usage should be an int")
    test.assertIsInstance(pre_alloc_mempool_usage_high, int, "before allocation: high-water usage should be an int")
    test.assertGreaterEqual(pre_alloc_mempool_usage_curr, 0, "before allocation: current usage should not be negative")
    test.assertGreaterEqual(
        pre_alloc_mempool_usage_high,
        pre_alloc_mempool_usage_curr,
        "before allocation: high-water usage should cover current usage",
    )

    # Allocate a 1 MiB array
    test_data = wp.empty(262144, dtype=wp.float32, device=device)
    wp.synchronize_device(device)

    # Query memory usage again
    post_alloc_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    post_alloc_mempool_usage_high = wp.get_mempool_used_mem_high(device)
    test.assertIsInstance(post_alloc_mempool_usage_curr, int, "after allocation: current usage should be an int")
    test.assertIsInstance(post_alloc_mempool_usage_high, int, "after allocation: high-water usage should be an int")
    test.assertGreaterEqual(post_alloc_mempool_usage_curr, 0, "after allocation: current usage should not be negative")
    test.assertGreaterEqual(
        post_alloc_mempool_usage_high,
        post_alloc_mempool_usage_curr,
        "after allocation: high-water usage should cover current usage",
    )
    test.assertGreaterEqual(
        post_alloc_mempool_usage_curr,
        pre_alloc_mempool_usage_curr,
        "Current usage should not decrease while the test allocation is alive.",
    )
    test.assertGreaterEqual(
        post_alloc_mempool_usage_high,
        pre_alloc_mempool_usage_high,
        "High-water mark should not decrease after allocation.",
    )

    # Free the allocation
    del test_data
    gc.collect()
    wp.synchronize_device(device)

    # Query memory usage
    post_free_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    post_free_mempool_usage_high = wp.get_mempool_used_mem_high(device)
    test.assertIsInstance(post_free_mempool_usage_curr, int, "after free: current usage should be an int")
    test.assertIsInstance(post_free_mempool_usage_high, int, "after free: high-water usage should be an int")
    test.assertGreaterEqual(post_free_mempool_usage_curr, 0, "after free: current usage should not be negative")
    test.assertGreaterEqual(
        post_free_mempool_usage_high,
        post_free_mempool_usage_curr,
        "after free: high-water usage should cover current usage",
    )
    test.assertGreaterEqual(
        post_free_mempool_usage_high,
        post_alloc_mempool_usage_high,
        "High-water mark should not decrease after free.",
    )


def test_mempool_exceptions(test, device):
    device = wp.get_device(device)

    assert not device.is_mempool_supported

    if device.is_cuda:
        expected_error = RuntimeError
    else:
        expected_error = ValueError

    with test.assertRaises(expected_error):
        wp.get_mempool_release_threshold(device)

    with test.assertRaises(expected_error):
        wp.set_mempool_release_threshold(device, 42000)


def test_mempool_access_self(test, device):
    device = wp.get_device(device)

    assert device.is_mempool_supported

    # setting mempool access to self is a no-op
    wp.set_mempool_access_enabled(device, device, True)
    wp.set_mempool_access_enabled(device, device, False)

    # should always be enabled
    enabled = wp.is_mempool_access_enabled(device, device)
    test.assertTrue(enabled)


@unittest.skipUnless(get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support")
def test_mempool_access(test, _):
    target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()

    was_enabled = wp.is_mempool_access_enabled(target_device, peer_device)

    if was_enabled:
        # try disabling
        wp.set_mempool_access_enabled(target_device, peer_device, False)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertFalse(is_enabled)

        # try re-enabling
        wp.set_mempool_access_enabled(target_device, peer_device, True)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertTrue(is_enabled)
    else:
        # try enabling
        wp.set_mempool_access_enabled(target_device, peer_device, True)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertTrue(is_enabled)

        # try re-disabling
        wp.set_mempool_access_enabled(target_device, peer_device, False)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertFalse(is_enabled)


@unittest.skipUnless(
    get_device_pair_without_mempool_access_support(), "Requires devices without mempool access support"
)
def test_mempool_access_exceptions_unsupported(test, _):
    """Handle memory-pool access between unsupported CUDA device pairs."""

    # get a CUDA device pair without mempool access support
    target_device, peer_device = get_device_pair_without_mempool_access_support()

    test.assertFalse(wp.is_mempool_access_enabled(target_device, peer_device))

    # enabling should raise RuntimeError
    with test.assertRaises(RuntimeError):
        wp.set_mempool_access_enabled(target_device, peer_device, True)

    # disabling should not raise an error
    wp.set_mempool_access_enabled(target_device, peer_device, False)


@unittest.skipUnless(wp.is_cpu_available() and wp.is_cuda_available(), "Requires both CUDA and CPU devices")
def test_mempool_access_exceptions_cpu(test, _):
    """Handle memory-pool access queries involving a CPU device."""

    test.assertFalse(wp.is_mempool_access_enabled("cuda:0", "cpu"))
    test.assertFalse(wp.is_mempool_access_enabled("cpu", "cuda:0"))

    # enabling should raise ValueError
    with test.assertRaises(ValueError):
        wp.set_mempool_access_enabled("cpu", "cuda:0", True)
    with test.assertRaises(ValueError):
        wp.set_mempool_access_enabled("cuda:0", "cpu", True)

    # disabling should not raise an error
    wp.set_mempool_access_enabled("cpu", "cuda:0", False)
    wp.set_mempool_access_enabled("cuda:0", "cpu", False)


@unittest.skipUnless(wp.is_cpu_available(), "Requires a CPU device")
def test_mempool_cpu_unsupported(test, _):
    """Verify CPU behavior for unsupported CUDA-style memory pools.

    CPU support and enablement are ``False``, and the public CUDA-only memory-pool
    query and toggle APIs raise ``ValueError``.
    """
    device = wp.get_device("cpu")

    test.assertFalse(wp.is_mempool_supported(device))
    test.assertFalse(device.is_mempool_supported)
    test.assertFalse(wp.is_mempool_enabled(device))

    with test.assertRaises(ValueError):
        wp.set_mempool_enabled(device, True)
    with test.assertRaises(ValueError):
        wp.set_mempool_release_threshold(device, 42000)
    with test.assertRaises(ValueError):
        wp.get_mempool_release_threshold(device)
    with test.assertRaises(ValueError):
        wp.get_mempool_used_mem_current(device)
    with test.assertRaises(ValueError):
        wp.get_mempool_used_mem_high(device)


@unittest.skipUnless(wp.is_cpu_available(), "Requires a CPU device")
def test_graph_capture_allocation_capability(test, _):
    """Verify the graph-capture allocation capability for each device type.

    The capture and allocation paths use this internal capability instead of the
    memory-pool flag. It is always ``True`` for CPU because of host allocation and
    APIC region retention. For CUDA, it mirrors the device's memory-pool support
    and enabled state.
    """
    from warp._src.context import (  # noqa: PLC0415
        _is_graph_capture_allocation_enabled,
        _is_graph_capture_allocation_supported,
    )

    cpu = wp.get_device("cpu")
    test.assertTrue(_is_graph_capture_allocation_supported(cpu))
    test.assertTrue(_is_graph_capture_allocation_enabled(cpu))

    for device in wp.get_cuda_devices():
        test.assertEqual(_is_graph_capture_allocation_supported(device), device.is_mempool_supported)
        test.assertEqual(_is_graph_capture_allocation_enabled(device), device.is_mempool_enabled)


@wp.kernel
def slow_copy_kernel(src: wp.array[float], dst: wp.array[float], iters: int, zero: float):
    # a few threads spin, then each copies a strided share of the array, leaving the rest of the device free
    tid = wp.tid()
    acc = float(0.0)
    for i in range(iters):
        acc = wp.sin(acc + float(i))
    # the start index depends on the loop through a runtime zero, so no load can be issued before it
    j = tid + int(acc * zero)
    stride = 64
    while j < src.shape[0]:
        dst[j] = src[j]
        j += stride


@wp.kernel
def fill_kernel(dst: wp.array[float], value: float):
    dst[wp.tid()] = value


def test_mempool_free_while_in_use(test, device):
    """A block freed while queued work still reads it must not be overwritten through a new allocation."""
    device = wp.get_device(device)
    n = 1 << 16
    reader = wp.Stream(device)
    writer = wp.Stream(device)
    with wp.ScopedMempool(device, True):
        for _ in range(3):
            src = wp.full(n, 1.0, dtype=float, device=device)
            dst = wp.zeros(n, dtype=float, device=device)
            wp.synchronize_device(device)
            wp.launch(slow_copy_kernel, dim=64, inputs=[src, dst, 1000000, 0.0], device=device, stream=reader)
            del src
            gc.collect()
            # many same-size allocations overwritten on another stream while the reader still runs
            fresh = []
            for _ in range(8):
                a = wp.empty(n, dtype=float, device=device)
                wp.launch(fill_kernel, dim=n, inputs=[a, 7.0], device=device, stream=writer)
                fresh.append(a)
            wp.synchronize_device(device)
            assert_np_equal(dst.numpy(), np.ones(n, dtype=np.float32))
            del fresh


class _HipAllocationCache:
    """Turn the opt-in HIP allocation cache on for one test; None outside HIP or while tracking allocations."""

    def __init__(self, device):
        from warp._src.context import CudaHipCaptureAwareAllocator  # noqa: PLC0415

        allocator = device.mempool_enabled_allocator
        self.allocator = allocator if isinstance(allocator, CudaHipCaptureAwareAllocator) else None

    def __enter__(self):
        if self.allocator is None:
            return None
        self.was_enabled = self.allocator._cache_enabled
        self.allocator._cache_enabled = True
        if not self.allocator._caching():
            self.allocator._cache_enabled = self.was_enabled
            return None
        return self.allocator

    def __exit__(self, *exc):
        if self.allocator is not None:
            self.allocator.release_cached()
            self.allocator._cache_enabled = self.was_enabled


def test_mempool_cache_concurrent_frees(test, device):
    """Frees from two threads while a synchronize is in progress never cache one block twice."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_concurrent_frees(test, device, allocator)


def _cache_concurrent_frees(test, device, allocator):
    n = 1024
    limit = allocator.PENDING_MAX_COUNT
    busy = wp.zeros(64, dtype=float, device=device)
    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with wp.ScopedMempool(device, True):
            for _ in range(100):
                gc.collect()
                allocator.release_cached()
                wp.synchronize_device(device)
                a = [wp.empty(n, dtype=float, device=device) for _ in range(limit - 1)]
                b = [wp.empty(n, dtype=float, device=device) for _ in range(300)]
                c = [wp.empty(n, dtype=float, device=device) for _ in range(300)]
                # keep the device busy so the synchronize releases the GIL for a while
                wp.launch(slow_copy_kernel, dim=1, inputs=[busy, busy, 20000, 0.0], device=device)
                del a  # one free short of a synchronize

                def free_b():
                    while b:
                        b.pop()

                t = threading.Thread(target=free_b)
                t.start()
                while c:
                    c.pop()
                t.join()
                wp.synchronize_device(device)
                cached = [p for blocks in allocator._ready.values() for p in blocks]
                cached += [p for p, _ in allocator._pending]
                test.assertEqual(len(cached), len(set(cached)), "a block was cached twice")
    finally:
        sys.setswitchinterval(switch)


def test_mempool_cache_external_capture(test, device):
    """Blocks freed during a capture Warp did not start are not reused without a synchronize."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_external_capture(test, device, allocator)


def _cache_external_capture(test, device, allocator):
    from warp._src.context import CaptureMode, runtime  # noqa: PLC0415

    n = 1024
    stream = wp.Stream(device)
    allocator.PENDING_MAX_COUNT = 16
    try:
        with wp.ScopedMempool(device, True):
            allocator.release_cached()
            arrays = [wp.empty(n, dtype=float, device=device) for _ in range(64)]
            freed = {x.ptr for x in arrays}
            with wp.ScopedStream(stream):
                # a capture started below Warp's capture registry, as another library would start one
                test.assertTrue(
                    runtime.core.wp_cuda_graph_begin_capture(
                        device.context, stream.cuda_stream, 0, int(CaptureMode.THREAD_LOCAL)
                    )
                )
                del arrays
                graph = ctypes.c_void_p()
                runtime.core.wp_cuda_graph_end_capture(device.context, stream.cuda_stream, ctypes.byref(graph))
                if graph:
                    runtime.core.wp_cuda_graph_destroy(device.context, graph)
            fresh = [wp.empty(n, dtype=float, device=device) for _ in range(64)]
            test.assertFalse(freed & {x.ptr for x in fresh}, "a block freed during the capture was reused")
            del fresh
    finally:
        del allocator.PENDING_MAX_COUNT


def test_mempool_cache_capture_other_stream(test, device):
    """Frees during a capture on another stream do not invalidate that capture."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_capture_other_stream(test, device, allocator)


def _cache_capture_other_stream(test, device, allocator):
    from warp._src.context import CaptureMode, runtime  # noqa: PLC0415

    n = 1024
    stream = wp.Stream(device)
    allocator.PENDING_MAX_COUNT = 16
    try:
        with wp.ScopedMempool(device, True):
            allocator.release_cached()
            arrays = [wp.empty(n, dtype=float, device=device) for _ in range(64)]
            test.assertTrue(
                runtime.core.wp_cuda_graph_begin_capture(
                    device.context, stream.cuda_stream, 0, int(CaptureMode.RELAXED)
                )
            )
            del arrays  # freed on the device's current stream, which is not capturing
            graph = ctypes.c_void_p()
            ended = runtime.core.wp_cuda_graph_end_capture(device.context, stream.cuda_stream, ctypes.byref(graph))
            if graph:
                runtime.core.wp_cuda_graph_destroy(device.context, graph)
            test.assertTrue(ended and graph, "the capture was invalidated")
    finally:
        del allocator.PENDING_MAX_COUNT


def test_mempool_cache_unregistered_capture(test, device):
    """Blocks freed during a capture started directly through HIP are not reused without a synchronize."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_unregistered_capture(test, device, allocator)


def _cache_unregistered_capture(test, device, allocator):
    hip = ctypes.CDLL("libamdhip64.so.7")
    n = 1024
    stream = wp.Stream(device)
    allocator.PENDING_MAX_COUNT = 16
    try:
        with wp.ScopedMempool(device, True):
            allocator.release_cached()
            arrays = [wp.empty(n, dtype=float, device=device) for _ in range(64)]
            freed = {x.ptr for x in arrays}
            with wp.ScopedStream(stream):
                # as another library would: neither Warp nor its native layer knows of this capture
                test.assertEqual(hip.hipStreamBeginCapture(ctypes.c_void_p(stream.cuda_stream), 2), 0)
                del arrays
                graph = ctypes.c_void_p()
                ended = hip.hipStreamEndCapture(ctypes.c_void_p(stream.cuda_stream), ctypes.byref(graph))
                if graph:
                    hip.hipGraphDestroy(graph)
                test.assertEqual(ended, 0, "the capture was invalidated")
            fresh = [wp.empty(n, dtype=float, device=device) for _ in range(64)]
            test.assertFalse(freed & {x.ptr for x in fresh}, "a block freed during the capture was reused")
            del fresh
    finally:
        del allocator.PENDING_MAX_COUNT


def test_mempool_cache_free_during_synchronize(test, device):
    """A block freed while a synchronize is in progress waits for the next one."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_free_during_synchronize(test, device, allocator)


def _cache_free_during_synchronize(test, device, allocator):
    n = 1024
    with wp.ScopedMempool(device, True):
        allocator.release_cached()
        before = [wp.empty(n, dtype=float, device=device) for _ in range(4)]
        during = [wp.empty(n, dtype=float, device=device)]
        late = during[0].ptr
        synchronize = allocator._synchronize

        def synchronize_and_free():
            done = synchronize()
            during.clear()  # freed after the synchronize was issued
            return done

        allocator._synchronize = synchronize_and_free
        try:
            del before
            allocator._retire_pending()
        finally:
            del allocator._synchronize
        ready = [p for blocks in allocator._ready.values() for p in blocks]
        test.assertNotIn(late, ready, "a block freed during the synchronize became reusable")
        test.assertIn(late, [p for p, _ in allocator._pending])


def test_mempool_cache_reuse_waits_for_reader(test, device):
    """A reused block is not overwritten while work on another stream still reads it."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_reuse_waits_for_reader(test, device, allocator)


def _cache_reuse_waits_for_reader(test, device, allocator):
    n = 1 << 16
    reader = wp.Stream(device)
    writer = wp.Stream(device)
    allocator.PENDING_MAX_COUNT = 8
    try:
        with wp.ScopedMempool(device, True):
            for _ in range(3):
                allocator.release_cached()
                src = wp.full(n, 1.0, dtype=float, device=device)
                dst = wp.zeros(n, dtype=float, device=device)
                filler = [wp.empty(n, dtype=float, device=device) for _ in range(8)]
                wp.synchronize_device(device)
                wp.launch(slow_copy_kernel, dim=64, inputs=[src, dst, 1000000, 0.0], device=device, stream=reader)
                del src
                filler.clear()  # the ninth free crosses the threshold while the reader runs
                fresh = []
                for _ in range(9):
                    a = wp.empty(n, dtype=float, device=device)
                    wp.launch(fill_kernel, dim=n, inputs=[a, 7.0], device=device, stream=writer)
                    fresh.append(a)
                wp.synchronize_device(device)
                assert_np_equal(dst.numpy(), np.ones(n, dtype=np.float32))
                del fresh
    finally:
        del allocator.PENDING_MAX_COUNT


def test_mempool_cache_failed_synchronize(test, device):
    """Blocks are not reused when the synchronize that would release them fails."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_failed_synchronize(test, device, allocator)


def _cache_failed_synchronize(test, device, allocator):
    n = 1024
    allocator.PENDING_MAX_COUNT = 8
    allocator._synchronize = lambda: False
    try:
        with wp.ScopedMempool(device, True):
            allocator.release_cached()
            arrays = [wp.empty(n, dtype=float, device=device) for _ in range(16)]
            freed = {x.ptr for x in arrays}
            del arrays
            fresh = [wp.empty(n, dtype=float, device=device) for _ in range(16)]
            test.assertFalse(freed & {x.ptr for x in fresh}, "a block was reused after a failed synchronize")
            del fresh
    finally:
        del allocator._synchronize
        del allocator.PENDING_MAX_COUNT


def test_mempool_cache_unregistered_capture_other_stream(test, device):
    """Blocks freed while an unknown capture on another stream blocks the synchronize are not reused."""
    device = wp.get_device(device)
    with _HipAllocationCache(device) as allocator:
        if allocator is None:
            test.skipTest("no HIP allocation cache on this device")
        _cache_unregistered_capture_other_stream(test, device, allocator)


def _cache_unregistered_capture_other_stream(test, device, allocator):
    hip = ctypes.CDLL("libamdhip64.so.7")
    n = 1024
    stream = wp.Stream(device)
    allocator.PENDING_MAX_COUNT = 16
    try:
        with wp.ScopedMempool(device, True):
            allocator.release_cached()
            arrays = [wp.empty(n, dtype=float, device=device) for _ in range(64)]
            freed = {x.ptr for x in arrays}
            # relaxed capture unknown to Warp on another stream: the device synchronize fails while it runs
            test.assertEqual(hip.hipStreamBeginCapture(ctypes.c_void_p(stream.cuda_stream), 2), 0)
            del arrays
            fresh = [wp.empty(n, dtype=float, device=device) for _ in range(64)]
            graph = ctypes.c_void_p()
            hip.hipStreamEndCapture(ctypes.c_void_p(stream.cuda_stream), ctypes.byref(graph))
            if graph:
                hip.hipGraphDestroy(graph)
            test.assertFalse(freed & {x.ptr for x in fresh}, "a block was reused without a successful synchronize")
            del fresh
    finally:
        del allocator.PENDING_MAX_COUNT


class TestMempool(unittest.TestCase):
    pass


# CUDA-only mempool semantics (threshold/usage/self-access). CPU has no memory pool, so it is
# excluded here and covered by test_mempool_exceptions / test_mempool_cpu_unsupported instead.
cuda_devices_with_mempools = get_cuda_test_devices_with_mempool()
devices_without_mempools = [d for d in get_test_devices() if not d.is_mempool_supported]

# test devices with mempool support
add_function_test(
    TestMempool, "test_mempool_release_threshold", test_mempool_release_threshold, devices=cuda_devices_with_mempools
)
add_function_test(
    TestMempool,
    "test_mempool_release_threshold_config",
    test_mempool_release_threshold_config,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool, "test_mempool_usage_queries", test_mempool_usage_queries, devices=cuda_devices_with_mempools
)
add_function_test(TestMempool, "test_mempool_access_self", test_mempool_access_self, devices=cuda_devices_with_mempools)
add_function_test(
    TestMempool, "test_mempool_free_while_in_use", test_mempool_free_while_in_use, devices=cuda_devices_with_mempools
)
add_function_test(
    TestMempool,
    "test_mempool_cache_reuse_waits_for_reader",
    test_mempool_cache_reuse_waits_for_reader,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool,
    "test_mempool_cache_failed_synchronize",
    test_mempool_cache_failed_synchronize,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool,
    "test_mempool_cache_unregistered_capture_other_stream",
    test_mempool_cache_unregistered_capture_other_stream,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool,
    "test_mempool_cache_concurrent_frees",
    test_mempool_cache_concurrent_frees,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool,
    "test_mempool_cache_external_capture",
    test_mempool_cache_external_capture,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool,
    "test_mempool_cache_capture_other_stream",
    test_mempool_cache_capture_other_stream,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool,
    "test_mempool_cache_unregistered_capture",
    test_mempool_cache_unregistered_capture,
    devices=cuda_devices_with_mempools,
)
add_function_test(
    TestMempool,
    "test_mempool_cache_free_during_synchronize",
    test_mempool_cache_free_during_synchronize,
    devices=cuda_devices_with_mempools,
)

# CPU has no CUDA-style mempool; it uses a separate graph-capture allocation capability.
add_function_test(TestMempool, "test_mempool_cpu_unsupported", test_mempool_cpu_unsupported)
add_function_test(TestMempool, "test_graph_capture_allocation_capability", test_graph_capture_allocation_capability)

# test devices without mempool support
add_function_test(TestMempool, "test_mempool_exceptions", test_mempool_exceptions, devices=devices_without_mempools)

# mempool access tests
add_function_test(TestMempool, "test_mempool_access", test_mempool_access)

# mempool access exceptions
add_function_test(TestMempool, "test_mempool_access_exceptions_unsupported", test_mempool_access_exceptions_unsupported)
add_function_test(TestMempool, "test_mempool_access_exceptions_cpu", test_mempool_access_exceptions_cpu)


if __name__ == "__main__":
    unittest.main(verbosity=2)
