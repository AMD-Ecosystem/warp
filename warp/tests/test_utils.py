# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import io
import unittest

from warp.tests.unittest_utils import *


def test_array_scan(test, device):
    rng = np.random.default_rng(123)

    for dtype in (int, float):
        if dtype == int:
            values = rng.integers(-1e6, high=1e6, size=100000, dtype=dtype)
        else:
            values = rng.uniform(low=-1e6, high=1e6, size=100000)

        expected = np.cumsum(values)

        values = wp.array(values, dtype=dtype, device=device)
        result_inc = wp.zeros_like(values)
        result_exc = wp.zeros_like(values)

        wp.utils.array_scan(values, result_inc, True)
        wp.utils.array_scan(values, result_exc, False)

        tolerance = 0 if dtype == int else 1e-3

        result_inc = result_inc.numpy().squeeze()
        result_exc = result_exc.numpy().squeeze()
        error_inc = np.max(np.abs(result_inc - expected)) / abs(expected[-1])
        error_exc = max(np.max(np.abs(result_exc[1:] - expected[:-1])), abs(result_exc[0])) / abs(expected[-2])

        test.assertTrue(error_inc <= tolerance)
        test.assertTrue(error_exc <= tolerance)


def test_array_scan_empty(test, device):
    values = wp.array((), dtype=int, device=device)
    result = wp.array((), dtype=int, device=device)
    wp.utils.array_scan(values, result)


def test_array_scan_error_sizes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device=device)
    result = wp.zeros(234, dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"In and out array storage sizes do not match \(123 vs 234\)$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_dtypes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device=device)
    result = wp.zeros(123, dtype=float, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"In and out array data types do not match \(int32 vs float32\)$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_unsupported_dtype(test, device):
    values = wp.zeros(123, dtype=wp.vec3, device=device)
    result = wp.zeros(123, dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type: vec3f$",
    ):
        wp.utils.array_scan(values, result, True)


def test_radix_sort_pairs(test, device):
    keyTypes = [int, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 8)
        assert_np_equal(keys.numpy()[:8], np.array((1, 2, 3, 4, 5, 6, 7, 8)))
        assert_np_equal(values.numpy()[:8], np.array((5, 2, 8, 4, 7, 6, 1, 3)))


def test_segmented_sort_pairs(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.segmented_sort_pairs(
            keys,
            values,
            8,
            wp.array((0, 4), dtype=int, device=device),
            wp.array((4, 8), dtype=int, device=device),
        )
        assert_np_equal(keys.numpy()[:8], np.array((2, 4, 7, 8, 1, 3, 5, 6)))
        assert_np_equal(values.numpy()[:8], np.array((2, 4, 1, 3, 5, 8, 7, 6)))


def test_radix_sort_pairs_empty(test, device):
    keyTypes = [int, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((), dtype=keyType, device=device)
        values = wp.array((), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 0)


def test_segmented_sort_pairs_empty(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((), dtype=keyType, device=device)
        values = wp.array((), dtype=int, device=device)
        wp.utils.segmented_sort_pairs(
            keys, values, 0, wp.array((), dtype=int, device=device), wp.array((), dtype=int, device=device)
        )


def test_radix_sort_pairs_error_insufficient_storage(test, device):
    keyTypes = [int, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((1, 2, 3), dtype=keyType, device=device)
        values = wp.array((1, 2, 3), dtype=int, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Keys and values array storage must be large enough to contain 2\*count elements$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 3)


def test_segmented_sort_pairs_error_insufficient_storage(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((1, 2, 3), dtype=keyType, device=device)
        values = wp.array((1, 2, 3), dtype=int, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Array storage must be large enough to contain 2\*count elements$",
        ):
            wp.utils.segmented_sort_pairs(
                keys,
                values,
                3,
                wp.array((0,), dtype=int, device=device),
                wp.array((3,), dtype=int, device=device),
            )


def test_radix_sort_pairs_error_unsupported_dtype(test, device):
    keyTypes = [wp.int32, wp.float32, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((1.0, 2.0, 3.0), dtype=keyType, device=device)
        values = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            rf"Unsupported keys and values data types: {keyType.__name__}, float32$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 1)


def test_segmented_sort_pairs_error_unsupported_dtype(test, device):
    keyTypes = [wp.int32, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((1.0, 2.0, 3.0), dtype=keyType, device=device)
        values = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            rf"Unsupported data type: {keyType.__name__}$",
        ):
            wp.utils.segmented_sort_pairs(
                keys,
                values,
                1,
                wp.array((0,), dtype=int, device=device),
                wp.array((3,), dtype=int, device=device),
            )


def test_array_sum(test, device):
    for dtype in (wp.float32, wp.float64):
        with test.subTest(dtype=dtype):
            values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
            test.assertEqual(wp.utils.array_sum(values), 6.0)

            values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
            result = wp.empty(shape=(1,), dtype=dtype, device=device)
            wp.utils.array_sum(values, out=result)
            test.assertEqual(result.numpy()[0], 6.0)


def test_array_sum_error_out_dtype_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(1,), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_out_shape_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(2,), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have shape \(1,\)$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_unsupported_dtype(test, device):
    values = wp.array((1, 2, 3), dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type: int32$",
    ):
        wp.utils.array_sum(values)


def test_array_inner(test, device):
    for dtype in (wp.float32, wp.float64, wp.vec3):
        a = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        b = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        test.assertEqual(wp.utils.array_inner(a, b), 14.0)

        a = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        b = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        result = wp.empty(shape=(1,), dtype=wp._src.types.type_scalar_type(dtype), device=device)
        wp.utils.array_inner(a, b, out=result)
        test.assertEqual(result.numpy()[0], 14.0)

    # test with different instances of same type
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.vec3, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.types.vector(3, float), device=device)
    test.assertEqual(wp.utils.array_inner(a, b), 14.0)


def test_array_inner_error_sizes_mismatch(test, device):
    a = wp.array((1.0, 2.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"A and b array storage sizes do not match \(2 vs 3\)$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_dtypes_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"A and b array data types do not match \(float32 vs float64\)$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_out_dtype_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(1,), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_out_shape_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(2,), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have shape \(1,\)$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_unsupported_dtype(test, device):
    a = wp.array((1, 2, 3), dtype=int, device=device)
    b = wp.array((1, 2, 3), dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type: int32$",
    ):
        wp.utils.array_inner(a, b)


def test_array_cast(test, device):
    values = wp.array((1, 2, 3), dtype=int, device=device)
    result = wp.empty(3, dtype=float, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (3,))
    assert_np_equal(result.numpy(), np.array((1.0, 2.0, 3.0), dtype=float))

    values = wp.array((1, 2, 3, 4), dtype=int, device=device)
    result = wp.empty((2, 2), dtype=float, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (2, 2))
    assert_np_equal(result.numpy(), np.array(((1.0, 2.0), (3.0, 4.0)), dtype=float))

    values = wp.array(((1, 2), (3, 4)), dtype=wp.vec2, device=device)
    result = wp.zeros(2, dtype=float, device=device)
    wp.utils.array_cast(values, result, count=1)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (2,))
    assert_np_equal(result.numpy(), np.array((1.0, 2.0), dtype=float))

    values = wp.array(((1, 2), (3, 4)), dtype=int, device=device)
    result = wp.zeros((2, 2), dtype=int, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.int32)
    test.assertEqual(result.shape, (2, 2))
    assert_np_equal(result.numpy(), np.array(((1, 2), (3, 4)), dtype=int))


def test_array_cast_error_unsupported_partial_cast(test, device):
    values = wp.array(((1, 2), (3, 4)), dtype=int, device=device)
    result = wp.zeros((2, 2), dtype=float, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Partial cast is not supported for arrays with more than one dimension$",
    ):
        wp.utils.array_cast(values, result, count=1)


@wp.func
def _map_split(x: float):
    return x * 2.0, x + 1.0


@wp.struct
class _MapPair:
    a: float
    b: float


@wp.func
def _map_make_pair(x: float):
    p = _MapPair()
    p.a = x
    p.b = -x
    return p


@wp.func
def _map_returns_nothing(x: float):
    y = x + 1.0  # noqa: F841


def _map_double(x):
    return x * 2.0


@wp.kernel
def _timing_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


devices = get_test_devices()


class TestUtils(unittest.TestCase):
    def test_warn(self):
        # Multiple warnings get printed out each time.
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp._src.utils.warn("hello, world!")
            wp._src.utils.warn("hello, world!")

        expected = "Warp UserWarning: hello, world!\nWarp UserWarning: hello, world!\n"

        self.assertEqual(f.getvalue(), expected)

        # Test verbose warnings
        saved_verbosity = wp.config.verbose_warnings
        try:
            wp.config.verbose_warnings = True
            with contextlib.redirect_stdout(io.StringIO()) as f:
                frame_info = inspect.getframeinfo(inspect.currentframe())
                wp._src.utils.warn("hello, world!")
                wp._src.utils.warn("hello, world!")

            expected = (
                f"Warp UserWarning: hello, world! ({frame_info.filename}:{frame_info.lineno + 1})\n"
                '  wp._src.utils.warn("hello, world!")\n'
                f"Warp UserWarning: hello, world! ({frame_info.filename}:{frame_info.lineno + 2})\n"
                '  wp._src.utils.warn("hello, world!")\n'
            )

            self.assertEqual(f.getvalue(), expected)

        finally:
            # make sure to restore warning verbosity
            wp.config.verbose_warnings = saved_verbosity

        # Multiple similar deprecation warnings get printed out only once.
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp._src.utils.warn("hello, world!", category=DeprecationWarning)
            wp._src.utils.warn("hello, world!", category=DeprecationWarning)

        expected = "Warp DeprecationWarning: hello, world!\n"

        self.assertEqual(f.getvalue(), expected)

        # Multiple different deprecation warnings get printed out each time.
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp._src.utils.warn("foo", category=DeprecationWarning)
            wp._src.utils.warn("bar", category=DeprecationWarning)

        expected = "Warp DeprecationWarning: foo\nWarp DeprecationWarning: bar\n"

        self.assertEqual(f.getvalue(), expected)

    def test_transform_expand(self):
        t = (1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0)
        self.assertEqual(
            wp.transform_expand(t),
            wp.transformf(p=(1.0, 2.0, 3.0), q=(4.0, 3.0, 2.0, 1.0)),
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_scan_error_devices_mismatch(self):
        values = wp.zeros(123, dtype=int, device="cpu")
        result = wp.zeros_like(values, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"In and out array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.array_scan(values, result, True)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_radix_sort_pairs_error_devices_mismatch(self):
        keys = wp.array((1, 2, 3), dtype=int, device="cpu")
        values = wp.array((1, 2, 3), dtype=int, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Keys and values array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 1)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_inner_error_out_device_mismatch(self):
        a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"out storage device should match values array$",
        ):
            wp.utils.array_inner(a, b, result)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_sum_error_out_device_mismatch(self):
        values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"out storage device should match values array$",
        ):
            wp.utils.array_sum(values, out=result)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_inner_error_devices_mismatch(self):
        a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"A and b array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.array_inner(a, b)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_cast_error_devices_mismatch(self):
        values = wp.array((1, 2, 3), dtype=int, device="cpu")
        result = wp.empty(3, dtype=float, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.array_cast(values, result)

    def test_mesh_adjacency(self):
        triangles = (
            (0, 3, 1),
            (0, 2, 3),
        )
        adj = wp._src.utils.MeshAdjacency(triangles, len(triangles))
        expected_edges = {
            (0, 3): (0, 3, 1, 2, 0, 1),
            (1, 3): (3, 1, 0, -1, 0, -1),
            (0, 1): (1, 0, 3, -1, 0, -1),
            (0, 2): (0, 2, 3, -1, 1, -1),
            (2, 3): (2, 3, 0, -1, 1, -1),
        }
        edges = {k: (e.v0, e.v1, e.o0, e.o1, e.f0, e.f1) for k, e in adj.edges.items()}
        self.assertDictEqual(edges, expected_edges)

    def test_mesh_adjacency_error_manifold(self):
        triangles = (
            (0, 3, 1),
            (0, 2, 3),
            (3, 0, 1),
        )

        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp._src.utils.MeshAdjacency(triangles, len(triangles))

        self.assertEqual(f.getvalue(), "Detected non-manifold edge\n")

    def test_scoped_timer(self):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            with wp.ScopedTimer("hello"):
                pass

        self.assertRegex(f.getvalue(), r"^hello took \d+\.\d+ ms$")

        with contextlib.redirect_stdout(io.StringIO()) as f:
            with wp.ScopedTimer("hello", detailed=True):
                pass

        self.assertRegex(f.getvalue(), r"^         4 function calls in \d+\.\d+ seconds")
        self.assertRegex(f.getvalue(), r"hello took \d+\.\d+ ms$")

    def test_broadcast_shapes(self):
        from warp._src.utils import broadcast_shapes

        self.assertEqual(broadcast_shapes([(3, 1, 4), (5, 4)]), (3, 5, 4))
        self.assertEqual(broadcast_shapes([(5,), (1,)]), (5,))
        self.assertEqual(broadcast_shapes([(2, 3)]), (2, 3))
        # second shape has more dimensions than the reference (else branch)
        self.assertEqual(broadcast_shapes([(4,), (5, 4)]), (5, 4))
        with self.assertRaisesRegex(ValueError, "not broadcastable"):
            broadcast_shapes([(3,), (4,)])

    def test_map_basic(self):
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        b = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device="cpu")

        # lambda
        r = wp.map(lambda x, y: x + 2.0 * y, a, b)
        assert_np_equal(r.numpy(), np.array([9.0, 12.0, 15.0], dtype=np.float32))

        # named Python function
        def sub(x, y):
            return x - y

        r2 = wp.map(sub, a, b)
        assert_np_equal(r2.numpy(), np.array([-3.0, -3.0, -3.0], dtype=np.float32))

        # existing Warp function over a vector array
        v = wp.array([wp.vec3(3.0, 0.0, 4.0), wp.vec3(1.0, 2.0, 2.0)], dtype=wp.vec3, device="cpu")
        r3 = wp.map(wp.length, v)
        assert_np_equal(r3.numpy(), np.array([5.0, 3.0], dtype=np.float32))

        # quaternion and matrix arrays exercise the remaining type_to_code branches
        q = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0), wp.quat(1.0, 0.0, 0.0, 0.0)], dtype=wp.quat, device="cpu")
        rq = wp.map(wp.normalize, q)
        self.assertEqual(rq.dtype, wp.quat)
        mats = wp.array([wp.mat22(1.0, 2.0, 3.0, 4.0)], dtype=wp.mat22, device="cpu")
        rm = wp.map(wp.transpose, mats)
        self.assertEqual(rm.dtype, wp.mat22)

        # non-array (transform) input broadcast against a vec3 array
        tf = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
        pts = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device="cpu")
        r4 = wp.map(wp.transform_point, tf, pts)
        assert_np_equal(r4.numpy(), np.array([[2.0, 2.0, 3.0], [1.0, 3.0, 3.0]], dtype=np.float32))

        # int and float scalar (non-array) inputs
        xs = wp.array([-1.0, 0.0, 1.0], dtype=wp.float32, device="cpu")
        wp.map(wp.clamp, xs, -0.5, 0.5, out=xs)
        assert_np_equal(xs.numpy(), np.array([-0.5, 0.0, 0.5], dtype=np.float32))

    def test_map_multiple_outputs_and_struct(self):
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")

        o1, o2 = wp.map(_map_split, a)
        assert_np_equal(o1.numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float32))
        assert_np_equal(o2.numpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32))

        # explicit multi-output arrays
        out0 = wp.empty(3, dtype=wp.float32, device="cpu")
        out1 = wp.empty(3, dtype=wp.float32, device="cpu")
        wp.map(_map_split, a, out=[out0, out1])
        assert_np_equal(out0.numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float32))

        # struct-returning function
        pairs = wp.map(_map_make_pair, a)
        self.assertEqual(pairs.shape, (3,))

    def test_map_broadcasting_and_return_kernel(self):
        m = wp.array(np.ones((3, 1), dtype=np.float32), dtype=wp.float32, device="cpu")
        n = wp.array(np.ones((1, 4), dtype=np.float32), dtype=wp.float32, device="cpu")
        rb = wp.map(lambda x, y: x + y, m, n)
        self.assertEqual(rb.shape, (3, 4))

        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        kernel = wp.map(lambda x: x * 2.0, a, return_kernel=True)
        self.assertIsInstance(kernel, wp.Kernel)

    def test_map_errors(self):
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")

        # not callable
        with self.assertRaisesRegex(TypeError, "must be a callable"):
            wp.map(42, a)

        # wrong number of arguments
        with self.assertRaisesRegex(TypeError, "does not match expected number"):
            wp.map(lambda x, y: x + y, a)

        # function that returns nothing
        with self.assertRaisesRegex(TypeError, "must return a value"):
            wp.map(_map_returns_nothing, a)

        # no array inputs at all
        with self.assertRaisesRegex(ValueError, "at least one warp.array"):
            wp.map(wp.clamp, 1.0, -0.5, 0.5)

        # output dtype mismatch
        with self.assertRaisesRegex(TypeError, "does not match expected dtype"):
            wp.map(_map_double, a, out=wp.empty(3, dtype=wp.int32, device="cpu"))

        # output shape mismatch
        with self.assertRaisesRegex(TypeError, "does not match expected shape"):
            wp.map(_map_double, a, out=wp.empty(5, dtype=wp.float32, device="cpu"))

        # wrong number of provided outputs for a multi-output function
        with self.assertRaisesRegex(TypeError, "does not match expected number of function outputs"):
            wp.map(_map_split, a, out=[wp.empty(3, dtype=wp.float32, device="cpu")])

        # per-element dtype mismatch in a multi-output list
        with self.assertRaisesRegex(TypeError, "does not match expected dtype"):
            wp.map(
                _map_split,
                a,
                out=[wp.empty(3, dtype=wp.int32, device="cpu"), wp.empty(3, dtype=wp.float32, device="cpu")],
            )

        # per-element shape mismatch in a multi-output list
        with self.assertRaisesRegex(TypeError, "does not match expected shape"):
            wp.map(
                _map_split,
                a,
                out=[wp.empty(5, dtype=wp.float32, device="cpu"), wp.empty(3, dtype=wp.float32, device="cpu")],
            )

        # multi-output function given a single (non-list) output
        with self.assertRaisesRegex(TypeError, "Invalid output provided"):
            wp.map(_map_split, a, out=wp.empty(3, dtype=wp.float32, device="cpu"))

    def test_timing_print(self):
        # empty results
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp.timing_print([])
        self.assertIn("No activity", f.getvalue())

        # constructed results exercise the aggregation/formatting branches
        from warp._src.utils import TimingResult

        device = wp.get_device("cpu")
        results = [
            TimingResult(device, "forward kernel foo", wp.TIMING_KERNEL, 1.5),
            TimingResult(device, "forward kernel foo", wp.TIMING_KERNEL, 2.5),
            TimingResult(device, "memcpy", wp.TIMING_MEMCPY, 0.25),
        ]
        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp.timing_print(results, indent="  ")
        out = f.getvalue()
        self.assertIn("CUDA timeline", out)
        self.assertIn("CUDA activity summary", out)
        self.assertIn("CUDA device summary", out)

    def test_scoped_memory_tracker(self):
        # inactive tracker is a no-op
        with contextlib.redirect_stdout(io.StringIO()) as f:
            with wp.ScopedMemoryTracker("inactive", active=False):
                pass
        self.assertEqual(f.getvalue(), "")

        # report_func routing + invalid sort order
        captured = []
        with wp.ScopedMemoryTracker("scope", active=True, print=False, report_func=captured.append) as tracker:
            _ = wp.zeros(128, dtype=wp.float32, device="cpu")
            tracker.report()
            with self.assertRaisesRegex(ValueError, "Invalid sort order"):
                tracker.report(sort="bogus")
            tracker.clear()

        # active tracker that prints on exit and reports to stdout (no report_func)
        with contextlib.redirect_stdout(io.StringIO()):
            with wp.ScopedMemoryTracker("printing", active=True, print=True) as tracker2:
                _ = wp.zeros(256, dtype=wp.float32, device="cpu")
                tracker2.report(sort="chronological")

    def test_sort_pairs_edge_cases(self):
        from warp._src.utils import radix_sort_pairs, segmented_sort_pairs

        # radix: count == 0 early return
        keys = wp.zeros(4, dtype=wp.int32, device="cpu")
        values = wp.zeros(4, dtype=wp.int32, device="cpu")
        radix_sort_pairs(keys, values, 0)

        # segmented: count == 0 early return
        seg_start = wp.array([0, 2], dtype=wp.int32, device="cpu")
        segmented_sort_pairs(keys, values, 0, seg_start)

        # segmented: inferred end indices (segment_end_indices=None) path
        skeys = wp.array([3, 1, 2, 0], dtype=wp.int32, device="cpu")
        svalues = wp.array([0, 1, 2, 3], dtype=wp.int32, device="cpu")
        segmented_sort_pairs(skeys, svalues, 2, wp.array([0, 2, 4], dtype=wp.int32, device="cpu"))

        # segmented: start indices must be int32
        with self.assertRaisesRegex(RuntimeError, "segment_start_indices array must be of type int32"):
            segmented_sort_pairs(keys, values, 2, wp.array([0.0, 2.0], dtype=wp.float32, device="cpu"))

        # segmented: end indices must be int32 when provided
        with self.assertRaisesRegex(RuntimeError, "segment_end_indices array must be of type int32"):
            segmented_sort_pairs(
                keys,
                values,
                2,
                wp.array([0, 2], dtype=wp.int32, device="cpu"),
                segment_end_indices=wp.array([2.0, 4.0], dtype=wp.float32, device="cpu"),
            )


def test_timing_begin_end(test, device):
    if not device.is_cuda:
        return
    a = wp.zeros(16, dtype=wp.float32, device=device)
    wp.timing_begin(synchronize=True)
    wp.launch(_timing_kernel, dim=16, inputs=[a], device=device)
    results = wp.timing_end(synchronize=True)
    test.assertIsInstance(results, list)
    with contextlib.redirect_stdout(io.StringIO()):
        wp.timing_print(results)


add_function_test(TestUtils, "test_timing_begin_end", test_timing_begin_end, devices=devices)
add_function_test(TestUtils, "test_array_scan", test_array_scan, devices=devices)
add_function_test(TestUtils, "test_array_scan_empty", test_array_scan_empty, devices=devices)
add_function_test(
    TestUtils, "test_array_scan_error_sizes_mismatch", test_array_scan_error_sizes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_scan_error_dtypes_mismatch", test_array_scan_error_dtypes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_scan_error_unsupported_dtype", test_array_scan_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_radix_sort_pairs", test_radix_sort_pairs, devices=devices)
add_function_test(TestUtils, "test_radix_sort_pairs_empty", test_radix_sort_pairs, devices=devices)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_insufficient_storage",
    test_radix_sort_pairs_error_insufficient_storage,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_unsupported_dtype",
    test_radix_sort_pairs_error_unsupported_dtype,
    devices=devices,
)
add_function_test(TestUtils, "test_segmented_sort_pairs", test_segmented_sort_pairs, devices=devices)
add_function_test(TestUtils, "test_segmented_sort_pairs_empty", test_segmented_sort_pairs, devices=devices)
add_function_test(
    TestUtils,
    "test_segmented_sort_pairs_error_insufficient_storage",
    test_segmented_sort_pairs_error_insufficient_storage,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_segmented_sort_pairs_error_unsupported_dtype",
    test_segmented_sort_pairs_error_unsupported_dtype,
    devices=devices,
)
add_function_test(TestUtils, "test_array_sum", test_array_sum, devices=devices)
add_function_test(
    TestUtils, "test_array_sum_error_out_dtype_mismatch", test_array_sum_error_out_dtype_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_sum_error_out_shape_mismatch", test_array_sum_error_out_shape_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_sum_error_unsupported_dtype", test_array_sum_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_array_inner", test_array_inner, devices=devices)
add_function_test(
    TestUtils, "test_array_inner_error_sizes_mismatch", test_array_inner_error_sizes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_dtypes_mismatch", test_array_inner_error_dtypes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_out_dtype_mismatch", test_array_inner_error_out_dtype_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_out_shape_mismatch", test_array_inner_error_out_shape_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_unsupported_dtype", test_array_inner_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_array_cast", test_array_cast, devices=devices)
add_function_test(
    TestUtils,
    "test_array_cast_error_unsupported_partial_cast",
    test_array_cast_error_unsupported_partial_cast,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
