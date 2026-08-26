# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so jacobian_plot never blocks

import warp as wp
import warp._src.autograd as _autograd
from warp.autograd import (
    gradcheck,
    gradcheck_tape,
    jacobian,
    jacobian_fd,
    jacobian_plot,
)
from warp.tests.unittest_utils import *


@wp.kernel
def kernel_3d(
    a: wp.array3d(dtype=Any),
    b: wp.array3d(dtype=Any),
    c: wp.array3d(dtype=Any),
    out1: wp.array3d(dtype=Any),
    out2: wp.array3d(dtype=Any),
):
    i, j, k = wp.tid()
    out1[i, j, k] = a[i, j, k] * b[i, j, k] + c[i, j, k]
    out2[i, j, k] = -a[i, j, k] * b[i, j, k] - c[i, j, k]


wp.overload(
    kernel_3d,
    [
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
        wp.array3d(dtype=wp.float32),
    ],
)

wp.overload(
    kernel_3d,
    [
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
        wp.array3d(dtype=wp.float64),
    ],
)


@wp.kernel
def kernel_mixed(
    a: wp.array(dtype=float),
    b: wp.array(dtype=wp.vec3),
    out1: wp.array(dtype=wp.vec2),
    out2: wp.array(dtype=wp.quat),
):
    tid = wp.tid()
    ai, bi = a[tid], b[tid]
    out1[tid] = wp.vec2(ai * wp.length(bi), -ai * wp.dot(bi, wp.vec3(0.1, 1.0, -0.1)))
    out2[tid] = wp.normalize(wp.quat(ai, bi[0], bi[1], bi[2]))


@wp.kernel
def vec_length_kernel(a: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    v = a[tid]
    # instead of wp.length(v), we use a trivial implementation that
    # fails when a division by zero occurs in the backward pass of sqrt
    out[tid] = wp.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)


@wp.func
def wrong_grad_func(x: float):
    return x * x


@wp.func_grad(wrong_grad_func)
def adj_wrong_grad_func(x: float, adj: float):
    wp.adjoint[x] -= 2.0 * x * adj


@wp.kernel
def wrong_grad_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wrong_grad_func(a[tid])


@wp.kernel
def transform_point_kernel(
    transforms: wp.array(dtype=wp.transform),
    points: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out[tid] = wp.transform_point(transforms[tid], points[tid])


def test_gradcheck_3d(test, device, dtype):
    # Adjust tolerances based on dtype precision
    if dtype == wp.float64:
        eps = 1e-5
        atol, rtol = 1e-7, 1e-7
    else:
        eps = 1e-4
        atol, rtol = 1e-2, 1e-2

    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    jacs_ad = jacobian(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_outputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
    )

    test.assertEqual(sorted(jacs_ad.keys()), [(0, 0), (1, 1)])
    test.assertEqual(jacs_ad[(0, 0)].shape, (6, 6))
    test.assertEqual(jacs_ad[(1, 1)].shape, (6, 6))
    # all entries beyond the max_outputs_per_var are NaN
    test.assertTrue(np.all(np.isnan(jacs_ad[(0, 0)].numpy()[4:])))
    test.assertTrue(np.all(np.isnan(jacs_ad[(1, 1)].numpy()[4:])))

    jacs_fd = jacobian_fd(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_inputs_per_var=4,
        # use integer indices instead of variable names
        input_output_mask=[(0, 0), (1, 1)],
        eps=eps,
    )

    test.assertEqual(sorted(jacs_fd.keys()), [(0, 0), (1, 1)])
    test.assertEqual(jacs_fd[(0, 0)].shape, (6, 6))
    test.assertEqual(jacs_fd[(1, 1)].shape, (6, 6))
    # all entries beyond the max_inputs_per_var are NaN
    test.assertTrue(np.all(np.isnan(jacs_fd[(0, 0)].numpy()[:, 4:])))
    test.assertTrue(np.all(np.isnan(jacs_fd[(1, 1)].numpy()[:, 4:])))

    # manual gradcheck
    np.testing.assert_allclose(jacs_ad[(0, 0)].numpy()[:4, :4], jacs_fd[(0, 0)].numpy()[:4, :4], atol=atol, rtol=rtol)
    np.testing.assert_allclose(jacs_ad[(1, 1)].numpy()[:4, :4], jacs_fd[(1, 1)].numpy()[:4, :4], atol=atol, rtol=rtol)

    passed = gradcheck(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_inputs_per_var=4,
        max_outputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
        show_summary=False,
    )
    test.assertTrue(
        passed,
        f"gradcheck failed for kernel_3d (dtype={dtype.__name__}, eps={eps}, atol={atol}, rtol={rtol})",
    )


def test_gradcheck_mixed(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    jacs_ad = jacobian(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2])
    jacs_fd = jacobian_fd(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], eps=1e-4)

    # manual gradcheck
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(jacs_ad[(i, j)].numpy(), jacs_fd[(i, j)].numpy(), atol=1e-2, rtol=1e-2)

    passed = gradcheck(
        kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], raise_exception=False, show_summary=False
    )

    test.assertTrue(passed, "gradcheck failed for kernel_mixed")


def test_gradcheck_nan(test, device):
    a = wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out = wp.array([0.0, 0.0], dtype=float, requires_grad=True, device=device)

    with test.assertRaises(ValueError):
        gradcheck(vec_length_kernel, dim=a.shape, inputs=[a], outputs=[out], raise_exception=True, show_summary=False)


def test_gradcheck_incorrect(test, device):
    a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device=device)
    out = wp.zeros_like(a)

    with test.assertRaises(ValueError):
        gradcheck(wrong_grad_kernel, dim=a.shape, inputs=[a], outputs=[out], raise_exception=True, show_summary=False)


def test_gradcheck_tape_basic(test, device, dtype):
    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(kernel_3d, dim=a_3d.shape, inputs=[a_3d, b_3d, c_3d], outputs=[out1_3d, out2_3d], device=device)

    passed = gradcheck_tape(tape, raise_exception=False, show_summary=False)

    test.assertTrue(passed, f"gradcheck_tape failed for kernel_3d (dtype={dtype.__name__})")


def test_gradcheck_tape_mixed(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], device=device)

    passed = gradcheck_tape(tape, raise_exception=False, show_summary=False)

    test.assertTrue(passed, "gradcheck_tape failed for kernel_mixed")


@wp.kernel(enable_backward=False)
def no_backward_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] * 2.0


@wp.kernel
def scale_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] * 3.0


def scale_function(a):
    """Regular Python function (not a kernel) that wraps a Warp kernel launch."""
    out = wp.zeros_like(a)
    out.requires_grad = True
    wp.launch(scale_kernel, dim=a.shape, inputs=[a], outputs=[out], device=a.device)
    return out


@wp.kernel
def vec_pair_kernel(
    x: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=wp.vec3),
    o1: wp.array(dtype=wp.vec3),
    o2: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    o1[tid] = x[tid] * 2.0 + y[tid]
    o2[tid] = x[tid] - y[tid] * 0.5


def vec_pair_function(x, y):
    """Vector-valued Python function used to exercise jacobian_plot (which
    requires vector-typed inputs/outputs to build its subplot grid)."""
    o1 = wp.zeros(len(x), dtype=wp.vec3, requires_grad=True, device=x.device)
    o2 = wp.zeros(len(x), dtype=wp.vec3, requires_grad=True, device=x.device)
    wp.launch(vec_pair_kernel, dim=len(x), inputs=[x, y], outputs=[o1, o2], device=x.device)
    return [o1, o2]


@wp.struct
class ArrayHolder:
    values: wp.array(dtype=float)


devices = get_test_devices()


class TestGradDebug(unittest.TestCase):
    def test_gradcheck_requires_inputs(self):
        # gradcheck raises if the inputs argument is omitted
        with self.assertRaisesRegex(ValueError, "inputs argument must be provided"):
            gradcheck(kernel_mixed, dim=1)

    def test_gradcheck_summary_pass_fail_nan(self):
        # Exercise the show_summary=True reporting path for passing, failing, and
        # NaN gradient cases (covers the summary table construction and printing).
        device = "cpu"
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device=device)
        out = wp.zeros_like(a)

        # passing case -> "PASS" row + success banner
        self.assertTrue(
            gradcheck(scale_kernel, dim=a.shape, inputs=[a], outputs=[out], show_summary=True)
        )

        # incorrect-gradient case -> "FAIL" row + failure banner (no exception raised)
        out2 = wp.zeros_like(a)
        self.assertFalse(
            gradcheck(
                wrong_grad_kernel,
                dim=a.shape,
                inputs=[a],
                outputs=[out2],
                raise_exception=False,
                show_summary=True,
            )
        )

        # NaN-gradient case -> "NaN" row (no exception raised)
        va = wp.array(
            [wp.vec3(1.0, 2.0, 3.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device
        )
        vout = wp.zeros(2, dtype=float, requires_grad=True, device=device)
        self.assertFalse(
            gradcheck(
                vec_length_kernel,
                dim=va.shape,
                inputs=[va],
                outputs=[vout],
                raise_exception=False,
                show_summary=True,
            )
        )

    def test_gradcheck_function_with_plots(self):
        # Use a plain Python function (not a kernel) so metadata is inferred via
        # update_from_function, and enable both plot options to exercise the
        # error-image computation and jacobian_plot calls inside gradcheck.
        # Vector-valued so jacobian_plot can build its subplot grid.
        x = wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(-1.0, 0.5, 2.0)], dtype=wp.vec3, requires_grad=True, device="cpu")
        y = wp.array([wp.vec3(0.5, -1.0, 1.0), wp.vec3(2.0, 1.0, -0.5)], dtype=wp.vec3, requires_grad=True, device="cpu")
        passed = gradcheck(
            vec_pair_function,
            inputs=[x, y],
            show_summary=True,
            plot_relative_error=True,
            plot_absolute_error=True,
        )
        self.assertTrue(passed)
        matplotlib.pyplot.close("all")

    def test_jacobian_function_input(self):
        # jacobian / jacobian_fd with a Python function input (exercises the
        # non-kernel branches, update_from_function, and the FD function path).
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device="cpu")
        jacs_ad = jacobian(scale_function, inputs=[a])
        jacs_fd = jacobian_fd(scale_function, inputs=[a], eps=1e-3)
        for key, jac in jacs_ad.items():
            np.testing.assert_allclose(jac.numpy(), jacs_fd[key].numpy(), atol=1e-2, rtol=1e-2)

    def test_jacobian_int_mask(self):
        # integer-index input/output mask (resolve_arg int branch in jacobian)
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device="cpu")
        out = wp.zeros_like(a)
        jacs = jacobian(scale_kernel, dim=a.shape, inputs=[a], outputs=[out], input_output_mask=[(0, 0)])
        self.assertIn((0, 0), jacs)

    def test_function_metadata_direct(self):
        # Directly exercise update_from_function branches: inferring outputs when
        # not provided, wrapping a single array output, and non-array in/outputs.
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device="cpu")

        md = _autograd.FunctionMetadata()
        md.update_from_function(scale_function, [a])  # outputs inferred + single-array wrap
        self.assertEqual(md.output_labels, ["output_0"])

        md2 = _autograd.FunctionMetadata()
        arr = wp.zeros(2, dtype=float, device="cpu")
        md2.update_from_function(lambda x, s: arr, [arr, 1.0], outputs=[arr, 2.0])
        self.assertEqual(md2.input_strides[1], None)  # non-array input
        self.assertEqual(md2.output_strides[1], None)  # non-array output

    def test_jacobian_kernel_validation(self):
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device="cpu")
        out = wp.zeros_like(a)

        # backward pass disabled
        with self.assertRaisesRegex(ValueError, "backward pass enabled"):
            jacobian(no_backward_kernel, dim=a.shape, inputs=[a], outputs=[out])
        with self.assertRaisesRegex(ValueError, "backward pass enabled"):
            jacobian_fd(no_backward_kernel, dim=a.shape, inputs=[a], outputs=[out])

        # missing outputs
        with self.assertRaisesRegex(ValueError, "output arguments must be provided"):
            jacobian(scale_kernel, dim=a.shape, inputs=[a], outputs=None)
        with self.assertRaisesRegex(ValueError, "output arguments must be provided"):
            jacobian_fd(scale_kernel, dim=a.shape, inputs=[a], outputs=None)

    def test_jacobian_skips_non_grad_args(self):
        # Inputs/outputs that are not requires_grad arrays are skipped (continue
        # branches in both jacobian and jacobian_fd).
        a = wp.array([1.0, 2.0], dtype=wp.float32, requires_grad=True, device="cpu")
        b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=False, device="cpu")
        out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device="cpu")
        out2 = wp.zeros(2, dtype=wp.quat, requires_grad=False, device="cpu")

        jacs_ad = jacobian(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2])
        jacs_fd = jacobian_fd(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], eps=1e-3)
        # Only the (a -> out1) pair has grad-enabled input and output
        self.assertIn((0, 0), jacs_ad)
        self.assertNotIn((1, 1), jacs_ad)
        self.assertIn((0, 0), jacs_fd)

    def test_gradcheck_tape_filters(self):
        device = "cpu"
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device=device)
        mid = wp.zeros_like(a)
        out = wp.zeros_like(a)

        with wp.Tape() as tape:
            wp.launch(scale_kernel, dim=a.shape, inputs=[a], outputs=[mid], device=device)
            wp.launch(scale_kernel, dim=a.shape, inputs=[mid], outputs=[out], device=device)
            # a kernel with the backward pass disabled is skipped
            wp.launch(no_backward_kernel, dim=a.shape, inputs=[out], outputs=[wp.zeros_like(out)], device=device)

        # blacklist / whitelist as sets, skip_to_launch_index and reverse
        self.assertTrue(
            gradcheck_tape(tape, blacklist_kernels=["scale_kernel"], show_summary=False)
        )
        self.assertTrue(
            gradcheck_tape(tape, whitelist_kernels=["scale_kernel"], show_summary=False, reverse_launches=True)
        )
        self.assertTrue(gradcheck_tape(tape, skip_to_launch_index=1, show_summary=False))
        self.assertTrue(
            gradcheck_tape(
                tape,
                input_output_masks={"scale_kernel": [("a", "out")]},
                show_summary=False,
            )
        )

        # Non-kernel entries on the tape are ignored (defensive continue branches).
        tape.launches.append("a-scope-marker")
        tape.launches.append([object(), 1, 2])
        self.assertTrue(gradcheck_tape(tape, whitelist_kernels=["scale_kernel"], show_summary=False))

    def test_infer_device_and_struct(self):
        # infer_device: struct with array field, and fallback to preferred device.
        holder = ArrayHolder()
        holder.values = wp.zeros(4, dtype=float, device="cpu")
        self.assertEqual(_autograd.infer_device([holder]), wp.get_device("cpu"))
        self.assertEqual(_autograd.get_struct_vars(holder)["values"].device, wp.get_device("cpu"))
        # no arrays / structs -> preferred device
        self.assertEqual(_autograd.infer_device([1.0, "x"]), wp.get_preferred_device())

    def test_scalarize_unsupported_dtype(self):
        bad = wp.zeros(2, dtype=wp.bool, device="cpu")
        with self.assertRaisesRegex(ValueError, "must be a scalar/vector/matrix array"):
            _autograd.scalarize_array_1d(bad)
        bad2d = wp.zeros((2, 2), dtype=wp.bool, device="cpu")
        with self.assertRaisesRegex(ValueError, "must be a scalar/vector/matrix array"):
            _autograd.scalarize_array_2d(bad2d)

    def test_jacobian_plot_options(self):
        # Directly exercise jacobian_plot option branches with a 2x2 grid of
        # Jacobians so that both present and absent (input, output) cells occur.
        x = wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(-1.0, 0.5, 2.0)], dtype=wp.vec3, requires_grad=True, device="cpu")
        y = wp.array([wp.vec3(0.5, -1.0, 1.0), wp.vec3(2.0, 1.0, -0.5)], dtype=wp.vec3, requires_grad=True, device="cpu")

        metadata = _autograd.FunctionMetadata()
        metadata.update_from_function(vec_pair_function, [x, y], vec_pair_function(x, y))
        jacs = jacobian(vec_pair_function, inputs=[x, y])

        # global color scale + colorbar, and exercise show_plot=True (no-op on Agg)
        fig = jacobian_plot(jacs, metadata, inputs=[x, y], show_plot=True, title="global")
        self.assertIsNotNone(fig)
        # per-submatrix scale + colorbar + log scale
        jacobian_plot(
            jacs,
            metadata,
            inputs=[x, y],
            show_plot=False,
            show_colorbar=True,
            scale_colors_per_submatrix=True,
            log_scale=True,
            title="custom",
        )
        # sparse jacobians -> some off-cells are turned off
        sparse = {(0, 0): jacs[(0, 0)], (1, 1): jacs[(1, 1)]}
        jacobian_plot(sparse, metadata, inputs=[x, y], show_plot=False, title="sparse")

        # all-NaN jacobians -> global vmin/vmax fall back to 0
        nan_jacs = {}
        for key, jac in jacs.items():
            nan_jac = wp.empty_like(jac)
            nan_jac.fill_(wp.nan)
            nan_jacs[key] = nan_jac
        jacobian_plot(nan_jacs, metadata, inputs=[x, y], show_plot=False, title="nan")

        # metadata with a non-array input and non-array output (continue branches
        # in the width/height ratio loops)
        md_mixed = _autograd.FunctionMetadata(
            key="mixed",
            input_labels=["v", "s"],
            output_labels=["o", "t"],
            input_strides=[(12,), None],
            output_strides=[(12,), None],
            input_dtypes=[wp.vec3, None],
            output_dtypes=[wp.vec3, None],
        )
        jac_single = wp.zeros((2, 2), dtype=wp.vec3, device="cpu")
        jacobian_plot({(0, 0): jac_single}, md_mixed, inputs=[x, 1.0], show_plot=False, title="mixed")

        # empty jacobians -> early return, for both metadata and kernel arguments
        self.assertIsNone(jacobian_plot({}, metadata, inputs=[x, y], show_plot=False, title="empty"))
        self.assertIsNone(jacobian_plot({}, scale_kernel, inputs=[x], show_plot=False, title="empty-kernel"))

        # invalid kernel argument type
        with self.assertRaisesRegex(ValueError, "must be a Warp kernel or a FunctionMetadata"):
            jacobian_plot(jacs, object(), inputs=[x, y], show_plot=False, title="bad")

        matplotlib.pyplot.close("all")


for dtype in [wp.float32, wp.float64]:
    add_function_test(
        TestGradDebug, f"test_gradcheck_3d_{dtype.__name__}", test_gradcheck_3d, devices=devices, dtype=dtype
    )
    add_function_test(
        TestGradDebug,
        f"test_gradcheck_tape_basic_{dtype.__name__}",
        test_gradcheck_tape_basic,
        devices=devices,
        dtype=dtype,
    )
add_function_test(TestGradDebug, "test_gradcheck_mixed", test_gradcheck_mixed, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_nan", test_gradcheck_nan, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_incorrect", test_gradcheck_incorrect, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_tape_mixed", test_gradcheck_tape_mixed, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
