# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def mul_constant(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()

    y[tid] = x[tid] * 2.0


@wp.struct
class Multiplicands:
    x: wp.array(dtype=float)
    y: wp.array(dtype=float)


@wp.kernel
def mul_variable(mutiplicands: Multiplicands, z: wp.array(dtype=float)):
    tid = wp.tid()

    z[tid] = mutiplicands.x[tid] * mutiplicands.y[tid]


@wp.kernel
def dot_product(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float)):
    tid = wp.tid()

    wp.atomic_add(z, 0, x[tid] * y[tid])


@wp.kernel
def _tape_step(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0


@wp.kernel(enable_backward=False)
def _tape_no_backward(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] + 1.0


@wp.kernel
def _tape_scalar_kernel(a: wp.array(dtype=float), s: float, b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * s


def test_tape_mul_constant(test, device):
    dim = 8
    iters = 16
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data
        x0 = wp.array(np.zeros(dim), dtype=wp.float32, device=device, requires_grad=True)
        x = x0

        for _i in range(iters):
            y = wp.empty_like(x, requires_grad=True)
            wp.launch(kernel=mul_constant, dim=dim, inputs=[x], outputs=[y], device=device)
            x = y

    # loss = wp.sum(x)
    x.grad = wp.array(np.ones(dim), device=device, dtype=wp.float32)

    # run backward
    tape.backward()

    # grad = 2.0^iters
    assert_np_equal(tape.gradients[x0].numpy(), np.ones(dim) * (2**iters))


def test_tape_mul_variable(test, device):
    dim = 8
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data (Note: We're intentionally testing structs in tapes here)
        multiplicands = Multiplicands()
        multiplicands.x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        multiplicands.y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros_like(multiplicands.x)

        wp.launch(kernel=mul_variable, dim=dim, inputs=[multiplicands], outputs=[z], device=device)

    # run backward with loss = wp.sum(z)
    tape.backward(grads={z: wp.ones_like(z)})

    # grad_x=y, grad_y=x
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), multiplicands.y.numpy())
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), multiplicands.x.numpy())

    # run backward again with different incoming gradient
    # should accumulate the same gradients again onto output
    # so gradients = 2.0*prev
    tape.backward(grads={z: wp.ones_like(z)})

    assert_np_equal(tape.gradients[multiplicands].x.numpy(), multiplicands.y.numpy() * 2.0)
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), multiplicands.x.numpy() * 2.0)

    # Clear launches and zero out the gradients
    tape.reset()
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), np.zeros_like(tape.gradients[multiplicands].x.numpy()))
    test.assertFalse(tape.launches)


def test_tape_dot_product(test, device):
    dim = 8
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data
        x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros(n=1, dtype=wp.float32, device=device, requires_grad=True)

        wp.launch(kernel=dot_product, dim=dim, inputs=[x, y], outputs=[z], device=device)

    # scalar loss
    tape.backward(loss=z)

    # grad_x=y, grad_y=x
    assert_np_equal(tape.gradients[x].numpy(), y.numpy())
    assert_np_equal(tape.gradients[y].numpy(), x.numpy())


@wp.kernel
def assign_chain_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid]
    z[tid] = y[tid]


def test_tape_zero_multiple_outputs(test, device):
    x = wp.array(np.arange(3), dtype=float, device=device, requires_grad=True)
    y = wp.zeros_like(x)
    z = wp.zeros_like(x)

    tape = wp.Tape()
    with tape:
        wp.launch(assign_chain_kernel, dim=3, inputs=[x, y, z], device=device)

    tape.backward(grads={y: wp.ones_like(x)})
    assert_np_equal(x.grad.numpy(), np.ones(3, dtype=float))
    tape.zero()

    tape.backward(grads={z: wp.ones_like(x)})
    assert_np_equal(x.grad.numpy(), np.ones(3, dtype=float))


@wp.struct
class NestedStruct:
    arr: wp.array(dtype=float)


@wp.struct
class WrapperStruct:
    nested: NestedStruct


@wp.kernel
def nested_loss_kernel(wrapper: WrapperStruct, loss: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(loss, 0, wrapper.nested.arr[i])


def test_tape_nested_struct(test, device):
    wrapper = WrapperStruct()
    wrapper.nested = NestedStruct()
    wrapper.nested.arr = wp.ones(shape=(1,), dtype=float, requires_grad=True, device=device)

    loss = wp.zeros(shape=(1,), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(nested_loss_kernel, dim=1, inputs=(wrapper, loss), device=device)

    assert_np_equal(loss.numpy(), [1.0])

    tape.backward(loss)
    assert_np_equal(wrapper.nested.arr.grad.numpy(), [1.0])

    tape.zero()

    assert_np_equal(wrapper.nested.arr.grad.numpy(), [0.0])


def test_tape_visualize(test, device):
    dim = 8
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data
        x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros(n=1, dtype=wp.float32, device=device, requires_grad=True)

        tape.record_scope_begin("my loop")
        for _ in range(16):
            wp.launch(kernel=dot_product, dim=dim, inputs=[x, y], outputs=[z], device=device)
        tape.record_scope_end()

    # generate GraphViz diagram code
    dot_code = tape.visualize(simplify_graph=True)

    assert "repeated 16x" in dot_code
    assert "my loop" in dot_code
    assert dot_code.count("dot_product") == 1


@wp.kernel
def dot_product_subscript(x: wp.array[float], y: wp.array[float], z: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(z, 0, x[tid] * y[tid])


# Subscript-style type hint variants (wp.array[dtype] syntax)
@wp.struct
class MultiplicandsSubscript:
    x: wp.array[float]
    y: wp.array[float]


@wp.kernel
def mul_variable_subscript(multiplicands: MultiplicandsSubscript, z: wp.array[float]):
    tid = wp.tid()
    z[tid] = multiplicands.x[tid] * multiplicands.y[tid]


@wp.struct
class NestedStructSubscript:
    arr: wp.array[float]


@wp.struct
class WrapperStructSubscript:
    nested: NestedStructSubscript


@wp.kernel
def nested_loss_kernel_subscript(wrapper: WrapperStructSubscript, loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, wrapper.nested.arr[i])


def test_tape_struct_subscript(test, device):
    """Test that struct fields using wp.array[float] subscript syntax work with Tape.backward() and Tape.zero()."""
    dim = 8
    tape = wp.Tape()

    with tape:
        multiplicands = MultiplicandsSubscript()
        multiplicands.x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        multiplicands.y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros_like(multiplicands.x)

        wp.launch(kernel=mul_variable_subscript, dim=dim, inputs=[multiplicands], outputs=[z], device=device)

    z.grad = wp.array(np.ones(dim), device=device, dtype=wp.float32)
    tape.backward()

    # grad_x=y, grad_y=x
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), multiplicands.y.numpy())
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), multiplicands.x.numpy())

    # zero should reset struct field gradients
    tape.zero()
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), np.zeros(dim))
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), np.zeros(dim))


def test_tape_nested_struct_subscript(test, device):
    """Test that nested struct fields using wp.array[float] subscript syntax work with Tape."""
    wrapper = WrapperStructSubscript()
    wrapper.nested = NestedStructSubscript()
    wrapper.nested.arr = wp.ones(shape=(1,), dtype=float, requires_grad=True, device=device)

    loss = wp.zeros(shape=(1,), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(nested_loss_kernel_subscript, dim=1, inputs=(wrapper, loss), device=device)

    assert_np_equal(loss.numpy(), np.ones(1))

    tape.backward(loss)
    assert_np_equal(wrapper.nested.arr.grad.numpy(), np.ones(1))

    tape.zero()
    assert_np_equal(wrapper.nested.arr.grad.numpy(), np.zeros(1))


def test_tape_visualize_subscript(test, device):
    """Test that tape visualization works with kernels using wp.array[float] subscript syntax."""
    dim = 8
    tape = wp.Tape()

    with tape:
        x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros(n=1, dtype=wp.float32, device=device, requires_grad=True)

        wp.launch(kernel=dot_product_subscript, dim=dim, inputs=[x, y], outputs=[z], device=device)

    dot_code = tape.visualize()

    # Array args should get "array: dtype=..." tooltip, not fall through to the scalar branch
    test.assertIn("array: dtype=", dot_code)


devices = get_test_devices()


class TestTape(unittest.TestCase):
    def test_tape_no_nested_tapes(self):
        with self.assertRaises(RuntimeError):
            with wp.Tape():
                with wp.Tape():
                    pass

    def test_tape_visualize_options(self):
        device = "cpu"
        dim = 4
        tape = wp.Tape()
        with tape:
            a = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
            # iterative chain where each output feeds the next launch (repeated
            # sequence + wrap-around connections)
            cur = a
            for _ in range(6):
                nxt = wp.zeros_like(cur)
                wp.launch(_tape_step, dim=dim, inputs=[cur], outputs=[nxt], device=device)
                cur = nxt
        out = cur

        code = tape.visualize(
            simplify_graph=True,
            hide_readonly_arrays=True,
            array_labels={a: "start", out: "end"},
            choose_longest_node_name=False,
            track_inputs=[a],
            track_outputs=[out],
            track_input_names=["x_in"],
            track_output_names=["x_out"],
            graph_direction="TB",
        )
        self.assertIn("digraph", code)
        # array_labels take precedence over the auto-generated track names
        self.assertIn("start", code)
        self.assertIn("end", code)
        self.assertIn("repeated 6x", code)

        # no simplification, ignore scopes
        code2 = tape.visualize(simplify_graph=False, ignore_graph_scopes=True)
        self.assertIn("_tape_step", code2)

        # writing to a file
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "tape.dot")
            tape.visualize(filename=fn)
            self.assertTrue(os.path.exists(fn))

    def test_tape_visualize_scalar_and_struct(self):
        device = "cpu"
        dim = 4

        # kernel with a scalar (non-array) input argument
        tape = wp.Tape()
        with tape:
            a = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
            b = wp.zeros_like(a)
            wp.launch(_tape_scalar_kernel, dim=dim, inputs=[a, 2.5], outputs=[b], device=device)
        self.assertIn("dtype=", tape.visualize())

        # kernel with a struct input argument
        tape2 = wp.Tape()
        with tape2:
            m = Multiplicands()
            m.x = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
            m.y = wp.array(np.ones(dim) * 2.0, dtype=wp.float32, device=device, requires_grad=True)
            z = wp.zeros_like(m.x)
            wp.launch(mul_variable, dim=dim, inputs=[m], outputs=[z], device=device)
        self.assertIn("mul_variable", tape2.visualize())

    def test_tape_visitors(self):
        from warp._src.tape import ArrayStatsVisitor, TapeVisitor, visit_tape

        device = "cpu"
        dim = 4
        tape = wp.Tape()
        with tape:
            a = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
            b = wp.zeros_like(a)
            wp.launch(_tape_step, dim=dim, inputs=[a], outputs=[b], device=device)
        b.grad = wp.array(np.ones(dim), dtype=wp.float32, device=device)
        tape.backward()

        # base visitor with no-op emit methods
        visit_tape(tape, TapeVisitor())

        # stats visitor computes numpy statistics over outputs and gradients
        stats = ArrayStatsVisitor()
        visit_tape(tape, stats)
        self.assertGreaterEqual(len(stats.launches), 1)
        self.assertGreaterEqual(len(stats.array_value_stats), 1)

    def test_tape_record_func_and_backward_errors(self):
        device = "cpu"
        dim = 4

        # record_func with a valid gradient-tracked array
        tape = wp.Tape()
        a = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
        called = []
        tape.record_func(lambda: called.append(True), [a])
        tape.backward()
        self.assertEqual(called, [True])

        # record_func error when an array has no gradient
        tape2 = wp.Tape()
        no_grad = wp.array(np.ones(dim), dtype=wp.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "missing a gradient"):
            tape2.record_func(lambda: None, [no_grad])

        # backward loss validation errors
        tape3 = wp.Tape()
        big_loss = wp.zeros(dim, dtype=wp.float32, device=device, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "scalar loss"):
            tape3.backward(loss=big_loss)

        loss_no_grad = wp.zeros(1, dtype=wp.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "requires_grad"):
            tape3.backward(loss=loss_no_grad)

        # supplying incoming gradients through the grads dict
        tape4 = wp.Tape()
        with tape4:
            x = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
            y = wp.zeros_like(x)
            wp.launch(_tape_step, dim=dim, inputs=[x], outputs=[y], device=device)
        g = wp.array(np.ones(dim), dtype=wp.float32, device=device)
        tape4.backward(grads={y: g})
        assert_np_equal(x.grad.numpy(), np.ones(dim) * 2.0)

    def test_tape_backward_enable_backward_false_warns(self):
        device = "cpu"
        dim = 4
        tape = wp.Tape()
        with tape:
            a = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
            b = wp.zeros_like(a)
            wp.launch(_tape_no_backward, dim=dim, inputs=[a], outputs=[b], device=device)
        # running backward on a kernel with enable_backward=False exercises the
        # warning path; it should complete without raising
        tape.backward()

    def test_tape_reset_and_scope_removal(self):
        device = "cpu"
        dim = 4
        tape = wp.Tape()
        with tape:
            a = wp.array(np.ones(dim), dtype=wp.float32, device=device, requires_grad=True)
            b = wp.zeros_like(a)
            # empty scope that should be removed on end
            tape.record_scope_begin("empty")
            tape.record_scope_end()
            tape.record_scope_begin("work")
            wp.launch(_tape_step, dim=dim, inputs=[a], outputs=[b], device=device)
            tape.record_scope_end()
        self.assertGreater(len(tape.launches), 0)
        tape.reset()
        self.assertEqual(tape.launches, [])
        self.assertEqual(tape.scopes, [])


add_function_test(TestTape, "test_tape_mul_constant", test_tape_mul_constant, devices=devices)
add_function_test(TestTape, "test_tape_mul_variable", test_tape_mul_variable, devices=devices)
add_function_test(TestTape, "test_tape_dot_product", test_tape_dot_product, devices=devices)
add_function_test(TestTape, "test_tape_zero_multiple_outputs", test_tape_zero_multiple_outputs, devices=devices)
add_function_test(TestTape, "test_tape_nested_struct", test_tape_nested_struct, devices=devices)
add_function_test(TestTape, "test_tape_visualize", test_tape_visualize, devices=devices)
add_function_test(TestTape, "test_tape_struct_subscript", test_tape_struct_subscript, devices=devices)
add_function_test(TestTape, "test_tape_nested_struct_subscript", test_tape_nested_struct_subscript, devices=devices)
add_function_test(TestTape, "test_tape_visualize_subscript", test_tape_visualize_subscript, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
