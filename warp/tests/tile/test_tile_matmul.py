# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64


@wp.kernel
def tile_grouped_gemm(A: wp.array3d[float], B: wp.array3d[float], C: wp.array3d[float]):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(A[i], shape=(TILE_M, TILE_K))
    b = wp.tile_load(B[i], shape=(TILE_K, TILE_N))

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    wp.tile_matmul(a, b, sum)

    wp.tile_store(C[i], sum)


def test_tile_grouped_gemm(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N
    K = TILE_K

    rng = np.random.default_rng(42)
    A = rng.random((batch_count, M, K), dtype=np.float32)
    B = rng.random((batch_count, K, N), dtype=np.float32)
    C = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((batch_count, TILE_M, TILE_N), requires_grad=True, device=device)

    with wp.Tape():
        wp.launch_tiled(
            tile_grouped_gemm, dim=[batch_count], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    # TODO: 32 mismatched elements
    assert_np_equal(C_wp.numpy(), C, 1e-6)


@wp.kernel
def tile_gemm(A: wp.array2d[Any], B: wp.array2d[Any], C: wp.array2d[Any]):
    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=A.dtype)

    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


wp.overload(tile_gemm, {"A": wp.array2d[wp.float16], "B": wp.array2d[wp.float16], "C": wp.array2d[wp.float16]})
wp.overload(tile_gemm, {"A": wp.array2d[wp.float32], "B": wp.array2d[wp.float32], "C": wp.array2d[wp.float32]})
wp.overload(tile_gemm, {"A": wp.array2d[wp.float64], "B": wp.array2d[wp.float64], "C": wp.array2d[wp.float64]})


def test_tile_gemm(dtype):
    def test(test, device):
        M = TILE_M * 7
        K = TILE_K * 6
        N = TILE_N * 5

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=float).astype(wp.dtype_to_numpy(dtype))
        B = rng.random((K, N), dtype=float).astype(wp.dtype_to_numpy(dtype))
        C = np.zeros((M, N), dtype=float).astype(wp.dtype_to_numpy(dtype))

        A_wp = wp.array(A, requires_grad=True, device=device)
        B_wp = wp.array(B, requires_grad=True, device=device)
        C_wp = wp.array(C, requires_grad=True, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                tile_gemm,
                dim=(int(M / TILE_M), int(N / TILE_N)),
                inputs=[A_wp, B_wp, C_wp],
                block_dim=TILE_DIM,
                device=device,
            )

        assert_np_equal(C_wp.numpy(), A @ B, tol=1.0e-1)

        adj_C = np.ones_like(C)

        tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

        assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1.0e-1)
        assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, 1.0e-1)

    return test


@wp.kernel
def test_tile_transpose_matmul_kernel(input: wp.array2d[float], output: wp.array2d[float]):
    x = wp.tile_load(input, shape=(TILE_M, TILE_N))
    y = wp.tile_transpose(x)

    z = wp.tile_zeros(dtype=float, shape=(TILE_N, TILE_N))
    wp.tile_matmul(y, x, z)

    wp.tile_store(output, z)


def test_tile_transpose_matmul(test, device):
    rng = np.random.default_rng(42)
    input = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), device=device)
    output = wp.zeros((TILE_N, TILE_N), dtype=float, device=device)

    wp.launch_tiled(
        test_tile_transpose_matmul_kernel, dim=[1], inputs=[input, output], block_dim=TILE_DIM, device=device
    )

    assert_np_equal(output.numpy(), input.numpy().T @ input.numpy(), 1e-6)


@wp.kernel
def test_tile_matmul_return_form_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    """Test the c = wp.tile_matmul(a, b) form which returns a fresh tile."""
    a = wp.tile_load(A, shape=(TILE_M, TILE_K))
    b = wp.tile_load(B, shape=(TILE_K, TILE_N))

    # Use the return form (not the accumulate form)
    # This tests that we don't read from the uninitialized output tile
    c = wp.tile_matmul(a, b)

    wp.tile_store(C, c)


def test_tile_matmul_return_form(test, device):
    """Test that c = wp.tile_matmul(a, b) works correctly with verify_fp.

    This specifically tests a fix where the return form was incorrectly
    reading from the uninitialized output tile (which could contain NaN
    when verify_fp is enabled, causing the result to be NaN).
    """
    # Enable verify_fp to trigger NaN initialization of tiles
    old_verify_fp = wp.config.verify_fp
    wp.config.verify_fp = True

    try:
        M = TILE_M
        K = TILE_K
        N = TILE_N

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=np.float32)
        B = rng.random((K, N), dtype=np.float32)
        expected = A @ B

        A_wp = wp.array(A, device=device)
        B_wp = wp.array(B, device=device)
        C_wp = wp.zeros((M, N), dtype=float, device=device)

        wp.launch_tiled(
            test_tile_matmul_return_form_kernel,
            dim=[1],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=TILE_DIM,
            device=device,
        )

        result = C_wp.numpy()

        # Check that result doesn't contain NaN (which would happen with the bug)
        test.assertFalse(np.any(np.isnan(result)), "Result contains NaN values")

        # Check correctness
        assert_np_equal(result, expected, tol=1e-5)
    finally:
        wp.config.verify_fp = old_verify_fp


@wp.kernel(module="unique")
def f32_to_bf16_kernel(input: wp.array2d[wp.float32], output: wp.array2d[wp.bfloat16]):
    i, j = wp.tid()
    output[i, j] = wp.bfloat16(input[i, j])


@wp.kernel(module="unique")
def bf16_to_f32_kernel(input: wp.array2d[wp.bfloat16], output: wp.array2d[wp.float32]):
    i, j = wp.tid()
    output[i, j] = wp.float32(input[i, j])


# cuBLASDx restricts the accumulator dtype to float16, float32, or float64. The kernel goes
# in its own module with enable_backward=False so that backward LTOs (which would use
# bfloat16 accumulators for adjA, adjB) are not generated.
@wp.kernel(module="unique", module_options={"enable_backward": False})
def tile_gemm_bf16(A: wp.array2d[wp.bfloat16], B: wp.array2d[wp.bfloat16], C: wp.array2d[wp.float32]):
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    K = A.shape[1]
    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


def test_tile_gemm_bf16(test, device):
    M = TILE_M * 7
    K = TILE_K * 6
    N = TILE_N * 5

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)

    # Convert float32 numpy data to bfloat16 on device via kernels
    A_f32 = wp.array(A, device=device)
    B_f32 = wp.array(B, device=device)
    A_wp = wp.zeros((M, K), dtype=wp.bfloat16, device=device)
    B_wp = wp.zeros((K, N), dtype=wp.bfloat16, device=device)
    C_wp = wp.zeros((M, N), dtype=wp.float32, device=device)

    wp.launch(f32_to_bf16_kernel, dim=(M, K), inputs=[A_f32, A_wp], device=device)
    wp.launch(f32_to_bf16_kernel, dim=(K, N), inputs=[B_f32, B_wp], device=device)

    wp.launch_tiled(
        tile_gemm_bf16,
        dim=(int(M / TILE_M), int(N / TILE_N)),
        inputs=[A_wp, B_wp, C_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    # Quantize reference inputs through bfloat16 to match what the kernel sees
    A_ref = wp.zeros((M, K), dtype=wp.float32, device=device)
    B_ref = wp.zeros((K, N), dtype=wp.float32, device=device)
    wp.launch(bf16_to_f32_kernel, dim=(M, K), inputs=[A_wp, A_ref], device=device)
    wp.launch(bf16_to_f32_kernel, dim=(K, N), inputs=[B_wp, B_ref], device=device)

    np.testing.assert_allclose(C_wp.numpy(), A_ref.numpy() @ B_ref.numpy(), rtol=1.0e-2)


def test_tile_matmul_bf16_out_rejected(test, device):
    """tile_matmul rejects a bfloat16 'out' tile because the accumulator must be float16,
    float32, or float64."""

    @wp.kernel(module="unique")
    def kernel_bf16_out(
        A: wp.array2d[wp.bfloat16],
        B: wp.array2d[wp.bfloat16],
        C: wp.array2d[wp.bfloat16],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.bfloat16)
        wp.tile_matmul(a, b, c)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.bfloat16, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.bfloat16, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.bfloat16, device=device)

    with test.assertRaisesRegex(TypeError, r"does not support a bfloat16 'out' tile"):
        wp.launch_tiled(kernel_bf16_out, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_bf16_out_rejected_return_form(test, device):
    """The 2-arg returning form ``c = wp.tile_matmul(a, b)`` synthesizes ``out`` with
    ``dtype=a.dtype``, so bf16 inputs produce a synthesized bf16 ``out`` that must be
    rejected by the same gate."""

    @wp.kernel(module="unique")
    def kernel_bf16_return(
        A: wp.array2d[wp.bfloat16],
        B: wp.array2d[wp.bfloat16],
        C: wp.array2d[wp.bfloat16],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_matmul(a, b)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.bfloat16, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.bfloat16, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.bfloat16, device=device)

    with test.assertRaisesRegex(TypeError, r"bfloat16 'out' tile"):
        wp.launch_tiled(kernel_bf16_return, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_bf16_a_with_backward_rejected(test, device):
    """tile_matmul rejects a bfloat16 'a' tile when the backward pass is enabled, because 'a'
    is the accumulator for adjA."""

    @wp.kernel(module="unique")
    def kernel_bf16_a(
        A: wp.array2d[wp.bfloat16],
        B: wp.array2d[wp.float32],
        C: wp.array2d[wp.float32],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        wp.tile_matmul(a, b, c)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.bfloat16, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.float32, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.float32, device=device)

    with test.assertRaisesRegex(TypeError, r"bfloat16 'a' or 'b' tiles when the backward pass is enabled"):
        wp.launch_tiled(kernel_bf16_a, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


def test_tile_matmul_bf16_b_with_backward_rejected(test, device):
    """tile_matmul rejects a bfloat16 'b' tile when the backward pass is enabled, because 'b'
    is the accumulator for adjB."""

    @wp.kernel(module="unique")
    def kernel_bf16_b(
        A: wp.array2d[wp.float32],
        B: wp.array2d[wp.bfloat16],
        C: wp.array2d[wp.float32],
    ):
        i, j = wp.tid()
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N))
        c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        wp.tile_matmul(a, b, c)
        wp.tile_store(C, c, offset=(i * TILE_M, j * TILE_N))

    A = wp.zeros((TILE_M, TILE_K), dtype=wp.float32, device=device)
    B = wp.zeros((TILE_K, TILE_N), dtype=wp.bfloat16, device=device)
    C = wp.zeros((TILE_M, TILE_N), dtype=wp.float32, device=device)

    with test.assertRaisesRegex(TypeError, r"bfloat16 'a' or 'b' tiles when the backward pass is enabled"):
        wp.launch_tiled(kernel_bf16_b, dim=(1, 1), inputs=[A, B, C], block_dim=TILE_DIM, device=device)


class TestTileMatmul(unittest.TestCase):
    pass


def create_matmul_kernel(dtype, M, N, K):
    @wp.kernel(module="unique")
    def mm(A: wp.array2d(dtype=dtype), B: wp.array2d(dtype=dtype), C: wp.array2d(dtype=dtype)):
        wp.tile_store(C, wp.tile_matmul(wp.tile_load(A, shape=(M, K)), wp.tile_load(B, shape=(K, N))))

    return mm


# shapes that fill 16x16 fragments, a non-square one, ones that do not fit, and rows of 2 and 4 fragments
FRAGMENT_SHAPES = ((16, 16, 16), (48, 16, 32), (24, 16, 16), (17, 33, 15), (16, 32, 16), (32, 64, 32))
# blocks smaller than a 32- or 64-wide wavefront, whole wavefronts, and blocks that are not a whole number of either
FRAGMENT_BLOCK_DIMS = (16, 32, 48, 64, 96)


def fragment_cases(device):
    # the cases cover the fragment boundaries of the matrix-core paths behind is_hip devices; elsewhere one case checks
    # the path (cuBLASDx on NVIDIA GPUs needs at least 32 threads per block and takes about 25 s to compile each case)
    if device.is_hip:
        return FRAGMENT_SHAPES, FRAGMENT_BLOCK_DIMS
    return FRAGMENT_SHAPES[:1], (32,)


def test_tile_matmul_fragment_shapes(dtype):
    # GPU results against the CPU
    def test(test, device):
        rng = np.random.default_rng(7)
        shapes, block_dims = fragment_cases(wp.get_device(device))
        for M, N, K in shapes:
            # the backward kernel keeps the three tiles and their adjoints in shared memory
            if not shared_memory_fits(
                device, 2 * wp.types.type_size_in_bytes(dtype) * (M * K + K * N + M * N), max(block_dims)
            ):
                continue
            mm = create_matmul_kernel(dtype, M, N, K)
            A, B = rng.uniform(-1, 1, (M, K)), rng.uniform(-1, 1, (K, N))
            for block_dim in block_dims:
                results = []
                for dev in (device, "cpu"):
                    a = wp.array(A, dtype=dtype, device=dev, requires_grad=True)
                    b = wp.array(B, dtype=dtype, device=dev, requires_grad=True)
                    c = wp.zeros((M, N), dtype=dtype, device=dev, requires_grad=True)
                    with wp.Tape() as tape:
                        wp.launch_tiled(mm, dim=[1], inputs=[a, b, c], block_dim=block_dim, device=dev)
                    c.grad = wp.ones_like(c)
                    tape.backward()
                    results.append([x.numpy().astype(np.float64) for x in (c, a.grad, b.grad)])
                tol = 2e-2 if dtype == wp.float16 else 1e-5
                for got, ref in zip(*results, strict=True):
                    assert_np_equal(got, ref, tol=tol)

    return test


def create_matmul_bf16_kernel(M, N, K):
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def mm(A: wp.array2d(dtype=wp.bfloat16), B: wp.array2d(dtype=wp.bfloat16), C: wp.array2d(dtype=wp.float32)):
        c = wp.tile_zeros(shape=(M, N), dtype=wp.float32)
        wp.tile_matmul(wp.tile_load(A, shape=(M, K)), wp.tile_load(B, shape=(K, N)), c)
        wp.tile_store(C, c)

    return mm


def test_tile_matmul_fragment_shapes_bf16(test, device):
    # bfloat16 inputs with a float32 output (forward only), GPU results against the CPU
    rng = np.random.default_rng(7)
    shapes, block_dims = fragment_cases(wp.get_device(device))
    for M, N, K in shapes:
        mm = create_matmul_bf16_kernel(M, N, K)
        A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
        B = rng.uniform(-1, 1, (K, N)).astype(np.float32)
        for block_dim in block_dims:
            results = []
            for dev in (device, "cpu"):
                a = wp.zeros((M, K), dtype=wp.bfloat16, device=dev)
                b = wp.zeros((K, N), dtype=wp.bfloat16, device=dev)
                wp.launch(f32_to_bf16_kernel, dim=(M, K), inputs=[wp.array(A, device=dev), a], device=dev)
                wp.launch(f32_to_bf16_kernel, dim=(K, N), inputs=[wp.array(B, device=dev), b], device=dev)
                c = wp.zeros((M, N), dtype=wp.float32, device=dev)
                wp.launch_tiled(mm, dim=[1], inputs=[a, b, c], block_dim=block_dim, device=dev)
                results.append(c.numpy())
            assert_np_equal(results[0], results[1], tol=1e-4)


devices = get_test_devices()

# bfloat16 requires CC >= 8.0 (Ampere+)
bf16_devices = []
if wp.is_cpu_available():
    bf16_devices.append("cpu")
for cuda_device in get_selected_cuda_test_devices():
    if cuda_device.arch >= 80:
        bf16_devices.append(cuda_device)

add_function_test(TestTileMatmul, "test_tile_gemm_fp16", test_tile_gemm(wp.float16), devices=devices)
add_function_test(TestTileMatmul, "test_tile_gemm_bf16", test_tile_gemm_bf16, devices=bf16_devices, check_output=False)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_out_rejected",
    test_tile_matmul_bf16_out_rejected,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_out_rejected_return_form",
    test_tile_matmul_bf16_out_rejected_return_form,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_a_with_backward_rejected",
    test_tile_matmul_bf16_a_with_backward_rejected,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_bf16_b_with_backward_rejected",
    test_tile_matmul_bf16_b_with_backward_rejected,
    devices=bf16_devices,
    check_output=False,
)
add_function_test(TestTileMatmul, "test_tile_gemm_fp32", test_tile_gemm(wp.float32), devices=devices)
add_function_test(TestTileMatmul, "test_tile_gemm_fp64", test_tile_gemm(wp.float64), devices=devices)
add_function_test(TestTileMatmul, "test_tile_grouped_gemm", test_tile_grouped_gemm, devices=devices)
add_function_test(TestTileMatmul, "test_tile_transpose_matmul", test_tile_transpose_matmul, devices=devices)
add_function_test(TestTileMatmul, "test_tile_matmul_return_form", test_tile_matmul_return_form, devices=devices)
for dtype in (wp.float16, wp.float32, wp.float64):
    add_function_test(
        TestTileMatmul,
        f"test_tile_matmul_fragment_shapes_{dtype.__name__}",
        test_tile_matmul_fragment_shapes(dtype),
        devices=get_cuda_test_devices(),
    )
add_function_test(
    TestTileMatmul,
    "test_tile_matmul_fragment_shapes_bfloat16",
    test_tile_matmul_fragment_shapes_bf16,
    devices=[d for d in bf16_devices if d != "cpu" and wp.get_device(d).is_cuda],
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
