# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp


class TestContext(unittest.TestCase):
    def test_context_type_str(self):
        self.assertEqual(wp._src.context.type_str(list[int]), "list[int]")
        self.assertEqual(wp._src.context.type_str(list[float]), "list[float]")

        self.assertEqual(wp._src.context.type_str(tuple[int]), "tuple[int]")
        self.assertEqual(wp._src.context.type_str(tuple[float]), "tuple[float]")
        self.assertEqual(wp._src.context.type_str(tuple[int, float]), "tuple[int, float]")
        self.assertEqual(wp._src.context.type_str(tuple[int, ...]), "tuple[int, ...]")

    def test_device_query_functions(self):
        import warp._src.context as ctx

        self.assertIsInstance(ctx.is_cpu_available(), bool)
        self.assertIsInstance(ctx.is_cuda_available(), bool)
        self.assertIsInstance(ctx.is_cuda_driver_initialized(), bool)
        self.assertIsInstance(ctx.is_cubql_available(), bool)

        devices = ctx.get_devices()
        self.assertGreater(len(devices), 0)

        cpu = ctx.get_device("cpu")
        self.assertTrue(ctx.is_device_available(cpu))
        self.assertTrue(cpu.is_cpu)

        self.assertEqual(ctx.get_cuda_device_count(), len(ctx.get_cuda_devices()))
        self.assertIsInstance(ctx.get_cuda_supported_archs(), list)

        pref = ctx.get_preferred_device()
        self.assertIn(pref, devices)

    def test_device_properties_and_methods(self):
        import warp._src.context as ctx

        for device in ctx.get_devices():
            self.assertIsInstance(str(device), str)
            self.assertIsInstance(repr(device), str)
            self.assertTrue(device == device)
            self.assertFalse(device == 12345)  # neither str nor Device
            self.assertTrue(device == str(device))
            self.assertTrue(device.can_access(device))

            self.assertIsInstance(device.is_cpu, bool)
            self.assertIsInstance(device.is_cuda, bool)
            self.assertIsInstance(device.is_hip, bool)
            self.assertIsInstance(device.is_capturing, bool)
            self.assertIsInstance(device.supports_graph_capture, bool)
            self.assertIsInstance(device.has_context, bool)
            self.assertIsInstance(device.has_stream, bool)

            self.assertGreaterEqual(device.total_memory, 0)
            self.assertGreaterEqual(device.free_memory, 0)

            self.assertIsNotNone(ctx.get_device_allocator(device))
            self.assertIsInstance(ctx.is_mempool_supported(device), bool)
            self.assertIsInstance(ctx.is_mempool_enabled(device), bool)

            if device.is_cuda:
                self.assertIn(device.get_cuda_output_format(), ("ptx", "cubin"))
                self.assertIsNotNone(device.stream)

    def test_mempool_and_allocator_errors(self):
        import warp._src.context as ctx

        # enabling memory pools on CPU is an error
        with self.assertRaisesRegex(ValueError, "only supported on CUDA"):
            ctx.set_mempool_enabled("cpu", True)

        # disabling memory pools on CPU is a no-op
        ctx.set_mempool_enabled("cpu", False)

        # custom allocators are only supported on CUDA devices
        with self.assertRaisesRegex(RuntimeError, "only supported on CUDA"):
            ctx.set_device_allocator("cpu", None)

    def test_export_generators(self):
        """The stub/doc/builtin generators must run for every registered builtin.

        These entry points back ``build_docs.py`` and the ``check_generated_files``
        pre-commit hook, and they exercise each builtin's generic return-type
        inference (``value_func(None, None)``) plus the vector/matrix/quat/transform
        stub formatting.
        """
        import io

        import warp._src.context as ctx

        for fn_name in ("export_stubs", "export_builtins", "export_functions_rst"):
            buf = io.StringIO()
            getattr(ctx, fn_name)(buf)
            self.assertGreater(len(buf.getvalue()), 0, f"{fn_name} produced no output")

    def test_cpu_compiler_flag_helpers(self):
        import warp._src.context as ctx

        self.assertEqual(ctx._resolve_cpu_compiler_flags(None, None), "-march=native")
        self.assertEqual(ctx._resolve_cpu_compiler_flags(None, "-O2"), "-O2")
        self.assertEqual(ctx._resolve_cpu_compiler_flags("-O1", "-O2"), "-O1")

        self.assertTrue(ctx._uses_march_native("-march=native -O2"))
        self.assertFalse(ctx._uses_march_native("-O2"))

        # these query the LLVM backend; verify only the documented return types
        self.assertIsInstance(ctx._get_cpu_feature_set(), frozenset)
        self.assertIsInstance(ctx._get_cpu_isa_hash(), str)
        self.assertIsInstance(ctx._get_host_cpu_name(), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
