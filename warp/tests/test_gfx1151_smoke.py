# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest

import warp as wp
from warp.tests.unittest_utils import run_python_subprocess


@wp.kernel
def vec_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = a[i] + b[i]


@wp.kernel
def vec_scale(a: wp.array(dtype=float), s: float):
    i = wp.tid()
    a[i] = a[i] * s


class TestGFX1151Smoke(unittest.TestCase):
    def test_device_and_arch(self):
        wp.init()

        devices = wp.get_devices()
        self.assertTrue(len(devices) > 0, "No Warp devices found")

        hip_devices = [d for d in devices if d.is_cuda or d.is_hip]

        self.assertTrue(len(hip_devices) > 0, "No HIP/CUDA device found")

        d = hip_devices[0]
        print("Using device:", d)

        # Check arch string if available (arch_str is the gfx string on HIP;
        # `arch` is a numeric value)
        arch = getattr(d, "arch_str", None) or str(getattr(d, "arch", ""))
        print("Detected arch:", arch)

        # The arch assertion is specific to the validated gfx1151 target; on
        # any other GPU (e.g. NVIDIA sm_*, other AMD gfx*) it is not a failure.
        if not arch or "gfx1151" not in arch.lower():
            self.skipTest(f"gfx1151-specific check (detected arch: {arch or 'unknown'})")

        self.assertIn("gfx1151", arch.lower(), f"Expected gfx1151, got {arch}")

    def test_vec_add_and_scale(self):
        wp.init()

        n = 1024
        device = wp.get_preferred_device()

        a = wp.array([1.0] * n, dtype=float, device=device)
        b = wp.array([2.0] * n, dtype=float, device=device)
        c = wp.zeros(n, dtype=float, device=device)

        # Launch add
        wp.launch(vec_add, dim=n, inputs=[a, b, c], device=device)

        # Verify result
        c_host = c.numpy()
        for v in c_host:
            self.assertAlmostEqual(v, 3.0)

        # Launch scale
        wp.launch(vec_scale, dim=n, inputs=[c, 2.0], device=device)

        c_host = c.numpy()
        for v in c_host:
            self.assertAlmostEqual(v, 6.0)

    def test_compile_writes_no_warning(self):
        """Compiling this module's kernels on HIP from an empty kernel cache writes no warning to stderr."""
        wp.init()

        device = wp.get_preferred_device()
        if not device.is_hip:
            self.skipTest("HIP compile path only")

        source = (
            "import warp as wp\n"
            "from warp.tests.test_gfx1151_smoke import vec_scale\n"
            f"a = wp.ones(4, dtype=float, device='{device.alias}')\n"
            f"wp.launch(vec_scale, dim=4, inputs=[a, 3.0], device='{device.alias}')\n"
            "print(a.numpy().sum())\n"
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            result = run_python_subprocess(source, timeout=600, env_updates={"WARP_CACHE_PATH": cache_dir})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(float(result.stdout.strip().splitlines()[-1]), 12.0)
        warnings = [line for line in result.stderr.splitlines() if "Warp warning" in line]
        self.assertEqual(warnings, [], result.stderr)


if __name__ == "__main__":
    unittest.main()
