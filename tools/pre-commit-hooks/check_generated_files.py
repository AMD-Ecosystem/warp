# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook that regenerates exports.h and __init__.pyi.

These files are derived from warp/_src/generated_files.py (which delegates to
warp/_src/context.py) and related source files.
When those sources change, the generated files must be updated to match.
This hook regenerates them in place; the pre-commit framework detects any
modifications and blocks the commit so the developer can stage them.

version.h is handled separately by the warp-check-version-consistency hook,
which has no heavy dependencies and can run on pre-commit.ci.

Usage (pre-commit hook):
    Configured in .pre-commit-config.yaml with language: python
    and additional_dependencies: [numpy]
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root is on sys.path so warp is importable in the
# pre-commit isolated virtualenv (which only has additional_dependencies).
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import warp  # noqa: E402 — populates builtin_functions via builtins.py
from warp._src.generated_files import generate_exports_header_file, generate_stubs_file  # noqa: E402


def _print_generation_status(path: str, changed: bool) -> None:
    """Print whether a generated file was updated."""
    if changed:
        print(f"Generated {path}")
    else:
        print(f"{path} is up to date")


def main() -> int:
    # Regenerate warp/native/exports.h
    exports_path = os.path.join(base_path, "warp", "native", "exports.h")
    _print_generation_status(exports_path, generate_exports_header_file(base_path))

    # Regenerate warp/__init__.pyi
    stubs_path = os.path.join(base_path, "warp", "__init__.pyi")
    _print_generation_status(stubs_path, generate_stubs_file(base_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
