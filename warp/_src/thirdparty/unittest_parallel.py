# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
unittest-parallel command-line script main module
"""

import argparse
import concurrent.futures  # NVIDIA Modification
import multiprocessing
import os
import re  # NVIDIA Modification (distributed GPU testing)
import shutil  # NVIDIA Modification (distributed GPU testing)
import subprocess  # NVIDIA Modification (distributed GPU testing)
import sys
import tempfile
import time
import unittest
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from io import StringIO

import warp.tests.unittest_suites  # NVIDIA Modification
import warp.tests.unittest_utils
from warp._src.thirdparty import appdirs
from warp.tests.unittest_utils import (  # NVIDIA modification
    ParallelJunitTestResult,
    write_junit_results,
)

try:
    import coverage

    COVERAGE_AVAILABLE = True  # NVIDIA Modification
except ImportError:
    COVERAGE_AVAILABLE = False  # NVIDIA Modification


# The following variables are NVIDIA Modifications
START_DIRECTORY = os.path.join(os.path.dirname(__file__), "..")  # The directory to start test discovery
_SUITE_TIMEOUT = (
    3600  # Timeout in seconds: total wall-clock limit for parallel execution, per-suite limit during isolated fallback
)

# NVIDIA Modification (distributed GPU testing) follows.
#
# Test subdirectories that are treated as a single "module" for the purposes of
# distributing whole modules onto a GPU. Every top-level ``test_*.py`` file is
# also treated as its own module (see ``_module_group_key``).
_TEST_SUBDIR_MODULES = {"aot", "cuda", "fem", "geometry", "interop", "matrix", "tile"}

# Module groups that contain tests requiring more than one visible GPU
# (e.g. ``cuda/test_multigpu.py``, ``test_peer.py``, multi-GPU stream/async
# tests). These are gathered into a dedicated bucket that runs with all the
# selected GPUs visible so their coverage is preserved instead of being
# skipped under single-GPU pinning.
_MULTI_GPU_MODULES = {"cuda"}


def main(argv=None):
    """
    unittest-parallel command-line script main entry point
    """

    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="unittest-parallel",
        # NVIDIA Modifications follow:
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
        python -m warp.tests -s autodetect -p 'test_a*.py'
        python -m warp.tests -s kit
        python -m warp.tests -k 'mgpu' -k 'cuda'
        """,
    )
    # parser.add_argument("-v", "--verbose", action="store_const", const=2, default=1, help="Verbose output")
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_const", const=0, default=2, help="Quiet output")
    parser.add_argument("-f", "--failfast", action="store_true", default=False, help="Stop on first fail or error")
    parser.add_argument(
        "-b", "--buffer", action="store_true", default=False, help="Buffer stdout and stderr during tests"
    )
    parser.add_argument(
        "-k",
        dest="testNamePatterns",
        action="append",
        type=_convert_select_pattern,
        help="Only run tests which match the given substring",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        metavar="PATTERN",
        default="test*.py",
        help="'autodetect' suite only: Pattern to match tests ('test*.py' default)",  # NVIDIA Modification
    )
    parser.add_argument(
        "-x",
        "--exclude",
        dest="exclude",
        action="append",
        metavar="SUBSTRING",
        help="Exclude any test whose id (module.Class.method) contains SUBSTRING. May be repeated.",
    )  # NVIDIA Modification
    parser.add_argument(
        "-t",
        "--top-level-directory",
        metavar="TOP",
        help="Top level directory of project (defaults to start directory)",
    )
    parser.add_argument(
        "--junit-report-xml", metavar="FILE", help="Generate JUnit report format XML file"
    )  # NVIDIA Modification
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        default="default",
        choices=["autodetect", "default", "debug", "kit"],
        help="Name of the test suite to run (default is 'default').",
    )  # NVIDIA Modification
    group_parallel = parser.add_argument_group("parallelization options")
    group_parallel.add_argument(
        "-j",
        "--jobs",
        metavar="COUNT",
        type=int,
        default=0,
        help="The number of test processes (default is 0, all cores)",
    )
    group_parallel.add_argument(
        "-m",
        "--maxjobs",
        metavar="MAXCOUNT",
        type=int,
        default=8,
        help="The maximum number of test processes (default is 8)",
    )  # NVIDIA Modification
    group_parallel.add_argument(
        "--level",
        choices=["module", "class", "test"],
        default="class",
        help="Set the test parallelism level (default is 'class')",
    )
    group_parallel.add_argument(
        "--disable-process-pooling",
        action="store_true",
        default=False,
        help="Do not reuse processes used to run test suites",
    )
    group_parallel.add_argument(
        "--disable-concurrent-futures",
        action="store_true",
        default=False,
        help="Use multiprocessing instead of concurrent.futures.",
    )  # NVIDIA Modification
    group_parallel.add_argument(
        "--serial-fallback",
        action="store_true",
        default=False,
        help="Run in a single-process (no spawning) mode without multiprocessing or concurrent.futures.",
    )  # NVIDIA Modification
    group_coverage = parser.add_argument_group("coverage options")
    group_coverage.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    group_coverage.add_argument("--coverage-branch", action="store_true", help="Run tests with branch coverage")
    group_coverage.add_argument(
        "--coverage-html",
        metavar="DIR",
        help="Generate coverage HTML report",
        default=os.path.join(START_DIRECTORY, "..", "..", "htmlcov"),
    )
    group_coverage.add_argument("--coverage-xml", metavar="FILE", help="Generate coverage XML report")
    group_coverage.add_argument(
        "--coverage-fail-under", metavar="MIN", type=float, help="Fail if coverage percentage under min"
    )
    group_warp = parser.add_argument_group("NVIDIA Warp options")  # NVIDIA Modification
    group_warp.add_argument(
        "--no-shared-cache", action="store_true", help="Use a separate kernel cache per test process."
    )
    group_warp.add_argument("--warp-debug", action="store_true", help="Set warp.config.mode to 'debug'")
    # NVIDIA Modification: distributed multi-GPU testing options
    group_dist = parser.add_argument_group("distributed GPU options")
    group_dist.add_argument(
        "--gpus",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Distribute whole test modules across N GPUs, pinning each module to a single GPU "
            "(all of a module's tests run on the same GPU). The effective GPU count is "
            "min(N, GPUs reported by rocm-smi). Use 0 or a negative value for all detected GPUs. "
            "When omitted, the legacy single-pool behavior is used."
        ),
    )
    group_dist.add_argument(
        "--gpu-ids",
        metavar="IDS",
        default=None,
        help=(
            "Comma-separated physical GPU ids to use for distribution (e.g. '0,2,3'), overriding "
            "automatic selection. Still capped by --gpus when both are given."
        ),
    )
    group_dist.add_argument(
        "--jobs-per-gpu",
        metavar="COUNT",
        type=int,
        default=0,
        help="Number of test processes per GPU (default 0: derive from --jobs/--maxjobs and the GPU count).",
    )
    args = parser.parse_args(args=argv)

    if args.coverage_branch:
        args.coverage = args.coverage_branch

    if args.coverage and not COVERAGE_AVAILABLE:
        parser.exit(
            status=2, message="--coverage was used, but coverage was not found. Is it installed?\n"
        )  # NVIDIA Modification

    process_count = max(0, args.jobs)
    if process_count == 0:
        process_count = multiprocessing.cpu_count()
    process_count = min(process_count, args.maxjobs)  # NVIDIA Modification

    # NVIDIA Modification: when distributing whole modules across GPUs, discover
    # tests with a single-GPU device set so only ``cuda:0`` device variants are
    # generated. Each worker is pinned (masked) to exactly one physical GPU that
    # appears as ``cuda:0``, so this keeps discovery-time and worker-time test
    # method names consistent. Must be set before test discovery below.
    if args.gpus is not None:
        warp.tests.unittest_utils.test_mode = "basic"

    import warp as wp  # noqa: PLC0415 NVIDIA Modification

    # Clear the Warp cache (NVIDIA Modification).  Honor WARP_CACHE_ROOT
    # before the clear so concurrent worktrees pinned to the same Warp
    # version do not wipe each other's default cache.  Workers key on the
    # same env var.
    if "WARP_CACHE_ROOT" in os.environ:
        wp.config.kernel_cache_dir = os.environ["WARP_CACHE_ROOT"]

    wp.clear_lto_cache()
    wp.clear_kernel_cache()
    print(f"Main process cleared Warp kernel cache: {wp.config.kernel_cache_dir}")

    # Create the temporary directory (for coverage files)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Discover tests
        with _coverage(args, temp_dir):
            test_loader = unittest.TestLoader()
            if args.testNamePatterns:
                test_loader.testNamePatterns = args.testNamePatterns

            auto_discover_suite = warp.tests.unittest_suites.auto_discover_suite(
                test_loader, args.pattern
            )  # NVIDIA Modification

            # NVIDIA Modification
            if args.suite != "autodetect":
                # Print notices for test classes missing from the suite when compared to auto-discovered tests
                discover_suite = warp.tests.unittest_suites.compare_unittest_suites(
                    test_loader, args.suite, auto_discover_suite
                )
            else:
                discover_suite = auto_discover_suite

            # NVIDIA Modification: drop tests whose id matches any --exclude substring
            if args.exclude:
                filtered_suite = _filter_excluded_suite(discover_suite, args.exclude)
                discover_suite = filtered_suite if filtered_suite is not None else unittest.TestSuite()

        # Get the parallelizable test suites
        if args.level == "test":
            test_suites = list(_iter_test_cases(discover_suite))
        elif args.level == "class":
            test_suites = list(_iter_class_suites(discover_suite))
        else:  # args.level == 'module'
            test_suites = list(_iter_module_suites(discover_suite))

        # Don't use more processes than test suites
        process_count = max(1, min(len(test_suites), process_count))

        if args.gpus is not None and not args.serial_fallback:
            # NVIDIA Modification: distribute whole modules across GPUs, pinning
            # each module to a single GPU.
            print(
                f"Running {len(test_suites)} test suites ({discover_suite.countTestCases()} total tests) "
                f"distributed across GPUs",
                file=sys.stderr,
            )
            if args.verbose > 1:
                print(file=sys.stderr)

            start_time = time.perf_counter()
            results = _run_distributed(args, temp_dir, test_suites, process_count)
        elif not args.serial_fallback:
            # Report test suites and processes
            print(
                f"Running {len(test_suites)} test suites ({discover_suite.countTestCases()} total tests) across {process_count} processes",
                file=sys.stderr,
            )
            if args.verbose > 1:
                print(file=sys.stderr)

            # Create the shared index object used in Warp caches (NVIDIA Modification)
            manager = multiprocessing.Manager()
            shared_index = manager.Value("i", -1)

            # Run the tests in parallel
            start_time = time.perf_counter()

            if args.disable_concurrent_futures:
                multiprocessing_context = multiprocessing.get_context(method="spawn")
                maxtasksperchild = 1 if args.disable_process_pooling else None
                with multiprocessing_context.Pool(
                    process_count,
                    maxtasksperchild=maxtasksperchild,
                    initializer=initialize_test_process,
                    initargs=(manager.Lock(), shared_index, args, temp_dir),
                ) as pool:
                    test_manager = ParallelTestManager(manager, args, temp_dir)
                    results = pool.map(test_manager.run_tests, test_suites)
            else:
                # NVIDIA Modification: added concurrent.futures with crash handling and per-suite isolated fallback
                results = []
                parallel_failed = False
                parallel_fail_reason = "unknown"

                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=process_count,
                        mp_context=multiprocessing.get_context(method="spawn"),
                        initializer=initialize_test_process,
                        initargs=(manager.Lock(), shared_index, args, temp_dir),
                    ) as executor:
                        test_manager = ParallelTestManager(manager, args, temp_dir)
                        # Iterate results explicitly so we can report which suite timed out
                        for result in executor.map(test_manager.run_tests, test_suites, timeout=_SUITE_TIMEOUT):
                            results.append(result)

                except TimeoutError:
                    pending_index = len(results)
                    total = len(test_suites)
                    suite_name = _get_suite_name(test_suites[pending_index]) if pending_index < total else "unknown"
                    print(
                        f"Warning: Parallel execution timed out (total timeout={_SUITE_TIMEOUT}s). "
                        f"Next pending result was suite "
                        f"{pending_index + 1}/{total} ({suite_name}), "
                        f"but a different suite may be the actual blocker. "
                        f"Switching to isolated single-process fallback.",
                        file=sys.stderr,
                    )
                    parallel_failed = True
                    parallel_fail_reason = "timed out"
                except BrokenProcessPool:
                    # Process pool is broken - switch to isolated single-process fallback
                    print(
                        "Warning: Process pool broken during parallel execution. Switching to isolated single-process fallback.",
                        file=sys.stderr,
                    )
                    parallel_failed = True
                    parallel_fail_reason = "process pool broken"
                except Exception as e:
                    # Handle other pool-level exceptions
                    print(
                        f"Warning: Process pool error: {e}. Switching to isolated single-process fallback.",
                        file=sys.stderr,
                    )
                    parallel_failed = True
                    parallel_fail_reason = str(e)

                # Fallback to isolated single-process execution if parallel failed
                # Skip fallback in CI/CD environments to respect job timeouts
                in_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or os.environ.get("GITLAB_CI")
                if parallel_failed and in_ci:
                    parser.exit(
                        status=1,
                        message=f"Error: Parallel execution failed ({parallel_fail_reason}) in CI/CD environment. Skipping single-process fallback due to job timeout constraints.\n",
                    )
                elif parallel_failed:
                    print("Running all tests in isolated single-process mode...", file=sys.stderr)
                    # Run all test suites in isolated single-process pools
                    results = []
                    for i, suite in enumerate(test_suites):
                        try:
                            # Create a new single-process pool for each test suite
                            with concurrent.futures.ProcessPoolExecutor(
                                max_workers=1,
                                mp_context=multiprocessing.get_context(method="spawn"),
                                initializer=initialize_test_process,
                                initargs=(manager.Lock(), shared_index, args, temp_dir),
                            ) as executor:
                                test_manager = ParallelTestManager(manager, args, temp_dir)
                                future = executor.submit(test_manager.run_tests, suite)
                                try:
                                    result = future.result(timeout=_SUITE_TIMEOUT)
                                    results.append(result)
                                except TimeoutError:
                                    suite_name = _get_suite_name(suite)
                                    print(
                                        f"Warning: Isolated test suite {i + 1}/{len(test_suites)} ({suite_name}) timed out (timeout={_SUITE_TIMEOUT}s). Marking tests as crashed.",
                                        file=sys.stderr,
                                    )
                                    crash_result = create_crash_result(
                                        suite, reason=f"Process timed out (timeout={_SUITE_TIMEOUT}s)"
                                    )
                                    results.append(crash_result)
                                except BrokenProcessPool:
                                    print(
                                        f"Warning: Process crashed or was terminated unexpectedly in isolated execution for test suite {i + 1}/{len(test_suites)}. Marking tests as crashed.",
                                        file=sys.stderr,
                                    )
                                    crash_result = create_crash_result(suite)
                                    results.append(crash_result)
                                except Exception as e:
                                    print(
                                        f"Warning: Error in isolated test suite {i + 1}/{len(test_suites)}: {e}. Marking tests as crashed.",
                                        file=sys.stderr,
                                    )
                                    error_result = create_crash_result(suite)
                                    results.append(error_result)
                        except Exception as e:
                            print(
                                f"Warning: Failed to create isolated process for test suite {i + 1}/{len(test_suites)}: {e}. Marking tests as crashed.",
                                file=sys.stderr,
                            )
                            error_result = create_crash_result(suite)
                            results.append(error_result)
        else:
            # This entire path is an NVIDIA Modification

            # Report test suites and processes
            print(f"Running {discover_suite.countTestCases()} total tests (serial fallback)", file=sys.stderr)
            if args.verbose > 1:
                print(file=sys.stderr)

            if args.warp_debug:
                wp.config.mode = "debug"

            # Run the tests in serial
            start_time = time.perf_counter()

            with multiprocessing.Manager() as manager:
                test_manager = ParallelTestManager(manager, args, temp_dir)
                results = [test_manager.run_tests(discover_suite)]

        stop_time = time.perf_counter()
        test_duration = stop_time - start_time

        # Aggregate parallel test run results
        tests_run = 0
        errors = []
        failures = []
        skipped = 0
        expected_failures = 0
        unexpected_successes = 0
        test_records = []  # NVIDIA Modification
        for result in results:
            tests_run += result[0]
            errors.extend(result[1])
            failures.extend(result[2])
            skipped += result[3]
            expected_failures += result[4]
            unexpected_successes += result[5]
            test_records += result[6]  # NVIDIA Modification
        is_success = not (errors or failures or unexpected_successes)

        # Compute test info
        infos = []
        if failures:
            infos.append(f"failures={len(failures)}")
        if errors:
            infos.append(f"errors={len(errors)}")
        if skipped:
            infos.append(f"skipped={skipped}")
        if expected_failures:
            infos.append(f"expected failures={expected_failures}")
        if unexpected_successes:
            infos.append(f"unexpected successes={unexpected_successes}")

        # Report test errors
        if errors or failures:
            print(file=sys.stderr)
            for error in errors:
                print(error, file=sys.stderr)
            for failure in failures:
                print(failure, file=sys.stderr)
        elif args.verbose > 0:
            print(file=sys.stderr)

        # Test report
        print(unittest.TextTestResult.separator2, file=sys.stderr)
        print(f"Ran {tests_run} {'tests' if tests_run > 1 else 'test'} in {test_duration:.3f}s", file=sys.stderr)
        print(file=sys.stderr)
        print(f"{'OK' if is_success else 'FAILED'}{' (' + ', '.join(infos) + ')' if infos else ''}", file=sys.stderr)

        if test_records and args.junit_report_xml:
            # NVIDIA modification to report results in Junit XML format
            write_junit_results(
                args.junit_report_xml,
                test_records,
                tests_run,
                len(failures) + unexpected_successes,
                len(errors),
                skipped,
                test_duration,
            )

        # Coverage?
        # NVIDIA Modification: generate the coverage report BEFORE the failure
        # exit below. Coverage data is valid regardless of whether tests pass,
        # so a single flaky/timed-out test should not discard the whole run's
        # report (which is expensive to reproduce).
        if args.coverage:
            # Combine the coverage files
            cov_options = {}
            cov_options["config_file"] = True  # Grab configuration from pyproject.toml (must install coverage[toml])
            cov = coverage.Coverage(**cov_options)
            cov.combine(data_paths=[os.path.join(temp_dir, x) for x in os.listdir(temp_dir)])

            # Coverage report
            print(file=sys.stderr)
            percent_covered = cov.report(ignore_errors=True, file=sys.stderr)
            print(f"Total coverage is {percent_covered:.2f}%", file=sys.stderr)

            # HTML coverage report
            if args.coverage_html:
                cov.html_report(directory=args.coverage_html, ignore_errors=True)

            # XML coverage report
            if args.coverage_xml:
                cov.xml_report(outfile=args.coverage_xml, ignore_errors=True)

            # Fail under
            if args.coverage_fail_under and percent_covered < args.coverage_fail_under:
                parser.exit(status=2)

        # Return an error status on failure (after the coverage report above has
        # been written, so flaky failures still yield a usable report).
        if not is_success:
            parser.exit(status=len(errors) + len(failures) + unexpected_successes)


def _convert_select_pattern(pattern):
    if "*" not in pattern:
        return f"*{pattern}*"
    return pattern


@contextmanager
def _coverage(args, temp_dir):
    # Running tests with coverage?
    if args.coverage:
        # Generate a random coverage data file name - file is deleted along with containing directory
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as coverage_file:
            pass

        # Create the coverage object
        cov_options = {
            "branch": args.coverage_branch,
            "data_file": coverage_file.name,
            # NVIDIA Modification removed unneeded options
        }
        cov_options["config_file"] = True  # Grab configuration from pyproject.toml (must install coverage[toml])
        cov = coverage.Coverage(**cov_options)
        try:
            # Start measuring code coverage
            cov.start()

            # Yield for unit test running
            yield cov
        finally:
            # Stop measuring code coverage
            cov.stop()

            # Save the collected coverage data to the data file
            cov.save()
    else:
        # Not running tests with coverage - yield for unit test running
        yield None


# NVIDIA Modification: recursively rebuild a suite, dropping test cases whose id
# contains any of the given substrings. Preserves the nested module/class structure
# so class-level parallelism is unaffected. Returns None if nothing remains.
def _filter_excluded_suite(test_suite, patterns):
    if isinstance(test_suite, unittest.TestCase):
        test_id = test_suite.id()
        if any(pattern in test_id for pattern in patterns):
            return None
        return test_suite

    filtered = unittest.TestSuite()
    for child in test_suite:
        kept = _filter_excluded_suite(child, patterns)
        if kept is not None:
            filtered.addTest(kept)

    return filtered if filtered.countTestCases() > 0 else None


# Iterate module-level test suites - all top-level test suites returned from TestLoader.discover
def _iter_module_suites(test_suite):
    for module_suite in test_suite:
        if module_suite.countTestCases():
            yield module_suite


# Iterate class-level test suites - test suites that contains test cases
def _iter_class_suites(test_suite):
    has_cases = any(isinstance(suite, unittest.TestCase) for suite in test_suite)
    if has_cases:
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_class_suites(suite)


def _get_suite_name(test_suite):
    """Return a human-readable name for a test suite (e.g. 'TestTileMatmul')."""
    first_test = next(_iter_test_cases(test_suite), None)
    return type(first_test).__name__ if first_test is not None else "unknown"


# Iterate test cases (methods)
def _iter_test_cases(test_suite):
    if isinstance(test_suite, unittest.TestCase):
        yield test_suite
    else:
        for suite in test_suite:
            yield from _iter_test_cases(suite)


def create_crash_result(test_suite, reason="Process crashed or was terminated unexpectedly"):
    """Create a result indicating the process failed while running this test suite.

    This entire function is an NVIDIA modification.
    """
    test_count = test_suite.countTestCases()
    crash_errors = []

    # Create error entries for each test in the suite
    # Note: We don't know which specific test caused the failure, just that the process failed
    for test in _iter_test_cases(test_suite):
        error_msg = f"{reason} while running this test suite (unknown which test caused the failure): {test}"
        crash_errors.append(
            "\n".join(
                [
                    unittest.TextTestResult.separator1,
                    str(test),
                    unittest.TextTestResult.separator2,
                    error_msg,
                ]
            )
        )

    # Return the same format as run_tests: (test_count, errors, failures, skipped, expected_failures, unexpected_successes, test_records)
    return (test_count, crash_errors, [], 0, 0, 0, [])


# ---------------------------------------------------------------------------
# NVIDIA Modification: distributed multi-GPU testing helpers
# ---------------------------------------------------------------------------


def _detect_rocm_gpu_count():
    """Return the number of GPUs reported by ``rocm-smi``, or ``None`` if it
    cannot be determined (rocm-smi missing or unparseable output)."""
    exe = shutil.which("rocm-smi")
    if not exe:
        return None

    try:
        proc = subprocess.run(  # noqa: PLW1510
            [exe, "--showid"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        output = proc.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None

    # Match lines like "GPU[0] : Device Name: ...". Count the unique ordinals.
    ordinals = {int(m) for m in re.findall(r"GPU\[(\d+)\]", output)}
    return len(ordinals) if ordinals else None


def _detect_gpu_count():
    """Best-effort GPU count: prefer rocm-smi, fall back to Warp's device count."""
    count = _detect_rocm_gpu_count()
    if count is not None:
        return count

    try:
        import warp as wp  # noqa: PLC0415

        return wp.get_cuda_device_count()
    except Exception:
        return 0


def _resolve_gpu_ids(args):
    """Resolve the list of physical GPU ordinals to distribute tests across.

    Honors ``--gpu-ids`` when provided, otherwise selects ``0..effective-1``
    where ``effective = min(--gpus, detected)``. ``--gpus <= 0`` means "use all
    detected GPUs".
    """
    detected = _detect_gpu_count()

    if args.gpu_ids:
        ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
        if args.gpus is not None and args.gpus > 0:
            ids = ids[: args.gpus]
        return ids

    if not detected:
        # No GPUs detected: fall back to a single (masked) device so the run
        # still executes CPU tests deterministically.
        return [0]

    if args.gpus is None or args.gpus <= 0:
        effective = detected
    else:
        effective = min(args.gpus, detected)

    effective = max(1, effective)
    return list(range(effective))


def _module_group_key(test_suite):
    """Return the "module" grouping key for a class/test suite.

    A test living in a test subdirectory (e.g. ``warp.tests.geometry.test_bvh``)
    is grouped by its subdirectory (``geometry``). A top-level test file
    (e.g. ``warp.tests.test_codegen``) is grouped by its module name
    (``test_codegen``). This keeps whole modules together so they can be pinned
    to a single GPU.
    """
    test_case = next(_iter_test_cases(test_suite), None)
    if test_case is None:
        return "unknown"

    module_path = type(test_case).__module__
    parts = module_path.split(".")

    if "tests" in parts:
        rest = parts[parts.index("tests") + 1 :]
    else:
        rest = parts[-1:]

    if len(rest) >= 2 and rest[0] in _TEST_SUBDIR_MODULES:
        return rest[0]

    return rest[-1] if rest else "unknown"


def _group_suites_by_module(test_suites):
    """Group class/test suites by module key, preserving discovery order.

    Returns a dict mapping ``module_key -> list[suite]``.
    """
    groups = {}
    for suite in test_suites:
        key = _module_group_key(suite)
        groups.setdefault(key, []).append(suite)
    return groups


def _suite_weight(suites):
    return sum(suite.countTestCases() for suite in suites)


def _assign_groups_to_gpus(groups, gpu_ids):
    """Greedily bin-pack whole module groups onto GPUs to balance test counts.

    Groups are never split: every suite of a module is assigned to the same GPU.
    Returns ``(assignment, group_to_gpu)`` where ``assignment`` maps
    ``gpu_id -> list[suite]`` and ``group_to_gpu`` maps ``module_key -> gpu_id``.
    """
    loads = {gpu_id: 0 for gpu_id in gpu_ids}
    assignment = {gpu_id: [] for gpu_id in gpu_ids}
    group_to_gpu = {}

    # Assign heaviest modules first so lighter ones can fill the gaps.
    for key in sorted(groups, key=lambda k: _suite_weight(groups[k]), reverse=True):
        target = min(gpu_ids, key=lambda g: (loads[g], g))
        assignment[target].extend(groups[key])
        loads[target] += _suite_weight(groups[key])
        group_to_gpu[key] = target

    return assignment, group_to_gpu


def initialize_test_process_pinned(lock, shared_index, args, temp_dir, visible_devices):
    """Process initializer that pins the worker to specific GPU(s) before Warp
    is initialized, then defers to :func:`initialize_test_process`.

    ``visible_devices`` is a string like ``"1"`` or ``"0,1"`` and is applied via
    both ``HIP_VISIBLE_DEVICES`` (ROCm) and ``CUDA_VISIBLE_DEVICES`` (NVIDIA) so
    the same mechanism works on either backend. Because ``import warp`` does not
    initialize the runtime (device enumeration happens lazily in ``wp.init()``),
    setting these here guarantees the worker only sees the assigned GPU(s).
    """
    os.environ["HIP_VISIBLE_DEVICES"] = visible_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    # Force a single-GPU device set so device-parameterized tests only produce a
    # ``cuda:0`` variant (which, under masking, is the assigned physical GPU).
    # This keeps the generated test method names identical to those produced
    # during discovery in the main process (see ``main``), avoiding a mismatch
    # where a worker is asked to run a ``cuda:1`` test that does not exist in a
    # single-GPU-pinned process. Multi-GPU-only tests are unaffected: they are
    # not device-parameterized by ``get_test_devices`` and instead gate on
    # ``len(wp.get_cuda_devices()) > 1`` at runtime, which still holds for the
    # multi-GPU bucket (where more than one GPU is visible).
    warp.tests.unittest_utils.test_mode = "basic"

    initialize_test_process(lock, shared_index, args, temp_dir)


def _run_distributed(args, temp_dir, test_suites, process_count):
    """Run tests distributed across GPUs, pinning whole modules to a single GPU.

    Returns a flat list of per-suite result tuples in the same format as
    :meth:`ParallelTestManager.run_tests`, so the caller can aggregate them
    exactly like the non-distributed path.
    """
    gpu_ids = _resolve_gpu_ids(args)
    num_gpus = len(gpu_ids)

    groups = _group_suites_by_module(test_suites)

    # Split the multi-GPU-requiring modules into their own bucket.
    multi_gpu_suites = []
    normal_groups = {}
    for key, suites in groups.items():
        if key in _MULTI_GPU_MODULES:
            multi_gpu_suites.extend(suites)
        else:
            normal_groups[key] = suites

    assignment, group_to_gpu = _assign_groups_to_gpus(normal_groups, gpu_ids)

    # Derive per-GPU worker count.
    if args.jobs_per_gpu > 0:
        jobs_per_gpu = args.jobs_per_gpu
    else:
        jobs_per_gpu = max(1, process_count // num_gpus)

    # Report the plan.
    print(
        f"Distributing {len(normal_groups)} module(s) across {num_gpus} GPU(s) "
        f"{gpu_ids} with up to {jobs_per_gpu} process(es) per GPU.",
        file=sys.stderr,
    )
    for gpu_id in gpu_ids:
        module_keys = sorted(k for k, g in group_to_gpu.items() if g == gpu_id)
        test_count = _suite_weight(assignment[gpu_id])
        print(
            f"  GPU {gpu_id}: {len(module_keys)} module(s), {test_count} test(s) -> {module_keys}",
            file=sys.stderr,
        )
    if multi_gpu_suites:
        visible_multi = ",".join(str(g) for g in gpu_ids)
        print(
            f"  Multi-GPU bucket (HIP_VISIBLE_DEVICES={visible_multi}): "
            f"{sorted(_MULTI_GPU_MODULES)}, {_suite_weight(multi_gpu_suites)} test(s)",
            file=sys.stderr,
        )

    manager = multiprocessing.Manager()
    shared_index = manager.Value("i", -1)
    lock = manager.Lock()
    test_manager = ParallelTestManager(manager, args, temp_dir)

    # Build the list of pools to run concurrently: one per GPU plus an optional
    # multi-GPU bucket. Each entry is (visible_devices, suites, worker_count).
    pools = []
    for gpu_id in gpu_ids:
        suites = assignment[gpu_id]
        if suites:
            pools.append((str(gpu_id), suites, min(jobs_per_gpu, len(suites))))
    if multi_gpu_suites:
        visible_multi = ",".join(str(g) for g in gpu_ids)
        pools.append((visible_multi, multi_gpu_suites, min(jobs_per_gpu, len(multi_gpu_suites))))

    if not pools:
        return []

    def run_pool(visible_devices, suites, worker_count):
        pool_results = []
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context(method="spawn"),
                initializer=initialize_test_process_pinned,
                initargs=(lock, shared_index, args, temp_dir, visible_devices),
            ) as executor:
                for result in executor.map(test_manager.run_tests, suites, timeout=_SUITE_TIMEOUT):
                    pool_results.append(result)
        except Exception as exc:  # noqa: BLE001  (BrokenProcessPool / TimeoutError / pool errors)
            # Mark any suites that did not report a result as crashed so their
            # tests are surfaced as errors instead of silently vanishing.
            for suite in suites[len(pool_results) :]:
                pool_results.append(
                    create_crash_result(
                        suite,
                        reason=f"GPU pool (HIP_VISIBLE_DEVICES={visible_devices}) failed: {exc}",
                    )
                )
        return pool_results

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pools)) as driver:
        futures = [driver.submit(run_pool, *pool) for pool in pools]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    return results


class ParallelTestManager:
    def __init__(self, manager, args, temp_dir):
        self.args = args
        self.temp_dir = temp_dir
        self.failfast = manager.Event()

    def run_tests(self, test_suite):
        # Fail fast?
        if self.failfast.is_set():
            return [0, [], [], 0, 0, 0, []]  # NVIDIA Modification

        # NVIDIA Modification for GitLab
        warp.tests.unittest_utils.coverage_enabled = self.args.coverage
        warp.tests.unittest_utils.coverage_temp_dir = self.temp_dir
        warp.tests.unittest_utils.coverage_branch = self.args.coverage_branch

        if self.args.junit_report_xml:
            resultclass = ParallelJunitTestResult
        else:
            resultclass = ParallelTextTestResult

        # Run unit tests
        with _coverage(self.args, self.temp_dir):
            runner = unittest.TextTestRunner(
                stream=StringIO(),
                resultclass=resultclass,  # NVIDIA Modification
                verbosity=self.args.verbose,
                failfast=self.args.failfast,
                buffer=self.args.buffer,
            )
            result = runner.run(test_suite)

            # Set failfast, if necessary
            if result.shouldStop:
                self.failfast.set()

            # Return (test_count, errors, failures, skipped_count, expected_failure_count, unexpected_success_count)
            return (
                result.testsRun,
                [self._format_error(result, error) for error in result.errors],
                [self._format_error(result, failure) for failure in result.failures],
                len(result.skipped),
                len(result.expectedFailures),
                len(result.unexpectedSuccesses),
                result.test_record,  # NVIDIA modification
            )

    @staticmethod
    def _format_error(result, error):
        return "\n".join(
            [
                unittest.TextTestResult.separator1,
                result.getDescription(error[0]),
                unittest.TextTestResult.separator2,
                error[1],
            ]
        )


class ParallelTextTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        stream = type(stream)(sys.stderr)
        super().__init__(stream, descriptions, verbosity)
        self.test_record = []  # NVIDIA modification

    def startTest(self, test):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ...")
            self.stream.flush()
        super(unittest.TextTestResult, self).startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        # Force garbage collection of CPU-side allocations to reduce peak
        # host RSS in parallel test runs.
        import gc  # noqa: PLC0415

        gc.collect()

    def _add_helper(self, test, dots_message, show_all_message):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ... {show_all_message}")
        elif self.dots:
            self.stream.write(dots_message)
        self.stream.flush()

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        self._add_helper(test, ".", "ok")

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._add_helper(test, "E", "ERROR")

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._add_helper(test, "F", "FAIL")

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        self._add_helper(test, "s", f"skipped {reason!r}")

    def addExpectedFailure(self, test, err):
        super(unittest.TextTestResult, self).addExpectedFailure(test, err)
        self._add_helper(test, "x", "expected failure")

    def addUnexpectedSuccess(self, test):
        super(unittest.TextTestResult, self).addUnexpectedSuccess(test)
        self._add_helper(test, "u", "unexpected success")

    def printErrors(self):
        pass


def initialize_test_process(lock, shared_index, args, temp_dir):
    """Necessary operations to be executed at the start of every test process.

    Currently this function can be used to set a separate Warp cache. (NVIDIA modification)
    If the environment variable `WARP_CACHE_ROOT` is detected, the cache will be placed in the provided path.

    It also ensures that Warp is initialized prior to running any tests.
    """

    with lock:
        shared_index.value += 1
        worker_index = shared_index.value

    with _coverage(args, temp_dir):
        import warp as wp  # noqa: PLC0415

        if args.warp_debug:
            wp.config.mode = "debug"

        # init_kernel_cache() appends warp.config.version, so we set
        # kernel_cache_dir to a base path and let Warp add the version segment.
        if args.no_shared_cache:
            if "WARP_CACHE_ROOT" in os.environ:
                cache_root_dir = os.path.join(os.getenv("WARP_CACHE_ROOT"), f"worker-{worker_index:03d}")
            else:
                cache_root_dir = appdirs.user_cache_dir(
                    appname="warp", appauthor="NVIDIA", version=f"worker-{worker_index:03d}"
                )

            wp.config.kernel_cache_dir = cache_root_dir

            wp.clear_lto_cache()
            wp.clear_kernel_cache()
        elif "WARP_CACHE_ROOT" in os.environ:
            # Using a shared cache for all test processes
            wp.config.kernel_cache_dir = os.getenv("WARP_CACHE_ROOT")


if __name__ == "__main__":  # pragma: no cover
    main()
