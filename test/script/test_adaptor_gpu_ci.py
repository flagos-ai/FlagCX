"""Check that hardware CI rejects missing or skipped GPU adaptor cases."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / ".github/scripts/ci/check_adaptor_gpu_tests.py"
SPEC = importlib.util.spec_from_file_location("check_adaptor_gpu_tests", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GPU_MR_SKIP = "<skipped>Selected net adaptor does not advertise GPU MR</skipped>"
GPU_MR_SKIP_ATTR = (
    '<skipped message="Selected net adaptor does not advertise GPU MR"/>'
)
UNRELATED_SKIP = "<skipped>some unrelated reason</skipped>"


def build_results(path, names, case_markup):
    """Write a GoogleTest XML report for the NetAdaptorLoopback suite.

    GoogleTest records GTEST_SKIP as result="skipped" with a <skipped> child,
    never as result="completed", while a failing case stays result="completed"
    with a <failure> child. Keep the fixture faithful to that format so the
    checker is exercised against the XML hardware CI actually produces.
    """
    cases = []
    for name in names:
        markup = case_markup.get(name, "")
        result = "skipped" if "<skipped" in markup else "completed"
        cases.append(
            f'<testcase name="{name}" status="run" result="{result}">'
            f"{markup}</testcase>"
        )
    path.write_text(
        '<testsuites><testsuite name="NetAdaptorLoopback">'
        f"{''.join(cases)}</testsuite></testsuites>"
    )


class AdaptorGpuCiTest(unittest.TestCase):
    def check(
        self,
        names,
        skipped=(),
        *,
        allow_gpu_mr_unsupported=False,
        markup=GPU_MR_SKIP,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            build_results(path, names, {name: markup for name in skipped})
            return MODULE.check_results(
                path,
                allow_gpu_mr_unsupported=allow_gpu_mr_unsupported,
            )

    def test_all_cases_pass(self):
        self.assertEqual(self.check(MODULE.REQUIRED_CASES), ([], []))

    def test_skipped_case_fails(self):
        failures, _ = self.check(
            MODULE.REQUIRED_CASES,
            skipped=("GpuIget",),
            markup=UNRELATED_SKIP,
        )
        self.assertEqual(failures, ["NetAdaptorLoopback.GpuIget: skipped or not run"])

    def test_runtime_confirmed_gpu_mr_capability_skip_is_reported(self):
        failures, unsupported = self.check(
            MODULE.REQUIRED_CASES,
            skipped=MODULE.REQUIRED_CASES,
            allow_gpu_mr_unsupported=True,
        )
        self.assertEqual(failures, [])
        self.assertEqual(unsupported, list(MODULE.REQUIRED_CASES))

    def test_iluvatar_policy_allows_only_gpu_mr_capability_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            for markup, expected in (
                (GPU_MR_SKIP, 0),
                (UNRELATED_SKIP, 1),
                ('<failure message="boom"/>', 1),
            ):
                with self.subTest(markup=markup):
                    build_results(
                        path, MODULE.REQUIRED_CASES,
                        {name: markup for name in MODULE.REQUIRED_CASES},
                    )
                    result = subprocess.run(
                        [sys.executable, str(CHECKER), str(path), "iluvatar"],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if expected == 0:
                        self.assertIn("skipped after runtime capability check", result.stdout)

    def test_gtest_skip_message_attribute_is_recognized(self):
        failures, unsupported = self.check(
            MODULE.REQUIRED_CASES,
            skipped=("GpuIget",),
            allow_gpu_mr_unsupported=True,
            markup=GPU_MR_SKIP_ATTR,
        )
        self.assertEqual(failures, [])
        self.assertEqual(unsupported, ["GpuIget"])

    def test_other_skip_reason_fails(self):
        failures, _ = self.check(
            MODULE.REQUIRED_CASES,
            skipped=("GpuIget",),
            allow_gpu_mr_unsupported=True,
            markup=UNRELATED_SKIP,
        )
        self.assertEqual(failures, ["NetAdaptorLoopback.GpuIget: skipped or not run"])

    def test_failed_case_fails_under_every_policy(self):
        failures, unsupported = self.check(
            MODULE.REQUIRED_CASES,
            skipped=("GpuIget",),
            markup='<failure message="boom"/>',
        )
        self.assertEqual(failures, ["NetAdaptorLoopback.GpuIget: failed"])
        self.assertEqual(unsupported, [])

    def test_missing_case_fails(self):
        failures, unsupported = self.check(MODULE.REQUIRED_CASES[:-1])
        self.assertEqual(unsupported, [])
        self.assertEqual(
            failures, ["NetAdaptorLoopback.GpuBounds: expected once, found 0"]
        )


if __name__ == "__main__":
    unittest.main()
