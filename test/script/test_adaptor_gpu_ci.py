"""Check that hardware CI rejects missing or skipped GPU adaptor cases."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / ".github/scripts/ci/check_adaptor_gpu_tests.py"
SPEC = importlib.util.spec_from_file_location("check_adaptor_gpu_tests", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AdaptorGpuCiTest(unittest.TestCase):
    def check(self, names, skipped=(), *, allow_gpu_mr_unsupported=False):
        cases = []
        for name in names:
            skipped_xml = (
                "<skipped>Selected net adaptor does not advertise GPU MR</skipped>"
                if name in skipped
                else ""
            )
            cases.append(
                f'<testcase name="{name}" status="run" result="completed">'
                f"{skipped_xml}</testcase>"
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            path.write_text(
                f'<testsuites><testsuite name="NetAdaptorLoopback">'
                f"{''.join(cases)}</testsuite></testsuites>"
            )
            return MODULE.check_results(
                path, allow_gpu_mr_unsupported=allow_gpu_mr_unsupported
            )

    def test_all_cases_pass(self):
        self.assertEqual(self.check(MODULE.REQUIRED_CASES), ([], []))

    def test_skipped_case_fails(self):
        failures, _ = self.check(MODULE.REQUIRED_CASES, skipped=("GpuIget",))
        self.assertEqual(failures, ["NetAdaptorLoopback.GpuIget: skipped or not run"])

    def test_explicit_unsupported_gpu_mr_skip_is_reported(self):
        failures, unsupported = self.check(
            MODULE.REQUIRED_CASES,
            skipped=MODULE.REQUIRED_CASES,
            allow_gpu_mr_unsupported=True,
        )
        self.assertEqual(failures, [])
        self.assertEqual(unsupported, list(MODULE.REQUIRED_CASES))

    def test_gtest_skip_message_attribute_is_recognized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            cases = []
            for name in MODULE.REQUIRED_CASES:
                skipped_xml = (
                    '<skipped message="Selected net adaptor does not advertise GPU MR"/>'
                    if name == "GpuIget"
                    else ""
                )
                cases.append(
                    f'<testcase name="{name}" status="run" result="completed">'
                    f"{skipped_xml}</testcase>"
                )
            path.write_text(
                '<testsuites><testsuite name="NetAdaptorLoopback">'
                f"{''.join(cases)}</testsuite></testsuites>"
            )
            failures, unsupported = MODULE.check_results(
                path, allow_gpu_mr_unsupported=True
            )
        self.assertEqual(failures, [])
        self.assertEqual(unsupported, ["GpuIget"])

    def test_other_skip_reason_fails_even_when_gpu_mr_is_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            cases = "".join(
                f'<testcase name="{name}" status="run" result="completed">'
                f'{"<skipped>some unrelated reason</skipped>" if name == "GpuIget" else ""}'
                "</testcase>"
                for name in MODULE.REQUIRED_CASES
            )
            path.write_text(
                '<testsuites><testsuite name="NetAdaptorLoopback">'
                f"{cases}</testsuite></testsuites>"
            )
            failures, _ = MODULE.check_results(path, allow_gpu_mr_unsupported=True)
        self.assertEqual(failures, ["NetAdaptorLoopback.GpuIget: skipped or not run"])

    def test_missing_case_fails(self):
        failures, unsupported = self.check(MODULE.REQUIRED_CASES[:-1])
        self.assertEqual(unsupported, [])
        self.assertEqual(
            failures, ["NetAdaptorLoopback.GpuBounds: expected once, found 0"]
        )


if __name__ == "__main__":
    unittest.main()
