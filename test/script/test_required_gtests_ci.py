"""The PR 2 hardware gate must reject missing, skipped, and failed cases."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / ".github/scripts/ci/check_required_gtests.py"
SPEC = importlib.util.spec_from_file_location("check_required_gtests", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RequiredGtestsCiTest(unittest.TestCase):
    def test_each_rank_must_run_all_required_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "rank-0.xml"
            report.write_text(
                '<testsuites><testsuite name="RmaTest">'
                '<testcase name="GetAsyncOffsets" status="run" '
                'result="completed"/>'
                '<testcase name="IpcGetCounterFollowsCopy" status="run" '
                'result="completed"/>'
                '</testsuite></testsuites>'
            )
            required = ["RmaTest.GetAsyncOffsets", "RmaTest.IpcGetCounterFollowsCopy"]
            self.assertEqual(MODULE.check_file(report, required), [])
            report.write_text(
                '<testsuites><testsuite name="RmaTest">'
                '<testcase name="GetAsyncOffsets" status="run" '
                'result="completed"/>'
                '<testcase name="IpcGetCounterFollowsCopy" status="run" '
                'result="skipped"><skipped/></testcase>'
                '</testsuite></testsuites>'
            )
            self.assertIn("skipped or not run", MODULE.check_file(report, required)[0])
            report.unlink()
            self.assertIn("cannot read", MODULE.check_file(report, required)[0])

    def test_duplicate_and_failed_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "adaptor.xml"
            report.write_text(
                '<testsuites><testsuite name="DeviceAdaptorTest">'
                '<testcase name="HostRegisterUnregister" status="run" '
                'result="completed"><failure/></testcase>'
                '<testcase name="StreamWaitWriteValue64" status="run" '
                'result="completed"/>'
                '<testcase name="StreamWaitWriteValue64" status="run" '
                'result="completed"/>'
                '</testsuite></testsuites>'
            )
            failures = MODULE.check_file(
                report,
                ["DeviceAdaptorTest.HostRegisterUnregister",
                 "DeviceAdaptorTest.StreamWaitWriteValue64",
                 "NetAdaptorLoopback.RegisterDmaBufMr"],
            )
            self.assertEqual(len(failures), 3)
            self.assertTrue(any("failed" in failure for failure in failures))
            self.assertTrue(any("expected once, found 2" in failure for failure in failures))
            self.assertTrue(any("expected once, found 0" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
