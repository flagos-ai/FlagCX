"""Exercise lifecycle selection and rank-complete XML gating without SDKs."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("timeout"), "GNU timeout is required")
class LifecycleCiTest(unittest.TestCase):
    def run_wrapper(self, mode="pass", inherited="*", selected="*"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "mpirun"
            launcher.write_text('''#!/usr/bin/env python3
import os, subprocess, sys
n = int(sys.argv[2])
for rank in range(n):
    if os.getenv('TEST_XML_MODE') == 'missing' and rank == 1:
        continue
    result = subprocess.run(sys.argv[3:], env={**os.environ, 'OMPI_COMM_WORLD_RANK': str(rank)})
    if result.returncode:
        sys.exit(result.returncode)
''')
            launcher.chmod(0o755)
            binary = root / "runner_mpi_tests"
            binary.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
print('FILTER ' + ' '.join(sys.argv[1:]), flush=True)
report = os.getenv('GTEST_OUTPUT', '')
if report:
    skipped = os.getenv('TEST_XML_MODE') == 'skip' and os.getenv('OMPI_COMM_WORLD_RANK') == '1'
    case = '<testcase name="FinalizeThenRankLocalDestroy" status="run" result="%s"/>' % ('skipped' if skipped else 'completed')
    Path(report.removeprefix('xml:')).write_text('<testsuites><testsuite name="FlagCXCollTest">' + case + '</testsuite></testsuites>')
'''.replace("report.removeprefix('xml:')", "report[4:]"))
            binary.chmod(0o755)
            return subprocess.run(
                ["bash", str(ROOT / ".github/scripts/ci/run_mpi_with_timeout.sh"),
                 "-np", "2", str(binary), "--gtest_filter=" + selected],
                env={**os.environ, "FLAGCX_CI_MPIRUN": str(launcher),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "FLAGCX_CI_TEST_DIAGNOSTICS_AFTER": "0",
                     "GTEST_FILTER": inherited, "TEST_XML_MODE": mode},
                text=True, capture_output=True, timeout=20,
            )

    def test_every_rank_must_report_success(self):
        self.assertEqual(self.run_wrapper().returncode, 0)
        for mode in ("missing", "skip"):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.run_wrapper(mode).returncode, 0)

    def test_platform_exclusion_survives_an_explicit_pxn_filter(self):
        case = "FlagCXCollTest.FinalizeThenRankLocalDestroy"
        result = self.run_wrapper("missing", "*-" + case, case)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--gtest_filter=" + case + "-" + case, result.stdout)
        self.assertNotIn("expected 2 lifecycle XML", result.stderr)


if __name__ == "__main__":
    unittest.main()
