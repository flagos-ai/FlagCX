"""Check the standalone Iluvatar SendRecv probe suite and CI matrix."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"


class IluvatarSendRecvProbeCiTest(unittest.TestCase):
    @staticmethod
    def script(path, body):
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)
        return path

    def test_runs_full_matrix_after_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "run_iluvatar_sendrecv_probe.sh"
            probe.write_text((CI / probe.name).read_text())
            binary = self.script(root / "sendrecv_probe_mpi_tests", "exit 0\n")
            self.script(root / "run_mpi_with_timeout.sh", '''
printf '%s|%s\n' "$FLAGCX_CI_MPI_LABEL" "$*" >> "$COMMAND_LOG"
case "$FLAGCX_CI_MPI_LABEL" in
  'SendRecv probe P2P F EARLY') exit 1 ;;
  'SendRecv probe NET R EARLY') exit 124 ;;
esac
''')
            result = subprocess.run(
                ["bash", str(probe), str(binary), "4", "2", "0,3,4,7"],
                env={**os.environ, "BASH_ENV": "", "COMMAND_LOG": str(root / "calls"),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "FLAGCX_CI_ENABLE_PXN": "1", "FLAGCX_CI_INSTALL_GDB": "0"},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            summary = (root / "logs/sendrecv-probe-summary.tsv").read_text()
            self.assertEqual(len(summary.splitlines()), 23)
            self.assertIn("P2P\tF\tEARLY\t1", summary)
            self.assertIn("NET\tR\tEARLY\t124", summary)
            self.assertIn("PXN\tPXN_D\tLATE\t0", summary)
            self.assertIn("failed in 2 cases; all cases were attempted", result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(len(calls), 22)
            self.assertTrue(any("FLAGCX_P2P_DISABLE=0" in call for call in calls))
            self.assertTrue(any("FLAGCX_P2P_DISABLE=1" in call for call in calls))
            self.assertTrue(any("CUDA_VISIBLE_DEVICES=0,3,4,7" in call
                                for call in calls))

    def test_probe_is_available_but_runner_is_scheduled(self):
        config = (ROOT / ".github/configs/iluvatar.yml").read_text()
        suites = config.split("unit_test_suites:\n", 1)[1].strip().splitlines()
        self.assertEqual(suites, ["- runner"])
        self.assertTrue((ROOT / "test/unittest/sendrecv_probe/Makefile").exists())
        self.assertFalse(
            (ROOT / "test/unittest/runner/coll_sendrecv_probe.cpp").exists()
        )


if __name__ == "__main__":
    unittest.main()
