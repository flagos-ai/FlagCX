"""Exercise runner isolation, timeout diagnostics and failure propagation."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"


class IluvatarRunnerCiTest(unittest.TestCase):
    def script(self, path, body):
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)
        return path

    def test_timeout_does_not_hide_failures_or_prevent_later_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = self.script(root / "runner_mpi_tests", '''
[[ "$1" == --gtest_list_tests ]]
printf 'FlagCXCollTest.\n  AllGather\n  SendRecv\n  AllReduce\n'
''')
            launcher = self.script(root / "mpirun", '''
printf '%s\n' "$*" >> "$COMMAND_LOG"
case "$*" in
  *FlagCXCollTest.AllGather*) exit 7 ;;
  *FlagCXCollTest.SendRecv*) exec sleep 10 ;;
esac
''')
            result = subprocess.run(
                ["bash", str(CI / "run_iluvatar_runner_mpi.sh"),
                 "-np", "4", str(binary)],
                env={**os.environ, "BASH_ENV": "", "COMMAND_LOG": str(root / "calls"),
                     "FLAGCX_CI_MPIRUN": str(launcher),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "FLAGCX_CI_RUNNER_SENDRECV_TIMEOUT": "0.4s",
                     "FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER": "0.03"},
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(sum("AllGather" in line for line in calls), 2)
            self.assertEqual(sum("SendRecv" in line for line in calls), 1)
            self.assertEqual(sum("AllReduce" in line for line in calls), 1)
            self.assertIn("FlagCXCollTest.SendRecv result=124", result.stdout)
            stacks = list((root / "logs").rglob("*-stacks.log"))
            self.assertEqual(len(stacks), 1)
            self.assertIn("Pre-timeout workload diagnostics", stacks[0].read_text())
            self.assertIn("name=sleep", stacks[0].read_text())

    def test_explicit_filter_is_used_for_discovery_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = self.script(root / "runner_mpi_tests", '''
[[ "$1" == --gtest_list_tests ]]
[[ "$2" == --gtest_filter=FlagCXCollTest.All*:FlagCXCollTest.SendRecv-FlagCXCollTest.AllGather ]]
printf 'FlagCXCollTest.\n  AllReduce\n'
''')
            launcher = self.script(root / "mpirun", '''
printf '%s\n' "$*" >> "$COMMAND_LOG"
''')
            result = subprocess.run(
                ["bash", str(CI / "run_iluvatar_runner_mpi.sh"),
                 "-np", "4", str(binary),
                 "--gtest_filter=FlagCXCollTest.All*:FlagCXCollTest.SendRecv-FlagCXCollTest.AllGather"],
                env={**os.environ, "BASH_ENV": "", "GTEST_FILTER": "unrelated.*",
                     "FLAGCX_CI_MPIRUN": str(launcher),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "COMMAND_LOG": str(root / "calls")},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("--gtest_filter=FlagCXCollTest.AllReduce", calls[0])
            self.assertNotIn("AllGather", calls[0])

    def test_timeout_result_is_not_changed_to_success_by_later_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = self.script(root / "runner_mpi_tests", '''
[[ "$2" == --gtest_filter=*-CoverageSupplemental.* ]]
printf 'FlagCXCollTest.\n  SendRecv\n  AllReduce\n'
''')
            launcher = self.script(root / "mpirun", '''
[[ "$*" != *SendRecv* ]] || exec sleep 10
''')
            result = subprocess.run(
                ["bash", str(CI / "run_iluvatar_runner_mpi.sh"), str(binary)],
                env={**os.environ, "BASH_ENV": "", "GTEST_FILTER": "*-CoverageSupplemental.*",
                     "FLAGCX_CI_MPIRUN": str(launcher),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "FLAGCX_CI_RUNNER_SENDRECV_TIMEOUT": "0.2s",
                     "FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER": "10"},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
            self.assertIn("FlagCXCollTest.AllReduce result=0", result.stdout)

    def test_default_timeout_wrapper_still_retries_and_reports_flaky(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.script(root / "flaky", '''
if [[ ! -f "$ATTEMPT_MARKER" ]]; then
  touch "$ATTEMPT_MARKER"
  exit 7
fi
''')
            result = subprocess.run(
                ["bash", str(CI / "run_with_timeout.sh"), str(command)],
                env={**os.environ, "BASH_ENV": "", "ATTEMPT_MARKER": str(root / "first"),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs")},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("classify as flaky", result.stdout)

    def test_stack_collector_does_not_inspect_unrelated_processes(self):
        child = subprocess.Popen(["sleep", "10"])
        unrelated = subprocess.Popen(["sleep", "10"])
        try:
            result = subprocess.run(
                [sys.executable, str(CI / "collect_process_stacks.py"),
                 "--pid", str(child.pid), "--delay", "0"],
                capture_output=True, text=True, timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"PID {child.pid} ", result.stdout)
            self.assertNotIn(f"PID {unrelated.pid} ", result.stdout)
        finally:
            child.terminate()
            unrelated.terminate()
            child.wait()
            unrelated.wait()


if __name__ == "__main__":
    unittest.main()
