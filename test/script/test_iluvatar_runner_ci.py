"""Exercise runner isolation, timeout diagnostics and failure propagation."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"


def require_linux(reason):
    """Let a developer machine skip, but never let CI skip.

    Hardware CI runs Linux, so a missing prerequisite there is a real defect
    rather than a platform difference. Raising keeps these cases from quietly
    turning green on the runners that actually gate the merge.
    """
    if reason and sys.platform.startswith("linux"):
        raise AssertionError(f"Linux CI must satisfy this prerequisite: {reason}")
    return reason


def timeout_wrapper_unsupported():
    """Report why run_with_timeout.sh cannot run here, or None when it can.

    Skip with a stated reason instead of failing: an unexplained red result on
    a developer machine teaches everyone to ignore this file, which costs more
    coverage than the skip does.
    """
    reason = None
    if shutil.which("timeout") is None:
        reason = "GNU timeout is not installed (macOS ships no coreutils timeout)"
    else:
        probe = subprocess.run(
            ["bash", "-c", "exec {fd}>/dev/null"], capture_output=True, text=True
        )
        if probe.returncode != 0:
            reason = (
                "run_with_timeout.sh needs the automatic file-descriptor "
                "allocation added in Bash 4.1; this bash is older"
            )
    return require_linux(reason)


def requires_proc():
    """collect_process_stacks.py reads /proc, which only Linux provides."""
    if Path("/proc").is_dir():
        return None
    return require_linux("/proc is not available")


class IluvatarRunnerCiTest(unittest.TestCase):
    def script(self, path, body):
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)
        return path

    def test_timeout_does_not_hide_failures_or_prevent_later_cases(self):
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
        reason = requires_proc()
        if reason:
            self.skipTest(reason)
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

    def test_every_suite_is_discovered_and_each_case_is_bounded(self):
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = self.script(root / "runner_mpi_tests", '''
[[ "$1" == --gtest_list_tests ]]
printf 'FlagCXCollTest.\n  AllReduce\nRunnerResult.\n  KeepsResult\n'
''')
            launcher = self.script(root / "mpirun", '''
printf '%s\n' "$*" >> "$COMMAND_LOG"
''')
            result = subprocess.run(
                ["bash", str(CI / "run_iluvatar_runner_mpi.sh"),
                 "-np", "4", str(binary)],
                env={**os.environ, "BASH_ENV": "", "COMMAND_LOG": str(root / "calls"),
                     "FLAGCX_CI_MPIRUN": str(launcher),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "FLAGCX_CI_RUNNER_PER_TEST_TIMEOUT": "7m"},
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # A suite other than FlagCXCollTest must not be dropped silently.
            self.assertIn("FlagCXCollTest.AllReduce result=0", result.stdout)
            self.assertIn("RunnerResult.KeepsResult result=0", result.stdout)
            # Each case carries its own bound instead of the 20m suite default.
            self.assertEqual(result.stdout.count("timeout=7m"), 2)

    def test_stage_budget_stops_the_stage_and_names_the_cases_left(self):
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = self.script(root / "runner_mpi_tests", '''
[[ "$1" == --gtest_list_tests ]]
printf 'FlagCXCollTest.\n  Slow\nRunnerResult.\n  NeverReached\n'
''')
            launcher = self.script(root / "mpirun", '''
printf '%s\n' "$*" >> "$COMMAND_LOG"
[[ "$*" != *Slow* ]] || sleep 2
''')
            result = subprocess.run(
                ["bash", str(CI / "run_iluvatar_runner_mpi.sh"),
                 "-np", "4", str(binary)],
                env={**os.environ, "BASH_ENV": "", "COMMAND_LOG": str(root / "calls"),
                     "FLAGCX_CI_MPIRUN": str(launcher),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "FLAGCX_CI_RUNNER_STAGE_BUDGET": "1"},
                capture_output=True, text=True, timeout=20,
            )
            # Running out of budget is a failure, never a quiet pass.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exhausted its 1s stage budget", result.stderr)
            self.assertIn("did not run: RunnerResult.NeverReached", result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertTrue(any("Slow" in line for line in calls))
            self.assertFalse(any("NeverReached" in line for line in calls))

    def test_explicit_filter_is_used_for_discovery_and_preserved(self):
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
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
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
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
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
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
        reason = requires_proc()
        if reason:
            self.skipTest(reason)
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
