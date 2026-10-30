import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"


class AscendCiTest(unittest.TestCase):
    def script(self, path, body):
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)
        return path

    def test_symmem_net_runs_after_unit_or_ipc_failure(self):
        for failed in ("unit", "IPC", "local NET fallback", "NET MR", "none"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                log = tmp / "commands"
                runner = self.script(tmp / "runner", '''
label=${FLAGCX_CI_TEST_LABEL:-${FLAGCX_CI_MPI_LABEL:-}}
printf '%s %s\n' "$label" "$*" >> "$COMMAND_LOG"
[[ "$FAILED_PHASE" == none || "$label" != *"$FAILED_PHASE"* ]] || exit 7
''')
                result = subprocess.run(
                    ["bash", "-c", 'set -e; source "$1"; TEST_RUNNER="$2"; '
                     'MPI_RUNNER="$2"; flagcx_ci_run_suite_override symmem /suite',
                     "bash", str(ROOT / ".github/scripts/set_env/ascend.sh"), str(runner)],
                    env={**os.environ, "BASH_ENV": "", "COMMAND_LOG": str(log),
                         "FAILED_PHASE": failed},
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0 if failed == "none" else 1,
                                 result.stdout + result.stderr)
                commands = log.read_text().splitlines()
                self.assertEqual(len(commands), 4)
                self.assertIn("local NET fallback", commands[-2])
                self.assertIn("FLAGCX_CI_REQUIRE_LOCAL_IPC_NET_FALLBACK=1", commands[-2])
                self.assertIn("Ascend symmem NET MR and fallback", commands[-1])
                self.assertIn("FLAGCX_CI_REQUIRE_NET_MR=1", commands[-1])
                self.assertIn("FLAGCX_HOSTID=node0", commands[-1])
                self.assertIn("FLAGCX_HOSTID=node1", commands[-1])
                self.assertIn("RankLocalIpcFailureUsesNetworkMrFallback", commands[-2])
                self.assertNotIn("FLAGCX_VMM_ENABLE=1", log.read_text())

    def test_runner_isolates_collectives_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            binary = self.script(tmp / "runner_mpi_tests", '''
[[ "$1" == --gtest_list_tests ]]
printf 'FlagCXCollTest.\n  AllGather\n  AllReduce\nOtherSuite.\n  OtherTest\n'
''')
            launcher = self.script(tmp / "mpirun", '''
printf '%s\n' "$*" >> "$COMMAND_LOG"
[[ "$*" != *FlagCXCollTest.AllGather* ]] || exit 7
''')
            log = tmp / "commands"
            env = {**os.environ, "BASH_ENV": "", "COMMAND_LOG": str(log),
                   "FLAGCX_CI_MPIRUN": str(launcher), "FLAGCX_CI_LOG_DIR": str(tmp)}
            result = subprocess.run(
                ["bash", str(CI / "run_ascend_runner_mpi.sh"), "-np", "4", str(binary)],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertIn("FlagCXCollTest.AllGather", log.read_text())
            self.assertIn("FlagCXCollTest.AllReduce", log.read_text())
            self.assertNotIn("OtherTest", log.read_text())
            log.unlink()
            result = subprocess.run(
                ["bash", str(CI / "run_ascend_runner_mpi.sh"), "-np", "4", str(binary),
                 "--gtest_filter=FlagCXCollTest.AllReduce"],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(log.read_text().splitlines()), 1)

    def test_rma_unit_failure_does_not_hide_ipc_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scripts = tmp / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            self.script(scripts / "run_with_timeout.sh", 'exec "$@"\n')
            self.script(scripts / "run_ascend_diagnostics.sh", 'exit 0\n')
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            self.script(bin_dir / "cmake", 'exit 0\n')
            self.script(bin_dir / "make", '''
printf '%s\n' "$*" >> "$COMMAND_LOG"
[[ "$*" != *run-unit* ]] || exit 7
[[ "$*" != *run-mpi-ipc* ]] || exit 8
''')
            platform = self.script(tmp / "ascend.sh", '''
FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_ASCEND=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_ASCEND=1)
export MPI_HOME=/nonexistent
flagcx_ci_validate_rdma() { return 0; }
''')
            log = tmp / "commands"
            result = subprocess.run(
                ["bash", str(CI / "run_unit_test.sh"), str(platform), "rma"],
                env={**os.environ, "BASH_ENV": "", "GITHUB_WORKSPACE": str(tmp),
                     "COMMAND_LOG": str(log), "PATH": f"{bin_dir}:{os.environ['PATH']}"},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("run-mpi-ipc", log.read_text())
            self.assertIn("run-mpi-net", log.read_text())
            self.assertIn("unit=7 IPC=8 network=0", result.stderr)

    def test_no_crash_does_not_start_debugger(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(CI / "run_ascend_diagnostics.sh"), "p2p"],
                env={**os.environ, "BASH_ENV": "", "FLAGCX_CI_LOG_DIR": tmp},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No SIGSEGV", result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [])



if __name__ == "__main__":
    unittest.main()
