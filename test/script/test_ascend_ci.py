import os
from pathlib import Path
import shutil
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

    def test_symmem_override_runs_unit_only_and_propagates_its_status(self):
        # The multi-rank phases stay skipped while the backend gaps they need
        # are unresolved, so only the single-process phase runs. A failing unit
        # phase must still fail the suite, otherwise the skip would turn a real
        # regression green.
        for unit_fails in (False, True):
            with self.subTest(unit_fails=unit_fails), \
                    tempfile.TemporaryDirectory() as tmp:
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
                         "FAILED_PHASE": "unit" if unit_fails else "none"},
                    capture_output=True, text=True,
                )
                # The unit status is propagated verbatim, so the original exit
                # code stays visible instead of being flattened to 1.
                self.assertEqual(result.returncode, 7 if unit_fails else 0,
                                 result.stdout + result.stderr)
                commands = log.read_text().splitlines()
                self.assertEqual(len(commands), 1, commands)
                self.assertIn("symmem_unit_tests", commands[0])
                # No multi-rank phase may be launched.
                self.assertNotIn("symmem_mpi_tests", log.read_text())
                self.assertNotIn("FLAGCX_CI_REQUIRE_NET_MR=1", log.read_text())
                self.assertNotIn("FLAGCX_CI_REQUIRE_LOCAL_IPC_NET_FALLBACK=1",
                                 log.read_text())
                # The skipped phases must still name what they wait on.
                self.assertIn("Ascend coverage gap", result.stdout)
                self.assertIn("flagcxProxyService", result.stdout)

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

    def test_runner_skips_only_the_affected_ascend_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ci = tmp / "ci"
            ci.mkdir()
            shutil.copy2(CI / "run_ascend_runner_mpi.sh",
                         ci / "run_ascend_runner_mpi.sh")
            self.script(ci / "run_mpi_with_timeout.sh",
                        'printf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
            tools = tmp / "tools"
            tools.mkdir()
            self.script(tools / "timeout", 'shift 2\nexec "$@"\n')
            binary = self.script(tmp / "runner_mpi_tests", '''
[[ "$1" == --gtest_list_tests ]]
printf 'FlagCXCollTest.\n  AllGather\n  AllReduce\n  AlltoAllvAsymmetric\n  Reduce\n  ReduceScatter\nOtherSuite.\n  OtherTest\n'
''')
            log = tmp / "commands"
            env = {**os.environ, "BASH_ENV": "", "COMMAND_LOG": str(log),
                   "PATH": f"{tools}:{os.environ['PATH']}"}
            for label in ("runner default", "runner heterogeneous",
                          "runner forced NET"):
                with self.subTest(label=label):
                    log.write_text("")
                    result = subprocess.run(
                        ["bash", str(ci / "run_ascend_runner_mpi.sh"),
                         "-np", "4", str(binary)],
                        env={**env, "FLAGCX_CI_MPI_LABEL": label},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)
                    commands = log.read_text()
                    self.assertIn("FlagCXCollTest.AllGather", commands)
                    self.assertNotIn("OtherTest", commands)
                    if label == "runner default":
                        self.assertNotIn("FlagCXCollTest.AlltoAllvAsymmetric",
                                         commands)
                        for test in ("AllReduce", "Reduce", "ReduceScatter"):
                            self.assertIn(f"FlagCXCollTest.{test}", commands)
                    else:
                        self.assertIn("FlagCXCollTest.AlltoAllvAsymmetric",
                                      commands)
                        for test in ("AllReduce", "Reduce", "ReduceScatter"):
                            self.assertNotIn(f"FlagCXCollTest.{test}", commands)
                    self.assertIn("Skipping", result.stdout)

    def test_rma_unit_failure_does_not_hide_ipc_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scripts = tmp / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            self.script(scripts / "run_with_timeout.sh", 'exec "$@"\n')
            self.script(scripts / "run_ascend_diagnostics.sh", 'exit 0\n')
            self.script(
                scripts / "filter_supplemental_tests.sh",
                'printf "%s\\n" "$1"\n',
            )
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
            for suite in ("p2p", "torch-api"):
                with self.subTest(suite=suite):
                    result = subprocess.run(
                        ["bash", str(CI / "run_ascend_diagnostics.sh"), suite],
                        env={**os.environ, "BASH_ENV": "", "FLAGCX_CI_LOG_DIR": tmp},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("No SIGSEGV", result.stdout)
                    self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_torch_backtrace_worker_saves_rank_specific_native_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tools = tmp / "tools"
            tools.mkdir()
            self.script(tools / "gdb", '''
printf '%s\\n' 'Program received signal SIGSEGV' '#0 flagcxCommInitRank()'
exit 11
''')
            result = subprocess.run(
                ["python3", str(CI / "torch_rank_backtrace.py"), "/tmp/example.py"],
                env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
                     "RANK": "2", "LOCAL_RANK": "2",
                     "FLAGCX_CI_DIAGNOSTIC_DIR": str(tmp / "traces")},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            trace = (tmp / "traces/rank-2.log").read_text()
            self.assertIn("rank=2 local_rank=2", trace)
            self.assertIn("#0 flagcxCommInitRank()", trace)
            self.assertIn("gdb exit=139", trace)

            propagated = subprocess.run(
                ["python3", str(CI / "torch_rank_backtrace.py"), "/tmp/example.py"],
                env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
                     "RANK": "2", "LOCAL_RANK": "2",
                     "FLAGCX_CI_DIAGNOSTIC_DIR": str(tmp / "traces"),
                     "FLAGCX_TORCH_GDB_PROPAGATE_FAILURE": "1"},
                capture_output=True, text=True,
            )
            self.assertEqual(propagated.returncode, 139, propagated.stderr)

            self.script(tools / "gdb", '''
printf '%s\\n' 'Thread 1 "python3" received signal SIGSEGV, Segmentation fault.'
exit 0
''')
            crashed = subprocess.run(
                ["python3", str(CI / "torch_rank_backtrace.py"), "/tmp/example.py"],
                env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
                     "RANK": "2", "LOCAL_RANK": "2",
                     "FLAGCX_CI_DIAGNOSTIC_DIR": str(tmp / "traces"),
                     "FLAGCX_TORCH_GDB_PROPAGATE_FAILURE": "1"},
                capture_output=True, text=True,
            )
            self.assertEqual(crashed.returncode, 139, crashed.stderr)

    def test_torch_diagnostics_count_gdb_thread_sigsegv_stacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "logs").mkdir()
            (tmp / "logs/attempt-1.log").write_text("Signal 11 (SIGSEGV)\n")
            self.script(tmp / "gdb", "exit 0\n")
            test_script = tmp / "test/script/torch_api_test.sh"
            test_script.parent.mkdir(parents=True)
            self.script(test_script, '''
for rank in 0 1 2 3; do
  printf '%s\\n' 'Thread 1 "python3" received signal SIGSEGV, Segmentation fault.' \\
    > "$FLAGCX_CI_DIAGNOSTIC_DIR/rank-$rank.log"
done
''')
            result = subprocess.run(
                ["bash", str(CI / "run_ascend_diagnostics.sh"), "torch-api"],
                env={**os.environ, "BASH_ENV": "", "GITHUB_WORKSPACE": str(tmp),
                     "FLAGCX_CI_LOG_DIR": str(tmp / "logs"),
                     "PATH": f"{tmp}:{os.environ['PATH']}"},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("4/4 rank logs, 4 SIGSEGV stacks", result.stdout)



if __name__ == "__main__":
    unittest.main()
