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

    def test_rma_runner_and_symmem_use_shared_suite_runner(self):
        result = subprocess.run(
            ["bash", "-c",
             'set -e; source "$1"; '
             'if declare -F flagcx_ci_run_suite_override >/dev/null; then '
             'exit 1; fi',
             "bash", str(ROOT / ".github/scripts/set_env/ascend.sh")],
            env={**os.environ, "BASH_ENV": ""},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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
            shutil.copy2(CI / "ignore_test.py", ci / "ignore_test.py")
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
                   "PATH": f"{tools}:{os.environ['PATH']}",
                   "FLAGCX_CI_IGNORE_TEST": (
                       '{"runner":[{"test":"FlagCXCollTest.AlltoAllvAsymmetric",'
                       '"modes":["runner default"],"reason":"HCCL failure"},'
                       '{"test":"FlagCXCollTest.AllReduce",'
                       '"modes":["runner heterogeneous","runner forced NET"],'
                       '"reason":"C2C failure"},'
                       '{"test":"FlagCXCollTest.Reduce",'
                       '"modes":["runner heterogeneous","runner forced NET"],'
                       '"reason":"C2C failure"},'
                       '{"test":"FlagCXCollTest.ReduceScatter",'
                       '"modes":["runner heterogeneous","runner forced NET"],'
                       '"reason":"C2C failure"}]}'
                   )}
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
            shutil.copyfile(CI / "ignore_test.py", scripts / "ignore_test.py")
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
