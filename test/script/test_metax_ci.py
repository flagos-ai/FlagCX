"""Hardware-free checks for the MetaX perf CI gates."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PREFLIGHT = Path(__file__).resolve().parents[2] / ".github/scripts/ci/check_metax_devices.py"
PERF_SUITE = Path(__file__).resolve().parents[2] / ".github/scripts/ci/run_host_perf_suite.sh"
METAX_ENV = PREFLIGHT.parent.parent / "set_env/metax.sh"


def load_preflight():
    spec = importlib.util.spec_from_file_location("metax_preflight_test", PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetaXPreflightTest(unittest.TestCase):
    def test_native_count_below_four_fails_perf(self):
        preflight = load_preflight()
        with mock.patch.object(sys, "argv", [str(PREFLIGHT), "perf"]), \
             mock.patch.object(preflight, "runtime_device_count", return_value=3), \
             mock.patch.object(preflight, "check_mccl_alltoall"), \
             mock.patch.object(preflight.shutil, "which", return_value=None):
            self.assertEqual(preflight.main(), 1)

    def test_missing_mccl_alltoall_fails_perf(self):
        preflight = load_preflight()
        with mock.patch.object(sys, "argv", [str(PREFLIGHT), "perf"]), \
             mock.patch.object(preflight, "runtime_device_count", return_value=4), \
             mock.patch.object(preflight, "check_mccl_alltoall",
                               side_effect=RuntimeError("symbol missing")), \
             mock.patch.object(preflight.shutil, "which", return_value=None):
            self.assertEqual(preflight.main(), 1)


class MetaXPerfSuiteTest(unittest.TestCase):
    def test_platform_skips_reductions_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy(PERF_SUITE, scripts / PERF_SUITE.name)
            shutil.copy(PERF_SUITE.parent / "ignore_test.py", scripts / "ignore_test.py")
            runner = scripts / "run_mpi_with_timeout.sh"
            runner.write_text(
                '#!/usr/bin/env bash\n'
                'printf "%s|%s\\n" "$FLAGCX_CI_MPI_LABEL" "$*" '
                '>> "$FLAGCX_CI_TEST_RUNNER_LOG"\n'
                'case "$FLAGCX_CI_MPI_LABEL" in '
                '*": $FLAGCX_CI_TEST_FAIL_OPERATION") exit 7;; esac\n'
            )
            runner.chmod(0o755)
            binaries = root / "bin"
            binaries.mkdir()
            for operation in ("alltoall", "alltoallv", "sendrecv", "allreduce",
                              "allgather", "reducescatter", "broadcast", "gather",
                              "scatter", "reduce"):
                binary = binaries / f"perf_{operation}"
                binary.write_text("#!/usr/bin/env bash\n")
                binary.chmod(0o755)
            calls = root / "calls.log"
            env = dict(os.environ,
                       FLAGCX_CI_PERF_PLATFORM="MetaX",
                       FLAGCX_CI_PERF_CONTINUE_ON_FAILURE="1",
                       MPI_HOME=str(root / "mpi"),
                       FLAGCX_CI_TEST_RUNNER_LOG=str(calls))
            for name in ("FLAGCX_CI_IGNORE_TEST", "FLAGCX_CI_PERF_SKIP_OPERATIONS",
                         "FLAGCX_CI_PERF_NP", "FLAGCX_CI_PERF_VERIFY",
                         "FLAGCX_CI_HETERO_SENDRECV_MAX_BYTES"):
                env.pop(name, None)
            for failed_operation in ("", "allgather", "alltoallv"):
                with self.subTest(failed_operation=failed_operation):
                    calls.write_text("")
                    env["FLAGCX_CI_TEST_FAIL_OPERATION"] = failed_operation
                    for mode, last_operation in (("homogeneous", "reduce"),
                                                 ("heterogeneous", "scatter")):
                        result = subprocess.run(
                            ["bash", "-e", "-c", 'source "$1"; bash "$2" "$3" "$4"',
                             "bash", str(METAX_ENV), str(scripts / PERF_SUITE.name),
                             mode, str(binaries)],
                            env=env, capture_output=True, text=True, check=False,
                        )
                        failed = bool(failed_operation)
                        self.assertEqual(result.returncode, int(failed),
                                         result.stdout + result.stderr)
                        if mode == "homogeneous":
                            for operation in ("allreduce", "reducescatter"):
                                self.assertIn(
                                    f"[PERF RESULT] MetaX homogeneous {operation}: SKIP (configured)",
                                    result.stdout,
                                )
                            self.assertEqual(result.stdout.count("SKIP"), 2)
                            self.assertIn("skipped=allreduce reducescatter", result.stdout)
                        else:
                            self.assertNotIn("SKIP", result.stdout)
                        self.assertIn(
                            f"[PERF RESULT] MetaX {mode} {last_operation}: PASS",
                            result.stdout,
                        )
                        self.assertIn(f"[PERF SUMMARY] MetaX {mode}:", result.stdout)
                        if failed:
                            self.assertIn(
                                f"[PERF RESULT] MetaX {mode} {failed_operation}: FAIL",
                                result.stderr,
                            )
                    records = [line.split("|", 1)
                               for line in calls.read_text().splitlines()]
                    labels = [label for label, _ in records]
                    self.assertEqual(len(labels), 15)
                    for operation in ("allreduce", "reducescatter"):
                        self.assertNotIn(f"MetaX homogeneous perf: {operation}", labels)
                    label = "MetaX homogeneous perf: allgather"
                    self.assertEqual(labels.count(label), 1)
                    arguments = records[labels.index(label)][1].split()
                    self.assertEqual(
                        arguments[:3], ["-np", "4", "--allow-run-as-root"],
                    )
                    self.assertEqual(
                        arguments[-8:],
                        ["-b", "128M", "-e", "1G", "-f", "2", "-p", "1"],
                    )
                    self.assertIn("MetaX heterogeneous perf: scatter", labels)


if __name__ == "__main__":
    unittest.main()
