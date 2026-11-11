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
    def test_skips_failing_reductions_and_reports_all_other_operations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy(PERF_SUITE, scripts / PERF_SUITE.name)
            runner = scripts / "run_mpi_with_timeout.sh"
            runner.write_text(
                '#!/usr/bin/env bash\n'
                'echo "$FLAGCX_CI_MPI_LABEL" >> "$FLAGCX_CI_TEST_RUNNER_LOG"\n'
                'case "$FLAGCX_CI_MPI_LABEL" in *alltoallv|*gather) exit 7;; esac\n'
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
                       FLAGCX_CI_PERF_SKIP_OPERATIONS="allreduce reducescatter",
                       FLAGCX_CI_PERF_CONTINUE_ON_FAILURE="1",
                       FLAGCX_CI_TEST_RUNNER_LOG=str(calls))
            for mode, last_operation in (("homogeneous", "reduce"),
                                         ("heterogeneous", "scatter")):
                result = subprocess.run(
                    ["bash", str(scripts / PERF_SUITE.name), mode, str(binaries)],
                    env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"[PERF RESULT] MetaX {mode} alltoallv: FAIL", result.stderr)
                self.assertIn(f"[PERF RESULT] MetaX {mode} {last_operation}: PASS",
                              result.stdout)
                self.assertIn(f"[PERF SUMMARY] MetaX {mode}:", result.stdout)
            labels = calls.read_text().splitlines()
            self.assertEqual(len(labels), 15)
            self.assertNotIn("MetaX homogeneous perf: allreduce", labels)
            self.assertNotIn("MetaX homogeneous perf: reducescatter", labels)
            self.assertIn("MetaX heterogeneous perf: scatter", labels)


if __name__ == "__main__":
    unittest.main()
