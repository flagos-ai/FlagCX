"""Validate shared lifecycle diagnostics without accelerator SDKs."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest

import yaml

from test_iluvatar_runner_ci import timeout_wrapper_unsupported


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"
PLATFORMS = yaml.safe_load((ROOT / ".github/configs/platforms.yml").read_text())["platforms"]


class LifecycleDiagnosticsTest(unittest.TestCase):
    def test_each_platform_collects_allgather_stacks_and_continues_after_timeout(self):
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
        for platform in PLATFORMS:
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root / "runner_mpi_tests"
                binary.write_text("#!/bin/sh\nprintf 'FlagCXCollTest.\\n  AllGather\\n  AllReduce\\n'\n")
                binary.chmod(0o755)
                launcher = root / "mpirun"
                launcher.write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$CALLS"
case "$*" in *AllGather*) exec sleep 10 ;; esac
printf '[  PASSED  ] 1 test.\\n'
''')
                launcher.chmod(0o755)
                result = subprocess.run(
                    ["bash", str(CI / "run_runner_mpi.sh"), "-np", "4", str(binary)],
                    env={**os.environ, "BASH_ENV": "", "GTEST_FILTER": "*",
                         "FLAGCX_CI_PLATFORM": platform, "FLAGCX_CI_MPIRUN": str(launcher),
                         "FLAGCX_CI_LOG_DIR": str(root / "logs"), "CALLS": str(root / "calls"),
                         "FLAGCX_CI_RUNNER_PER_TEST_TIMEOUT": "1s",
                         "FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER": "0.02",
                         "FLAGCX_CI_MPI_KILL_AFTER": "0.1s"},
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
                self.assertIn("AllGather result=124", result.stdout)
                self.assertIn("AllReduce result=0", result.stdout)
                calls = (root / "calls").read_text().splitlines()
                self.assertEqual(len(calls), 2)
                stacks = list((root / "logs").rglob("*-stacks.log"))
                self.assertTrue(stacks, result.stderr)
                self.assertTrue(any("Invocation: runner FlagCXCollTest.AllGather" in p.read_text()
                                    and "name=sleep" in p.read_text() for p in stacks))

    def test_default_sampler_preserves_nonzero_status_without_explicit_enablement(self):
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                ["bash", str(CI / "run_with_timeout.sh"), "bash", "-c", "sleep 1.4; exit 19"],
                env={**os.environ, "FLAGCX_CI_TEST_TIMEOUT": "1.6s",
                     "FLAGCX_CI_TEST_MAX_ATTEMPTS": "1",
                     "FLAGCX_CI_TEST_KILL_AFTER": "0.1s", "FLAGCX_CI_LOG_DIR": str(root / "logs")},
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
            self.assertTrue(list((root / "logs").rglob("*-stacks.log")))
            self.assertNotIn("watchdog expired", result.stderr)

    def test_sampler_failure_and_cancellation_keep_workload_status(self):
        reason = timeout_wrapper_unsupported()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("run_with_timeout.sh", "timeouts.sh"):
                shutil.copy2(CI / filename, root / filename)
            (root / "collect_process_stacks.py").write_text("raise RuntimeError('sampler unavailable')\n")
            env = {**os.environ, "FLAGCX_CI_TEST_DIAGNOSTICS_AFTER": "0",
                   "FLAGCX_CI_TEST_TIMEOUT": "4s", "FLAGCX_CI_TEST_KILL_AFTER": "0.1s",
                   "FLAGCX_CI_TEST_MAX_ATTEMPTS": "1", "FLAGCX_CI_LOG_DIR": str(root / "logs")}
            result = subprocess.run(
                ["bash", str(root / "run_with_timeout.sh"), "bash", "-c", "sleep 0.3; exit 19"],
                env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
            self.assertIn("sampler unavailable", result.stderr)
            process = subprocess.Popen(
                ["bash", str(CI / "run_with_timeout.sh"), "bash", "-c",
                 'touch "$READY"; exec sleep 10'],
                env=dict(env, READY=str(root / "ready")), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / "ready").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue((root / "ready").exists())
                process.send_signal(signal.SIGTERM)
                output, errors = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 143, output + errors)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    @unittest.skipUnless(shutil.which("mpic++") and shutil.which("mpirun"), "MPI toolchain required")
    def test_rank_phases_distinguish_completion_from_agreement_and_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "lifecycle.cpp"
            source.write_text('''
#include "comm_lifecycle_check.hpp"
static int rank, finalizes, destroys;
flagcxResult_t flagcxCommFinalize(flagcxComm_t) { return flagcxSuccess; }
flagcxResult_t flagcxCommDestroy(flagcxComm_t) { return flagcxSuccess; }
static flagcxResult_t finish(flagcxComm_t) {
  return ++finalizes == 1 && rank == 0 ? flagcxInProgress : flagcxSuccess;
}
static flagcxResult_t destroy(flagcxComm_t) {
  return ++destroys == 1 && rank == 0 ? flagcxSystemError : flagcxSuccess;
}
static flagcxResult_t synchronize(flagcxStream_t) { return flagcxSuccess; }
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  flagcxDeviceHandle device = {};
  device.streamSynchronize = synchronize;
  int bad = flagcxTestSynchronizeStream(&device, nullptr) != flagcxSuccess;
  flagcxTestLifecycleCalls calls{finish, destroy};
  flagcxComm_t comm = reinterpret_cast<flagcxComm_t>(1);
  bad |= flagcxTestFinalizeAndDestroyAcrossRanks(&comm, MPI_COMM_WORLD, calls) != flagcxSystemError;
  bad |= (rank == 0 ? comm == nullptr : comm != nullptr);
  bad |= flagcxTestFinalizeAndDestroyAcrossRanks(&comm, MPI_COMM_WORLD, calls) != flagcxSuccess;
  bad |= comm != nullptr || destroys != (rank == 0 ? 2 : 1);
  int common = 0;
  MPI_Allreduce(&bad, &common, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return common;
}
''')
            binary = root / "lifecycle"
            compiled = subprocess.run(
                ["mpic++", "-std=c++17", "-DOMPI_SKIP_MPICXX", "-DMPICH_SKIP_MPICXX",
                 f"-I{ROOT / 'test/fixtures'}", f"-I{ROOT / 'flagcx/include'}",
                 str(source), "-o", str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            for ranks in (2, 4):
                result = subprocess.run(
                    ["mpirun", "-np", str(ranks), str(binary)],
                    env={**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
                         "OMPI_MCA_rmaps_base_oversubscribe": "1", "OMPI_MCA_pml": "ob1",
                         "OMPI_MCA_btl": "self,tcp", "FLAGCX_CI_LIFECYCLE_CASE": "retry-fixture"},
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for rank in range(ranks):
                    for phase in ("stream", "finalize", "finalize-agree", "destroy", "destroy-agree"):
                        for edge in ("enter", "return"):
                            self.assertIn(f"rank={rank} case=retry-fixture phase={phase} edge={edge}", result.stderr)


if __name__ == "__main__":
    unittest.main()
