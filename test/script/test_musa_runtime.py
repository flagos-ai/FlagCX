"""Exercise actual MTML loading and its MUSA workload preparation gate."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / ".github/scripts/ci/check_musa_runtime.py"
ENVIRONMENT = ROOT / ".github/scripts/set_env/mthreads.sh"


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cc"),
                     "Linux ELF loader and C compiler required")
class MusaRuntimeTest(unittest.TestCase):
    def build_library(self, root, *, missing_dependency=False):
        source = root / "mtml.c"
        flags = []
        if missing_dependency:
            dependency = root / "dependency.c"
            dependency.write_text("int mtml_test_dependency(void) { return 0; }\n")
            subprocess.run(["cc", "-shared", "-fPIC", str(dependency),
                            "-Wl,-soname,libmtml_test_dependency.so",
                            "-o", str(root / "libmtml_test_dependency.so")], check=True,
                           capture_output=True, text=True)
            source.write_text("extern int mtml_test_dependency(void);\n"
                              "int mtml_test(void) { return mtml_test_dependency(); }\n")
            flags = [f"-L{root}", "-lmtml_test_dependency"]
        else:
            source.write_text("int mtml_test(void) { return 0; }\n")
        subprocess.run(["cc", "-shared", "-fPIC", str(source), *flags,
                        "-Wl,-soname,libmtml.so", "-o", str(root / "libmtml.so")],
                       check=True, capture_output=True, text=True)
        if missing_dependency:
            (root / "libmtml_test_dependency.so").unlink()

    def environment(self, root):
        return {**os.environ, "LD_LIBRARY_PATH": str(root),
                "MTHREADS_VISIBLE_DEVICES": "0,1,2,3",
                "MTHREADS_DRIVER_CAPABILITIES": "compute,utility"}

    def test_actual_shared_library_load_and_resolved_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_library(root)
            result = subprocess.run([sys.executable, str(CHECK)], env=self.environment(root),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("MUSA MTML mapped: " + str(root / "libmtml.so"), result.stdout)
            self.assertIn("MTHREADS_VISIBLE_DEVICES=0,1,2,3", result.stdout)

    def test_existing_library_with_missing_dependency_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_library(root, missing_dependency=True)
            result = subprocess.run([sys.executable, str(CHECK)], env=self.environment(root),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("MUSA MTML preflight failed:", result.stderr)
            self.assertIn("libmtml_test_dependency.so", result.stderr)
            self.assertIn("MTML candidate: " + str(root / "libmtml.so"), result.stderr)
            self.assertNotIn("MUSA MTML loaded:", result.stdout)

    def test_prepare_stops_native_workloads_but_keeps_host_checks(self):
        native = ("runner", "device_api_host", "device_api", "device_api_unified_ir",
                  "rma", "symmem", "perf", "torch-api", "perf-test-experimental")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_library(root, missing_dependency=True)
            sdk = root / "sdk/bin"
            sdk.mkdir(parents=True)
            compiler = sdk / "mcc"
            compiler.write_text("#!/bin/sh\nexit 0\n")
            compiler.chmod(0o755)
            marker = root / "build-started"
            for suite in (*native, "adaptor", "core", "p2p", "service"):
                with self.subTest(suite=suite):
                    marker.unlink(missing_ok=True)
                    result = subprocess.run(["bash", "-c", '''set -euo pipefail
source "$MUSA_ENV_SCRIPT"
mpirun() { :; }
g++() { :; }
mthreads-gmi() { :; }
flagcx_ci_prepare "$MUSA_TEST_SUITE"
touch "$MUSA_BUILD_MARKER"
'''], env={**self.environment(root), "MUSA_ENV_SCRIPT": str(ENVIRONMENT),
           "MUSA_TEST_SUITE": suite, "MUSA_BUILD_MARKER": str(marker),
           "DEVICE_HOME": str(root / "sdk"), "FLAGCX_CI_MUSA_DEVICES": "0,1,2,3"},
                        capture_output=True, text=True)
                    if suite in native:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("MUSA MTML preflight failed:", result.stderr)
                        self.assertFalse(marker.exists())
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
