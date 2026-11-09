import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
HANDLER = REPO_ROOT / ".github/scripts/ci/run_kunlunxin_workload.sh"


class KunlunXinWorkloadTest(unittest.TestCase):
    def run_workload(self, workload, *, sdk=True, make_status=0):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripts = root / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            handler = scripts / HANDLER.name
            shutil.copyfile(HANDLER, handler)
            (root / "platform.sh").write_text(
                "FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_KUNLUNXIN=1)\n"
                "FLAGCX_CI_TEST_MAKE_ARGS=(USE_KUNLUNXIN=1)\n"
                'DEVICE_HOME="$GITHUB_WORKSPACE/xpu"\n'
                "flagcx_ci_prepare() { :; }\n"
                "flagcx_ci_validate_rdma() { :; }\n"
            )
            # Deliberately provide no XTDK compiler or toolchain finder.
            # These host workloads must reach their build/test commands.
            (scripts / "run_host_perf_suite.sh").write_text(
                '#!/bin/bash\nprintf "perf %s\\n" "$1" >> "$CALL_LOG"\n'
            )
            (scripts / "run_host_perf_suite.sh").chmod(0o755)
            torch_script = root / "test/script/torch_api_test.sh"
            torch_script.parent.mkdir(parents=True)
            torch_script.write_text('echo torch-tests >> "$CALL_LOG"\n')
            binaries = root / "bin"
            binaries.mkdir()
            (binaries / "make").write_text(
                '#!/bin/bash\nprintf "make %s\\n" "$*" >> "$CALL_LOG"\n'
                f"exit {make_status}\n"
            )
            (binaries / "python3").write_text(
                '#!/bin/bash\nprintf "python %s\\n" "$*" >> "$CALL_LOG"\n'
                'printf "CPATH=%s\\nLIBRARY_PATH=%s\\nHOST_CXXFLAGS=%s\\n" '
                '"${CPATH:-}" "${LIBRARY_PATH:-}" "${HOST_CXXFLAGS:-}" '
                '>> "$CALL_LOG"\n'
                'if [[ "${1:-}" == - ]]; then cat >/dev/null; fi\n'
            )
            for executable in binaries.iterdir():
                executable.chmod(0o755)
            include = root / "cuda/include"
            include.mkdir(parents=True)
            if sdk:
                (include / "cuda.h").touch()
                (include / "cuda_fp16.h").touch()
            log = root / "calls.log"
            env = os.environ.copy()
            env.update(
                GITHUB_WORKSPACE=str(root),
                FLAGCX_CI_SET_ENV="platform.sh",
                CUDA_HOME=str(root / "cuda"),
                PATH=f"{binaries}:{env['PATH']}",
                CALL_LOG=str(log),
                CPATH="/existing/include",
                LIBRARY_PATH="/existing/lib",
                HOST_CXXFLAGS="-O2",
            )
            result = subprocess.run(
                ["bash", str(handler), workload],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            return result, log.read_text() if log.exists() else "", str(root)

    def test_host_perf_does_not_require_xtdk(self):
        result, calls, _ = self.run_workload("perf", sdk=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count("make "), 2)
        self.assertIn("perf homogeneous\nperf heterogeneous\n", calls)

    def test_perf_build_failure_is_not_hidden(self):
        result, calls, _ = self.run_workload("perf", make_status=23)
        self.assertEqual(result.returncode, 23)
        self.assertNotIn("perf homogeneous", calls)

    def test_torch_build_uses_sdk_and_compatibility_runtime_without_xtdk(self):
        result, calls, root = self.run_workload("torch-api")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"CPATH={root}/cuda/include:/existing/include\n", calls)
        self.assertIn(
            f"LIBRARY_PATH=/usr/local/xcudart/lib:{root}/xpu/so:"
            f"{root}/xpu/lib:/existing/lib\n",
            calls,
        )
        self.assertIn("HOST_CXXFLAGS=-O2 -include cstddef\n", calls)
        self.assertIn("python setup.py build_ext --inplace\n", calls)
        self.assertIn("torch-tests\n", calls)

    def test_missing_cuda_headers_fail_before_torch_build(self):
        result, calls, _ = self.run_workload("torch-api", sdk=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KunlunXin Torch build requires", result.stderr)
        self.assertNotIn("python setup.py", calls)
        self.assertNotIn("torch-tests", calls)


if __name__ == "__main__":
    unittest.main()
