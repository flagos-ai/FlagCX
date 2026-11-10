import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
HANDLER = REPO_ROOT / ".github/scripts/ci/run_kunlunxin_workload.sh"


class KunlunXinUnitDispatchTest(unittest.TestCase):
    def test_device_suites_enable_kernel_build_without_affecting_host_suites(self):
        env_script = REPO_ROOT / ".github/scripts/set_env/kunlunxin.sh"
        for suite in ("device_api", "device_api_unified_ir", "kernel", "core", "device_api_host"):
            with self.subTest(suite=suite):
                result = subprocess.run(
                    ["bash", "-c", 'set -euo pipefail; source "$1"; '
                     'flagcx_ci_configure_suite "$2"; '
                     'printf "%s\\n" "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"; '
                     'printf "TEST_ARGS\\n"; '
                     'printf "%s\\n" "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"',
                     "bash", str(env_script), suite],
                    check=True, capture_output=True, text=True,
                )
                project_args, test_args = result.stdout.split("TEST_ARGS\n")
                for args in (project_args, test_args):
                    if suite in ("device_api", "device_api_unified_ir", "kernel"):
                        self.assertIn("COMPILE_KERNEL=1\n", args)
                        self.assertIn("USE_SHMEM=1\n", args)
                    else:
                        self.assertNotIn("COMPILE_KERNEL=1\n", args)
                        self.assertNotIn("USE_SHMEM=1\n", args)

    def test_other_platform_build_hook_receives_directory_and_make_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripts = root / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            for name in ("run_unit_test.sh", "filter_supplemental_tests.sh"):
                shutil.copyfile(REPO_ROOT / ".github/scripts/ci" / name, scripts / name)
            (scripts / "run_with_timeout.sh").write_text('#!/bin/bash\n"$@"\n')
            (scripts / "run_with_timeout.sh").chmod(0o755)
            platform = root / "metax.sh"
            platform.write_text(
                'MPI_HOME="$GITHUB_WORKSPACE/mpi"\n'
                'FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_METAX=1)\n'
                'FLAGCX_CI_TEST_MAKE_ARGS=(USE_METAX=1)\n'
                'flagcx_ci_build_suite_override() {\n'
                '  [[ $# -eq 3 ]] || return 41\n'
                '  [[ "$2" == "$GITHUB_WORKSPACE/test/unittest/core" ]] || return 42\n'
                '  [[ "$3" == USE_METAX=1 ]] || return 43\n'
                '  echo build-hook >> "$CALL_LOG"\n'
                '  FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1\n'
                '}\n'
            )
            binaries = root / "bin"
            binaries.mkdir()
            for name in ("cmake", "make"):
                command = binaries / name
                command.write_text(f'#!/bin/bash\necho {name} "$@" >> "$CALL_LOG"\n')
                command.chmod(0o755)
            log = root / "calls.log"
            env = os.environ.copy()
            env.update(GITHUB_WORKSPACE=str(root), CALL_LOG=str(log),
                       PATH=f"{binaries}:{env['PATH']}", FLAGCX_CI_COVERAGE="0")
            result = subprocess.run(
                ["bash", str(scripts / "run_unit_test.sh"), str(platform), "core"],
                env=env, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text()
            self.assertEqual(calls.count("build-hook\n"), 1)
            self.assertIn("run-unit USE_METAX=1", calls)


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
            (scripts / "run_mpi_with_timeout.sh").write_text(
                '#!/bin/bash\nprintf "mpi %s\\n" "$*" >> "$CALL_LOG"\n'
            )
            (scripts / "run_mpi_with_timeout.sh").chmod(0o755)
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
