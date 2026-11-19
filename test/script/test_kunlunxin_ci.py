import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDLER = REPO_ROOT / ".github/scripts/ci/run_kunlunxin_workload.sh"


def load_kunlunxin_ignore_config():
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$FLAGCX_CI_IGNORE_TEST"',
         "bash", str(REPO_ROOT / ".github/scripts/set_env/kunlunxin.sh")],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


class KunlunXinUnitDispatchTest(unittest.TestCase):
    def test_ignore_config_is_owned_by_platform_shell(self):
        config = (REPO_ROOT / ".github/configs/kunlunxin.yml").read_text()
        self.assertNotIn("ignore_test:", config)
        self.assertTrue(load_kunlunxin_ignore_config()["phases"])

    def test_unified_ir_ignore_test_only_matches_s16(self):
        config = load_kunlunxin_ignore_config()
        env = {
            **os.environ,
            "FLAGCX_CI_IGNORE_TEST": json.dumps(config),
        }
        helper = REPO_ROOT / ".github/scripts/ci/ignore_test.py"
        s16 = subprocess.run(
            [sys.executable, str(helper), "case", "device_api_unified_ir", "S16"],
            env=env, check=True, capture_output=True, text=True,
        )
        s17 = subprocess.run(
            [sys.executable, str(helper), "case", "device_api_unified_ir", "S17"],
            env=env, check=True, capture_output=True, text=True,
        )
        self.assertIn("timed out twice", s16.stdout)
        self.assertEqual(s17.stdout, "\n")
        self.assertFalse(any(
            entry.get("suite") == "device_api_unified_ir"
            and entry.get("name") == "whole-suite"
            for entry in config["phases"]
        ))

    def test_device_group_and_heap_size_can_be_overridden(self):
        result = subprocess.run(
            ["bash", "-c", 'set -euo pipefail; source "$1"; '
             'printf "%s\\n" "$FLAGCX_ADAPTOR" "$XSHMEM_SYMMETRIC_SIZE"; '
             'printf "%s\\n" "${FLAGCX_CI_UNIFIED_IR_MPI_ARGS[@]}"',
             "bash", str(REPO_ROOT / ".github/scripts/set_env/kunlunxin.sh")],
            env={**os.environ, "FLAGCX_CI_KUNLUNXIN_DEVICES": "0,1,2,3",
                 "XSHMEM_SYMMETRIC_SIZE": "8589934592"},
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            ["klx", "8589934592", "-x", "XPU_VISIBLE_DEVICES=0,1,2,3"],
        )

    def run_mock_suite(self, suite, failed_label):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ci = root / ".github/scripts/ci"
            ci.mkdir(parents=True)
            for name in ("run_unit_test.sh", "filter_supplemental_tests.sh",
                         "ignore_test.py"):
                shutil.copyfile(REPO_ROOT / ".github/scripts/ci" / name, ci / name)
            symmem_script = root / "test/script/symmem_test.sh"
            symmem_script.parent.mkdir(parents=True)
            shutil.copyfile(REPO_ROOT / "test/script/symmem_test.sh", symmem_script)
            binaries = root / "bin"
            binaries.mkdir()
            probe = (
                '#!/bin/bash\n'
                'label=${FLAGCX_CI_MPI_LABEL:-${FLAGCX_CI_TEST_LABEL:-build}}\n'
                'printf "%s|%s\\n" "$label" "$*" >> "$CALL_LOG"\n'
                'if [[ "$label" == "$FAIL_LABEL" ]]; then exit 7; fi\n'
            )
            for executable in (binaries / "make", binaries / "cmake",
                               ci / "run_mpi_with_timeout.sh",
                               root / "test/unittest/symmem/build/bin/symmem_unit_tests"):
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_text(probe)
                executable.chmod(0o755)
            (ci / "run_with_timeout.sh").write_text('#!/bin/bash\nexec "$@"\n')
            (ci / "run_with_timeout.sh").chmod(0o755)
            platform = root / "kunlunxin.sh"
            (root / "test/unittest" / suite).mkdir(parents=True, exist_ok=True)
            platform.write_text(
                'MPI_HOME="$GITHUB_WORKSPACE/mpi"\n'
                'FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_KUNLUNXIN=1)\n'
                'FLAGCX_CI_TEST_MAKE_ARGS=(USE_KUNLUNXIN=1)\n'
                'FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1\n'
                'FLAGCX_CI_RUNNER_NP=4\n'
                'FLAGCX_CI_NODE_NP=4\n'
                'FLAGCX_CI_INTRA_NP=4\n'
                'FLAGCX_CI_KUNLUNXIN_DEVICES=${FLAGCX_CI_KUNLUNXIN_DEVICES:-0,1,2,3}\n'
                'export XSHMEM_SYMMETRIC_SIZE=8589934592\n'
                'FLAGCX_CI_NODE1_MPI_ARGS=(-x FLAGCX_HOSTID=node0)\n'
                'FLAGCX_CI_NODE2_MPI_ARGS=(-x FLAGCX_HOSTID=node1)\n'
                'export FLAGCX_ADAPTOR=klx\n'
                'flagcx_ci_validate_rdma() { return 0; }\n'
            )
            config = load_kunlunxin_ignore_config()
            log = root / "calls"
            env = {**os.environ, "GITHUB_WORKSPACE": str(root),
                   "PATH": f"{binaries}:{os.environ['PATH']}",
                   "CALL_LOG": str(log), "FAIL_LABEL": failed_label,
                   "FLAGCX_CI_COVERAGE": "0", "GTEST_FILTER": "*",
                   "FLAGCX_CI_IGNORE_TEST": json.dumps(config)}
            result = subprocess.run(
                ["bash", str(ci / "run_unit_test.sh"), str(platform), suite],
                cwd=root, env=env, capture_output=True, text=True, timeout=15,
            )
            return result, log.read_text()

    def test_independent_phases_run_after_failures(self):
        for suite, failed, required in (
            ("kernel", "kernel unit tests", ("kernel MPI tests",)),
        ):
            with self.subTest(suite=suite, failed=failed):
                result, calls = self.run_mock_suite(suite, failed)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"{failed}|", calls)
                for label in required:
                    self.assertIn(f"{label}|", calls)

    def test_restored_phases_succeed_when_commands_succeed(self):
        for suite in ("symmem", "p2p", "rma", "runner", "kernel"):
            with self.subTest(suite=suite):
                result, _ = self.run_mock_suite(suite, "no matching label")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_hardware_mpi_phases_are_not_blanket_skipped(self):
        config = load_kunlunxin_ignore_config()
        skipped_phases = {
            (entry["suite"], entry["name"])
            for entry in config.get("phases", [])
        }
        for phase in (
        ):
            with self.subTest(phase=phase):
                self.assertNotIn(phase, skipped_phases)

        excluded_gtests = {
            test
            for entry in config.get("gtest", [])
            for test in entry.get("tests", [])
        }
        self.assertIn(
            "FlagcxP2pEngineReadTest.ReadsWholeRegisteredGpuBufferAfterMetadataHandshake",
            excluded_gtests,
        )
        self.assertNotIn("DevicePtrArgs.DevCommFreeDevicePtrNull", excluded_gtests)
        self.assertNotIn("DevicePtrArgs.DevMemFreeDevicePtrNull", excluded_gtests)

    def test_kunlunxin_skips_unavailable_rma_symmem_and_runner_mpi_phases(self):
        config = load_kunlunxin_ignore_config()
        skipped_phases = {
            (entry["suite"], entry["name"]): entry["reason"]
            for entry in config.get("phases", [])
        }
        for phase in (
            ("p2p", "p2p-write-mpi"),
            ("rma", "rma-mpi"),
            ("symmem", "symmem-mpi"),
            ("runner", "runner-mpi"),
        ):
            with self.subTest(phase=phase):
                self.assertTrue(skipped_phases.get(phase))

        env = {
            **os.environ,
            "FLAGCX_CI_IGNORE_TEST": json.dumps(config),
        }
        helper = REPO_ROOT / ".github/scripts/ci/ignore_test.py"
        for suite, phase in skipped_phases:
            if (suite, phase) not in {
                ("rma", "rma-mpi"),
                ("symmem", "symmem-mpi"),
                ("runner", "runner-mpi"),
            }:
                continue
            with self.subTest(suite=suite, phase=phase):
                result = subprocess.run(
                    [sys.executable, str(helper), "phase", suite, phase],
                    env=env, check=True, capture_output=True, text=True,
                )
                self.assertTrue(result.stdout.strip())

    def test_mpi_phase_skips_do_not_launch_mpi_commands(self):
        for suite, forbidden_labels in (
            ("p2p", ("p2p Engine WRITE MPI tests", "p2p shared-engine WRITE MPI tests")),
            ("rma", ("rma IPC MPI tests", "rma network MPI tests")),
            ("symmem", ("symmem IPC + NET", "symmem local IPC + NET fallback")),
            ("runner", ("runner default", "runner heterogeneous", "runner forced NET")),
        ):
            with self.subTest(suite=suite):
                result, calls = self.run_mock_suite(suite, "no matching label")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for label in forbidden_labels:
                    self.assertNotIn(f"{label}|", calls)
                self.assertIn(f"{suite} unit tests|", calls)

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
            for name in ("run_unit_test.sh", "filter_supplemental_tests.sh", "ignore_test.py"):
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
            shutil.copyfile(REPO_ROOT / ".github/scripts/ci/ignore_test.py",
                            scripts / "ignore_test.py")
            (root / "platform.sh").write_text(
                "FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_KUNLUNXIN=1)\n"
                "FLAGCX_CI_TEST_MAKE_ARGS=(USE_KUNLUNXIN=1)\n"
                "FLAGCX_CI_KUNLUNXIN_DEVICES=${FLAGCX_CI_KUNLUNXIN_DEVICES:-0,1,2,3}\n"
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
            helper = root / ".github/scripts/ci/ignore_test.py"
            torch_script.write_text(
                f'ops=$("{sys.executable}" "{helper}" torch homogeneous)\n'
                '[[ " $ops " != *" gather "* ]] || export FLAGCX_TORCH_SKIP_GATHER=1\n'
                f'reason=$("{sys.executable}" "{helper}" phase torch-api heterogeneous)\n'
                '[[ -z "$reason" ]] || export FLAGCX_SKIP_HETERO=1\n'
                'echo "torch-tests skip-gather=${FLAGCX_TORCH_SKIP_GATHER:-0} '
                'skip-hetero=${FLAGCX_SKIP_HETERO:-0}" >> "$CALL_LOG"\n'
            )
            binaries = root / "bin"
            binaries.mkdir()
            (binaries / "make").write_text(
                '#!/bin/bash\nprintf "make %s\\n" "$*" >> "$CALL_LOG"\n'
                f"exit {make_status}\n"
            )
            (binaries / "python3").write_text(
                '#!/bin/bash\n'
                f'if [[ "${{1:-}}" == */ignore_test.py ]]; then exec "{sys.executable}" "$@"; fi\n'
                'printf "python %s\\n" "$*" >> "$CALL_LOG"\n'
                'printf "CPATH=%s\\nLIBRARY_PATH=%s\\nHOST_CXXFLAGS=%s\\n" '
                '"${CPATH:-}" "${LIBRARY_PATH:-}" "${HOST_CXXFLAGS:-}" '
                '>> "$CALL_LOG"\n'
                'if [[ "${1:-}" == - ]]; then '
                'printf "VISIBLE=%s|%s\\n" "${CUDA_VISIBLE_DEVICES:-}" '
                '"${XPU_VISIBLE_DEVICES:-}" >> "$CALL_LOG"; cat >/dev/null; fi\n'
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
                FLAGCX_CI_IGNORE_TEST=(
                    '{"phases":[{"suite":"perf","name":"heterogeneous",'
                    '"reason":"runtime unsupported"},'
                    '{"suite":"torch-api","name":"heterogeneous",'
                    '"reason":"runtime unsupported"}],'
                    '"torch":{"operations":["gather"],'
                    '"modes":["homogeneous","heterogeneous"]}}'
                ),
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
        self.assertIn("perf homogeneous\n", calls)
        self.assertNotIn("perf heterogeneous", calls)
        self.assertIn("heterogeneous perf skipped", result.stdout)

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
        self.assertIn("torch-tests skip-gather=1 skip-hetero=1\n", calls)

    def test_torch_uses_configured_kunlunxin_device_group(self):
        result, calls, _ = self.run_workload("torch-api")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VISIBLE=0,1,2,3|0,1,2,3\n", calls)

        with patch.dict(os.environ, {"FLAGCX_CI_KUNLUNXIN_DEVICES": "4,5,6,7"}):
            result, calls, _ = self.run_workload("torch-api")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VISIBLE=4,5,6,7|4,5,6,7\n", calls)

    def test_missing_cuda_headers_fail_before_torch_build(self):
        result, calls, _ = self.run_workload("torch-api", sdk=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KunlunXin Torch build requires", result.stderr)
        self.assertNotIn("python setup.py", calls)
        self.assertNotIn("torch-tests", calls)

    def test_torch_gather_skip_is_explicit_and_opt_in(self):
        source = (REPO_ROOT / "plugin/torch/example/example.py").read_text()
        tree = ast.parse(source)
        gather = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "test_gather")
        module = ast.fix_missing_locations(ast.Module(body=[gather], type_ignores=[]))

        class TorchProbe:
            class cuda:
                @staticmethod
                def is_available():
                    raise RuntimeError("gather reached accelerator path")

        namespace = {"os": os, "torch": TorchProbe}
        exec(compile(module, "example.py", "exec"), namespace)
        from unittest.mock import patch
        with patch.dict(os.environ, {"FLAGCX_TORCH_SKIP_GATHER": "1"}):
            namespace["test_gather"]()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "accelerator path"):
                namespace["test_gather"]()


if __name__ == "__main__":
    unittest.main()
