import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("torch_api_test.sh")


class TorchApiEnvironmentTest(unittest.TestCase):
    def run_script(self, *, skip_hetero=False, diagnostic_only=False, ci=False,
                   continue_on_failure=False, fail_homogeneous=False,
                   hetero_skip_ops=""):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_path = temp_path / "environment.log"
            library_log = temp_path / "library.log"
            python_path = temp_path / "python"
            python_path.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    if [[ "${1:-}" == "-c" ]]; then
                      echo 29500
                      exit 0
                    fi
                    printf '%s|%s|%s|%s|%s|%s|%s\\n' \\
                      "${FLAGCX_USE_HETERO_COMM-}" \\
                      "${FLAGCX_CLUSTER_SPLIT_LIST-}" \\
                      "${FLAGCX_MEM_ENABLE-}" \\
                      "${FLAGCX_VMM_ENABLE-}" \\
                      "${FLAGCX_P2P_TRANSPORT-}" \\
                      "${FLAGCX_P2P_DISABLE-}" \\
                      "${FLAGCX_TORCH_SKIP_OPS-}" >> "${FLAGCX_TEST_ENV_LOG}"
                    printf '%s\\n' "${FLAGCX_CI_EXPECT_FLAGCX_LIBRARY-}" > "${FLAGCX_TEST_LIBRARY_LOG}"
                    if [[ "${FLAGCX_TEST_FAIL_HOMOGENEOUS:-0}" == 1 &&
                          -z "${FLAGCX_USE_HETERO_COMM-}" ]]; then
                      exit 7
                    fi
                    """
                )
            )
            python_path.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": str(python_path),
                    "FLAGCX_TORCH_TEST_RUNNER": "/usr/bin/env",
                    "FLAGCX_TORCH_TEST_INTER_MODE_DELAY": "0",
                    "FLAGCX_TEST_ENV_LOG": str(log_path),
                    "FLAGCX_TEST_LIBRARY_LOG": str(library_log),
                    "CI": "true" if ci else "false",
                    "FLAGCX_USE_HETERO_COMM": "1",
                    "FLAGCX_CLUSTER_SPLIT_LIST": "2",
                    "FLAGCX_MEM_ENABLE": "1",
                    "FLAGCX_VMM_ENABLE": "0",
                    "FLAGCX_P2P_TRANSPORT": "accl",
                    "FLAGCX_P2P_DISABLE": "0",
                    "FLAGCX_TORCH_HETERO_DIAGNOSTIC_ONLY": "1" if diagnostic_only else "0",
                    "FLAGCX_TORCH_RANK_BACKTRACE": "1" if diagnostic_only else "0",
                    "FLAGCX_TORCH_CONTINUE_ON_FAILURE": "1" if continue_on_failure else "0",
                    "FLAGCX_TEST_FAIL_HOMOGENEOUS": "1" if fail_homogeneous else "0",
                    "FLAGCX_TORCH_HETERO_SKIP_OPS": hetero_skip_ops,
                    "FLAGCX_TORCH_SKIP_OPS": "alltoall",
                }
            )
            if skip_hetero:
                environment["FLAGCX_SKIP_HETERO"] = "1"
            else:
                environment.pop("FLAGCX_SKIP_HETERO", None)
            environment.pop("FLAGCX_CI_EXPECT_FLAGCX_LIBRARY", None)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            lines = log_path.read_text().splitlines()
            return result, lines, library_log.read_text().strip()

    def test_homogeneous_and_heterogeneous_environments_are_isolated(self):
        result, lines, _ = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines[0], "||||||")
        self.assertEqual(lines[1], "1|2|1|0|accl|0|")
        self.assertEqual(result.stdout.count("--nproc_per_node 4"), 2)

    def test_skip_heterogeneous_runs_only_homogeneous_environment(self):
        result, lines, _ = self.run_script(skip_hetero=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, ["||||||"])
        self.assertIn("Skipping heterogeneous", result.stdout)

    def test_diagnostic_runs_only_heterogeneous_workers_under_gdb(self):
        result, lines, _ = self.run_script(diagnostic_only=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, ["1|2|1|0|accl|0|"])
        self.assertIn("torch_rank_backtrace.py", result.stdout)
        self.assertNotIn("in homogeneous mode", result.stdout)

    def test_ci_expects_the_library_built_by_this_checkout(self):
        result, _, library = self.run_script(ci=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(library, str(SCRIPT.parents[2] / "build/lib/libflagcx.so"))

    def test_diagnostic_runs_both_modes_and_preserves_first_failure(self):
        result, lines, _ = self.run_script(
            continue_on_failure=True, fail_homogeneous=True)

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(lines, ["||||||", "1|2|1|0|accl|0|"])
        self.assertIn("homogeneous=7 heterogeneous=0", result.stdout)

    def test_heterogeneous_operation_skip_does_not_affect_homogeneous_mode(self):
        result, lines, _ = self.run_script(
            hetero_skip_ops="reduce,allreduce,reducescatter")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, [
            "||||||", "1|2|1|0|accl|0|reduce,allreduce,reducescatter",
        ])


class TorchApiOperationSelectionTest(unittest.TestCase):
    def test_all_runs_remaining_operations_when_three_are_skipped(self):
        example = SCRIPT.parents[2] / "plugin/torch/example/example.py"
        spec = importlib.util.spec_from_file_location("torch_api_example_test", example)
        module = importlib.util.module_from_spec(spec)
        torch = types.ModuleType("torch")
        torch.__path__ = []
        distributed = types.ModuleType("torch.distributed")
        torch.distributed = distributed
        sys.path.insert(0, str(example.parent))
        try:
            with patch.dict(sys.modules, {
                    "torch": torch,
                    "torch.distributed": distributed,
                    "flagcx": types.ModuleType("flagcx"),
                    "torch_mlu": None,
                    "torch_npu": None,
                    "torch_musa": None,
                    "torch_txda": None,
                    "torch_gcu": None,
            }), \
                    patch.dict(os.environ, {
                        "FLAGCX_TORCH_SKIP_OPS": "reduce,allreduce,reducescatter",
                        "FLAGCX_CI_EXPECT_FLAGCX_LIBRARY": "",
                    }):
                spec.loader.exec_module(module)
                called = []
                for name in ("broadcast", "reduce", "allreduce", "sendrecv",
                             "allgather", "reducescatter", "gather", "scatter",
                             "alltoall"):
                    setattr(module, f"test_{name}",
                            lambda operation=name: called.append(operation))
                module.test_all()
        finally:
            sys.path.remove(str(example.parent))

        self.assertEqual(called, ["broadcast", "sendrecv", "allgather",
                                  "gather", "scatter", "alltoall"])


if __name__ == "__main__":
    unittest.main()
