import ast
import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from unittest import mock

import yaml
from pathlib import Path

SCRIPT = Path(__file__).with_name("torch_api_test.sh")


class TorchApiEnvironmentTest(unittest.TestCase):
    def test_pair_teardown_allocates_on_the_selected_local_device(self):
        example = SCRIPT.resolve().parents[2] / "plugin/torch/example/example.py"
        tree = ast.parse(example.read_text())
        teardown = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "test_pair_teardown")
        code = compile(ast.Module(body=[teardown], type_ignores=[]), str(example), "exec")
        for platform in ("cuda", "musa"):
            with self.subTest(platform=platform), mock.patch.dict(os.environ, LOCAL_RANK="2"):
                torch = mock.Mock()
                torch.device.side_effect = lambda name, index: f"{name}:{index}"
                torch.testing.assert_close = mock.Mock()
                expected_device = f"{platform}:2"

                def full(*args, **kwargs):
                    # CPU reference tensors omit device. Accelerator factories
                    # must select the local device, as required by torch_musa.
                    self.assertEqual(kwargs.get("device", expected_device), expected_device)
                    return mock.Mock()

                torch.full.side_effect = full
                dist = mock.Mock()
                work = mock.Mock()
                dist.batch_isend_irecv.return_value = [work]
                backend = dist.new_group.return_value._get_backend.return_value
                backend.barrier.side_effect = RuntimeError("backend has been shut down")
                backend.allreduce_coalesced.side_effect = RuntimeError("backend has been shut down")
                flagcx = mock.Mock()
                namespace = dict(os=os, torch=torch, dist=dist, flagcx=flagcx,
                                 dev_name=platform, MY_RANK=0, WORLD_SIZE=2)
                exec(code, namespace)
                namespace["test_pair_teardown"]()
                torch.empty.assert_called_once_with(8, device=expected_device)
                dist.new_group.return_value._get_backend.assert_called_once_with(expected_device)
                work.wait.assert_called_once_with()
                self.assertEqual(flagcx.shutdown_backend.call_args_list,
                                 [mock.call(backend), mock.call(backend)])

    def run_script(self, *, skip_hetero=False, hetero_skip_ops="", ignore_config=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_path = temp_path / "environment.log"
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
                    [[ -z "${UCX_HOME+x}" ]]
                    [[ "$LD_LIBRARY_PATH" == /sdk/ucx/lib:/sdk/mpi/lib ]]
                    printf '%s|%s|%s|%s|%s|%s|%s\\n' \\
                      "${FLAGCX_USE_HETERO_COMM-}" \\
                      "${FLAGCX_CLUSTER_SPLIT_LIST-}" \\
                      "${FLAGCX_MEM_ENABLE-}" \\
                      "${FLAGCX_VMM_ENABLE-}" \\
                      "${FLAGCX_P2P_TRANSPORT-}" \\
                      "${FLAGCX_P2P_DISABLE-}" \\
                      "${FLAGCX_TORCH_SKIP_OPS-}" >> "${FLAGCX_TEST_ENV_LOG}"
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
                    "FLAGCX_USE_HETERO_COMM": "1",
                    "FLAGCX_CLUSTER_SPLIT_LIST": "2",
                    "FLAGCX_MEM_ENABLE": "1",
                    "FLAGCX_VMM_ENABLE": "0",
                    "FLAGCX_P2P_TRANSPORT": "accl",
                    "FLAGCX_P2P_DISABLE": "0",
                    "FLAGCX_TORCH_HETERO_SKIP_OPS": hetero_skip_ops,
                    "FLAGCX_TORCH_SKIP_OPS": "alltoall",
                    "UCX_HOME": "/sdk/ucx",
                    "LD_LIBRARY_PATH": "/sdk/ucx/lib:/sdk/mpi/lib",
                    "FLAGCX_CI_IGNORE_TEST": json.dumps(ignore_config or {}),
                }
            )
            if skip_hetero:
                environment["FLAGCX_SKIP_HETERO"] = "1"
            else:
                environment.pop("FLAGCX_SKIP_HETERO", None)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            lines = log_path.read_text().splitlines()
            return result, lines

    def test_homogeneous_and_heterogeneous_environments_are_isolated(self):
        result, lines = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines[0], "||||||")
        self.assertEqual(lines[1], "1|2|1|0|accl|0|")
        self.assertEqual(result.stdout.count("--nproc_per_node 4"), 2)

    def test_skip_heterogeneous_runs_only_homogeneous_environment(self):
        result, lines = self.run_script(skip_hetero=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, ["||||||"])
        self.assertIn("Skipping heterogeneous", result.stdout)

    def test_heterogeneous_operation_skip_does_not_affect_homogeneous_mode(self):
        result, lines = self.run_script(
            hetero_skip_ops="reduce,allreduce,reducescatter")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, [
            "||||||", "1|2|1|0|accl|0|reduce,allreduce,reducescatter",
        ])

    def test_ascend_config_excludes_only_selected_heterogeneous_operations(self):
        config = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / ".github/configs/ascend.yml").read_text()
        )["ignore_test"]
        result, lines = self.run_script(ignore_config=config)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, [
            "||||||", "1|2|1|0|accl|0|reduce,allreduce,reducescatter",
        ])


if __name__ == "__main__":
    unittest.main()
