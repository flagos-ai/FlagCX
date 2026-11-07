"""Hardware-free checks for the MetaX CI gates."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / ".github/scripts/ci/check_metax_devices.py"
TORCH_ASSERTIONS = ROOT / ".github/scripts/ci/metax_torch_assertions.py"
TORCH_ASSERTIONS_RUNNER = ROOT / ".github/scripts/ci/run_metax_torch_assertions.sh"
TORCH_WORKFLOW = ROOT / ".github/workflows/torch-api-test.yml"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetaXPreflightTest(unittest.TestCase):
    def test_native_count_below_four_fails_perf(self):
        preflight = load_module(PREFLIGHT, "metax_preflight_perf_test")
        with mock.patch.object(sys, "argv", [str(PREFLIGHT), "perf"]), \
             mock.patch.object(preflight, "runtime_device_count", return_value=3), \
             mock.patch.object(preflight, "check_mccl_alltoall"), \
             mock.patch.object(preflight.shutil, "which", return_value=None):
            self.assertEqual(preflight.main(), 1)

    def test_unusable_torch_devices_fail_torch_workload(self):
        preflight = load_module(PREFLIGHT, "metax_preflight_torch_test")
        with mock.patch.object(sys, "argv", [str(PREFLIGHT), "torch-api"]), \
             mock.patch.object(preflight, "runtime_device_count", return_value=4), \
             mock.patch.object(preflight, "check_mccl_alltoall"), \
             mock.patch.object(preflight, "torch_device_count",
                               side_effect=RuntimeError("cuda:2 failed")), \
             mock.patch.object(preflight.shutil, "which", return_value=None):
            self.assertEqual(preflight.main(), 1)

    def test_torch_preflight_allocates_on_each_device(self):
        preflight = load_module(PREFLIGHT, "metax_preflight_allocation_test")
        allocated = []
        synchronized = []
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
            synchronize=synchronized.append,
        )
        fake_torch = types.SimpleNamespace(
            __file__=str(ROOT / "torch/__init__.py"),
            __version__="test",
            version=types.SimpleNamespace(cuda="maca-test"),
            cuda=fake_cuda,
            ones=lambda size, device: allocated.append(device)
            or types.SimpleNamespace(item=lambda: 1),
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(preflight.torch_device_count(), 4)
        self.assertEqual(allocated, [f"cuda:{device}" for device in range(4)])
        self.assertEqual(synchronized, list(range(4)))


class MetaXTorchAssertionsTest(unittest.TestCase):
    def test_workflow_rejects_skip_and_missing_rank_result(self):
        workflow = TORCH_WORKFLOW.read_text()
        section = workflow.split(
            "      - name: Verify MetaX Torch API ran both modes\n", 1
        )[1].split("\n      - name:", 1)[0]
        script = textwrap.dedent(section.split("        run: |\n", 1)[1])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            example_log = root / "metax-torch-api.log"
            assertions_log = root / "metax-torch-assertions.log"
            example_log.write_text(
                "[INFO] Completed PyTorch API tests in homogeneous mode\n"
                "[INFO] Completed PyTorch API tests in heterogeneous mode\n"
            )
            markers = [
                f"[ASSERT] MetaX Torch {mode} rank={rank}: data passed"
                for mode in ("homogeneous", "heterogeneous")
                for rank in range(4)
            ]
            assertions_log.write_text("\n".join(markers) + "\n")
            environment = dict(os.environ, FLAGCX_CI_LOG_DIR=str(root))

            def check():
                return subprocess.run(["bash", "-e", "-c", script],
                                      capture_output=True, text=True,
                                      env=environment, check=False)

            self.assertEqual(check().returncode, 0)
            example_log.write_text(example_log.read_text() + "[SKIP] no GPU\n")
            self.assertNotEqual(check().returncode, 0)
            example_log.write_text(example_log.read_text().replace("[SKIP] no GPU\n", ""))
            assertions_log.write_text("\n".join(markers[:-1]) + "\n")
            self.assertNotEqual(check().returncode, 0)

    def test_runner_checks_four_ranks_in_both_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ci_dir = root / ".github/scripts/ci"
            ci_dir.mkdir(parents=True)
            fake_runner = ci_dir / "run_with_timeout.sh"
            fake_runner.write_text("#!/usr/bin/env bash\nexec \"$@\"\n")
            fake_runner.chmod(0o755)
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == -c ]]; then echo 29500; exit 0; fi\n"
                "printf '%s|%s|%s|%s\\n' \"${FLAGCX_CLUSTER_SPLIT_LIST-}\" "
                "\"${FLAGCX_MEM_ENABLE-}\" \"${FLAGCX_USE_HETERO_COMM-}\" "
                "\"$*\" >> \"$FLAGCX_TEST_ENV_LOG\"\n"
            )
            fake_python.chmod(0o755)
            output = root / "environment.log"
            environment = dict(os.environ,
                               GITHUB_WORKSPACE=str(root),
                               PYTHON_BIN=str(fake_python),
                               FLAGCX_CI_LOG_DIR=str(root),
                               FLAGCX_TEST_ENV_LOG=str(output),
                               FLAGCX_CLUSTER_SPLIT_LIST="9",
                               FLAGCX_MEM_ENABLE="9",
                               FLAGCX_USE_HETERO_COMM="1")
            result = subprocess.run(["bash", str(TORCH_ASSERTIONS_RUNNER)],
                                    capture_output=True, text=True,
                                    env=environment, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith("|||"), lines[0])
            self.assertTrue(lines[1].startswith("2|1||"), lines[1])
            self.assertIn("--nproc_per_node=4", lines[0])
            self.assertIn("homogeneous", lines[0])
            self.assertIn("heterogeneous", lines[1])

    def test_received_data_mismatch_fails(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        with mock.patch.dict(sys.modules, {"flagcx": types.ModuleType("flagcx")}):
            assertions = load_module(TORCH_ASSERTIONS, "metax_torch_assertions_test")
        assertions.require_equal("all_to_all_single", torch.tensor([1]),
                                 torch.tensor([1]), 0)
        with self.assertRaises(AssertionError):
            assertions.require_equal("all_to_all_single", torch.tensor([0]),
                                     torch.tensor([1]), 0)


if __name__ == "__main__":
    unittest.main()
