import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/ci/check_metax_devices.py"
SPEC = importlib.util.spec_from_file_location("check_metax_devices", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
RESULT_SPEC = importlib.util.spec_from_file_location(
    "check_metax_torch_results", SCRIPT.with_name("check_metax_torch_results.py")
)
RESULTS = importlib.util.module_from_spec(RESULT_SPEC)
RESULT_SPEC.loader.exec_module(RESULTS)


class MetaXTorchPreflightTest(unittest.TestCase):
    def run_check(self, *, native_count=8, native_status=0, torch_count=4,
                  available=True, init_error=None, result=120):
        runtime = mock.Mock()

        def get_count(pointer):
            pointer._obj.value = native_count
            return native_status

        runtime.mcGetDeviceCount.side_effect = get_count
        torch = mock.MagicMock()
        torch.__version__ = "2.10.0+metax3.8.1.0"
        torch.__file__ = "/flagos/lib/python3.12/site-packages/torch/__init__.py"
        torch.cuda.init.side_effect = init_error
        torch.cuda.device_count.return_value = torch_count
        torch.cuda.is_available.return_value = available
        torch.arange.return_value.sum.return_value.item.return_value = result
        output = io.StringIO()
        with mock.patch.object(CHECK.ctypes, "CDLL", return_value=runtime), \
                mock.patch.object(CHECK.importlib.metadata, "distribution"), \
                mock.patch.object(CHECK.shutil, "which", return_value=None), \
                mock.patch.dict(sys.modules, {"torch": torch}), \
                contextlib.redirect_stdout(output):
            status = CHECK.main(["torch-api"])
        return torch, status, output.getvalue()

    def test_checks_all_four_devices(self):
        torch, status, output = self.run_check()
        self.assertEqual(status, 0, output)
        self.assertEqual(
            [call.kwargs["device"] for call in torch.arange.call_args_list],
            [f"cuda:{i}" for i in range(4)],
        )
        self.assertEqual(torch.cuda.synchronize.call_count, 4)

    def test_native_errors_and_insufficient_devices_fail(self):
        for kwargs in ({"native_status": 3}, {"native_count": 0},
                       {"torch_count": 0}, {"torch_count": 3},
                       {"available": False}):
            with self.subTest(kwargs=kwargs):
                _, status, _ = self.run_check(**kwargs)
                self.assertEqual(status, 1)

    def test_preserves_torch_initialization_error(self):
        _, status, output = self.run_check(init_error=RuntimeError("wrong driver"))
        self.assertEqual(status, 1)
        self.assertIn("wrong driver", output)

    def test_bad_tensor_result_fails(self):
        _, status, output = self.run_check(result=0)
        self.assertEqual(status, 1)
        self.assertIn("read/write check failed", output)


class MetaXTorchResultsTest(unittest.TestCase):
    def mode_log(self, mode, ranks=(0, 1, 2, 3)):
        lines = [f"[INFO] Launching PyTorch API tests in {mode} mode"]
        lines += [
            f"rank {rank} after all_to_all_single with FLAGCX_GROUP1 (with splits)"
            for rank in ranks
        ]
        lines.append(f"[INFO] Completed PyTorch API tests in {mode} mode")
        return "\n".join(lines) + "\n"

    def test_both_modes_must_complete_on_all_ranks(self):
        RESULTS.validate(self.mode_log("homogeneous") + self.mode_log("heterogeneous"))
        for log in (
            "", self.mode_log("homogeneous"),
            self.mode_log("homogeneous") + self.mode_log("heterogeneous", (0, 1, 2)),
            self.mode_log("homogeneous") + self.mode_log("heterogeneous") + "[SKIP]",
        ):
            with self.subTest(log=log), self.assertRaises(RuntimeError):
                RESULTS.validate(log)

    def test_retry_cannot_hide_missing_rank(self):
        log = self.mode_log("homogeneous")
        log += "[INFO] Launching PyTorch API tests in heterogeneous mode\n"
        log += "rank 3 after all_to_all_single with FLAGCX_GROUP1 (with splits)\n"
        log += "Starting PyTorch API tests (heterogeneous) (attempt 2/2)\n"
        log += self.mode_log("heterogeneous", (0, 1, 2)).split("\n", 1)[1]
        with self.assertRaises(RuntimeError):
            RESULTS.validate(log)
