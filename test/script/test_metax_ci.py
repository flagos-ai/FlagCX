"""Hardware-free checks for the MetaX perf CI preflight."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


PREFLIGHT = Path(__file__).resolve().parents[2] / ".github/scripts/ci/check_metax_devices.py"


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


if __name__ == "__main__":
    unittest.main()
