import os
import sys
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "flagcx"
sys.path.insert(0, str(PACKAGE_DIR))

import _backend_loader as backend_loader


class BackendLoaderTest(unittest.TestCase):
    def backend_environment(self, value=None):
        environment = dict(os.environ)
        environment.pop("FLAGCX_TORCH_BACKEND", None)
        if value is not None:
            environment["FLAGCX_TORCH_BACKEND"] = value
        return mock.patch.dict(os.environ, environment, clear=True)

    def test_vendor_is_the_runtime_default(self):
        with self.backend_environment():
            self.assertEqual(backend_loader.selected_torch_backend(), "vendor")

    def test_flagos_loads_only_torch_fl(self):
        with self.backend_environment("flagos"), mock.patch.object(
            backend_loader.importlib, "import_module"
        ) as import_module:
            loaded = backend_loader.load_torch_device_backend()

        self.assertEqual(loaded, "torch_fl")
        import_module.assert_called_once_with("torch_fl")

    def test_vendor_keeps_existing_probe_order(self):
        def import_package(package):
            if package == "torch_gcu":
                return mock.sentinel.torch_gcu
            raise ImportError(package)

        with self.backend_environment("vendor"), mock.patch.object(
            backend_loader.importlib,
            "import_module",
            side_effect=import_package,
        ) as import_module:
            loaded = backend_loader.load_torch_device_backend()

        self.assertEqual(loaded, "torch_gcu")
        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            ["torch_npu", "torch_mlu", "torch_musa", "torch_txda", "torch_gcu"],
        )

    def test_invalid_backend_is_rejected_before_import(self):
        with self.backend_environment("torch_npu"), mock.patch.object(
            backend_loader.importlib, "import_module"
        ) as import_module:
            with self.assertRaisesRegex(RuntimeError, "Valid values"):
                backend_loader.load_torch_device_backend()

        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
