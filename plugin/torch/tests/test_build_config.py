import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

import _build_config as build_config


BACKEND_ENV_VARS = {
    "FLAGCX_TORCH_BACKEND",
    "FLAGOS_INSTALL_PATH",
    "FLAGOS_INCLUDE_DIR",
    "FLAGOS_LIBRARY_DIR",
    "ASCEND_HOME_PATH",
    "DEVICE_HOME",
}


class TorchBackendConfigTest(unittest.TestCase):
    def clean_environment(self, **values):
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in BACKEND_ENV_VARS
        }
        environment.update(values)
        return mock.patch.dict(os.environ, environment, clear=True)

    def test_vendor_is_the_default(self):
        with self.clean_environment():
            ascend = build_config.resolve_torch_backend("ascend")
            enflame = build_config.resolve_torch_backend("enflame")

        self.assertEqual(ascend.name, "vendor")
        self.assertEqual(ascend.python_package, "torch_npu")
        self.assertEqual(ascend.device_name, "npu")
        self.assertEqual(ascend.compile_flags, ())
        self.assertEqual(enflame.name, "vendor")
        self.assertEqual(enflame.python_package, "torch_gcu")
        self.assertEqual(enflame.device_name, "gcu")
        self.assertEqual(enflame.compile_flags, ())

    def test_flagos_support_matrix(self):
        with self.clean_environment(FLAGCX_TORCH_BACKEND="flagos"):
            for adaptor in ("ascend", "enflame"):
                backend = build_config.resolve_torch_backend(adaptor)
                self.assertEqual(backend.name, "flagos")
                self.assertEqual(backend.python_package, "torch_fl")
                self.assertEqual(backend.device_name, "flagos")
                self.assertEqual(
                    backend.compile_flags,
                    ("-DFLAGCX_TORCH_BACKEND_FLAGOS",),
                )

            with self.assertRaisesRegex(RuntimeError, "supports only"):
                build_config.resolve_torch_backend("nvidia")

    def test_invalid_backend_is_rejected(self):
        with self.clean_environment(FLAGCX_TORCH_BACKEND="torch_gcu"):
            with self.assertRaisesRegex(RuntimeError, "Valid values"):
                build_config.resolve_torch_backend("enflame")

    def test_vendor_device_configs_remain_vendor_only(self):
        torch_npu = types.ModuleType("torch_npu")
        torch_npu.__file__ = "/opt/torch_npu/torch_npu/__init__.py"
        torch_gcu = types.ModuleType("torch_gcu")
        torch_gcu.__file__ = "/opt/torch_gcu/torch_gcu/__init__.py"

        with self.clean_environment(), mock.patch.dict(
            sys.modules,
            {"torch_npu": torch_npu, "torch_gcu": torch_gcu},
        ):
            ascend_backend = build_config.resolve_torch_backend("ascend")
            ascend = build_config.get_device_config(
                build_config.ADAPTOR_MAP["ascend"], ascend_backend
            )
            enflame_backend = build_config.resolve_torch_backend("enflame")
            enflame = build_config.get_device_config(
                build_config.ADAPTOR_MAP["enflame"], enflame_backend
            )

        self.assertEqual(ascend[0], ["/opt/torch_npu/torch_npu/include"])
        self.assertEqual(ascend[1], ["/opt/torch_npu/torch_npu/lib"])
        self.assertEqual(ascend[2], ["torch_npu"])
        self.assertEqual(
            enflame,
            (
                ["/opt/tops/include", "/opt/torch_gcu/torch_gcu/include"],
                ["/opt/tops/lib", "/opt/torch_gcu/torch_gcu/lib"],
                ["topsrt", "torch_gcu"],
            ),
        )
        self.assertNotIn("flagos", ascend[2])
        self.assertNotIn("flagos", enflame[2])

    def test_flagos_device_configs_do_not_load_vendor_packages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            install_path = Path(temp_dir)
            include_dir = install_path / "include"
            library_dir = install_path / "lib"
            include_dir.mkdir()
            library_dir.mkdir()
            (include_dir / "flagos.h").touch()
            (library_dir / "libflagos.so").touch()

            with self.clean_environment(
                FLAGCX_TORCH_BACKEND="flagos",
                FLAGOS_INSTALL_PATH=str(install_path),
                ASCEND_HOME_PATH="/opt/ascend",
            ):
                ascend_backend = build_config.resolve_torch_backend("ascend")
                ascend = build_config.get_device_config(
                    build_config.ADAPTOR_MAP["ascend"], ascend_backend
                )
                enflame_backend = build_config.resolve_torch_backend("enflame")
                enflame = build_config.get_device_config(
                    build_config.ADAPTOR_MAP["enflame"], enflame_backend
                )

        self.assertEqual(
            ascend,
            (
                ["/opt/ascend/include", str(include_dir)],
                ["/opt/ascend/lib64", str(library_dir)],
                ["ascendcl", "flagos"],
            ),
        )
        self.assertEqual(
            enflame,
            (
                ["/opt/tops/include", str(include_dir)],
                ["/opt/tops/lib", str(library_dir)],
                ["topsrt", "flagos"],
            ),
        )
        self.assertNotIn("torch_npu", sys.modules)
        self.assertNotIn("torch_gcu", sys.modules)

    def test_rpaths_preserve_flagos_library_directory(self):
        paths = ["/opt/device/lib", "/opt/flagos/lib"]
        self.assertEqual(
            build_config.get_device_rpath_dirs("unused", paths), paths
        )


if __name__ == "__main__":
    unittest.main()
