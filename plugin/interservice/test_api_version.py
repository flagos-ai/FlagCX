"""Regression tests for FlagCX integer version code decoding, no native .so."""

import ctypes
import importlib.util
from pathlib import Path
import sys
import unittest

api_path = Path(__file__).resolve().parents[2] / "src" / "flagcx" / "api.py"
spec = importlib.util.spec_from_file_location("flagcx_api_version_test", api_path)
api_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api_module
spec.loader.exec_module(api_module)
FLAGCXLibrary = api_module.FLAGCXLibrary


class FlagcxVersionTest(unittest.TestCase):
    def _decode(self, code, status=0):
        def fake_get_version(version_ptr):
            ctypes.cast(version_ptr, ctypes.POINTER(ctypes.c_int))[0] = code
            return status

        class FakeLibrary:
            _funcs = {"flagcxGetVersion": fake_get_version}

            def FLAGCX_CHECK(self, result):
                if result:
                    raise RuntimeError(f"FLAGCX status {result}")

        return FLAGCXLibrary.flagcxGetVersion(FakeLibrary())

    def test_version_encoding_round_trip(self):
        for code, expected in (
            (0, "0.0.0"),
            (1, "0.0.1"),
            (1300, "0.13.0"),
            (10000, "1.0.0"),
            (10101, "1.1.1"),
            (21903, "2.19.3"),
            (999999, "99.99.99"),
        ):
            with self.subTest(code=code):
                self.assertEqual(self._decode(code), expected)

    def test_api_errors_are_still_propagated(self):
        with self.assertRaisesRegex(RuntimeError, "FLAGCX status 3"):
            self._decode(10000, status=3)


if __name__ == "__main__":
    unittest.main()
