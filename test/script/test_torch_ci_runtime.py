import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "plugin/torch/example/ci_runtime.py"
SPEC = importlib.util.spec_from_file_location("ci_runtime", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TorchCiRuntimeTest(unittest.TestCase):
    def test_rejects_preinstalled_library_even_when_current_library_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "build/lib/libflagcx.so"
            package = root / "src/flagcx/lib/libflagcx.so"
            old = root / "installed/libflagcx.so"
            for library in (built, package, old):
                library.parent.mkdir(parents=True, exist_ok=True)
            built.write_bytes(b"current library")
            package.write_bytes(b"current library")
            old.write_bytes(b"old library")
            maps = root / "maps"
            with patch.dict(os.environ, {"FLAGCX_CI_EXPECT_FLAGCX_LIBRARY": str(built)}):
                maps.write_text(f"0-1 r-xp 0 00:00 0 {package}\n")
                MODULE.verify_ci_flagcx_library(maps)

                maps.write_text(
                    f"0-1 r-xp 0 00:00 0 {package}\n"
                    f"1-2 r-xp 0 00:00 0 {old}\n"
                )
                with self.assertRaisesRegex(RuntimeError, "differs from this checkout"):
                    MODULE.verify_ci_flagcx_library(maps)

                old.write_bytes(b"current library")
                MODULE._digest.cache_clear()
                self.assertTrue(MODULE.verify_ci_flagcx_library(maps))

                maps.write_text("")
                self.assertFalse(MODULE.verify_ci_flagcx_library(
                    maps, stage="after imports", require_loaded=False))
                with self.assertRaisesRegex(RuntimeError, "no mapped libflagcx.so"):
                    MODULE.verify_ci_flagcx_library(maps)


if __name__ == "__main__":
    unittest.main()
