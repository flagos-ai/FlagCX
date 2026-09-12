import ctypes
import os
import tempfile
import unittest
from unittest import mock

import plugin.interservice.flagcx_wrapper as flagcx_wrapper
from plugin.interservice.flagcx_wrapper import (
    FLAGCX_UNIQUE_ID_BYTES,
    FLAGCXLibrary,
    flagcxUniqueId,
)


class FlagcxUniqueIdTest(unittest.TestCase):
    def test_size_matches_public_abi(self):
        self.assertEqual(FLAGCX_UNIQUE_ID_BYTES, 256)
        self.assertEqual(ctypes.sizeof(flagcxUniqueId), FLAGCX_UNIQUE_ID_BYTES)

    def test_round_trip_preserves_public_id(self):
        data = bytes(i % 256 for i in range(FLAGCX_UNIQUE_ID_BYTES))

        unique_id = FLAGCXLibrary.unique_id_from_bytes(None, data)

        round_trip = bytes(unique_id.internal)
        self.assertEqual(round_trip, data)

    def test_rejects_incorrect_sizes(self):
        for size in (FLAGCX_UNIQUE_ID_BYTES - 1, FLAGCX_UNIQUE_ID_BYTES + 1):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, str(size)):
                    FLAGCXLibrary.unique_id_from_bytes(None, bytes(size))


class FlagcxDefaultLibraryTest(unittest.TestCase):
    """The resolver must keep searching when FLAGCX_PATH is set but unusable."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("FLAGCX_PATH", None)
        # Move the package directory out of the source tree so the search is
        # not satisfied by this checkout's own build/ directory.
        self.module_file = os.path.join(
            self.root, "src", "plugin", "interservice", "flagcx_wrapper.py"
        )
        module_file = mock.patch.object(
            flagcx_wrapper, "__file__", self.module_file
        )
        module_file.start()
        self.addCleanup(module_file.stop)

    @staticmethod
    def _touch(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w"):
            pass
        return path

    def test_flagcx_path_lib_dir_wins(self):
        os.environ["FLAGCX_PATH"] = self.root
        expected = self._touch(
            os.path.join(self.root, "lib", "libflagcx.so")
        )

        self.assertEqual(FLAGCXLibrary._find_default_library(), expected)

    def test_packaged_soname_under_flagcx_path(self):
        os.environ["FLAGCX_PATH"] = self.root
        expected = self._touch(
            os.path.join(self.root, "lib", "libflagcx.so.0")
        )

        self.assertEqual(FLAGCXLibrary._find_default_library(), expected)

    def test_unusable_flagcx_path_still_searches_package_dir(self):
        barren = self._touch(os.path.join(self.root, "keep", "file"))
        os.environ["FLAGCX_PATH"] = os.path.dirname(barren)
        expected = self._touch(
            os.path.join(os.path.dirname(self.module_file), "libflagcx.so")
        )

        self.assertEqual(FLAGCXLibrary._find_default_library(), expected)

    def test_falls_back_to_system_library(self):
        with mock.patch(
            "ctypes.util.find_library", return_value="libflagcx.so.0"
        ) as find_library:
            found = FLAGCXLibrary._find_default_library()

        find_library.assert_called_once_with("flagcx")
        self.assertEqual(found, "libflagcx.so.0")

    def test_reports_every_searched_location(self):
        os.environ["FLAGCX_PATH"] = self.root
        with mock.patch("ctypes.util.find_library", return_value=None):
            with self.assertRaises(FileNotFoundError) as caught:
                FLAGCXLibrary._find_default_library()

        message = str(caught.exception)
        for path in (
            os.path.join(self.root, "lib", "libflagcx.so"),
            os.path.join(self.root, "build", "lib", "libflagcx.so"),
            os.path.join(self.root, "lib", "libflagcx.so.0"),
            os.path.join(
                os.path.dirname(self.module_file), "libflagcx.so"
            ),
            os.path.join(self.root, "src", "build", "lib", "libflagcx.so"),
        ):
            self.assertIn(path, message)
        self.assertIn("system library lookup", message)


if __name__ == "__main__":
    unittest.main()
