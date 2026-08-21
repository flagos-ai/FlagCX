import ctypes
import unittest

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


if __name__ == "__main__":
    unittest.main()
