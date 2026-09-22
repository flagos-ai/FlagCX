import json
import struct
import unittest
from unittest import mock

import p2p_config


class _FakeSocket:
    def __init__(self, response):
        self._response = bytearray(response)
        self.sent = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def sendall(self, data):
        self.sent.extend(data)

    def recv(self, size):
        data = self._response[:size]
        del self._response[:size]
        return bytes(data)


class P2pConfigTest(unittest.TestCase):
    def _configure(self, endpoint):
        result = {
            "status": 0,
            "slice_size": 65536,
            "fragment_limit": 16384,
            "error": "",
        }
        payload = json.dumps(result).encode("ascii")
        response = struct.pack("=ii", p2p_config._TAG, len(payload)) + payload
        sock = _FakeSocket(response)
        with mock.patch.object(
            p2p_config.socket, "create_connection", return_value=sock
        ) as connect:
            actual = p2p_config.configure(endpoint, timeout=1.0)
        return actual, connect.call_args, sock

    def test_ipv4_endpoint(self):
        result, call, sock = self._configure("127.0.0.1:1234")
        self.assertEqual(result["slice_size"], 65536)
        self.assertEqual(call.args, (("127.0.0.1", 1234),))
        self.assertEqual(call.kwargs, {"timeout": 1.0})
        self.assertTrue(sock.sent.endswith(b"GET"))

    def test_bracketed_ipv6_endpoint(self):
        _, call, _ = self._configure("[::1]:2345")
        self.assertEqual(call.args, (("::1", 2345),))

    def test_ipv6_scope_endpoint(self):
        _, call, _ = self._configure("[fe80::1%eth0]:3456")
        self.assertEqual(call.args, (("fe80::1%eth0", 3456),))

    def test_only_matching_brackets_are_removed(self):
        _, call, _ = self._configure("[localhost:4567")
        self.assertEqual(call.args, (("[localhost", 4567),))


if __name__ == "__main__":
    unittest.main()
