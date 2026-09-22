"""Configure an existing FlagCX IB engine via its bootstrap/RPC port.

No FlagCX library, torch, RDMA device or SGLang import is needed by the client.
The bootstrap envelope is native endian, matching FlagCX's existing protocol;
client and server must have the same byte order.
"""

import argparse
import json
import os
import socket
import struct

_MAGIC = 0x564AB9F2FC4B9D6C
_TAG = 0x46585431
_HEADER = struct.Struct("=ii")
_RUNTIME_KEYS = ("FLAGCX_P2P_SLICE_SIZE", "FLAGCX_P2P_FRAGMENT_LIMIT")


def environment_values():
    """Only explicitly exported runtime keys are sent; absent keys stay unchanged."""
    values = {}
    for key in _RUNTIME_KEYS:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            value = int(raw, 16 if raw.lower().startswith("0x") else 10)
        except ValueError:
            raise ValueError(f"invalid {key}={raw!r}: expected integer bytes") from None
        if not 0 <= value <= 1 << 30:
            raise ValueError(f"{key} must be in [0, 1 GiB]")
        values[key] = value
    return values


def _receive(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise RuntimeError("engine closed the connection; check runtime-control support")
        chunks.extend(chunk)
    return bytes(chunks)


def configure(endpoint, values=None, timeout=5.0):
    """SET a dict of environment-variable names to integers, or GET if empty.

    Success acknowledges publication for subsequent submissions, not draining
    existing transfers. A timeout has an uncertain outcome: query with GET.
    """
    host, port = endpoint.rsplit(":", 1)
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    assignments = []
    for key, value in (values or {}).items():
        if key not in _RUNTIME_KEYS:
            raise ValueError(f"parameter is not runtime tunable: {key}")
        if type(value) is not int or not 0 <= value <= 1 << 30:
            raise ValueError(f"{key} must be an integer in [0, 1 GiB]")
        assignments.append(f"{key}={value}")
    payload = ("\n".join(assignments) if assignments else "GET").encode("ascii")
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        # Socket handshake: uint64 magic, int32 Bootstrap type. It has no ACK.
        sock.sendall(struct.pack("=Qi", _MAGIC, 1) +
                     _HEADER.pack(_TAG, len(payload)) + payload)
        tag, length = _HEADER.unpack(_receive(sock, _HEADER.size))
        if tag != _TAG or not 0 < length <= 256:
            raise RuntimeError("invalid control response (wrong port or old engine)")
        result = json.loads(_receive(sock, length))
    if result.get("status") != 0:
        raise RuntimeError(f"{endpoint}: {result.get('error', 'control request failed')}")
    if not all(type(result.get(k)) is int for k in ("slice_size", "fragment_limit")):
        raise RuntimeError("engine response is missing effective configuration")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="append", required=True,
                        help="engine host:port; repeat for all sender ranks")
    parser.add_argument("--get", action="store_true",
                        help="query only; ignore exported runtime parameters")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    try:
        values = {} if args.get else environment_values()
    except ValueError as error:
        parser.error(str(error))
    for endpoint in args.engine:
        try:
            result = configure(endpoint, values, args.timeout)
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"{endpoint}: {error}. Earlier engines may already be updated; "
                           "do not start the benchmark until all ranks agree.\n")
        print(json.dumps({"engine": endpoint, **result}), flush=True)


if __name__ == "__main__":
    main()
