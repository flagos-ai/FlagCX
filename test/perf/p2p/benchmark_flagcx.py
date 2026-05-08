#!/usr/bin/env python3
"""
NIXL P2P benchmark — FlagCX backend.

Mirrors uccl/p2p/benchmarks/benchmark_nixl.py (Mooncake/UCCL paths), using a
ZMQ sidechannel for agent metadata and descriptor exchange so the FlagCX
plugin's 256-byte notification cap is never hit.

FlagCX P2P is internode-only today (intra-node IPC not yet implemented).
Run server and client on different hosts with RDMA NICs.

Requirements
------------
- NIXL built with the FlagCX plugin  (libplugin_FLAGCX.so on the plugin path)
- libflagcx.so reachable via LD_LIBRARY_PATH
- pyzmq, torch
- Matching PyTorch CUDA major version with installed nixl-cuXX wheel

Usage
-----
  # server (IP 10.8.2.169)
  python3 benchmark_flagcx.py --role=server --remote-ip=10.8.2.169 \\
      --device=gpu --local-gpu-idx=0 --op-type=read \\
      --sizes=1048576,4194304,16777216,104857600 --iters=20

  # client
  python3 benchmark_flagcx.py --role=client --remote-ip=10.8.2.169 \\
      --device=gpu --local-gpu-idx=0 --op-type=read \\
      --sizes=1048576,4194304,16777216,104857600 --iters=20
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import traceback
from typing import List

import torch
import zmq

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:
    sys.stderr.write(
        "Failed to import NIXL. Is the nixl Python package installed in this env?\n"
    )
    raise


ZMQ_PORT = 9000  # coordination channel; distinct from the RDMA dataplane


# ------------------------------------------------------------------ dataset

def create_dataset(
    role: str, size: int, num_kvblocks: int, device: str, gpu_idx: int = 0
) -> List[torch.Tensor]:
    """Allocate `num_kvblocks` tensors totalling >= `size` bytes.

    server buffers start at 0; client buffers start at 1. After a WRITE
    (client->server) the server should hold 1s; after a READ (client<-server)
    the client should hold 0s.
    """
    dtype = torch.float32 if size >= 4 else torch.uint8
    value = 0 if "server" in role else 1

    element_size = torch.tensor([], dtype=dtype).element_size()
    n_elems_per_block = max(size // (element_size * num_kvblocks), 1)
    dev = f"cuda:{gpu_idx}" if device == "gpu" else "cpu"

    blocks = [
        torch.full((n_elems_per_block,), value, device=dev, dtype=dtype)
        for _ in range(num_kvblocks)
    ]

    total = sum(t.numel() * t.element_size() for t in blocks)
    if total < size:
        extra = (size - total) // element_size
        if extra > 0:
            blocks.append(torch.full((extra,), value, device=dev, dtype=dtype))
    return blocks


# ------------------------------------------------------------------ zmq

def init_zmq(host: str, port: int, role: str) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PAIR)
    if "server" in role:
        sock.bind(f"tcp://{host}:{port}")
    else:
        sock.connect(f"tcp://{host}:{port}")
        sock.setsockopt(zmq.LINGER, 0)
    return sock


# ------------------------------------------------------------------ agent

def create_flagcx_agent(role: str, dataset, zmq_sock: zmq.Socket):
    """Instantiate a NIXL agent bound to the FlagCX backend, exchange
    opaque agent-metadata blobs via ZMQ, and register `dataset` for RDMA."""
    config = nixl_agent_config(backends=["FLAGCX"])
    agent = nixl_agent(role, config)

    reg_descs = agent.register_memory(agent.get_reg_descs(dataset))
    local_meta = agent.get_agent_metadata()

    if "client" in role:
        zmq_sock.send(local_meta)
        remote_meta = zmq_sock.recv()
    else:
        remote_meta = zmq_sock.recv()
        zmq_sock.send(local_meta)

    agent.add_remote_agent(remote_meta)
    return agent, reg_descs


# ------------------------------------------------------------------ transfer

def init_transfer(role: str, op: str, agent, reg_descs, zmq_sock: zmq.Socket):
    """Per-iteration handshake: server ships its serialized descriptors to the
    client via ZMQ, client builds a transfer handle. Returns the handle on
    client, None on server."""
    local_xfer = reg_descs.trim()

    if "server" in role:
        msg = zmq_sock.recv().decode("utf-8")
        if msg != "START":
            raise RuntimeError(f"server got unexpected handshake: {msg!r}")
        zmq_sock.send(agent.get_serialized_descs(local_xfer))
        return None

    zmq_sock.send(b"START")
    remote_ser = zmq_sock.recv()
    remote_xfer = agent.deserialize_descs(remote_ser)
    return agent.initialize_xfer(op, local_xfer, remote_xfer, "server")


def run_transfer(role: str, agent, handle, zmq_sock: zmq.Socket,
                 uid: str = "DONE") -> None:
    """Client posts and polls the transfer; server blocks on a ZMQ ack from
    the client so the benchmark can include end-to-end completion."""
    if "client" in role:
        state = agent.transfer(handle)
        assert state != "ERR", "transfer post failed"
        while True:
            state = agent.check_xfer_state(handle)
            assert state != "ERR", "transfer errored"
            if state == "DONE":
                zmq_sock.send(uid.encode("utf-8"))
                return
    else:
        while zmq_sock.recv().decode("utf-8") != uid:
            pass


# ------------------------------------------------------------------ driver

def benchmark_size(size: int, num_kvblocks: int, args) -> None:
    op = "WRITE" if args.op_type == "write" else "READ"
    zmq_sock = init_zmq(args.remote_ip, ZMQ_PORT, args.role)

    agent = None
    handle = None
    reg_descs = None

    try:
        dataset = create_dataset(
            args.role, size, num_kvblocks, args.device, args.local_gpu_idx
        )

        # Silence NIXL's chatty agent-init logs so the perf numbers stand out.
        saved_stdout = sys.stdout
        sys.stdout = io.StringIO()
        agent, reg_descs = create_flagcx_agent(args.role, dataset, zmq_sock)
        sys.stdout = saved_stdout

        warmup = 1 if args.iters > 1 else 0
        total_bytes = 0
        total_time = 0.0
        last_time = 0.0

        for i in range(args.iters):
            handle = init_transfer(args.role, op, agent, reg_descs, zmq_sock)

            t0 = time.perf_counter()
            run_transfer(args.role, agent, handle, zmq_sock)
            last_time = time.perf_counter() - t0

            if i >= warmup:
                total_time += last_time
                total_bytes += size

            if "client" in args.role:
                iter_bw = size / last_time / 1e9
                tag = "WARM" if i < warmup else f"{i:4d}"
                print(
                    f"  [{tag}] {last_time*1000:8.3f} ms   {iter_bw:7.2f} GB/s",
                    flush=True,
                )

            if handle is not None:
                agent.release_xfer_handle(handle)
                handle = None

        if total_time == 0.0:
            total_time = last_time
            total_bytes = size
        effective = max(args.iters - warmup, 1)
        avg_lat = total_time / effective
        gb_sec = total_bytes / total_time / 1e9
        gbps = gb_sec * 8

        print(
            f"[{args.role}] {_pretty_size(size):>8}  op={op}  "
            f"{gbps:7.2f} Gbps  {gb_sec:7.2f} GB/s  "
            f"lat={avg_lat*1e3:.3f} ms",
            flush=True,
        )

        _verify(op, args.role, dataset)

    except KeyboardInterrupt:
        return
    except Exception:
        print(
            f"[{args.role}] error at size {size}: {traceback.format_exc()}",
            flush=True,
        )
    finally:
        if handle is not None and agent is not None:
            try:
                agent.release_xfer_handle(handle)
            except Exception:
                pass
        if reg_descs is not None and agent is not None:
            try:
                agent.deregister_memory(reg_descs)
            except Exception:
                pass
        if agent is not None:
            peer = "server" if agent.name == "client" else "client"
            try:
                agent.remove_remote_agent(peer)
            except Exception:
                pass
        zmq_sock.close()


def _verify(op: str, role: str, dataset) -> None:
    """Cheap correctness check on the destination buffer."""
    if op == "WRITE" and "server" in role:
        expected = 1.0
        who = "server"
    elif op == "READ" and "client" in role:
        expected = 0.0
        who = "client"
    else:
        return
    for idx, blk in enumerate(dataset):
        mean = torch.mean(blk.float()).item()
        if abs(mean - expected) > 1e-6:
            print(
                f"[{role}] VERIFY FAIL: {who} block {idx} mean={mean} "
                f"expected={expected}",
                flush=True,
            )
            return


# ------------------------------------------------------------------ cli

def _pretty_size(n: int) -> str:
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024 or unit == "GB":
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{n} B"


def _parse_sizes(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--sizes must be comma-separated integers (bytes)"
        ) from e


def main() -> None:
    p = argparse.ArgumentParser(
        description="NIXL P2P benchmark using the FlagCX backend"
    )
    p.add_argument("--role", choices=["server", "client"], required=True,
                   help="server listens; client initiates the transfer")
    p.add_argument("--remote-ip", default="0.0.0.0",
                   help="server IP — client dials it, server binds it")
    p.add_argument("--local-gpu-idx", type=int, default=0,
                   help="CUDA device index for the local buffer")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument(
        "--sizes", type=_parse_sizes,
        default=[1 << s for s in (10, 14, 18, 20, 22, 24, 26, 27)],
        help="comma-separated message sizes in bytes",
    )
    p.add_argument("--iters", type=int, default=10,
                   help="iterations per size (first is a warm-up)")
    p.add_argument("--num-kvblocks", type=int, default=1,
                   help="number of tensor blocks per transfer")
    p.add_argument("--op-type", choices=["write", "read"], default="read",
                   help="one-sided WRITE (put) or READ (get)")
    args = p.parse_args()

    print(f"NIXL P2P benchmark (backend=FLAGCX)  role={args.role}")
    print(
        f"device={args.device}  gpu={args.local_gpu_idx}  "
        f"op={args.op_type}  iters={args.iters}  num_kvblocks={args.num_kvblocks}"
    )
    print("sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print("-" * 72)

    for size in args.sizes:
        benchmark_size(size, args.num_kvblocks, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(1)
