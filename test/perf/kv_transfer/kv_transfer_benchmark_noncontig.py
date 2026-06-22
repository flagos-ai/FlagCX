#!/usr/bin/env python3
"""
Non-contiguous KV transfer microbenchmark (NIXL / Mooncake / FlagCX).

For each requested transfer size, the payload is split into fixed-size blocks
(default 8 KB, ``--block-bytes``) scattered at random, non-overlapping,
block-aligned offsets inside a large pool buffer, and one batched RDMA transfer
moves them all. Measures wire latency + bandwidth. ZMQ coordinates the two
sides out of band.

The pool is registered ONCE, split into <= max_mr_size chunks (a single
oversized MR is silently shrunk by the NIC, leaving high offsets unreachable).

Usage
-----
  # server (receiver)            # client (sender / initiator)
  python kv_transfer_noncontiguous.py --connector=mooncake --role=server \\
      --remote-ip=10.8.2.169
  python kv_transfer_noncontiguous.py --connector=mooncake --role=client \\
      --remote-ip=10.8.2.169

  # connectors: --connector=nixl|mooncake|flagcx
  # tune block size: --block-bytes 16384
  # tune sizes:      --sizes 64M,256M,1G
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import torch
import zmq

# A single RDMA MR is capped by the NIC's max_mr_size; register the pool as
# several <= 1 GiB chunks instead of one huge region. Boundaries are aligned to
# block_bytes so no single-block WR ever straddles two regions.
_MR_CHUNK_BYTES = 1 << 30


@dataclass
class Pattern:
    """One step: ``n`` scattered block transfers (each ``block_bytes``)."""
    src_offsets: List[int]   # byte offset of each WR in the src pool
    dst_offsets: List[int]   # byte offset of each WR in the dst pool
    tags: List[int]          # 1..255, written by src / checked by dst
    block_bytes: int
    pool_bytes: int

    @property
    def n(self) -> int:
        return len(self.tags)

    @property
    def total_bytes(self) -> int:
        return self.n * self.block_bytes


def make_pattern(total_bytes: int, block_bytes: int, pool_bytes: int,
                 seed: int) -> Pattern:
    """Scatter ``total_bytes/block_bytes`` blocks at random offsets in the pool.

    Deterministic from ``seed`` so both sides build the identical mapping
    without exchanging it. Src and dst use independent scatters (different
    engines fragment differently); offsets are distinct (no overlap).
    """
    pool_blocks = pool_bytes // block_bytes
    n = max(1, total_bytes // block_bytes)
    if n > pool_blocks:
        raise ValueError(f"need {n} blocks but pool holds {pool_blocks}")
    src = random.Random(seed).sample(range(pool_blocks), n)
    dst = random.Random(seed ^ 0x9E3779B9).sample(range(pool_blocks), n)
    return Pattern(
        src_offsets=[b * block_bytes for b in src],
        dst_offsets=[b * block_bytes for b in dst],
        tags=[(i % 255) + 1 for i in range(n)],
        block_bytes=block_bytes,
        pool_bytes=pool_bytes,
    )


def _mr_chunks(base: int, total: int, block_bytes: int) -> Tuple[List[int], List[int]]:
    """Split ``[base, base+total)`` into block-aligned, <= chunk-sized regions.

    First entry's address is ``base``, so callers advertising one base VA stay
    correct. ``base+offset`` addressing is unaffected: each VA lands in exactly
    one region (Mooncake resolves by VA; FlagCX exchanges every region's rkey).
    """
    chunk = max(block_bytes, (_MR_CHUNK_BYTES // block_bytes) * block_bytes)
    ptrs: List[int] = []
    lens: List[int] = []
    off = 0
    while off < total:
        n = min(chunk, total - off)
        ptrs.append(base + off)
        lens.append(n)
        off += n
    return ptrs, lens


def _init_zmq(host: str, port: int, role: str) -> zmq.Socket:
    sock = zmq.Context.instance().socket(zmq.PAIR)
    if "server" in role:
        sock.bind(f"tcp://{host}:{port}")
    else:
        sock.connect(f"tcp://{host}:{port}")
        sock.setsockopt(zmq.LINGER, 0)
    return sock


def _pretty(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{n} B"


def _parse_sizes(val: str) -> List[int]:
    mul = {"k": 1024, "m": 1024**2, "g": 1024**3}
    out = []
    for tok in val.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok[-1] in mul:
            out.append(int(float(tok[:-1]) * mul[tok[-1]]))
        else:
            out.append(int(tok))
    return out


# ---------------------------------------------------------------------------


class Transport(ABC):
    """Common interface. Source side fills/holds the data; dest side verifies."""

    SOURCE_ROLE = "client"   # which role owns the source data

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.role = args.role
        self.device = args.device
        self.gpu_idx = args.local_gpu_idx
        self.remote_ip = args.remote_ip
        self.zmq_port = args.zmq_port
        self.buffer: "torch.Tensor | None" = None
        self._pat: "Pattern | None" = None

    @property
    def is_source(self) -> bool:
        return self.SOURCE_ROLE in self.role

    def _alloc_pool(self, pool_bytes: int) -> None:
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"
        self.buffer = torch.zeros((pool_bytes,), device=dev, dtype=torch.uint8)

    def _refresh(self, pat: Pattern) -> None:
        """Dest zeroes its pool; source zeroes then stamps each WR's tag."""
        self._pat = pat
        self.buffer.zero_()
        if self.is_source:
            for off, tag in zip(pat.src_offsets, pat.tags):
                self.buffer[off:off + pat.block_bytes] = tag
        if self.device == "gpu" and torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu_idx)

    def verify(self) -> None:
        if self.is_source:
            return
        pat = self._pat
        for i, (off, tag) in enumerate(zip(pat.dst_offsets, pat.tags)):
            region = self.buffer[off:off + pat.block_bytes]
            if not torch.all(region == tag).item():
                bad = int((region != tag).sum().item())
                raise AssertionError(
                    f"[{self.role}] VERIFY FAIL: WR {i} @dst {off} "
                    f"(tag={tag}) has {bad}/{pat.block_bytes} bytes wrong.")

    @abstractmethod
    def setup_pool(self, pool_bytes: int) -> None: ...
    @abstractmethod
    def prepare_step(self, pat: Pattern) -> None: ...
    @abstractmethod
    def run_transfer(self) -> None: ...
    @abstractmethod
    def teardown(self) -> None: ...


# ---------------------------------------------------------------------------


class NixlTransport(Transport):
    """NIXL agent, READ semantics: server holds source, client reads it."""

    SOURCE_ROLE = "server"

    def __init__(self, args):
        super().__init__(args)
        from nixl._api import nixl_agent, nixl_agent_config
        self._agent_cls = nixl_agent
        self._cfg_cls = nixl_agent_config
        self.backend = args.nixl_backend
        self.agent = None
        self.reg = None
        self.handle = None
        self.sock = None
        self._xfer = None

    def setup_pool(self, pool_bytes):
        self._alloc_pool(pool_bytes)
        self.sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self.agent = self._agent_cls(self.role, self._cfg_cls(backends=[self.backend]))
        self.reg = self.agent.register_memory(self.agent.get_reg_descs([self.buffer]))
        meta = self.agent.get_agent_metadata()
        if "client" in self.role:
            self.sock.send(meta)
            peer = self.sock.recv()
        else:
            peer = self.sock.recv()
            self.sock.send(meta)
        self.agent.add_remote_agent(peer)

    def prepare_step(self, pat):
        self._refresh(pat)
        offs = pat.src_offsets if self.is_source else pat.dst_offsets
        base = self.buffer.data_ptr()
        mem = "VRAM" if self.device == "gpu" else "DRAM"
        devid = self.gpu_idx if self.device == "gpu" else 0
        descs = [(base + o, pat.block_bytes, devid) for o in offs]
        self._xfer = self.agent.get_xfer_descs(descs, mem)

    def run_transfer(self):
        if "server" in self.role:
            assert self.sock.recv().decode() == "START"
            self.sock.send(self.agent.get_serialized_descs(self._xfer))
            while self.sock.recv().decode() != "DONE":
                pass
            return
        self.sock.send(b"START")
        remote = self.agent.deserialize_descs(self.sock.recv())
        self.handle = self.agent.initialize_xfer("READ", self._xfer, remote, "server")
        assert self.agent.transfer(self.handle) != "ERR"
        while True:
            st = self.agent.check_xfer_state(self.handle)
            assert st != "ERR"
            if st == "DONE":
                break
        self.sock.send(b"DONE")
        self.agent.release_xfer_handle(self.handle)
        self.handle = None

    def teardown(self):
        if self.reg is not None:
            try: self.agent.deregister_memory(self.reg)
            except Exception: pass
        if self.agent is not None:
            try: self.agent.remove_remote_agent(
                "server" if "client" in self.role else "client")
            except Exception: pass
        if self.sock is not None:
            self.sock.close()


# ---------------------------------------------------------------------------


class MooncakeTransport(Transport):
    """Mooncake TransferEngine, WRITE: client writes into server's pool."""

    def __init__(self, args):
        super().__init__(args)
        from mooncake.engine import TransferEngine
        self._engine_cls = TransferEngine
        self.protocol = args.mooncake_protocol
        self.engine = None
        self.sock = None
        self.remote_session = ""
        self._remote_base = 0
        self._wr = ([], [], [])  # src VAs, dst VAs, sizes

    def setup_pool(self, pool_bytes):
        import json, socket
        self._alloc_pool(pool_bytes)
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            local_ip = "127.0.0.1"
        self.engine = self._engine_cls()
        if self.engine.initialize(local_ip, "P2PHANDSHAKE", self.protocol, "") != 0:
            raise RuntimeError("Mooncake initialize failed")
        rpc_port = self.engine.get_rpc_port()

        base = self.buffer.data_ptr()
        ptrs, lens = _mr_chunks(base, pool_bytes, self.args.block_bytes)
        if self.engine.batch_register_memory(ptrs, lens) != 0:
            raise RuntimeError("Mooncake registration failed")

        self.sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        info = json.dumps({"session": f"{local_ip}:{rpc_port}", "base": base}).encode()
        if "server" in self.role:
            remote = json.loads(self.sock.recv().decode()); self.sock.send(info)
        else:
            self.sock.send(info); remote = json.loads(self.sock.recv().decode())
        self.remote_session = remote["session"]
        self._remote_base = remote["base"]

    def prepare_step(self, pat):
        self._refresh(pat)
        if self.is_source:
            base, rb, bs = self.buffer.data_ptr(), self._remote_base, pat.block_bytes
            self._wr = ([base + o for o in pat.src_offsets],
                        [rb + o for o in pat.dst_offsets],
                        [bs] * pat.n)
            self.sock.send(b"READY")          # barrier: receiver zeroed its pool
        else:
            self.sock.recv()

    def run_transfer(self):
        if "client" in self.role:
            if self.engine.batch_transfer_sync_write(self.remote_session, *self._wr) != 0:
                raise RuntimeError("Mooncake transfer failed")
            self.sock.send(b"DONE")
        else:
            while self.sock.recv().decode() != "DONE":
                pass

    def teardown(self):
        if self.sock is not None:
            self.sock.close()


# ---------------------------------------------------------------------------


class FlagCXTransport(Transport):
    """FlagCX one-sided RDMA, WRITE: client writes into server's pool."""

    def __init__(self, args):
        super().__init__(args)
        import os
        path = args.flagcx_path or os.getenv("FLAGCX_PATH") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        wrapper = os.path.join(path, "plugin", "interservice")
        if wrapper not in sys.path:
            sys.path.insert(0, wrapper)
        from flagcx_wrapper import FLAGCXLibrary
        lib = args.flagcx_lib_path or os.path.join(path, "build", "lib", "libflagcx.so")
        self.flagcx = FLAGCXLibrary(lib)
        self.engine = None
        self.conn = None
        self.sock = None
        self._remote_base = 0
        self._wr = ([], [], [])

    def setup_pool(self, pool_bytes):
        import json, socket
        self._alloc_pool(pool_bytes)
        self.sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self.engine = self.flagcx.flagcxP2pEngineCreate()

        base = self.buffer.data_ptr()
        # Register the pool in chunks BEFORE the connection handshake so the
        # desc-table exchange ships every region's rkey to the peer.
        for ptr, length in zip(*_mr_chunks(base, pool_bytes, self.args.block_bytes)):
            self.flagcx.flagcxP2pRegister(self.engine, ptr, length)

        if "server" in self.role:
            self.flagcx.flagcxP2pStartRpcServer(self.engine)
            rpc_port = self.flagcx.flagcxP2pGetRpcPort(self.engine)
            host = self.remote_ip
            if host in ("", "0.0.0.0", None):
                host = socket.gethostbyname(socket.gethostname())
            self.sock.send(json.dumps({"session": f"{host}:{rpc_port}",
                                       "base": base}).encode())
            self.sock.recv()                  # wait for sender to connect
        else:
            remote = json.loads(self.sock.recv().decode())
            self._remote_base = int(remote["base"])
            self.conn = self.flagcx.flagcxP2pGetConn(self.engine, remote["session"])
            self.sock.send(b"READY")

    def prepare_step(self, pat):
        self._refresh(pat)
        if self.is_source:
            base, rb, bs = self.buffer.data_ptr(), self._remote_base, pat.block_bytes
            self._wr = ([base + o for o in pat.src_offsets],
                        [rb + o for o in pat.dst_offsets],
                        [bs] * pat.n)
            self.sock.send(b"READY")          # barrier: receiver zeroed its pool
        else:
            self.sock.recv()

    def run_transfer(self):
        if "server" in self.role:
            while self.sock.recv() != b"DONE":
                pass
            return
        self.flagcx.flagcxP2pBatchWriteSync(self.conn, *self._wr)
        self.sock.send(b"DONE")

    def teardown(self):
        if self.engine is not None:
            try: self.flagcx.flagcxP2pEngineDestroy(self.engine)
            except Exception: pass
        if self.sock is not None:
            self.sock.close()


# ---------------------------------------------------------------------------


def run_size(t: Transport, pat: Pattern, iters: int, warmup: int) -> None:
    t.prepare_step(pat)
    for _ in range(warmup):
        t.run_transfer()
    if torch.cuda.is_available() and t.device == "gpu":
        torch.cuda.synchronize(t.gpu_idx)
    start = time.perf_counter()
    for _ in range(iters):
        t.run_transfer()
    if torch.cuda.is_available() and t.device == "gpu":
        torch.cuda.synchronize(t.gpu_idx)
    avg = (time.perf_counter() - start) / iters
    bw = (pat.total_bytes / avg) / (1024**3) if avg else 0
    print(f"  {_pretty(pat.total_bytes):>9s}  |  lat={avg*1000:8.3f} ms  |  "
          f"BW={bw:7.2f} GB/s  ({bw*8*1024**3/1e9:7.2f} Gbps)  |  "
          f"WRs={pat.n} block={_pretty(pat.block_bytes)}")
    t.verify()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connector", choices=["nixl", "mooncake", "flagcx"], required=True)
    p.add_argument("--role", choices=["server", "client"], required=True)
    p.add_argument("--remote-ip", default="0.0.0.0")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--block-bytes", type=int, default=8192,
                   help="size of each scattered transfer block (default 8 KB)")
    p.add_argument("--sizes", type=_parse_sizes, default=_parse_sizes("16M,64M,256M,1G"),
                   help="comma-separated total transfer sizes (K/M/G suffixes ok)")
    p.add_argument("--pool-gb", type=float, default=8.0,
                   help="pool buffer size; blocks are scattered within it")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--zmq-port", type=int, default=4566)
    p.add_argument("--nixl-backend", default="UCX")
    p.add_argument("--mooncake-protocol", default="rdma")
    p.add_argument("--flagcx-lib-path", default=None)
    p.add_argument("--flagcx-path", default=None)
    args = p.parse_args()

    t = {"nixl": NixlTransport, "mooncake": MooncakeTransport,
         "flagcx": FlagCXTransport}[args.connector](args)

    pool_bytes = int(args.pool_gb * 1024**3)
    pool_blocks = pool_bytes // args.block_bytes
    print(f"connector={args.connector} role={args.role} device={args.device} "
          f"block={_pretty(args.block_bytes)} pool={_pretty(pool_bytes)}")
    print("-" * 72)

    t.setup_pool(pool_bytes)
    try:
        for size in args.sizes:
            if size // args.block_bytes > pool_blocks:
                print(f"  size={_pretty(size):>9s} | SKIP: "
                      f"{size // args.block_bytes} blocks > pool {pool_blocks}")
                continue
            pat = make_pattern(size, args.block_bytes, pool_bytes, args.seed)
            run_size(t, pat, args.iters, args.warmup)
    finally:
        t.teardown()
    print("-" * 72)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(1)
