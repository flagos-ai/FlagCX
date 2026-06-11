#!/usr/bin/env python3
"""
KV Transfer Connector Benchmark — unified benchmark for NIXL, Mooncake, and FlagCX.

Measures raw transport-level bandwidth and latency for KV cache transfers
using the same underlying libraries as vLLM's KV connectors, without the
full vLLM scheduler/worker overhead.

Supports:
  - NIXL with UCX, UCCL, or FLAGCX backends
  - Mooncake TransferEngine (RDMA)
  - FlagCX direct one-sided RDMA

Uses ZMQ for out-of-band coordination between server and client.

Modes:
  - uniform: simple sizes x num_blocks sweep (one value per byte)
  - noncontig: models a PD step — a batch of requests whose KV blocks are
    scattered across a large (total_req) block pool, swept over request length

Usage
-----
  # NIXL with UCX backend (server on 10.8.2.169)
  python kv_transfer_benchmark.py --connector=nixl --role=server \\
      --remote-ip=10.8.2.169 --device=gpu --nixl-backend=UCX

  python kv_transfer_benchmark.py --connector=nixl --role=client \\
      --remote-ip=10.8.2.169 --device=gpu --nixl-backend=UCX

  # FlagCX (direct library)
  python kv_transfer_benchmark.py --connector=flagcx --role=server \\
      --remote-ip=10.8.2.169 --device=gpu

  python kv_transfer_benchmark.py --connector=flagcx --role=client \\
      --remote-ip=10.8.2.169 --device=gpu

  # Non-contiguous step mode
  python kv_transfer_benchmark.py --connector=flagcx --role=client \\
      --remote-ip=10.8.2.169 --mode=noncontig --contiguity=0.3
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import torch
import zmq

# bytes per KV element, selectable via --dtype.
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "fp32": 4}


@dataclass
class StepPattern:
    """A modeled single-step transfer: a list of variable-size, scattered WRs.

    Each WR carries a 1-byte ``tag`` (1..255). The source fills its region with
    the tag; after the transfer the destination region must equal it, so
    verification checks the per-WR src->dst mapping (untouched bytes stay 0).
    """

    sizes: List[int]                                       # bytes per WR
    src_offsets: List[int] = field(default_factory=list)   # byte offset in src
    dst_offsets: List[int] = field(default_factory=list)   # byte offset in dst
    tags: List[int] = field(default_factory=list)          # 1..255 per WR
    pool_bytes: int = 0                                     # full buffer span

    @property
    def wr_count(self) -> int:
        return len(self.sizes)

    @property
    def total_bytes(self) -> int:
        """Bytes actually transferred this step (sum of WRs)."""
        return sum(self.sizes)


def compute_block_len(block_size: int, num_kv_heads: int, head_dim: int,
                      tp: int, dtype_bytes: int) -> int:
    """Bytes of one KV block, for one layer, on one tp rank.

    Each rank owns ``num_kv_heads // tp`` heads (floored to >= 1 for GQA); the
    factor 2 is K and V. Matches FlagCXConnector's per-rank ``block_len``.
    """
    heads_per_rank = max(1, num_kv_heads // tp)
    return 2 * block_size * heads_per_rank * head_dim * dtype_bytes


def _request_run_lengths(blocks_per_req: int, contiguity: float,
                         rng: random.Random) -> List[int]:
    """Split ``blocks_per_req`` blocks into contiguous runs.

    ``contiguity`` is the probability the next block extends the current run.
    1.0 => one run (fully contiguous); 0.0 => every block its own run.
    """
    if blocks_per_req <= 0:
        return []
    if contiguity >= 1.0:
        return [blocks_per_req]
    runs: List[int] = []
    cur = 1
    for _ in range(blocks_per_req - 1):
        if rng.random() < contiguity:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def generate_step_pattern(block_len: int, num_layers: int, blocks_per_req: int,
                          batch: int, total_req: int, contiguity: float,
                          seed: int, coalesce: bool = True) -> StepPattern:
    """Build one step's WRs for ``batch`` requests, scattered across a pool.

    The pool holds ``total_req`` requests' blocks (per layer); only ``batch``
    requests are transferred, with their runs scattered across the pool to
    model fragmentation. ``coalesce=False`` (NIXL) emits one WR per block.
    """
    rng = random.Random(seed)
    if coalesce:
        per_req_runs = [_request_run_lengths(blocks_per_req, contiguity, rng)
                        for _ in range(batch)]
    else:
        per_req_runs = [[1] * blocks_per_req for _ in range(batch)]

    run_blocks: List[int] = []
    for _layer in range(num_layers):
        for runs in per_req_runs:
            run_blocks.extend(runs)

    sizes = [r * block_len for r in run_blocks]
    tags = [(i % 255) + 1 for i in range(len(sizes))]

    pool_blocks = total_req * blocks_per_req * num_layers
    # src and dst are independent pool allocations -> different scatters.
    src_offsets = _scatter_offsets(run_blocks, pool_blocks, block_len,
                                   random.Random(seed ^ 0xA5A5))
    dst_offsets = _scatter_offsets(run_blocks, pool_blocks, block_len,
                                   random.Random(seed ^ 0x5A5A))
    return StepPattern(sizes=sizes, src_offsets=src_offsets,
                       dst_offsets=dst_offsets, tags=tags,
                       pool_bytes=pool_blocks * block_len)


def _scatter_offsets(run_blocks: List[int], pool_blocks: int, block_len: int,
                     rng: random.Random) -> List[int]:
    """Place each run at a block-aligned, non-overlapping, scattered offset.

    Distributes the free blocks (pool minus transferred) as random gaps before
    each run, laying runs out in a shuffled order. Returns one byte offset/run.
    """
    k = len(run_blocks)
    if k == 0:
        return []
    order = list(range(k))
    rng.shuffle(order)
    free = max(0, pool_blocks - sum(run_blocks))
    cuts = sorted(rng.randint(0, free) for _ in range(k))
    offsets = [0] * k
    block = 0
    prev = 0
    for pos, idx in enumerate(order):
        block += cuts[pos] - prev
        prev = cuts[pos]
        offsets[idx] = block * block_len
        block += run_blocks[idx]
    return offsets


def _blocks_from_pattern(pattern: StepPattern, is_source: bool,
                         device: str, gpu_idx: int) -> List[torch.Tensor]:
    """One byte-exact uint8 tensor per WR. Source fills its tag; dest fills 0."""
    dev = f"cuda:{gpu_idx}" if device == "gpu" else "cpu"
    return [
        torch.full((sz,), tag if is_source else 0, device=dev, dtype=torch.uint8)
        for sz, tag in zip(pattern.sizes, pattern.tags)
    ]


def _verify_dest_blocks(dataset: List[torch.Tensor], pattern: StepPattern,
                        role: str) -> None:
    """Dest side: every WR tensor must equal its tag."""
    for i, (blk, tag) in enumerate(zip(dataset, pattern.tags)):
        if not torch.all(blk == tag).item():
            bad = int((blk != tag).sum().item())
            raise AssertionError(
                f"[{role}] STEP VERIFY FAIL: WR {i} (size={blk.numel()}, "
                f"tag={tag}) has {bad} bytes != tag."
            )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TransportBenchmark(ABC):
    """Common interface for KV transfer benchmarks."""

    # Subclasses set this to the fixed op type for their connector.
    OP_TYPE: str = ""
    COALESCE: bool = True
    SOURCE_ROLE: str = "client"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.role = args.role
        self.device = args.device
        self.gpu_idx = args.local_gpu_idx
        self.remote_ip = args.remote_ip
        self.op_type = self.OP_TYPE
        self.zmq_port = args.zmq_port
        self._pattern: "StepPattern | None" = None

    @property
    def is_source(self) -> bool:
        """True if this process holds the source data for the transfer."""
        return self.SOURCE_ROLE in self.role

    @abstractmethod
    def setup(self, size: int, num_blocks: int,
              pattern: "StepPattern | None" = None) -> None:
        """Initialize transport, allocate buffers, exchange metadata.

        When ``pattern`` is given (step mode), its WR sizes/offsets override the
        uniform ``size``/``num_blocks`` split.
        """

    @abstractmethod
    def run_transfer(self) -> None:
        """Execute one transfer and wait for completion."""

    @abstractmethod
    def verify_strict(self) -> None:
        """Strict element-by-element verification. Raises AssertionError on mismatch."""

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup resources for this size iteration."""


# ---------------------------------------------------------------------------
# NIXL Benchmark
# ---------------------------------------------------------------------------


class NixlBenchmark(TransportBenchmark):
    """Benchmark using NIXL agent with configurable backend (UCX/FLAGCX)."""

    OP_TYPE = "read"
    COALESCE = False
    SOURCE_ROLE = "server"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.backend = args.nixl_backend
        try:
            from nixl._api import nixl_agent, nixl_agent_config
            self._nixl_agent_cls = nixl_agent
            self._nixl_config_cls = nixl_agent_config
        except ImportError:
            sys.stderr.write(
                "Failed to import NIXL. Is the nixl Python package installed?\n"
            )
            raise

        self.agent = None
        self.reg_descs = None
        self.handle = None
        self.zmq_sock = None
        self.dataset: List[torch.Tensor] = []

    def setup(self, size: int, num_blocks: int,
              pattern: "StepPattern | None" = None) -> None:
        self._pattern = pattern
        if pattern is not None:
            self.dataset = _blocks_from_pattern(
                pattern, self.is_source, self.device, self.gpu_idx)
        else:
            self.dataset = self._create_dataset(size, num_blocks)

        # ZMQ coordination
        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)

        # Create NIXL agent
        config = self._nixl_config_cls(backends=[self.backend])
        self.agent = self._nixl_agent_cls(self.role, config)

        # Register memory
        self.reg_descs = self.agent.register_memory(
            self.agent.get_reg_descs(self.dataset)
        )

        # Exchange agent metadata
        local_meta = self.agent.get_agent_metadata()
        if "client" in self.role:
            self.zmq_sock.send(local_meta)
            remote_meta = self.zmq_sock.recv()
        else:
            remote_meta = self.zmq_sock.recv()
            self.zmq_sock.send(local_meta)
        self.agent.add_remote_agent(remote_meta)

    def _init_transfer_handle(self) -> None:
        """Per-iteration handshake: exchange descriptors and build handle."""
        local_xfer = self.reg_descs.trim()

        if "server" in self.role:
            msg = self.zmq_sock.recv().decode("utf-8")
            if msg != "START":
                raise RuntimeError(f"server got unexpected handshake: {msg!r}")
            self.zmq_sock.send(self.agent.get_serialized_descs(local_xfer))
            self.handle = None
        else:
            self.zmq_sock.send(b"START")
            remote_ser = self.zmq_sock.recv()
            remote_xfer = self.agent.deserialize_descs(remote_ser)
            self.handle = self.agent.initialize_xfer(
                "READ", local_xfer, remote_xfer, "server"
            )

    def run_transfer(self) -> None:
        # Per-iteration: init handle, transfer, release handle
        self._init_transfer_handle()

        if "client" in self.role:
            state = self.agent.transfer(self.handle)
            assert state != "ERR", "transfer post failed"
            while True:
                state = self.agent.check_xfer_state(self.handle)
                assert state != "ERR", "transfer errored"
                if state == "DONE":
                    self.zmq_sock.send(b"DONE")
                    break
        else:
            # Server waits for client completion signal
            while self.zmq_sock.recv().decode("utf-8") != "DONE":
                pass

        # Release handle
        if self.handle is not None:
            self.agent.release_xfer_handle(self.handle)
            self.handle = None

    def verify_strict(self) -> None:
        """Verify on the dest side. Raises on mismatch."""
        if self.is_source:  # source side has nothing to check
            return
        if self._pattern is not None:
            _verify_dest_blocks(self.dataset, self._pattern, self.role)
            return

        expected = 0.0
        for idx, blk in enumerate(self.dataset):
            matches = blk == expected
            if not torch.all(matches).item():
                bad = ~matches
                first_idx = int(bad.flatten().to(torch.int8).argmax().item())
                first_value = blk.flatten()[first_idx].item()
                bad_count = int(bad.sum().item())
                raise AssertionError(
                    f"[{self.role}] VERIFY FAIL: block {idx} has {bad_count}/"
                    f"{blk.numel()} mismatched elements; first mismatch at "
                    f"element {first_idx}: got {first_value}, expected {expected}"
                )

    def teardown(self) -> None:
        if self.handle is not None and self.agent is not None:
            try:
                self.agent.release_xfer_handle(self.handle)
            except Exception:
                pass
        if self.reg_descs is not None and self.agent is not None:
            try:
                self.agent.deregister_memory(self.reg_descs)
            except Exception:
                pass
        if self.agent is not None:
            peer = "server" if self.agent.name == "client" else "client"
            try:
                self.agent.remove_remote_agent(peer)
            except Exception:
                pass
        if self.zmq_sock is not None:
            self.zmq_sock.close()
            self.zmq_sock = None
        self.agent = None
        self.reg_descs = None
        self.handle = None
        self.dataset = []

    def _create_dataset(self, size: int, num_blocks: int) -> List[torch.Tensor]:
        """Allocate tensor blocks. Server=0s, Client=1s."""
        dtype = torch.float32 if size >= 4 else torch.uint8
        value = 0 if "server" in self.role else 1
        element_size = torch.tensor([], dtype=dtype).element_size()
        n_elems_per_block = max(size // (element_size * num_blocks), 1)
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"

        blocks = [
            torch.full((n_elems_per_block,), value, device=dev, dtype=dtype)
            for _ in range(num_blocks)
        ]
        total = sum(t.numel() * t.element_size() for t in blocks)
        if total < size:
            extra = (size - total) // element_size
            if extra > 0:
                blocks.append(
                    torch.full((extra,), value, device=dev, dtype=dtype)
                )
        return blocks


# ---------------------------------------------------------------------------
# Mooncake Benchmark
# ---------------------------------------------------------------------------


class MooncakeBenchmark(TransportBenchmark):
    """Benchmark using Mooncake TransferEngine."""

    OP_TYPE = "write"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        try:
            from mooncake.engine import TransferEngine
            self._engine_cls = TransferEngine
        except ImportError:
            sys.stderr.write(
                "Failed to import Mooncake. Install mooncake following:\n"
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md\n"
            )
            raise

        self.engine = None
        self.zmq_sock = None
        self.dataset: List[torch.Tensor] = []
        self.protocol = args.mooncake_protocol
        self.remote_session: str = ""
        self.local_ptrs: List[int] = []
        self.local_lens: List[int] = []
        self.remote_ptrs: List[int] = []

    def setup(self, size: int, num_blocks: int,
              pattern: "StepPattern | None" = None) -> None:
        self._pattern = pattern
        if pattern is not None:
            self.dataset = _blocks_from_pattern(
                pattern, self.is_source, self.device, self.gpu_idx)
        else:
            self.dataset = self._create_dataset(size, num_blocks)

        # Initialize TransferEngine
        import socket
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            local_ip = "127.0.0.1"

        self.engine = self._engine_cls()
        ret = self.engine.initialize(local_ip, "P2PHANDSHAKE", self.protocol, "")
        if ret != 0:
            raise RuntimeError(
                f"Mooncake TransferEngine initialization failed (ret={ret})"
            )
        self.rpc_port = self.engine.get_rpc_port()

        # Register memory
        self.local_ptrs = []
        self.local_lens = []
        for blk in self.dataset:
            ptr = blk.data_ptr()
            nbytes = blk.numel() * blk.element_size()
            self.local_ptrs.append(ptr)
            self.local_lens.append(nbytes)

        ret = self.engine.batch_register_memory(self.local_ptrs, self.local_lens)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake memory registration failed (ret={ret})"
            )

        # ZMQ coordination — exchange session info and remote addresses
        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self._exchange_metadata(local_ip)

    def _exchange_metadata(self, local_ip: str) -> None:
        """Exchange session hostname:port and buffer addresses via ZMQ."""
        import json

        local_info = json.dumps({
            "session": f"{local_ip}:{self.rpc_port}",
            "ptrs": self.local_ptrs,
            "lens": self.local_lens,
        }).encode("utf-8")

        if "server" in self.role:
            remote_info_raw = self.zmq_sock.recv()
            self.zmq_sock.send(local_info)
        else:
            self.zmq_sock.send(local_info)
            remote_info_raw = self.zmq_sock.recv()

        remote_info = json.loads(remote_info_raw.decode("utf-8"))
        self.remote_session = remote_info["session"]
        self.remote_ptrs = remote_info["ptrs"]

    def run_transfer(self) -> None:
        """Client writes its data to server's buffers."""
        if "client" in self.role:
            ret = self.engine.batch_transfer_sync_write(
                self.remote_session,
                self.local_ptrs,
                self.remote_ptrs,
                self.local_lens,
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake transfer failed (ret={ret})")
            self.zmq_sock.send(b"DONE")
        else:
            # Server waits for client completion
            while self.zmq_sock.recv().decode("utf-8") != "DONE":
                pass

    def verify_strict(self) -> None:
        """Verify on the dest side (server). Raises on mismatch."""
        if self.is_source:  # client is the source for write semantics
            return
        if self._pattern is not None:
            _verify_dest_blocks(self.dataset, self._pattern, self.role)
            return

        expected = 1.0
        for idx, blk in enumerate(self.dataset):
            matches = blk == expected
            if not torch.all(matches).item():
                bad = ~matches
                first_idx = int(bad.flatten().to(torch.int8).argmax().item())
                first_value = blk.flatten()[first_idx].item()
                bad_count = int(bad.sum().item())
                raise AssertionError(
                    f"[{self.role}] VERIFY FAIL: block {idx} has {bad_count}/"
                    f"{blk.numel()} mismatched elements; first mismatch at "
                    f"element {first_idx}: got {first_value}, expected {expected}"
                )

    def teardown(self) -> None:
        if self.zmq_sock is not None:
            self.zmq_sock.close()
            self.zmq_sock = None
        self.engine = None
        self.dataset = []
        self.local_ptrs = []
        self.local_lens = []
        self.remote_ptrs = []

    def _create_dataset(self, size: int, num_blocks: int) -> List[torch.Tensor]:
        """Allocate tensor blocks. Server=0s, Client=1s."""
        dtype = torch.float32 if size >= 4 else torch.uint8
        value = 0 if "server" in self.role else 1
        element_size = torch.tensor([], dtype=dtype).element_size()
        n_elems_per_block = max(size // (element_size * num_blocks), 1)
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"

        blocks = [
            torch.full((n_elems_per_block,), value, device=dev, dtype=dtype)
            for _ in range(num_blocks)
        ]
        total = sum(t.numel() * t.element_size() for t in blocks)
        if total < size:
            extra = (size - total) // element_size
            if extra > 0:
                blocks.append(
                    torch.full((extra,), value, device=dev, dtype=dtype)
                )
        return blocks


# ---------------------------------------------------------------------------
# FlagCX Benchmark
# ---------------------------------------------------------------------------


class FlagCXBenchmark(TransportBenchmark):
    """Benchmark using FlagCX one-sided RDMA (direct library, no NIXL)."""

    OP_TYPE = "write"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        import os
        flagcx_path = args.flagcx_wrapper_path or os.getenv("FLAGCX_PATH", "")
        if not flagcx_path:
            # Default to repo-relative path (script is at test/perf/kv_transfer/)
            flagcx_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )

        wrapper_dir = os.path.join(flagcx_path, "plugin", "interservice")
        if wrapper_dir not in sys.path:
            sys.path.insert(0, wrapper_dir)

        from flagcx_wrapper import FLAGCXLibrary
        self._FLAGCXLibrary = FLAGCXLibrary

        lib_path = args.flagcx_lib_path
        if not lib_path:
            lib_path = os.path.join(flagcx_path, "build", "lib", "libflagcx.so")
        self.flagcx = FLAGCXLibrary(lib_path)

        self.engine = None
        self.buffer: torch.Tensor = None
        self.zmq_sock = None
        self.conn = None
        self._src_vas: List[int] = []
        self._dst_vas: List[int] = []
        self._sizes: List[int] = []

    def _fill_source_buffer(self, pattern: StepPattern) -> None:
        """Source side: write each WR's tag into its src_offset region."""
        for off, sz, tag in zip(pattern.src_offsets, pattern.sizes, pattern.tags):
            self.buffer[off:off + sz] = tag

    def setup(self, size: int, num_blocks: int,
              pattern: "StepPattern | None" = None) -> None:
        import json
        import socket

        is_server = "server" in self.role  # server == receiver / write target

        # ZMQ coordination
        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self.engine = self.flagcx.flagcxP2pEngineCreate()

        self._pattern = pattern
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"
        if pattern is not None:
            self.buffer = torch.zeros(
                (pattern.pool_bytes,), device=dev, dtype=torch.uint8
            )
            if self.is_source:
                self._fill_source_buffer(pattern)
        else:
            value = 0 if is_server else 1
            total_bytes = size * num_blocks
            n_elems = total_bytes // 4  # float32
            self.buffer = torch.full(
                (n_elems,), value, device=dev, dtype=torch.float32
            )

        base_addr = self.buffer.data_ptr()
        reg_bytes = self.buffer.numel() * self.buffer.element_size()
        self.flagcx.flagcxP2pRegister(self.engine, base_addr, reg_bytes)

        if is_server:
            # Receiver: start the RPC accept server and advertise its session
            # ("host:rpc_port") + buffer base VA so the sender can write here.
            self.flagcx.flagcxP2pStartRpcServer(self.engine)
            rpc_port = self.flagcx.flagcxP2pGetRpcPort(self.engine)
            host = self.remote_ip
            if host in ("", "0.0.0.0", None):
                host = socket.gethostbyname(socket.gethostname())
            self.zmq_sock.send(json.dumps({
                "session": f"{host}:{rpc_port}",
                "base_addr": base_addr,
            }).encode("utf-8"))
            # Barrier: wait for the sender to finish connecting.
            self.zmq_sock.recv()
        else:
            # Sender: learn the receiver's session + remote base VA, then open
            # the P2P connection (QP + desc-table handshake on first call).
            remote_info = json.loads(self.zmq_sock.recv().decode("utf-8"))
            remote_session = remote_info["session"]
            remote_base = int(remote_info["base_addr"])
            self.conn = self.flagcx.flagcxP2pGetConn(self.engine, remote_session)

            # Precompute the absolute-VA write lists once.
            if pattern is not None:
                self._sizes = list(pattern.sizes)
                self._src_vas = [base_addr + o for o in pattern.src_offsets]
                self._dst_vas = [remote_base + o for o in pattern.dst_offsets]
            elif num_blocks <= 1:
                self._sizes = [total_bytes]
                self._src_vas = [base_addr]
                self._dst_vas = [remote_base]
            else:
                block_size = total_bytes // num_blocks
                self._sizes = [block_size] * num_blocks
                self._src_vas = [base_addr + i * block_size
                                 for i in range(num_blocks)]
                self._dst_vas = [remote_base + i * block_size
                                 for i in range(num_blocks)]
            self.zmq_sock.send(b"READY")

    def run_transfer(self) -> None:
        """Sender (client) RDMA-writes its buffer into the receiver's buffer."""
        if "server" in self.role:  # receiver: nothing to drive, just wait
            while self.zmq_sock.recv() != b"DONE":
                pass
            return

        # flagcxP2pBatchWriteSync is synchronous: it returns once the batched
        # one-sided write has completed on the wire.
        self.flagcx.flagcxP2pBatchWriteSync(
            self.conn, self._src_vas, self._dst_vas, self._sizes
        )
        self.zmq_sock.send(b"DONE")

    def verify_strict(self) -> None:
        """Verify on the receiver (server). Raises on mismatch."""
        if self.is_source:  # client is the source for write semantics
            return
        if self._pattern is not None:
            p = self._pattern
            for i, (off, sz, tag) in enumerate(
                    zip(p.dst_offsets, p.sizes, p.tags)):
                region = self.buffer[off:off + sz]
                if not torch.all(region == tag).item():
                    bad = int((region != tag).sum().item())
                    raise AssertionError(
                        f"[{self.role}] STEP VERIFY FAIL: WR {i} @dst {off} "
                        f"(size={sz}, tag={tag}) has {bad} bytes != tag."
                    )
            return

        expected = 1.0
        matches = self.buffer == expected
        if not torch.all(matches).item():
            bad = ~matches
            first_idx = int(bad.flatten().to(torch.int8).argmax().item())
            first_value = self.buffer.flatten()[first_idx].item()
            bad_count = int(bad.sum().item())
            raise AssertionError(
                f"[{self.role}] VERIFY FAIL: buffer has {bad_count}/"
                f"{self.buffer.numel()} mismatched elements; first mismatch at "
                f"element {first_idx}: got {first_value}, expected {expected}"
            )

    def teardown(self) -> None:
        if self.engine is not None:
            try:
                self.flagcx.flagcxP2pEngineDestroy(self.engine)
            except Exception:
                pass
        if self.zmq_sock is not None:
            self.zmq_sock.close()
            self.zmq_sock = None
        self.engine = None
        self.conn = None
        self.buffer = None
        self._src_vas = []
        self._dst_vas = []
        self._sizes = []


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _init_zmq(host: str, port: int, role: str) -> zmq.Socket:
    """Create a ZMQ PAIR socket for coordination."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PAIR)
    if "server" in role:
        sock.bind(f"tcp://{host}:{port}")
    else:
        sock.connect(f"tcp://{host}:{port}")
        sock.setsockopt(zmq.LINGER, 0)
    return sock


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
            "expected comma-separated integers"
        ) from e


def _gpu_buffer_fits(required: int, args: argparse.Namespace) -> bool:
    """True if ``required`` bytes fit in the local GPU (90% of free mem)."""
    if args.device != "gpu" or not torch.cuda.is_available():
        return True
    free, _ = torch.cuda.mem_get_info(args.local_gpu_idx)
    return required <= free * 0.9


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------


def benchmark_size(bench: TransportBenchmark, size: int, num_blocks: int,
                   iters: int, warmup: int) -> None:
    """Run benchmark for a single message size."""
    bench.setup(size, num_blocks)

    # Warmup iterations
    for _ in range(warmup):
        bench.run_transfer()

    # Timed iterations
    if torch.cuda.is_available() and bench.device == "gpu":
        torch.cuda.synchronize(bench.gpu_idx)

    t_start = time.perf_counter()
    for _ in range(iters):
        bench.run_transfer()
    if torch.cuda.is_available() and bench.device == "gpu":
        torch.cuda.synchronize(bench.gpu_idx)
    elapsed = time.perf_counter() - t_start

    # Stats
    avg_s = elapsed / iters
    total_bytes = size * num_blocks
    bw_GBs = (total_bytes / avg_s) / (1024**3) if avg_s > 0 else 0
    bw_Gbps = (total_bytes * 8 / avg_s) / 1e9 if avg_s > 0 else 0

    print(
        f"  {_pretty_size(total_bytes):>10s}  |  "
        f"lat={avg_s * 1000:8.3f} ms  |  "
        f"BW={bw_GBs:7.2f} GB/s  ({bw_Gbps:7.2f} Gbps)  |  "
        f"iters={iters}"
    )

    # Verify correctness
    bench.verify_strict()
    bench.teardown()


def benchmark_step(bench: TransportBenchmark, pattern: StepPattern,
                   contiguity: float, req_tokens: int,
                   iters: int, warmup: int) -> None:
    """Run benchmark for one non-contiguous step pattern."""
    bench.setup(0, 0, pattern=pattern)

    for _ in range(warmup):
        bench.run_transfer()

    if torch.cuda.is_available() and bench.device == "gpu":
        torch.cuda.synchronize(bench.gpu_idx)

    t_start = time.perf_counter()
    for _ in range(iters):
        bench.run_transfer()
    if torch.cuda.is_available() and bench.device == "gpu":
        torch.cuda.synchronize(bench.gpu_idx)
    elapsed = time.perf_counter() - t_start

    avg_s = elapsed / iters
    total = pattern.total_bytes
    bw_GBs = (total / avg_s) / (1024**3) if avg_s > 0 else 0
    bw_Gbps = (total * 8 / avg_s) / 1e9 if avg_s > 0 else 0
    avg_wr = total / pattern.wr_count if pattern.wr_count else 0

    print(
        f"  reqlen={req_tokens // 1024:>3d}k | "
        f"contig={contiguity:4.2f} | "
        f"WRs={pattern.wr_count:6d} | "
        f"WRsize[min/avg/max]={_pretty_size(min(pattern.sizes))}/"
        f"{_pretty_size(int(avg_wr))}/{_pretty_size(max(pattern.sizes))} | "
        f"xfer={_pretty_size(total):>9s} | "
        f"lat={avg_s * 1000:8.3f} ms | "
        f"BW={bw_GBs:7.2f} GB/s ({bw_Gbps:7.2f} Gbps)"
    )

    bench.verify_strict()
    bench.teardown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="KV Transfer Connector Benchmark (NIXL / Mooncake / FlagCX)"
    )
    p.add_argument(
        "--connector", choices=["nixl", "mooncake", "flagcx"], required=True,
        help="Transport backend to benchmark"
    )
    p.add_argument(
        "--role", choices=["server", "client"], required=True,
        help="server listens; client initiates the transfer"
    )
    p.add_argument(
        "--mode", choices=["uniform", "noncontig"], default="uniform",
        help="uniform: --sizes x --num-blocks sweep; "
             "noncontig: scattered PD-step sweep over request length"
    )
    p.add_argument(
        "--remote-ip", default="0.0.0.0",
        help="server IP — client dials it, server binds it"
    )
    p.add_argument(
        "--device", choices=["cpu", "gpu"], default="gpu",
        help="memory device for buffers"
    )
    p.add_argument(
        "--local-gpu-idx", type=int, default=0,
        help="CUDA device index for the local buffer"
    )
    p.add_argument(
        "--sizes", type=_parse_sizes,
        default=[1 << s for s in (10, 12, 14, 16, 18, 20, 22, 24, 26)],
        help="comma-separated message sizes in bytes"
    )
    p.add_argument(
        "--num-blocks", type=int, default=1,
        help="number of tensor blocks per transfer"
    )
    p.add_argument(
        "--iters", type=int, default=100,
        help="timed iterations per size"
    )
    p.add_argument(
        "--warmup-iters", type=int, default=10,
        help="number of warmup iterations before timed runs"
    )
    p.add_argument(
        "--req-lens", type=_parse_sizes,
        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="[noncontig] comma-separated request lengths in tokens to sweep"
    )
    p.add_argument(
        "--total-req", type=int, default=500,
        help="[noncontig] requests in the block pool (sets total buffer size)"
    )
    p.add_argument(
        "--batch", type=int, default=8,
        help="[noncontig] requests transferred per step"
    )
    p.add_argument(
        "--block-size", type=int, default=16,
        help="[noncontig] KV tokens per block (blocks_per_req = req_len/this)"
    )
    p.add_argument(
        "--num-kv-heads", type=int, default=8,
        help="[noncontig] model num_key_value_heads (sharded across tp ranks)"
    )
    p.add_argument(
        "--head-dim", type=int, default=128,
        help="[noncontig] attention head_dim"
    )
    p.add_argument(
        "--tp", type=int, default=8,
        help="[noncontig] tensor-parallel size; each rank owns num_kv_heads//tp"
    )
    p.add_argument(
        "--dtype", choices=list(_DTYPE_BYTES), default="bf16",
        help="[noncontig] KV element dtype"
    )
    p.add_argument(
        "--num-layers", type=int, default=48,
        help="[noncontig] number of layers (one WR group each)"
    )
    p.add_argument(
        "--contiguity", type=float, default=1.0,
        help="[noncontig] block run-extension probability in [0,1]: "
             "1.0=fully contiguous, 0.0=fully scattered (coalescing backends)"
    )
    p.add_argument(
        "--seed", type=int, default=1234,
        help="[noncontig] RNG seed for the scatter pattern (same on both sides)"
    )
    p.add_argument(
        "--nixl-backend", default="UCX",
        help="NIXL backend plugin (UCX, UCCL, FLAGCX, etc.)"
    )
    p.add_argument(
        "--zmq-port", type=int, default=4566,
        help="ZMQ coordination port"
    )
    p.add_argument(
        "--mooncake-protocol", default="rdma",
        help="Mooncake transport protocol (rdma, tcp)"
    )
    p.add_argument(
        "--flagcx-lib-path", default=None,
        help="Path to libflagcx.so (default: $FLAGCX_PATH/build/lib/libflagcx.so)"
    )
    p.add_argument(
        "--flagcx-wrapper-path", default=None,
        help="Path to FlagCX wrapper directory (default: $FLAGCX_PATH)"
    )
    args = p.parse_args()

    # Select benchmark implementation
    if args.connector == "nixl":
        bench = NixlBenchmark(args)
    elif args.connector == "mooncake":
        bench = MooncakeBenchmark(args)
    elif args.connector == "flagcx":
        bench = FlagCXBenchmark(args)
    else:
        raise ValueError(f"Unknown connector: {args.connector}")

    print(f"KV Transfer Benchmark  connector={args.connector}  role={args.role}")
    print(
        f"  mode={args.mode}  device={args.device}  gpu={args.local_gpu_idx}  "
        f"op={bench.op_type}  iters={args.iters}  warmup={args.warmup_iters}"
    )
    if args.connector == "nixl":
        print(f"  nixl-backend={args.nixl_backend}")
    elif args.connector == "mooncake":
        print(f"  protocol={args.mooncake_protocol}")
    elif args.connector == "flagcx":
        lib = args.flagcx_lib_path or "$FLAGCX_PATH/build/lib/libflagcx.so"
        print(f"  lib={lib}")

    if args.mode == "uniform":
        print(f"  num_blocks={args.num_blocks}")
        print(f"  sizes: {', '.join(_pretty_size(s) for s in args.sizes)}")
        print("-" * 72)
        for size in args.sizes:
            benchmark_size(bench, size, args.num_blocks,
                              args.iters, args.warmup_iters)
    else:
        _run_noncontig(bench, args)

    print("-" * 72)
    print("Done.")


def _run_noncontig(bench: TransportBenchmark, args: argparse.Namespace) -> None:
    """Sweep the non-contiguous step pattern over request length."""
    dtype_bytes = _DTYPE_BYTES[args.dtype]
    block_len = compute_block_len(
        args.block_size, args.num_kv_heads, args.head_dim, args.tp, dtype_bytes,
    )
    coalesce = type(bench).COALESCE
    effective_contiguity = args.contiguity if coalesce else 0.0
    heads_per_rank = max(1, args.num_kv_heads // args.tp)

    print(
        f"  block_len={_pretty_size(block_len)} "
        f"(block_size={args.block_size} x heads/rank={heads_per_rank}"
        f"[={args.num_kv_heads}//{args.tp}] x head_dim={args.head_dim} "
        f"x 2(KV) x {args.dtype})"
    )
    print(
        f"  num_layers={args.num_layers}  total_req={args.total_req}"
        f"  batch={args.batch}  seed={args.seed}"
        f"  coalesce={coalesce}  contiguity={args.contiguity}"
    )
    print("-" * 72)

    for req_tokens in args.req_lens:
        blocks_per_req = max(1, math.ceil(req_tokens / args.block_size))
        # FlagCX allocates the whole pool; per-WR backends only the transfer.
        pool_bytes = args.total_req * blocks_per_req * args.num_layers * block_len
        xfer_bytes = args.batch * blocks_per_req * args.num_layers * block_len
        required = pool_bytes if args.connector == "flagcx" else xfer_bytes
        if not _gpu_buffer_fits(required, args):
            print(
                f"  reqlen={req_tokens // 1024:>3d}k | SKIP: needs "
                f"{_pretty_size(required)} > GPU free mem"
            )
            continue

        pattern = generate_step_pattern(
            block_len=block_len,
            num_layers=args.num_layers,
            blocks_per_req=blocks_per_req,
            batch=args.batch,
            total_req=args.total_req,
            contiguity=effective_contiguity,
            seed=args.seed,
            coalesce=coalesce,
        )
        benchmark_step(bench, pattern, args.contiguity, req_tokens,
                       args.iters, args.warmup_iters)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(1)
