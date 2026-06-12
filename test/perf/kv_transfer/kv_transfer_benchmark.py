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
  - uniform(default): simple sizes x num_blocks sweep (one value per byte)
  - noncontig: models a PD step — a batch of requests whose KV blocks are
    scattered across a large (total_req) block pool, swept over request length

Usage
-----
  # NIXL with UCX backend (server on 10.8.2.169)
  python kv_transfer_benchmark.py --connector=nixl --role=server \\
      --remote-ip=10.8.2.169 --device=gpu --nixl-backend=UCX

  python kv_transfer_benchmark.py --connector=nixl --role=client \\
      --remote-ip=10.8.2.169 --device=gpu --nixl-backend=UCX

  # NIXL with FLAGCX backend
  python kv_transfer_benchmark.py --connector=nixl --role=server \\
      --remote-ip=10.8.2.169 --device=gpu --nixl-backend=FLAGCX

  # Mooncake
  python kv_transfer_benchmark.py --connector=mooncake --role=server \\
      --remote-ip=10.8.2.169 --device=gpu

  python kv_transfer_benchmark.py --connector=mooncake --role=client \\
      --remote-ip=10.8.2.169 --device=gpu

  # FlagCX (direct library)
  python kv_transfer_benchmark.py --connector=flagcx --role=server \\
      --remote-ip=10.8.2.169 --device=gpu

  python kv_transfer_benchmark.py --connector=flagcx --role=client \\
      --remote-ip=10.8.2.169 --device=gpu

  # Non-contiguous step mode (fragmented KV-cache block pool)
  python kv_transfer_benchmark.py --connector=flagcx --role=client \\
      --remote-ip=10.8.2.169 --mode=noncontig --pool-util=0.9
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import zmq

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "fp32": 4, "fp4": 0.5}


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
    return int(2 * block_size * heads_per_rank * head_dim * dtype_bytes)


class _BlockPool:
    """Minimal port of vLLM's BlockPool free queue (v1/core/block_pool.py).

    A fixed pool of ``num_blocks`` ids, reused by every request. Allocation pops
    from the head (``popleft_n``); freeing appends to the tail (``append_n``).
    vLLM's doubly-linked FreeKVCacheBlockQueue is collapsed to a deque — only
    its FIFO order matters for which ids a request receives.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free = deque(range(num_blocks))

    def alloc(self, n: int) -> List[int]:
        return [self.free.popleft() for _ in range(n)]

    def free_blocks(self, ids: List[int]) -> None:
        self.free.extend(ids)

    @property
    def used(self) -> int:
        return self.num_blocks - len(self.free)


def _fragmented_alloc(num_blocks: int, bpr: int, batch: int, total_req: int,
                      util: float, bg_choices: List[int],
                      rng: random.Random) -> List[List[int]]:
    """Churn the pool with mixed-size traffic, then allocate the transfer batch.

    Models a busy engine: ``total_req`` background requests of mixed sizes
    (``bg_choices``, in blocks) arrive and finish in random order, kept near
    ``util`` occupancy. Uniform-size traffic never fragments — the size mix is
    what scatters the free queue; the batch then pops scattered ids from it.
    Returns one block-id list per transferred request.
    """
    pool = _BlockPool(num_blocks)
    active: List[List[int]] = []
    target = int(num_blocks * util)

    for _ in range(total_req):
        need = rng.choice(bg_choices)
        if need > num_blocks:
            continue
        while (pool.used + need > target or need > len(pool.free)) and active:
            pool.free_blocks(active.pop(rng.randrange(len(active))))
        if need <= len(pool.free):
            active.append(pool.alloc(need))

    batch_ids: List[List[int]] = []
    for _ in range(batch):
        while bpr > len(pool.free) and active:
            pool.free_blocks(active.pop(rng.randrange(len(active))))
        if bpr > len(pool.free):
            raise RuntimeError("block pool too small for the transfer batch")
        batch_ids.append(pool.alloc(bpr))
    return batch_ids


def _joint_runs(src_ids: List[int], dst_ids: List[int],
                coalesce: bool) -> List[Tuple[int, int, int]]:
    """Coalesce a request's blocks into (src_start, dst_start, run) WRs.

    A run extends only while BOTH src and dst ids stay contiguous (a write maps
    one contiguous src span to one contiguous dst span), matching the
    connector's _group_contiguous. ``coalesce=False`` (NIXL) => one WR/block.
    """
    runs: List[Tuple[int, int, int]] = []
    i, n = 0, len(src_ids)
    while i < n:
        j = i + 1
        if coalesce:
            while (j < n and src_ids[j] == src_ids[j - 1] + 1
                   and dst_ids[j] == dst_ids[j - 1] + 1):
                j += 1
        runs.append((src_ids[i], dst_ids[i], j - i))
        i = j
    return runs


def build_step_pattern(block_len: int, num_layers: int, num_blocks: int,
                       bpr: int, batch: int, total_req: int, util: float,
                       bg_choices: List[int], seed: int,
                       coalesce: bool = True) -> StepPattern:
    """Build one PD-step transfer from real (fragmented) block allocations.

    Sender and receiver are independent engines, so their pools are fragmented
    with different seeds; a request's i-th block maps src_ids[i] -> dst_ids[i].
    The block-id set is shared across layers (one block table per request).
    """
    src_batch = _fragmented_alloc(num_blocks, bpr, batch, total_req, util,
                                  bg_choices, random.Random(seed ^ 0xA5A5))
    dst_batch = _fragmented_alloc(num_blocks, bpr, batch, total_req, util,
                                  bg_choices, random.Random(seed ^ 0x5A5A))

    layer_region = num_blocks * block_len
    sizes: List[int] = []
    src_offsets: List[int] = []
    dst_offsets: List[int] = []
    for layer in range(num_layers):
        base = layer * layer_region
        for src_ids, dst_ids in zip(src_batch, dst_batch):
            for s_start, d_start, run in _joint_runs(src_ids, dst_ids, coalesce):
                sizes.append(run * block_len)
                src_offsets.append(base + s_start * block_len)
                dst_offsets.append(base + d_start * block_len)

    tags = [(i % 255) + 1 for i in range(len(sizes))]
    return StepPattern(sizes=sizes, src_offsets=src_offsets,
                       dst_offsets=dst_offsets, tags=tags,
                       pool_bytes=num_layers * layer_region)


def _verify_dest_pool(buffer: torch.Tensor, pattern: StepPattern,
                      role: str) -> None:
    """Dest side (pool buffer): each WR's dst_offset region must equal its tag."""
    for i, (off, sz, tag) in enumerate(
            zip(pattern.dst_offsets, pattern.sizes, pattern.tags)):
        region = buffer[off:off + sz]
        if not torch.all(region == tag).item():
            bad = int((region != tag).sum().item())
            raise AssertionError(
                f"[{role}] STEP VERIFY FAIL: WR {i} @dst {off} (size={sz}, "
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
        # In noncontig mode this is one large contiguous pool buffer
        # (pool_bytes), allocated and registered ONCE via setup_pool() and
        # reused across every reqlen — each step only re-derives WR offsets.
        self.buffer: "torch.Tensor | None" = None

    @property
    def is_source(self) -> bool:
        """True if this process holds the source data for the transfer."""
        return self.SOURCE_ROLE in self.role

    def _fill_source_buffer(self, pattern: StepPattern) -> None:
        """Source side: write each WR's tag into its src_offset region."""
        for off, sz, tag in zip(pattern.src_offsets, pattern.sizes,
                                pattern.tags):
            self.buffer[off:off + sz] = tag

    def _refresh_pool_for_step(self, pattern: StepPattern) -> None:
        """Reset the reused pool buffer for a new step pattern.

        Dest side zeroes the pool so verification only passes if this step's
        WRs actually land; source side zeroes then stamps this step's tags.
        A device sync makes the reset visible before any remote write/read.
        """
        self.buffer.zero_()
        if self.is_source:
            self._fill_source_buffer(pattern)
        if self.device == "gpu" and torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu_idx)

    @abstractmethod
    def setup(self, size: int, num_blocks: int) -> None:
        """[uniform mode] Init transport, allocate per-size buffers, exchange
        metadata. Called once per swept message size."""

    @abstractmethod
    def setup_pool(self, pool_bytes: int) -> None:
        """[noncontig mode] One-time init: create the engine/agent, allocate
        and register the full ``pool_bytes`` buffer, and establish the
        connection. Reused across every reqlen step."""

    @abstractmethod
    def prepare_step(self, pattern: "StepPattern") -> None:
        """[noncontig mode] Per-reqlen: refresh the pool buffer for ``pattern``
        and re-derive this step's WR address/size lists. No (de)registration."""

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
        self._local_xfer = None  # this side's xfer descriptor list for a step

    def _create_agent(self) -> None:
        """Create the ZMQ socket and NIXL agent. Shared by both modes."""
        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        config = self._nixl_config_cls(backends=[self.backend])
        self.agent = self._nixl_agent_cls(self.role, config)

    def _exchange_agent_meta(self) -> None:
        """Swap NIXL agent metadata and register the peer. One-time."""
        local_meta = self.agent.get_agent_metadata()
        if "client" in self.role:
            self.zmq_sock.send(local_meta)
            remote_meta = self.zmq_sock.recv()
        else:
            remote_meta = self.zmq_sock.recv()
            self.zmq_sock.send(local_meta)
        self.agent.add_remote_agent(remote_meta)

    def _make_xfer_descs(self, offsets: List[int], sizes: List[int]):
        """Build a NIXL xfer descriptor list for sub-regions of the pool.

        Each entry is (abs_addr, len, dev_id) within the already-registered
        pool buffer, so no per-step (de)registration is needed.
        """
        base = self.buffer.data_ptr()
        mem = "VRAM" if self.device == "gpu" else "DRAM"
        devid = self.gpu_idx if self.device == "gpu" else 0
        descs = [(base + o, s, devid) for o, s in zip(offsets, sizes)]
        return self.agent.get_xfer_descs(descs, mem)

    def setup(self, size: int, num_blocks: int) -> None:
        """[uniform] Per-size dataset alloc + register + metadata exchange."""
        self.dataset = self._create_dataset(size, num_blocks)
        self._create_agent()
        self.reg_descs = self.agent.register_memory(
            self.agent.get_reg_descs(self.dataset)
        )
        self._exchange_agent_meta()
        # Uniform transfers the whole registered set, one WR per block.
        self._local_xfer = self.reg_descs.trim()

    def setup_pool(self, pool_bytes: int) -> None:
        """[noncontig] One-time: alloc + register the full pool, swap metadata.

        The pool is registered as a single region exactly once; per-step xfer
        descriptors are sliced from it in ``prepare_step`` without re-reg.
        """
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"
        self.buffer = torch.zeros((pool_bytes,), device=dev, dtype=torch.uint8)
        self._create_agent()
        self.reg_descs = self.agent.register_memory(
            self.agent.get_reg_descs([self.buffer])
        )
        self._exchange_agent_meta()

    def prepare_step(self, pattern: StepPattern) -> None:
        """[noncontig] Refresh the pool + build this side's per-step descs.

        Server holds the READ source (src_offsets); client is the READ dest
        (dst_offsets). The per-iteration handshake in ``run_transfer`` then
        swaps these and serializes source-fill before the read.
        """
        self._pattern = pattern
        self._refresh_pool_for_step(pattern)
        offsets = (pattern.src_offsets if self.is_source
                   else pattern.dst_offsets)
        self._local_xfer = self._make_xfer_descs(offsets, list(pattern.sizes))

    def _init_transfer_handle(self) -> None:
        """Per-iteration handshake: exchange descriptors and build handle."""
        local_xfer = self._local_xfer

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
            _verify_dest_pool(self.buffer, self._pattern, self.role)
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
        self.buffer = None
        self._local_xfer = None

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
        self._base_addr: int = 0      # local pool base VA (noncontig)
        self._remote_base: int = 0    # receiver's pool base VA (noncontig)
        self._wr_src: List[int] = []  # per-transfer local addrs
        self._wr_dst: List[int] = []  # per-transfer remote addrs
        self._wr_len: List[int] = []  # per-transfer sizes

    def _init_engine(self) -> str:
        """Init the TransferEngine and return the local IP. Shared by both modes."""
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
        return local_ip

    def setup(self, size: int, num_blocks: int) -> None:
        """[uniform] Per-size dataset alloc + register + metadata exchange."""
        self.dataset = self._create_dataset(size, num_blocks)
        local_ip = self._init_engine()

        # Register each block as its own region.
        self.local_ptrs = []
        self.local_lens = []
        for blk in self.dataset:
            self.local_ptrs.append(blk.data_ptr())
            self.local_lens.append(blk.numel() * blk.element_size())

        ret = self.engine.batch_register_memory(self.local_ptrs, self.local_lens)
        if ret != 0:
            raise RuntimeError(f"Mooncake memory registration failed (ret={ret})")

        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self.remote_session, self.remote_ptrs, _ = self._exchange_metadata(
            local_ip, self.local_ptrs, self.local_lens)
        # Uniform transfers every registered block, one WR each.
        self._wr_src = self.local_ptrs
        self._wr_dst = self.remote_ptrs
        self._wr_len = self.local_lens

    def setup_pool(self, pool_bytes: int) -> None:
        """[noncontig] One-time: alloc + register the full pool, exchange base.

        The whole pool is registered as a single region exactly once; each step
        only re-derives base+offset addresses, so registration is not redone.
        """
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"
        self.buffer = torch.zeros((pool_bytes,), device=dev, dtype=torch.uint8)
        local_ip = self._init_engine()

        self._base_addr = self.buffer.data_ptr()
        self.local_ptrs = [self._base_addr]
        self.local_lens = [pool_bytes]
        ret = self.engine.batch_register_memory(self.local_ptrs, self.local_lens)
        if ret != 0:
            raise RuntimeError(f"Mooncake memory registration failed (ret={ret})")

        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self.remote_session, remote_ptrs, _ = self._exchange_metadata(
            local_ip, self.local_ptrs, self.local_lens)
        self._remote_base = remote_ptrs[0]

    def prepare_step(self, pattern: StepPattern) -> None:
        """[noncontig] Refresh the pool + re-derive base+offset WR lists."""
        self._pattern = pattern
        self._refresh_pool_for_step(pattern)
        if self.is_source:  # client writes
            base, rb = self._base_addr, self._remote_base
            self._wr_src = [base + o for o in pattern.src_offsets]
            self._wr_dst = [rb + o for o in pattern.dst_offsets]
            self._wr_len = list(pattern.sizes)
            # Barrier: only write after the receiver has zeroed its pool.
            self.zmq_sock.send(b"READY")
        else:
            self.zmq_sock.recv()  # receiver: signal pool reset is done

    def _exchange_metadata(self, local_ip: str, ptrs: List[int],
                           lens: List[int]) -> "Tuple[str, List[int], List[int]]":
        """Exchange session hostname:port and buffer addresses via ZMQ.

        Returns the remote ``(session, ptrs, lens)``.
        """
        import json

        local_info = json.dumps({
            "session": f"{local_ip}:{self.rpc_port}",
            "ptrs": ptrs,
            "lens": lens,
        }).encode("utf-8")

        if "server" in self.role:
            remote_info_raw = self.zmq_sock.recv()
            self.zmq_sock.send(local_info)
        else:
            self.zmq_sock.send(local_info)
            remote_info_raw = self.zmq_sock.recv()

        remote_info = json.loads(remote_info_raw.decode("utf-8"))
        return (remote_info["session"], remote_info["ptrs"],
                remote_info["lens"])

    def run_transfer(self) -> None:
        """Client writes its data to the server's buffer(s)."""
        if "client" in self.role:
            ret = self.engine.batch_transfer_sync_write(
                self.remote_session,
                self._wr_src,
                self._wr_dst,
                self._wr_len,
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
            _verify_dest_pool(self.buffer, self._pattern, self.role)
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
        self.buffer = None
        self.local_ptrs = []
        self.local_lens = []
        self.remote_ptrs = []
        self._wr_src = []
        self._wr_dst = []
        self._wr_len = []

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
        self._base_addr: int = 0     # local pool/buffer base VA
        self._remote_base: int = 0   # sender: receiver's pool base VA
        self._src_vas: List[int] = []
        self._dst_vas: List[int] = []
        self._sizes: List[int] = []

    def _register_and_connect(self) -> None:
        """Register ``self.buffer`` and establish the P2P connection ONCE.

        Sets ``self._base_addr`` and (sender) ``self._remote_base``/``self.conn``.
        Shared by uniform ``setup`` and noncontig ``setup_pool``.
        """
        import json
        import socket

        base_addr = self.buffer.data_ptr()
        reg_bytes = self.buffer.numel() * self.buffer.element_size()
        self.flagcx.flagcxP2pRegister(self.engine, base_addr, reg_bytes)
        self._base_addr = base_addr

        if "server" in self.role:
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
            self._remote_base = int(remote_info["base_addr"])
            self.conn = self.flagcx.flagcxP2pGetConn(
                self.engine, remote_info["session"])
            self.zmq_sock.send(b"READY")

    def setup(self, size: int, num_blocks: int) -> None:
        """[uniform] Per-size buffer alloc + one-shot register/connect."""
        is_server = "server" in self.role  # server == receiver / write target
        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self.engine = self.flagcx.flagcxP2pEngineCreate()

        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"
        value = 0 if is_server else 1
        total_bytes = size * num_blocks
        n_elems = total_bytes // 4  # float32
        self.buffer = torch.full(
            (n_elems,), value, device=dev, dtype=torch.float32
        )

        self._register_and_connect()

        if not is_server:
            base_addr, remote_base = self._base_addr, self._remote_base
            if num_blocks <= 1:
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

    def setup_pool(self, pool_bytes: int) -> None:
        """[noncontig] One-time: alloc + register the full pool, then connect.

        The 70+ GB MR registration (``flagcxP2pRegister`` -> ``ibv_reg_mr``)
        happens here exactly once and is reused across every reqlen step.
        """
        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)
        self.engine = self.flagcx.flagcxP2pEngineCreate()
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"
        self.buffer = torch.zeros((pool_bytes,), device=dev, dtype=torch.uint8)
        self._register_and_connect()

    def prepare_step(self, pattern: StepPattern) -> None:
        """[noncontig] Refresh the pool for this step + re-derive WR VAs.

        No (de)registration — only a buffer reset and pointer arithmetic.
        """
        self._pattern = pattern
        self._refresh_pool_for_step(pattern)
        if self.is_source:
            base, remote_base = self._base_addr, self._remote_base
            self._sizes = list(pattern.sizes)
            self._src_vas = [base + o for o in pattern.src_offsets]
            self._dst_vas = [remote_base + o for o in pattern.dst_offsets]
            # Barrier: only write after the receiver has zeroed its pool.
            self.zmq_sock.send(b"READY")
        else:
            self.zmq_sock.recv()  # receiver: signal pool reset is done

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
            _verify_dest_pool(self.buffer, self._pattern, self.role)
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


def _kv_cache_num_blocks(block_len: int, num_layers: int,
                         args: argparse.Namespace) -> int:
    """Fixed pool size (blocks) sized to the KV-cache memory budget.

    Like a real engine: fill the available memory with KV cache. Budget is
    --kv-cache-gb if set, else 90% of free GPU mem (8 GiB fallback on CPU).
    """
    per_block = block_len * num_layers
    if args.kv_cache_gb:
        budget = int(args.kv_cache_gb * (1024 ** 3))
    elif args.device == "gpu" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(args.local_gpu_idx)
        budget = int(free * 0.9)
    else:
        budget = 8 * (1024 ** 3)
    return max(1, budget // per_block)


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
                   req_tokens: int, iters: int, warmup: int) -> None:
    """Run benchmark for one non-contiguous step pattern.

    The engine, pool buffer, and MR registration are set up ONCE by the
    caller (``setup_pool``); here we only refresh per-step WR offsets, so the
    expensive 70+ GB registration is not redone between reqlens.
    """
    bench.prepare_step(pattern)

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
        f"WRs={pattern.wr_count:6d} | "
        f"WRsize[min/avg/max]={_pretty_size(min(pattern.sizes))}/"
        f"{_pretty_size(int(avg_wr))}/{_pretty_size(max(pattern.sizes))} | "
        f"xfer={_pretty_size(total):>9s} | "
        f"lat={avg_s * 1000:8.3f} ms | "
        f"BW={bw_GBs:7.2f} GB/s ({bw_Gbps:7.2f} Gbps)"
    )

    # Per-step verify only; pool teardown is deferred to the end of the sweep.
    bench.verify_strict()


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
        "--total-req", type=int, default=300,
        help="[noncontig] background requests churned through the pool to "
             "fragment its free queue (does not size the buffer)"
    )
    p.add_argument(
        "--batch", type=int, default=8,
        help="[noncontig] requests transferred per step"
    )
    p.add_argument(
        "--pool-util", type=float, default=0.9,
        help="[noncontig] target pool occupancy during churn; higher => more "
             "fragmentation"
    )
    p.add_argument(
        "--kv-cache-gb", type=float, default=None,
        help="[noncontig] KV-cache buffer budget in GiB (default: 90%% of free "
             "GPU mem). Sets the fixed, reused block pool size."
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
        "--seed", type=int, default=1234,
        help="[noncontig] RNG seed for the churn pattern (same on both sides)"
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
    heads_per_rank = max(1, args.num_kv_heads // args.tp)

    # Fixed, reused block pool sized to the memory budget (like a real engine).
    num_blocks = _kv_cache_num_blocks(block_len, args.num_layers, args)
    pool_bytes = num_blocks * args.num_layers * block_len
    # Background size mix (in blocks) = the swept request lengths themselves.
    bg_choices = [max(1, math.ceil(L / args.block_size)) for L in args.req_lens]

    print(
        f"  block_len={_pretty_size(block_len)} "
        f"(block_size={args.block_size} x heads/rank={heads_per_rank}"
        f"[={args.num_kv_heads}//{args.tp}] x head_dim={args.head_dim} "
        f"x 2(KV) x {args.dtype})"
    )
    print(
        f"  num_layers={args.num_layers}  num_gpu_blocks={num_blocks}"
        f"  pool={_pretty_size(pool_bytes)}  util={args.pool_util}"
    )
    print(
        f"  total_req={args.total_req}  batch={args.batch}"
        f"  coalesce={coalesce}  seed={args.seed}"
    )
    print("-" * 72)

    # One-time: create the engine, allocate + register the full pool buffer,
    # and establish the connection. The pool size is fixed across all reqlens,
    # so the costly MR registration happens exactly once here (not per step).
    bench.setup_pool(pool_bytes)
    try:
        for req_tokens in args.req_lens:
            bpr = max(1, math.ceil(req_tokens / args.block_size))
            if args.batch * bpr > num_blocks:
                print(
                    f"  reqlen={req_tokens // 1024:>3d}k | SKIP: batch working "
                    f"set {args.batch * bpr} blocks > pool {num_blocks} blocks"
                )
                continue

            pattern = build_step_pattern(
                block_len=block_len,
                num_layers=args.num_layers,
                num_blocks=num_blocks,
                bpr=bpr,
                batch=args.batch,
                total_req=args.total_req,
                util=args.pool_util,
                bg_choices=bg_choices,
                seed=args.seed,
                coalesce=coalesce,
            )
            benchmark_step(bench, pattern, req_tokens, args.iters,
                           args.warmup_iters)
    finally:
        bench.teardown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(1)
