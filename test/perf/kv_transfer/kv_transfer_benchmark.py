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
  - perf (default): Pure performance measurement, no verification overhead
  - correctness: Strict element-by-element verification with detailed error reporting

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

  # Correctness mode
  python kv_transfer_benchmark.py --connector=nixl --role=client \\
      --remote-ip=10.8.2.169 --mode=correctness
"""

from __future__ import annotations

import argparse
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, List

import torch
import zmq

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TransportBenchmark(ABC):
    """Common interface for KV transfer benchmarks."""

    # Subclasses set this to the fixed op type for their connector.
    OP_TYPE: str = ""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.role = args.role
        self.device = args.device
        self.gpu_idx = args.local_gpu_idx
        self.remote_ip = args.remote_ip
        self.op_type = self.OP_TYPE
        self.zmq_port = args.zmq_port

    @abstractmethod
    def setup(self, size: int, num_blocks: int) -> None:
        """Initialize transport, allocate buffers, exchange metadata."""

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

    def setup(self, size: int, num_blocks: int) -> None:
        # Allocate dataset
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
        """Strict element-by-element verification. Raises on mismatch."""
        if "client" not in self.role:
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

    def setup(self, size: int, num_blocks: int) -> None:
        # Allocate dataset
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
        """Strict element-by-element verification. Raises on mismatch."""
        if "server" not in self.role:
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

        from flagcx_wrapper import FLAGCXLibrary, flagcxUniqueId
        self._FLAGCXLibrary = FLAGCXLibrary
        self._flagcxUniqueId = flagcxUniqueId

        lib_path = args.flagcx_lib_path
        if not lib_path:
            lib_path = os.path.join(flagcx_path, "build", "lib", "libflagcx.so")
        self.flagcx = FLAGCXLibrary(lib_path)

        self.comm = None
        self.buffer: torch.Tensor = None
        self.zmq_sock = None

    def setup(self, size: int, num_blocks: int) -> None:
        import ctypes

        # ZMQ coordination
        self.zmq_sock = _init_zmq(self.remote_ip, self.zmq_port, self.role)

        # Exchange unique ID and init communicator
        rank = 0 if "server" in self.role else 1
        if rank == 0:
            uid_ptr = self.flagcx.flagcxGetUniqueId()
            uid_bytes = bytes(uid_ptr.contents.internal)
            self.zmq_sock.send(uid_bytes)
        else:
            uid_bytes = self.zmq_sock.recv()
            uid_ptr = ctypes.pointer(
                self.flagcx.unique_id_from_bytes(uid_bytes)
            )

        self.comm = self.flagcx.flagcxCommInitRank(2, uid_ptr, rank)

        # Allocate data buffer
        total_bytes = size * num_blocks
        dev = f"cuda:{self.gpu_idx}" if self.device == "gpu" else "cpu"
        value = 0 if "server" in self.role else 1
        n_elems = total_bytes // 4  # float32
        self.buffer = torch.full(
            (n_elems,), value, device=dev, dtype=torch.float32
        )

        # Register data MR (index 0)
        self.flagcx.flagcxOneSideRegister(
            self.comm, self.buffer.data_ptr(),
            self.buffer.numel() * self.buffer.element_size()
        )

        # Allocate and register signal buffer
        # TODO: if we support flagcxWaitCounter(sender) and flagcxWaitSignal(receiver)
        #       then we need to implement a flagcxOneSideSignalRegister, like below:
        # self.signal_buffer = torch.zeros(1, dtype=torch.int64).pin_memory()
        # self.flagcx.flagcxOneSideSignalRegister(
        #     self.comm, self.signal_buffer.data_ptr(),
        #     self.signal_buffer.numel() * self.signal_buffer.element_size(), 1
        # )

        # Sync ready
        if "server" in self.role:
            self.zmq_sock.send(b"READY")
            self.zmq_sock.recv()
        else:
            self.zmq_sock.recv()
            self.zmq_sock.send(b"READY")

        self._total_bytes = total_bytes
        self._num_blocks = num_blocks

    def run_transfer(self) -> None:
        """Client puts data to server."""
        rank = 0 if "server" in self.role else 1
        peer = 1 - rank
        block_size = self._total_bytes // self._num_blocks

        if rank == 1:  # client
            before = self.flagcx.flagcxReadCounter(self.comm)
            if self._num_blocks == 1:
                self.flagcx.flagcxPut(
                    self.comm, peer, 0, 0, self._total_bytes, 0, 0
                )
                n_ops = 1
            else:
                offsets = [i * block_size for i in range(self._num_blocks)]
                sizes = [block_size] * self._num_blocks
                mr_idxs = [0] * self._num_blocks
                self.flagcx.flagcxBatchPut(
                    self.comm, peer, offsets, offsets, sizes,
                    mr_idxs, mr_idxs
                )
                n_ops = self._num_blocks
            self.flagcx.flagcxWaitCounter(self.comm, before + n_ops)
            self.zmq_sock.send(b"DONE")
        else:  # server waits
            while self.zmq_sock.recv() != b"DONE":
                pass

    def verify_strict(self) -> None:
        """Strict element-by-element verification. Raises on mismatch."""
        if "server" not in self.role:
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
        if self.comm is not None:
            try:
                self.flagcx.flagcxCommDestroy(self.comm)
            except Exception:
                pass
        if self.zmq_sock is not None:
            self.zmq_sock.close()
            self.zmq_sock = None
        self.comm = None
        self.buffer = None


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
            "--sizes must be comma-separated integers (bytes)"
        ) from e


# ---------------------------------------------------------------------------
# Benchmark driver
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
    print(f"  ✓ Verification passed")

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
        "--iters", type=int, default=100,
        help="timed iterations per size"
    )
    p.add_argument(
        "--warmup-iters", type=int, default=10,
        help="number of warmup iterations before timed runs"
    )
    p.add_argument(
        "--num-blocks", type=int, default=1,
        help="number of tensor blocks per transfer"
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

    # Print header
    print(f"KV Transfer Benchmark  connector={args.connector}  role={args.role}")
    print(
        f"  device={args.device}  gpu={args.local_gpu_idx}  "
        f"op={bench.op_type}  iters={args.iters}  warmup={args.warmup_iters}  "
        f"num_blocks={args.num_blocks}"
    )
    print(f"  NOTE: {args.connector} connector uses op_type={bench.op_type} "
          f"(matching vLLM connector semantics)")
    if args.connector == "nixl":
        print(f"  nixl-backend={args.nixl_backend}")
    elif args.connector == "mooncake":
        print(f"  protocol={args.mooncake_protocol}")
    elif args.connector == "flagcx":
        lib = args.flagcx_lib_path or "$FLAGCX_PATH/build/lib/libflagcx.so"
        print(f"  lib={lib}")
    print(f"  sizes: {', '.join(_pretty_size(s) for s in args.sizes)}")
    print("-" * 72)

    # Run benchmark for each size
    for size in args.sizes:
        benchmark_size(bench, size, args.num_blocks, args.iters, args.warmup_iters)

    print("-" * 72)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(1)
