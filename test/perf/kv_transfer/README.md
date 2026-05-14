# KV Transfer Connector Benchmark

Unified benchmark for measuring raw transport-level performance of vLLM KV transfer connectors (NIXL, Mooncake, and FlagCX).

## Overview

This benchmark measures bandwidth and latency for KV cache transfers using the same underlying transport libraries as vLLM's KV connectors, without the full vLLM scheduler/worker overhead.

**Supported Connectors:**
- **NIXL** with UCX or FLAGCX backends
- **Mooncake** TransferEngine (RDMA)
- **FlagCX** direct one-sided RDMA (no NIXL abstraction layer)

## Requirements

### NIXL
```bash
pip install nixl-cu12  # or nixl-cu11 depending on CUDA version
pip install pyzmq torch
```

For FLAGCX backend:
```bash
# Ensure libflagcx.so is in LD_LIBRARY_PATH
# Ensure libplugin_FLAGCX.so is on the NIXL plugin path
```

### Mooncake
```bash
# Install Mooncake following:
# https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md
pip install pyzmq torch
```

### FlagCX (Direct)
```bash
# Build FlagCX from source:
# https://github.com/FlagOpen/FlagCX
# Set FLAGCX_PATH to the FlagCX root directory
export FLAGCX_PATH=/path/to/FlagCX
pip install pyzmq torch
```

## Usage

### NIXL with UCX Backend

**Server (Node A, IP: 10.8.2.169):**
```bash
python kv_transfer_benchmark.py --connector=nixl --role=server \
    --remote-ip=10.8.2.169 --device=gpu --nixl-backend=UCX \
    --sizes=1048576,16777216,104857600 --iters=20
```

**Client (Node B):**
```bash
python kv_transfer_benchmark.py --connector=nixl --role=client \
    --remote-ip=10.8.2.169 --device=gpu --nixl-backend=UCX \
    --sizes=1048576,16777216,104857600 --iters=20
```

### NIXL with FLAGCX Backend

Same as above, but add `--nixl-backend=FLAGCX`:
```bash
python kv_transfer_benchmark.py --connector=nixl --role=server \
    --remote-ip=10.8.2.169 --device=gpu --nixl-backend=FLAGCX \
    --sizes=1048576,16777216,104857600 --iters=20
```

### Mooncake

**Server:**
```bash
python kv_transfer_benchmark.py --connector=mooncake --role=server \
    --remote-ip=10.8.2.169 --device=gpu \
    --sizes=1048576,16777216,104857600 --iters=20
```

**Client:**
```bash
python kv_transfer_benchmark.py --connector=mooncake --role=client \
    --remote-ip=10.8.2.169 --device=gpu \
    --sizes=1048576,16777216,104857600 --iters=20
```

### FlagCX (Direct)

**Server:**
```bash
cd ~/WORKDIR/FlagCX
python3 test/perf/kv_transfer/kv_transfer_benchmark.py --connector=flagcx --role=server \
    --remote-ip=10.8.2.169 --device=gpu \
    --sizes=1048576,16777216,104857600 --iters=20
```

**Client:**
```bash
cd ~/WORKDIR/FlagCX
python3 test/perf/kv_transfer/kv_transfer_benchmark.py --connector=flagcx --role=client \
    --remote-ip=10.8.2.169 --device=gpu \
    --sizes=1048576,16777216,104857600 --iters=20
```

**Custom FlagCX paths:**
```bash
python3 test/perf/kv_transfer/kv_transfer_benchmark.py --connector=flagcx --role=server \
    --remote-ip=10.8.2.169 --device=gpu \
    --flagcx-lib-path=/custom/path/libflagcx.so \
    --flagcx-wrapper-path=/custom/path/FlagCX
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--connector` | Transport backend: `nixl`, `mooncake`, or `flagcx` | **required** |
| `--role` | `server` or `client` | **required** |
| `--remote-ip` | Server IP address | `0.0.0.0` |
| `--device` | Memory device: `cpu` or `gpu` | `gpu` |
| `--local-gpu-idx` | CUDA device index | `0` |
| `--sizes` | Comma-separated transfer sizes in bytes | `1KB,4KB,16KB,64KB,256KB,1MB,4MB,16MB,64MB` |
| `--iters` | Number of timed iterations per size | `100` |
| `--warmup-iters` | Number of warmup iterations before timed runs | `10` |
| `--num-blocks` | Number of tensor blocks per transfer | `1` |
| `--nixl-backend` | NIXL backend plugin (e.g., `UCX`, `UCCL`, `FLAGCX`) | `UCX` |
| `--zmq-port` | ZMQ coordination port | `9000` |
| `--mooncake-protocol` | Mooncake protocol: `rdma` or `tcp` | `rdma` |
| `--flagcx-lib-path` | Path to `libflagcx.so` | `$FLAGCX_PATH/build/lib/libflagcx.so` |
| `--flagcx-wrapper-path` | Path to FlagCX root directory | `$FLAGCX_PATH` |

**Note**: Operation type (read/write) is automatically determined by the connector:
- **NIXL**: Uses `read` (client reads from server)
- **Mooncake**: Uses `write` (client writes to server)
- **FlagCX**: Uses `write` (client writes to server)

## Output Format

```
KV Transfer Benchmark  connector=nixl  role=client
  device=gpu  gpu=0  op=read  iters=100  warmup=10  num_blocks=1
  NOTE: nixl connector uses op_type=read (matching vLLM connector semantics)
  nixl-backend=UCX
  sizes: 1.0 KB, 4.0 KB, 16.0 KB, 64.0 KB, 256.0 KB, 1.0 MB, 4.0 MB, 16.0 MB, 64.0 MB
------------------------------------------------------------------------
       1.0 KB  |  lat=   0.123 ms  |  BW=   0.01 GB/s  (   0.08 Gbps)  |  iters=100
  ✓ Verification passed
       4.0 KB  |  lat=   0.130 ms  |  BW=   0.03 GB/s  (   0.25 Gbps)  |  iters=100
  ✓ Verification passed
      16.0 KB  |  lat=   0.145 ms  |  BW=   0.11 GB/s  (   0.88 Gbps)  |  iters=100
  ✓ Verification passed
      64.0 KB  |  lat=   0.178 ms  |  BW=   0.36 GB/s  (   2.88 Gbps)  |  iters=100
  ✓ Verification passed
     256.0 KB  |  lat=   0.234 ms  |  BW=   1.09 GB/s  (   8.72 Gbps)  |  iters=100
  ✓ Verification passed
       1.0 MB  |  lat=   0.456 ms  |  BW=   2.19 GB/s  (  17.52 Gbps)  |  iters=100
  ✓ Verification passed
       4.0 MB  |  lat=   1.234 ms  |  BW=   3.24 GB/s  (  25.92 Gbps)  |  iters=100
  ✓ Verification passed
      16.0 MB  |  lat=   4.567 ms  |  BW=   3.50 GB/s  (  28.00 Gbps)  |  iters=100
  ✓ Verification passed
      64.0 MB  |  lat=  18.234 ms  |  BW=   3.51 GB/s  (  28.08 Gbps)  |  iters=100
  ✓ Verification passed
------------------------------------------------------------------------
Done.
```

## Verification

The benchmark always performs strict element-by-element verification after each transfer size. Verification runs after timing completes, so it does not affect performance measurements. If any mismatch is detected, an `AssertionError` is raised with detailed diagnostics (which block, which element, expected vs actual value).

Verification semantics per connector:
- **NIXL (read)**: Client checks that received data matches server's pattern (0s)
- **Mooncake (write)**: Server checks that received data matches client's pattern (1s)
- **FlagCX (write)**: Server checks that received data matches client's pattern (1s)

## Architecture

```
┌─────────────────────────────────────────────┐
│           kv_transfer_benchmark.py          │
├─────────────────────────────────────────────┤
│  TransportBenchmark (ABC)                   │
│    setup() → init transport, register mem   │
│    run_transfer() → execute one transfer    │
│    verify() → check correctness             │
│    teardown() → cleanup                     │
├─────────────────────────────────────────────┤
│  NixlBenchmark                              │
│    Uses nixl_agent + nixl_agent_config      │
│    ZMQ for metadata exchange                │
│    Per-iteration handle init/release        │
├─────────────────────────────────────────────┤
│  MooncakeBenchmark                          │
│    Uses mooncake.engine.TransferEngine      │
│    ZMQ for session/address exchange         │
│    batch_transfer_sync_write()              │
├─────────────────────────────────────────────┤
│  FlagCXBenchmark                            │
│    Uses FLAGCXLibrary (ctypes wrapper)      │
│    ZMQ for unique ID exchange               │
│    flagcxPut/flagcxBatchPut + signal/wait   │
└─────────────────────────────────────────────┘
```

## Comparison with vLLM Connectors

This benchmark measures **transport-level** performance without vLLM overhead:
- No scheduler/worker coordination
- No KV cache block management
- No attention metadata
- Direct memory registration and transfer

For **end-to-end** vLLM KV transfer benchmarks, use:
```bash
vllm serve <model> --kv-transfer-config '{...}'
vllm bench serve --base-url http://127.0.0.1:8000 ...
```

## Troubleshooting

### NIXL Import Error
```
Failed to import NIXL. Is the nixl Python package installed?
```
**Solution**: Install NIXL with `pip install nixl-cu12` (or `nixl-cu11`)

### FLAGCX Backend Not Found
```
NIXL agent initialization failed
```
**Solution**: Ensure `libflagcx.so` is in `LD_LIBRARY_PATH` and `libplugin_FLAGCX.so` is on the NIXL plugin path.

### Mooncake Import Error
```
Failed to import Mooncake.
```
**Solution**: Install Mooncake following https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md

### FlagCX Import Error
```
FLAGCX_PATH not set and --flagcx-wrapper-path not provided.
```
**Solution**: Set `FLAGCX_PATH` environment variable or use `--flagcx-wrapper-path` to point to the FlagCX root directory.

### FlagCX Library Not Found
```
OSError: libflagcx.so: cannot open shared object file
```
**Solution**: Build FlagCX and ensure `libflagcx.so` exists at `$FLAGCX_PATH/build/lib/libflagcx.so`, or use `--flagcx-lib-path` to specify the path.

### ZMQ Connection Timeout
**Solution**: Ensure both server and client use the same `--remote-ip` and `--zmq-port`. Check firewall rules.

### CUDA Out of Memory
**Solution**: Reduce `--sizes` or use `--device=cpu` for testing.
