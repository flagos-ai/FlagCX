# NCCL Wrapper for FlagCX

This plugin builds a drop-in `libnccl.so` that intercepts NCCL API calls and routes them through FlagCX. Any application or framework that uses NCCL (e.g. PyTorch, DeepSpeed) can transparently use FlagCX without code changes.

## How It Works

The wrapper exports all public NCCL 2.27.7 symbols. When an application calls an NCCL function (e.g. `ncclAllReduce`), the wrapper translates the call into the equivalent FlagCX API.

Because FlagCX's internal NCCL adaptor itself calls back into `nccl*` functions, the wrapper uses a **thread-local recursive guard** to prevent infinite recursion. On the first call, the wrapper delegates to FlagCX. If FlagCX re-enters an `nccl*` symbol on the same thread, the wrapper forwards the call to the **real** NCCL library (loaded via `dlopen` at runtime).

```
Application
    |
    v
ncclAllReduce()          <-- wrapper (first entry, delegates to FlagCX)
    |
    v
FlagCX
    |
    v
ncclAllReduce()          <-- wrapper (re-entry detected, forwards to real NCCL)
    |
    v
Real NCCL
```

## Prerequisites

- **FlagCX** built and installed (the wrapper links against `libflagcx.so`)
- **CUDA toolkit** (for `cuda_runtime.h` and `libcudart`)
- **Real NCCL >= 2.27.7** installed somewhere on the system (the wrapper loads it via `dlopen` at runtime). The wrapper exposes NCCL 2.27.7 symbols, so the underlying real NCCL must be at least this version to provide all required functions.

## Building

```bash
cd plugin/nccl
make NCCL_HOME=/path/to/nccl CUDA_HOME=/path/to/cuda
```

### Build Variables

| Variable | Default | Description |
|---|---|---|
| `FLAGCX_HOME` | `../../` (auto-detected) | Path to the FlagCX source/build tree |
| `CUDA_HOME` | `/usr/local/cuda` | Path to the CUDA toolkit |
| `NCCL_HOME` | `/usr/local/nccl` | Path to the real NCCL installation |
| `REAL_NCCL_LIB` | `$(NCCL_HOME)/lib/libnccl.so.2` | Full path to the real `libnccl.so.2` |

On systems where NCCL is not installed under a standard `NCCL_HOME/lib/` layout, set `REAL_NCCL_LIB` directly:

```bash
# Debian/Ubuntu with system NCCL
make REAL_NCCL_LIB=/usr/lib/x86_64-linux-gnu/libnccl.so.2

# Custom location
make REAL_NCCL_LIB=/opt/libs/libnccl.so.2
```

Build outputs go to `build/`:

```
build/
  lib/
    libnccl.so.2.27.7    # the wrapper library
    libnccl.so.2          # symlink
    libnccl.so            # symlink
  include/
    nccl.h
```

## Usage

Use `LD_PRELOAD` to inject the wrapper library so that it intercepts all NCCL calls made by the application:

```bash
LD_PRELOAD=/path/to/plugin/nccl/build/lib/libnccl.so.2 python train.py
```

You can also point `LD_LIBRARY_PATH` to the wrapper's lib directory so that the dynamic linker picks it up before the real NCCL:

```bash
export LD_LIBRARY_PATH=/path/to/plugin/nccl/build/lib:$LD_LIBRARY_PATH
python train.py
```

Make sure the FlagCX library is also loadable at runtime (it is set via `rpath` during linking, but you may need to add it to `LD_LIBRARY_PATH` if you moved things around):

```bash
export LD_LIBRARY_PATH=/path/to/flagcx/build/lib:/path/to/plugin/nccl/build/lib:$LD_LIBRARY_PATH
python train.py
```

## Supported APIs

The following NCCL APIs are intercepted and routed through FlagCX:

- **Version / Error**: `ncclGetVersion`, `ncclGetErrorString`, `ncclGetLastError`
- **Unique ID**: `ncclGetUniqueId`
- **Communicator**: `ncclCommInitRank`, `ncclCommInitRankConfig`, `ncclCommFinalize`, `ncclCommDestroy`, `ncclCommAbort`
- **Communicator Query**: `ncclCommCount`, `ncclCommCuDevice`, `ncclCommUserRank`, `ncclCommGetAsyncError`
- **Memory**: `ncclMemAlloc`, `ncclMemFree`
- **Buffer Registration**: `ncclCommRegister`, `ncclCommDeregister`, `ncclCommWindowRegister`, `ncclCommWindowDeregister`
- **Collectives**: `ncclAllReduce`, `ncclBroadcast`, `ncclReduce`, `ncclAllGather`, `ncclReduceScatter`
- **Point-to-Point**: `ncclSend`, `ncclRecv`
- **Group**: `ncclGroupStart`, `ncclGroupEnd`

The following APIs are exported but return `ncclInvalidUsage` (no FlagCX equivalent):

`ncclBcast`, `ncclCommInitAll`, `ncclCommSplit`, `ncclCommShrink`, `ncclCommInitRankScalable`, `ncclRedOpCreatePreMulSum`, `ncclRedOpDestroy`, `ncclGroupSimulateEnd`
