# MetaX UniRunner and experimental CI

Build the native shared executor and Default Device API backend with:

```bash
make -j USE_METAX=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1
```

The MACA translation unit includes the same executor implementation used by
CUDA, Hygon/DU and PPU. Runtime device attributes and occupancy determine the
cooperative launch budget; no fixed occupancy or ordinary-launch substitute
is used. Scalar reductions support int8, uint8, int32, uint32, int64, uint64,
FP32 and FP64. FP16/BF16 reductions are rejected before execution. Fused GEMM
uses FP32. The native executor is separate from the Device API bitcode exports;
existing native, Scalar IR, Unified IR and bitcode tests remain required.

## Allocation purposes

The internal latest adaptor appends an optional `deviceMallocWithFlags` callback.
The frozen v1 plugin ABI, public device handle and plugin version are unchanged.
Callers use `flagcxDeviceAllocWithFlags`, which validates purpose bits and routes
`DEFAULT` to the original allocator. `UNIRUNNER` invokes the optional policy;
when no callback exists, including upgraded v1 plugins, it uses the original
allocator with the same arguments. An error from an implemented policy is
returned without retrying a different allocator.

| Adaptor | Synchronous device allocation for `UNIRUNNER` |
|---|---|
| MetaX | `mcMalloc` |
| CUDA, Hygon/DU, HIP, PPU, Iluvatar, MUSA, Ascend, MLU, Kunlunxin, TOPS, TSMicro, PTPU | Original `deviceMalloc` |

The first consumer is the local DAG `streamFlagsPool`. This purpose is not
applied to IPC scratch, shared ready/progress or network signal allocations.
MetaX ordinary synchronous device allocations keep finegrained. Host, managed
and asynchronous allocations retain the original path. No HOST SIGNAL or
host-mapped DAG flag pool is introduced. GDR/VMM allocation, SYNC_MEMOPS and
remote-write flush policies are unchanged.

Earlier CI passed baseline memops on `mcMalloc` and rejected the tested
extended-allocation addresses, including finegrained and Default. This is the
reason for the MetaX policy; it does not establish a permanent restriction for
every finegrained address alias or SDK version.

Pool growth is transactional. It initializes replacement storage before
releasing the old allocation, retains failed rollback allocations, and retries
cleanup before another growth. Existing execution and recovery drains remain
required before reset, replacement or destruction.

## Capabilities and execution

Compiled kernel availability, device-control memops, host-mapped semaphore
memops and remote-write flush are separate capabilities. A lazy check uses the
actual `UNIRUNNER` device allocator on a private stream, writes and waits for a
64-bit value including high bits, synchronizes and checks readback. Its result
is cached separately from the existing host semaphore check. No allocation
matrix or RDMA probe is added. Real algorithms verify stream/kernel interaction.
`InvalidValue` and other unexpected errors remain errors; only an explicit
`NotSupported` operation result is classified as unavailable.

MetaX performs optional checks only when UniRunner is selected. Preparation
results converge across ranks before dependent execution. Explicit GroupedAG
(algorithm 7) fails if required capabilities are absent; automatic selection
can retain the existing P2pAG fallback. Logs identify actual AllReduce/AllGather
selection, device allocation policy, residency budget and native launchers.

## Experimental CI

The `perf-test-experimental (metax)` job follows the existing L2/L3 platform
routing. It uses four MPI ranks, `MACA_VISIBLE_DEVICES=0,2,4,6`, the selected
MetaX HCAs and `COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1`. Symmem's separate GPU
selection and PXN's GPU/HCA override remain unchanged.

| Algorithm | Required coverage |
|---|---|
| 2 | IPC Ring AllReduce, 64K–4M; IPC buffer rebinding |
| 4 | RingAG AllReduce with an uneven 40-byte count |
| 6 | SlicedAR, 64K–64M, multiple slices and repeated invocations |
| 7 | GroupedAG, group size 1, 64K–4M, no silent fallback |
| 8 | P2pAG inside a user group; not native-kernel evidence |
| 9 | FP32 AllGatherGemm and rank-local size failure convergence |
| 11 | FP32 GemmReduceScatter and rank-local size failure convergence |
| Error paths | Outer-group rejection and rank-local IPC preparation failure |

`FLAGCX_CI_METAX_SLICED_AR_MAX_BYTES` overrides the SlicedAR upper bound.
GTest cases require passing XML from every rank. Kernel, device-control and
local-P2P expectations are explicit. Successful runs must also contain the
actual algorithm and `mcMalloc` policy logs; algorithms 2, 6, 9 and 11 require
native-launcher evidence. The job uses existing stage/workload
deadlines and MPI cleanup, records each result and stops after the first failed
invocation; remaining cases are `BLOCKED`, never passes. The performance gate
requires both ordinary and experimental MetaX jobs when MetaX is selected.

Always-uploaded artifacts contain build/invocation logs, runtime/driver and
GPU PCI information, HCA selection, per-rank XML and
`metax-experimental/results.tsv`. Directories are traversable and reports are
readable by the host uploader.

The existing exclusions remain:

- `DeviceAdaptorTest.StreamWaitWriteValue64`
- `DeviceAdaptorTest.StreamWaitValue64RemoteFlushCapability`
- `RmaTest.IpcGetSameStreamCompletion`

These still use the original allocation policy. Device API intra remains the
IPC visibility acceptance; RDMA probe sources and entry points stay absent.

## Validation

Hardware validation is required before declaring support complete:

1. Build MetaX native and Device API bitcode targets, and the other platform
   builds affected by the latest-adaptor extension.
2. Run adaptor compatibility, core stream-value probe, and runner stream-flag
   pool lifecycle unit tests.
3. Run the experimental job and require every matrix entry to pass with actual
   native launcher evidence for the kernel algorithms.
4. Re-run ordinary MetaX CI, especially Device API intra/native/IR/bitcode and
   communicator lifecycle checks. Check that unsupported remote flush does not
   become advertised as available.

Source and script checks do not establish MACA compilation, cooperative
residency, stream/kernel coherence, IPC visibility or hardware performance.
