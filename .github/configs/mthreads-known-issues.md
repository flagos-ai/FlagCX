# Mthreads CI known capability gaps

These records come from the Mthreads CI image and hardware run. The affected
cases are recorded as capability skips so the remaining tests can complete.
The CI artifact under `capability-skips/` includes the environment and device
inventory for each skip. A timeout or hang is collected by
`run_with_timeout.sh` in the same artifact directory with `/proc` thread
stacks and a gdb `thread apply all bt` backtrace when gdb is available.

## device_api and device_api_unified_ir

The current MUSA backend has no `test/kernel/musa` target or kernel sources;
`makefiles/musa.mk` leaves `PLATFORM_KERNEL_DIR` and `DEVICE_COMPILER` unset.
The regular device API tests therefore cannot build their required
`device_api.o`. The unified IR tests additionally have no `bindings/ir/musa`
implementation. Both suites are included in the Mthreads matrix and reported
as capability skips until those backend pieces exist. `device_api_host` remains
the runnable host-only coverage.

## adaptor GPU memory registration

The selected IBRC/SHCA adaptor does not advertise GPU MR. The observed
message is:

```text
Selected RDMA adaptor does not advertise GPU MR
```

Observed test locations include `test/net_adaptor/test_net_adaptor.cpp` near
the GPU MR registration, put/get, batch, and bounds cases (around lines 1245,
1611, 1644, 1677, and 1756 in the captured run). The adaptor suite keeps host
and NET coverage and records GPU-MR cases as optional capability skips.

## P2P WRITE MPI

MUSA GPU buffer registration/classification fails before the WRITE operation:

```text
flagcx/core/flagcx_p2p.cc:3277 ... failed to classify addr ...: 9
[rank] P2P WRITE: registering GPU buffer failed
MPI_ABORT
```

The Mthreads CI path records this evidence and skips the WRITE MPI workload;
the P2P unit tests still run.

## RMA IPC and direct-consumer visibility

The Mthreads IPC data window cannot be created. Observed errors include:

```text
RMA data window is unavailable
flagcx/adaptor/flagcx_device.cc:697 buildIpcPeerPointers: ipcMemHandleCreate failed
```

The affected IPC cases include `GetAsyncOffsets`,
`IpcGetCounterFollowsCopy`, `IpcResolvedPeerPointerSupportsDirectCopy`,
`IpcPutWithoutNetworkMr`, `PutSignalSmall`, `PutSignalLarge`,
`SignalOnlyNoData`, and `MultipleSignals`. Direct-consumer visibility has the
same unavailable IPC/data-window prerequisite and is skipped with a separate
artifact record. RMA NET tests remain enabled.

## Runner heterogeneous P2P

Local P2P setup fails at IPC handle creation. The observed call chain is:

```text
flagcx/core/p2p.cc:792 deviceAdaptor->ipcMemHandleCreate failed
flagcx/core/p2p.cc:692 -> 9
flagcx/core/proxy.cc:1349 -> 9
flagcx/runner/c2c_algo.cc:2543 -> 9
```

This prevents the heterogeneous `AllGather`, `AllReduce`, `AlltoAll`, and
`AlltoAllvAsymmetric` cases from completing. The CI path records the issue and
skips heterogeneous P2P while retaining homogeneous and forced-NET runner
coverage.
