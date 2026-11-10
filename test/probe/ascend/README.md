# Standalone Ascend capability probes

The VMM and IPC probes link CANN and MPI; the optional NIC probe also links
libibverbs. They do not initialize FlagCX, create a FlagCX communicator, or
run the proxy service. Run them on the target Ascend host with two visible
NPUs before implementing the SymMem and RMA callbacks.

## VMM mapping

```sh
make -C test/probe/ascend DEVICE_HOME=/usr/local/Ascend/ascend-toolkit/latest
make -C test/probe/ascend run-same MPI_ARGS=--allow-run-as-root
make -C test/probe/ascend run-cross MPI_ARGS=--allow-run-as-root DEVICE_PAIR=0,1
```

`run-same` establishes a same-device two-process baseline. `run-cross` is the
required probe for the SymMem design: rank 0 selects the first device and rank
1 selects the second. Both ranks create two independent physical allocations,
export 64-bit CANN shareable handles, import the peer's allocations, map them
individually, and then try to map all four allocations into one VA reservation.
The probe checks data at a subrange and near each segment boundary, including
the peer-owned segments. It unmaps and releases imported handles before either
rank releases its owned physical memory.

Each stage prints `PROBE stage=... status=PASS|UNSUPPORTED|FAIL`. On a CANN
failure, the rank also prints the API name and raw `acl_error`. Exit codes are
0 for full PASS, 3 for an explicit CANN feature-not-supported result, and 2 for
other failures. Neither unsupported nor a partial pass is acceptance. Save
both ranks' output, the installed CANN/driver versions, and `npu-smi info` with
the result. A mixed flat-map failure is evidence to resolve before advertising
SymMem VMM support; a successful independent peer map alone is insufficient.

The VMM probe does not establish VMM-backed IB MR support or a device-kernel
data path.

## Stream value operations and remote peer visibility

CANN exposes `aclrtValueWrite` and `aclrtValueWait`; the latter accepts
`ACL_STREAM_WAIT_VALUE_GEQ`. The stream probe builds a small AscendC producer
and consumer kernel, so it needs `SOC_VERSION` and `ASCENDC_CMAKE_DIR` from the
installed toolkit. For example:

```sh
make -C test/probe/ascend stream-run-same \
  SOC_VERSION=Ascend910_9362 \
  ASCENDC_CMAKE_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake \
  MPI_ARGS=--allow-run-as-root
make -C test/probe/ascend stream-run-cross \
  SOC_VERSION=Ascend910_9362 \
  ASCENDC_CMAKE_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake \
  MPI_ARGS=--allow-run-as-root DEVICE_PAIR=0,1
```

Adjust both toolkit paths and `SOC_VERSION` to the installed CANN package.
`stream-run-same` runs two processes on NPU 0; `stream-run-cross` places them
on different NPUs. Each run first checks same-stream producer → ValueWrite(2)
→ ValueWait(GEQ 1) → consumer ordering. The default remote mode is
`valuewrite`; set `STREAM_MODE=kernel-readback` or
`STREAM_MODE=kernel-wait` for a Makefile run. The CI job runs all three modes
in separate MPI processes:

* `--mode valuewrite` reports local and IPC-imported pointer attributes, then
  submits `aclrtValueWrite` to the imported signal. A failure includes the
  recent CANN error message. If submission succeeds, rank 1 reads the signal.
  Each rank also queries `drvMemGetAttribute` for the same pointer and logs
  the raw `mem_type`, device, page size, and loaded driver/ACL library paths. If
  the driver symbol is unavailable, the raw query prints `UNAVAILABLE` and
  the remaining probe still runs.
* `--mode kernel-readback` has rank 0 write the imported payload and signal with
  ordered device kernels. After the writer completes, rank 1 runs a direct
  device payload consumer and device signal reader before copying either region
  to Host for diagnosis.
* `--mode kernel-wait` has rank 1 queue ValueWait(GEQ 1) immediately followed
  by the device consumer before rank 0 writes. There is no Host copy or extra
  synchronization between the wait and consumer.

The writer-completion and Host-readback steps in `kernel-readback` identify
where data stops being visible; they are not acceptance of the unsynchronized
wait contract. Every stream sync has a 10-second timeout. Separate processes
keep a timed-out wait from affecting another mode.

This is a **remote peer-device** visibility test through CANN IPC. It does not
prove that `aclrtValueWait` acquires incoming **NIC RDMA** writes: its public
flag controls comparison mode, and the API does not document a remote-write
flush flag.

## NIC RDMA WRITE visibility

The optional `ib_visibility_probe` requires userspace `libibverbs`, an active
RC-capable port, and a provider that can register `aclrtMalloc` device memory
with `ibv_reg_mr`. On a host with those dependencies:

```sh
make -C test/probe/ascend ib-run-cross \
  SOC_VERSION=Ascend910_9362 \
  ASCENDC_CMAKE_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake \
  IB_DEVICE=mlx5_0 IB_PORT=1 IB_GID_INDEX=0 DEVICE_PAIR=0,1 \
  MPI_ARGS=--allow-run-as-root
```

Set `IB_DEVICE` to the local verbs device name on each rank. The Makefile
example supplies one shared name; for multi-node systems with different local
names, launch the built executable with rank-specific environments. The sender
posts ordered RC RDMA WRITEs for the payload and then an 8-byte signal to the
receiver's device allocation. `--mode readback` (`IB_MODE=readback` with Make)
records the sender's CQ result,
then runs a direct device payload consumer and signal reader before
recording diagnostic Host copies. `--mode wait`
(the default) queues `aclrtValueWait` before the sender posts and runs the
consumer immediately after the wait. CQ completion and wait completion have
separate stages. No Host copy or extra device synchronization is inserted
between wait and consumer. A GPU MR registration failure, CQ error, wait
timeout, stale data, or missing consumer result fails the relevant mode.
Success establishes visibility only
for the tested CANN, driver, NIC provider, and topology; it does not imply
support on every Ascend system or set a FlagCX capability automatically.

## Ascend CI

For automatic PR runs on `ascend-symmem`, the main CI workflow schedules only
the `ascend_probe` job. Manual workflow dispatch and `/test` requests retain
the original test suites. The probe job builds and runs VMM and stream IPC on
the same device and across devices, then runs NIC RDMA visibility across
devices. IPC runs the three modes above; NIC runs `readback` and `wait`.
`results-detail.tsv` records each mode, while `results.tsv` records the five
top-level `PASS`, `FAIL`, `UNSUPPORTED`, or `NOT_RUN` cases and their totals.
A completed five-case report succeeds as a diagnostic
CI job even when a probe fails or reports unsupported. Build failures or
missing hardware leave dependent cases `NOT_RUN` and fail the incomplete
report. Tagged rank output, build logs, host RDMA inventory, and the TSV
result summary are uploaded in the `attempts-ascend-probe-*` artifact. To
select another topology, set `FLAGCX_CI_PROBE_DEVICE_PAIR`,
`FLAGCX_CI_PROBE_SOC_VERSION`,
`FLAGCX_CI_PROBE_ASCENDC_CMAKE_DIR`, `FLAGCX_CI_PROBE_IB_DEVICE`,
`FLAGCX_CI_PROBE_IB_PORT`, or `FLAGCX_CI_PROBE_IB_GID_INDEX` in the container
environment.
