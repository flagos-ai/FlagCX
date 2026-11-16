# MUSA Device API Backend

This document describes the Moore Threads MUSA backend for the FlagCX Device
API: how to build it, how to run the device test suite, and which device
capabilities are supported on the current hardware/driver generation.

The backend targets MTT S5000 GPUs (`mp_31`) with the MUSA SDK 4.3.5 toolchain
(`mcc` 5.1.0). It plugs into the platform-neutral `DefaultBackend` through
`PlatformTraits<MusaPlatform>` and `CommTraits<DefaultBackend<MusaPlatform>>`;
no Device API kernel code is forked.

## Requirements

| Item | Value |
|---|---|
| GPU | MTT S5000 (reports compute capability 3.1, target `mp_31`) |
| SDK | MUSA 4.3.5, `mcc` 5.1.0 |
| Host compiler | g++ with C++17 |
| MPI | any MPI implementation with `mpirun`/`mpicxx` |
| Kernel driver | MUSA kernel driver matching the SDK (`mthreads-gmi` for inventory) |

`--offload-arch` must match the device. `mp_32` compiles but every launch then
fails with `musaErrorInvalidDeviceFunction`; the makefile defaults to `mp_31`
and exposes `MUSA_ARCH` for override.

## Build

Build the library and the device test programs:

```shell
make USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa -j
make -C test/unittest/device_api mpi unit USE_MUSA=1 COMPILE_KERNEL=1 \
     DEVICE_HOME=/usr/local/musa -j
```

`COMPILE_KERNEL=1` is required for the Device API tests: the Device API
implementation is compiled into `libflagcx.so` as device code, and the test
binaries additionally link their own device kernels.

MUSA has no separate device link step. `mcc -dlink` is rejected by the driver,
so `makefiles/musa.mk` sets `DEVICE_NEEDS_DLINK := 0` and a single `.mu` object
already carries a complete device image.

Like the other device platforms, the compiler emits an executable for every
source file that includes the platform traits. Keep `COMPILE_KERNEL` and the
toolchain path consistent between the library build and the test build; the
build-configuration stamp rebuilds objects when those variables change.

## Running the Device API tests

Single node, 8 GPUs:

```shell
cd test/unittest/device_api/build/bin
mpirun --allow-run-as-root -np 8 ./test_device_api_intra -b 1K -e 16M -f 2
mpirun --allow-run-as-root -np 8 ./test_device_ir_unified_intra -b 1K -e 16M -f 2
```

Two nodes, 4 GPUs each:

```shell
mpirun --allow-run-as-root -np 4 -x MUSA_VISIBLE_DEVICES=0,1,2,3 ./test_device_api_inter ...
                   : -np 4 -x MUSA_VISIBLE_DEVICES=0,1,2,3 ./test_device_api_inter ...
```

Do not set `FLAGCX_HOSTID` for the two-node runs; the device API paths used by
these tests select peers through the communicator, and a forced host id splits
the job into unrelated single-node groups.

Inter-node operation needs a real RDMA path (`/dev/infiniband` visible in the
container and a NIC with GPUDirect support). Without it FlagCX falls back to
Socket, which cannot carry GPU memory and is not evidence for the inter-node
Device API cases.

### Environment variables

| Variable | Effect |
|---|---|
| `FLAGCX_SIGNAL_HOST_ENABLE=1` | Allocate signal/counter planes in host-pinned memory instead of device memory. The two-node gate uses the default device-memory plane; both planes show the same intermittent stall (limitation 7). |
| `FLAGCX_DMABUF_ENABLE=1` | Allow the RDMA transport to register device buffers through DMA-BUF instead of a raw VA. Falls back to VA registration when a range cannot be exported. |
| `MUSA_VISIBLE_DEVICES` | Restrict the visible GPUs per process; used to pin four ranks to each node. |

`FLAGCX_SIGNAL_HOST_ENABLE=0` (device-memory signal plane) is the default for the
cross-node gate: remote signals are delivered by an RDMA fetch-and-add into the
peer's signal buffer, and keeping that buffer in device memory avoids a second
host-pinned registration path. The host-pinned plane (`=1`) works the same way
and shows the same intermittent stall described under limitations, so it is not
gated separately.

## Supported capabilities

| Area | Status on `mp_31` |
|---|---|
| Device memory (`deviceMalloc`/`deviceFree`, managed, host-pinned) | supported |
| Host-pinned memory | registered heap memory; `musaHostGetDevicePointer` gives the device alias |
| P2P / IPC handles | supported; the adaptor exports and opens peer allocations |
| DMA-BUF export for GPUDirect RDMA | supported for page-aligned allocation bases; other ranges fall back to VA registration |
| GDR allocation | supported; allocation is rounded to the device granularity so the base is exportable |
| Stream/event operations | supported |
| One-sided device operations (`Put`/`Get`/`Signal`/`Wait`/counter) | supported through the Default Device API; cross-node is functional but has an unresolved intermittent wait stall (limitation 7); the Device IR bindings are single-node only for now |
| Symmetric memory (VMM/multicast) | not supported (returns `flagcxNotSupported`) |
| 32/64-bit atomics, acquire/release fences | supported at device scope (see limitations) |
| Named barriers | whole block and single 32-lane wave only; other group sizes trap |
| Cooperative tile sync | whole wave; a sub-warp mask is widened to the whole wave |

## Known limitations

1. **Fence scope.** On this toolchain `__threadfence_system()` is an alias of
   `__threadfence()`, and the backend has no documented system-scope membar.
   Release/acquire for `Device` and `System` scope therefore both lower to the
   device (GL) fence. Cross-device ordering of published signals and counters
   relies on the transport; it is not provided by a system-scope fence. Any
   future path that publishes device-written data straight to a peer GPU or to
   host memory without going through the transport has to be validated on its
   own.

2. **Scoped atomics.** Address-space-3 (shared memory) scoped atomic loads
   cannot be selected by the `mp_31` backend, so the traits use the hardware
   relaxed RMW plus explicit fences instead of the scoped builtins. The
   observable semantics match a device-scope acquire/release sequence.

3. **Cooperative tile sync.** `__syncwarp()` is a whole-wave barrier and ignores
   its mask argument. A sub-warp tile is served by that whole-wave barrier,
   which is a stronger sync and is valid only when every lane of the wave
   reaches the call, as the unified API's `CoopTile` groups do. A genuinely
   divergent partial-mask group must not call it.

4. **Named barriers.** `__barrier_sync`/`__barrier_sync_count` make `mcc` 5.1.0
   loop forever, so only the whole-block and single-wave granularities are
   exposed. Other spans trap instead of hanging.

5. **DMA-BUF alignment.** The DMA-BUF export ioctl accepts only page-aligned
   allocation bases. The GDR allocation path compensates, and a range that
   cannot be exported falls back to VA registration rather than failing the
   connection.

6. **Inter-node Device IR signal path.** The two inter-node Device IR suites do
   not pass on MUSA yet, so they are excluded from the gated stages
   (`REAL_INTER_IR=1` runs them for reproduction):

   - `test_device_ir_inter` passes S1 and S2 on all eight ranks, then aborts in
     S3 (`PutSignalIncS`) with `err=700` (illegal memory access) on the first
     `streamSynchronize`.
   - `test_device_ir_unified_inter` passes S16–S19 (barrier, team resolution,
     put, get) on all eight ranks, then aborts in S20
     (`DevSignalStandalone`) with the same `err=700`.

   Both failures reproduce with the device-memory and the host-pinned signal
   plane, so they are not a configuration artifact and not a wedge: a minimal
   kernel runs on every GPU of both nodes immediately afterwards. The
   single-node IR suites do pass (`test_device_ir_intra` completes all of
   S1–S14). The failing cases are the first ones that drive a *remote* signal
   through the device IR bindings, which points at that path rather than at the
   Default Device API backend exercised by T1/T2.

   The logical-node variant of the inter-node suite (two `FLAGCX_HOSTID` groups
   on one host) does not reach a runnable state on the eight-GPU node either:
   `flagcxDevCommCreate` returns `flagcxNotSupported` because neither the signal
   IPC plane nor the Net signal fallback becomes ready. Only the
   two-physical-node gate is supported.

7. **Intermittent cross-node wait stall.** A full cross-node T2 sweep
   occasionally stops at a random case with `err=702` ("launch timed out and was
   terminated") on one node's ranks while the other node's ranks have already
   completed the case. The waiting kernel never observes the RDMA fetch-and-add
   that should complete its signal, so it spins until the driver kills the
   launch. Nothing is logged on the host side, the run cannot be recovered, and
   the same configuration passes when repeated.

   Measured on the two-node pair, nine full 1 KiB–16 MiB sweeps with the
   device-memory signal plane: six passed and three stalled with `err=702`
   (at K4, at K14, and around K13/K14 hitting both nodes). A sweep with the
   host-pinned plane stalled as well, so the plane is not the cause. Every
   stalled sweep is a *functional* failure of the cross-node signal path, not a
   test-harness artifact, and it is not yet fixed.

   A minimal kernel still runs on every GPU of both nodes right after a stall,
   so the devices are healthy; the fault is in signal delivery/observation, not
   in the device. The relay posts the completion as `FLAGCX_RMA_PUT_SIGNAL`
   (RDMA fetch-and-add into the peer signal buffer), and that buffer is polled
   by a spinning device loop. Two candidates remain: a lost/never-posted relay
   completion, and a polling load that does not observe a PCIe write into
   device memory. Both planes are affected, so the backend cannot yet claim a
   stable cross-node signal path.

## Validation status

The backend is validated on two physical nodes with four MTT S5000 GPUs each
(eight ranks total) over RDMA/DMA-BUF. Both nodes build from the same commit and
produce a byte-identical `libflagcx.so`
(`sha256 1dbbc2dc6625b399fa59f2cd18e529039c4db14d6cd8ba277166b9dbc50c34ce`) and
byte-identical test binaries.

| Gate | Program | Rounds | Result |
|---|---|---|---|
| T1 | `test_device_api_intra` (8 GPUs, one node) | 3 | PASS |
| T2 | `test_device_api_inter` (4+4 ranks, two nodes, RDMA) | 9 | 6 PASS, 3 stall (`err=702`, see limitation 7) |
| T3 | `test_device_ir_unified_intra` (8 GPUs, one node) | 3 | PASS |
| T4 | `test_device_ir_unified_inter` (4+4 ranks, two nodes) | 3 | known failure at S20 |
| — | `test_device_ir_inter` (4+4 ranks, two nodes) | 3 | known failure at S3 |

Every recorded T1/T2/T3 pass runs the full 1 KiB to 16 MiB sweep and passes
every official case on every rank; the per-case assertions (`K14
OneSidedAlltoAll`, `=== Overall: PASS ===`, `S10`, `=== Final Result: ALL
PASS ===`) are checked by `scripts/musa_device_api/`, which also asserts the
topology report, the IB transport line, and the per-node library hash.

Two stability observations are worth recording:

- `test_device_ir_unified_intra` occasionally exceeds a five-minute budget after
  S21 (`DevPutSignalWait`). S21 and later cases pass when the run is given more
  time; the sweep is simply slow on this driver.
- Single-node gates (T1/T3) are deterministic across rounds; the instability is
  confined to cross-node signal delivery (limitation 7).

No RDMA bandwidth or GPU Direct RDMA stress test was run as part of this
validation; the inter-node evidence is functional, not a performance claim.

## Reproducing the validation

```shell
# T1 / T3: single node, eight GPUs
mpirun --allow-run-as-root -np 8 ./test_device_api_intra -b 1K -e 16M -f 2
mpirun --allow-run-as-root -np 8 ./test_device_ir_unified_intra -b 1K -e 16M -f 2

# T2: two physical nodes, four GPUs each (hostfile with one line per node)
REAL_INTER_HOSTFILE=/tmp/flagcx.hosts \
  bash scripts/musa_device_api/run_real_inter.sh

# Known inter-node IR failure, same two nodes
REAL_INTER_HOSTFILE=/tmp/flagcx.hosts REAL_INTER_IR=1 \
  bash scripts/musa_device_api/run_real_inter.sh
```
