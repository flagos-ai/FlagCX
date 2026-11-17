# MUSA Default Device API

This integration selects `CommTraits<DefaultBackend<MusaPlatform>>` and the
Default Host backend. It targets `mp_31`; set `DEVICE_HOME` and `MUSA_ARCH`
when using a different compatible SDK installation or architecture.

## Build and dependencies

The build needs the MUSA SDK/runtime, MCCL, a Host C++ compiler, MPI and the
project's IB dependencies. Initialize the existing dependency submodules:

```bash
git submodule update --init --recursive
make -B -j1 USE_MUSA=1 COMPILE_KERNEL=1 \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
make -B -C test/kernel/musa all PROJECT_ROOT="$PWD" \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
bin="$PWD/test/unittest/device_api/build/bin"
make -B -C test/unittest/device_api PROJECT_ROOT="$PWD" \
  USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31 \
  "$bin/test_device_api_intra" "$bin/test_device_api_inter" \
  "$bin/test_device_ir_unified_intra" "$bin/test_device_ir_unified_inter"
```

Main-library `.mu` files use compiler-managed independent device images. The
linker map keeps generated image symbols local to `libflagcx.so`. The public
test kernel objects use linked RDC images. Rebuild the library, device objects
and Host test executables together after changing Device API headers.

## Acceptance

Run inside the container with eight available devices and usable RDMA devices:

```bash
ROUNDS=3 BUILD=1 bash test/unittest/device_api/musa_all_acceptance.sh "$PWD"
```

This invokes the four existing public programs without changing their Host
drivers, data checks or device test cases:

| Test | Program | Topology | Required result |
| --- | --- | --- | --- |
| T1 | `test_device_api_intra` | 1 x 8 ranks | K10 and Overall PASS |
| T2 | `test_device_api_inter` | 2 x 4 ranks | K14 and Overall PASS |
| T3 | `test_device_ir_unified_intra` | 1 x 8 ranks | S16-S25 and ALL PASS |
| T4 | `test_device_ir_unified_inter` | 2 x 4 ranks | S16-S25 and ALL PASS |

All runs cover 1K through 16M with factor 2. T1/T2 use `-R 2`; T3/T4 register
their own windows. Open MPI `-x` supplies the same environment variables as
MPICH/Hydra `--genv`. The runner uses Open MPI and a 30-minute timeout per run.
The Inter groups have different `FLAGCX_HOSTID`/`NCCL_HOSTID` values. Each
public driver binds its world rank to a device; eight visible devices map
ranks 0-7 to devices 0-7.

The runner records and exports the MUSA runtime configuration, including
`FLAGCX_USE_HETERO_COMM=1`, `FLAGCX_VMM_ENABLE=0`, DMA-BUF, mapped Host signals,
and default MUSA rank-local CTA synchronization. FlagCX IB is enabled for
T2/T4 and disabled for T1/T3; MCCL IB is disabled. Copies submitted with a
stream use asynchronous MUSA copies. Explicit `FLAGCX_IB_HCA`,
`FLAGCX_IB_GID_INDEX`, and `FLAGCX_SOCKET_IFNAME` selections are forwarded;
choose them for the actual NIC topology.

Each required case must have 15 successful sizes on every rank. A nonzero
launcher exit, incomplete coverage, or failed required case stops the runner.
Existing SKIP cases in the public source remain SKIP, not functional coverage.

## Environment and result artifacts

`OUTPUT=` names a unique directory containing complete stdout/stderr logs,
the exact build/run commands, source and binary hashes, start/end timestamps,
launcher exits, `results.tsv`, and `summary.json`. Keep that directory as an
external acceptance artifact; generated logs and binaries are not source files.

`environment-report.json` follows the vendor environment template:

- FlagCX: full commit, branch/tag, backend, registration, build arguments.
- Host: OS, kernel, CPU architecture, Host compiler.
- Device: model/count/topology, SDK/runtime, driver, firmware and compiler,
  target architecture. Query failures are recorded explicitly; firmware not
  exposed by the platform is N/A.
- Communication runtime: MPI, MCCL; SHMEM is N/A for the Default backend.
- Artifacts: build commands, library path, actual `ldd` output for all four
  test binaries. The runner rejects resolution to a different `libflagcx.so`.

To collect those fields from an already built acceptance tree without starting
a build or tests:

```bash
METADATA_ONLY=1 bash test/unittest/device_api/musa_all_acceptance.sh "$PWD"
```

## Supported path and constraints

The Default path provides native IPC, mapped Host registration, DMA-BUF export,
Proxy Net operations and communicator/device-view lifecycle handling. Unsupported
VMM and multicast entry points report `flagcxNotSupported`; the validated run
uses VMM disabled. Vendor CCL/SHMEM Device API backends are not selected.

MUSA FIFO publication uses a system fence before the valid header. Proxy
PUT/Signal operations complete their submitted context work before releasing
the participating Coop. When available, Signal uses Host Proxy transport;
INTRA communicators without Net retain direct delivery. WORLD synchronization
uses local, Inter, then local phases. These conservative waits trade throughput
for ordering; they are not a replacement for the existing network visibility
flushes or caller participation rules.

The MUSA WORLD barrier includes rank-local CTA synchronization by default.
This requires every CTA to enter in the same order, full-CTA participation,
concurrent progress, and no concurrent kernels sharing the counter. It is not
a general grid barrier for arbitrary launches. Memory
allocation/free, IPC close, registration and error-word ownership retain their
separate lifetimes. Device/FIFO errors propagate to Host completion checks.
