# Iluvatar SymMem probes

This diagnostic suite is independent of `adaptor`, `rma`, and `symmem`.
Iluvatar runs the same SymMem test phases and filters as CUDA, including VMM
local mapping, segmented mapping, remote-without-NET, VMM+NET and route union.
DMA-BUF is explicitly disabled and network registration must use VA MR;
POSIX VMM allocation FDs are still required for local peer mappings.

## Run on CoreX

When Iluvatar is selected for PR CI, the platform suite matrix runs
`symmem_probe` as its own job alongside `symmem`. It builds FlagCX,
GoogleTest, and this suite in the existing CoreX image, then uploads inventory,
attempt logs, pre-timeout stacks, and per-rank XML. Suite jobs use
`fail-fast: false`, so a failure in `core` or `symmem` does not cancel the probes.
The existing SymMem tests do not execute this diagnostic suite.

The **Iluvatar SymMem probes** manual Actions workflow is also available for
running only this suite on the branch being tested.

Inside a prepared CoreX environment, the same entry point is:

```bash
bash .github/scripts/ci/run_unit_test.sh .github/scripts/set_env/iluvatar.sh symmem_probe
```

PR CI and the manual workflow run all 19 independent cases: the VMM
prerequisite, eleven component cases and seven MPI cases. This includes
multi-segment VA MR READ/WRITE and all five registration controls below.
There is no coverage mode selector. Every case must actually pass; an MR
failure fails the diagnostic job and retains its XML/logs. Failures are not
converted to success or skips. The formal `symmem` matrix is unchanged.

Every case runs in its own process with one attempt and a finite timeout.
The job first writes compile-only SDK diagnostics before building the library.
The `iluvatar-vmm-sdk` artifact directory contains the compiler/version,
check source, commands, errors and exit codes for types, basic VMM APIs, POSIX
FD sharing and the complete check. The host library and test builds use the
same check source. A zero result preserves legacy allocation compatibility;
it does not pass the required VMM probes.

`SymmemProbePrerequisites.VmmCallbacks` checks the compiled VMM flag and provider
callbacks without calling the GPU driver. Failure blocks component and MPI
probes with one explicit prerequisite failure. When that check passes,
allocation failure blocks MPI probes and leaves the suite failed. Other
component failures do not hide independent diagnostics. Missing, skipped,
failed, or absent-rank results cannot pass the XML gate.
Result directories and files remain readable by the host artifact uploader
even when the root container starts with a restrictive umask.

The compile-only diagnostic can also be run directly, without building or
executing GPU code:

```bash
make USE_ILUVATAR=1 iluvatar-vmm-diagnostics \
  ILUVATAR_VMM_DIAGNOSTICS_DIR=/tmp/iluvatar-vmm-sdk
```

| Probe | What it isolates |
|---|---|
| `SymmemProbePrerequisites.VmmCallbacks` | Compiled VMM support and required provider callbacks |
| `VmmProbe.AllocationAndRange` | Physical VMM allocation, data, interior pointer range |
| `VmmProbe.MultiSegmentSubrange` | Real segments and data spanning a boundary |
| `VmmProbe.AliasAndRetryableTeardown` | Alias data, invalid duplicate map, split teardown |
| `VmmProbe.LsaArenaOffsetsAndFdLifetime` | Map at reserved offsets, reuse, exported FD lifetime |
| `VmmProbe.VaMrReadWrite` | Direct VA MR and RDMA GET/PUT, without a FlagCX communicator |
| `VmmProbe.VaMrMultiSegmentReadWrite` | The same direct data path across physical segments |
| `VmmProbe.VaMrSingleSegmentSameSize` | Register one physical segment with the same total size as the three-segment controls |
| `VmmProbe.VaMrEachPhysicalSegment` | Register/deregister each of three physical segments independently; report every segment |
| `VmmProbe.VaMrCrossSegmentBoundary` | Register two pages straddling the first physical boundary |
| `VmmProbe.VaMrWholeMultiSegment` | Register/deregister the complete three-segment VA range |
| `VmmProbe.VaMrMultiSegmentStrongOrdering` | Register the complete range with FORCE_SO to compare against relaxed ordering |
| `MpiVmmProbe.LocalFdPeerMapping` | Two GPU processes, actual FD exchange and peer data, no FlagCX communicator |
| `MpiVmmProbe.LocalArenaWindowReuse` | Communicator arena, concurrent windows, released offset reuse |
| `MpiVmmProbe.MapFailureAndRetry` | Rank-local failure after actual mapping and successful retry |
| `MpiVmmProbe.RemoteWithoutNetRollsBack` | Expected remote registration rejection without NET |
| `MpiVmmProbe.WindowVaGetPut` | Public VMM window, actual VA route, remote GET/PUT payloads |
| `MpiVmmProbe.MrFailureAndRetry` | Rank-local VA MR failure before metadata publication |
| `MpiVmmProbe.DeregistrationFailureAndRetry` | Failed unmap retains the window, then cleanup retries |

`DeregistrationFailureAndRetry` follows public rank-local deregistration:
only rank zero receives the injected failure and retains its mapping and
allocation lease. Other ranks release their state successfully. Only rank zero
retries deregistration; the probe then checks lease release, public-handle
reuse and final allocation cleanup. Registration-failure probes still require
collective error convergence.

The component suite uses topology-local `mlx5` selection through the existing
net adaptor helper. A successful registration alone is not a data-path pass.
The GET/PUT probes exercise the same MR with READ and WRITE before
deregistration. No extra
`deviceSynchronize` is added after RMA completion to hide payload visibility
problems.

The five registration controls allocate exactly three VMM granules. The
single-segment control uses one physical handle; the other four use three.
They log GPU/HCA, the allocation and requested ranges, physical segment count,
granularity, MR flags, return code and errno observed after registration.
Registration failures remain failures, and each control runs independently so
one unsupported range does not hide the other comparisons. These controls do
not claim a passing RDMA data path; the existing GET/PUT probes check data.

## Relationship to the SymMem acceptance suite

Probes isolate failures. Final functional acceptance uses the corresponding
`symmem` cases with their required modes actually enabled:

| Capability / diagnostic probes | SymMem coverage |
|---|---|
| Allocation/range, alias, `LocalFdPeerMapping` | `RegisterSymmetricWindow`, `CrossGpuReadViaPeerPtr`, `CrossGpuWriteViaPeerPtr` |
| `LsaArenaOffsetsAndFdLifetime`, `LocalArenaWindowReuse` | `LsaWindowsShareArenaAndReuseReleasedOffset`, repeated registration and window cleanup |
| `VaMrReadWrite`, `WindowVaGetPut` | `VmmNetRouteCapabilityUnion`, `PublicRegistrationUsesVmmMrRouting`, `DirectGdrVmmPreservesMrRoute`, `HybridLocalAndRemoteAccess` |
| `RemoteWithoutNetRollsBack` | `RemotePeersWithoutNetworkDoNotPublishWindow` |
| `MapFailureAndRetry` | `VmmLsaMapFailureDoesNotPublishNetworkOnlyWindow`, `VmmRollbackFailureConvergesBeforeMrMetadataExchange` |
| `MrFailureAndRetry` | `RankLocalMrFailureDoesNotPublishPartialWindow`, `MrRollbackFailureRetainsWindowLeaseUntilRetry` |
| `DeregistrationFailureAndRetry` | `LocalDeregisterAndReusePublicHandle`, `CrossNodeLocalCleanupFailureRetries` |
| `MultiSegmentSubrange` and the registration controls | Local segmented geometry tests and multi-segment registration/failure-convergence tests; not an unconditional requirement that one MR span all physical segments |

The Iluvatar configuration enables the same SymMem phases as CUDA. Its
VMM+NET auto invocation requires VA routing, and route union exercises the
VA route only and requires successful registration. CUDA retains its existing
DMA-BUF/VA route union. With the default allocation settings, the route-union
case uses one physical segment.
`CrossNodeMixedMultiSegmentVmmRejectsSymmetricAndRetriesVmm` accepts a converged
registration failure and validates cleanup, while
`SegmentedMapFailureDoesNotPublishNetworkOnlyWindow` validates failure rollback.
A successful whole-range multi-segment MR is therefore not an unconditional
requirement of the existing SymMem suite. The controls diagnose this additional
capability without adding a segmented MR metadata/data-path implementation.

IXCUDA descriptions borrow original create handles, matching CUDA. The source
allocation stays alive until its mappings and bundles have been released;
describing an allocation does not create an export/import reference cycle.

## Diagnosing an untracked-allocation IPC crash

The formal `UntrackedVmmMappingIsRejected` case still uses the same range query
and legacy IPC trial as CUDA. No pointer-capability query or allocation
whitelist bypasses that trial. IXCUDA logs the range result, original/base
addresses, requested/allocation sizes and the IPC handle ABI size immediately
before calling the runtime.

If that formal case fails on Iluvatar or NVIDIA, CI runs an SDK-only
`ipc_vmm_repro` binary under GDB and uploads `ipc-vmm-diagnostics` with the
attempt artifacts. This binary links only the device SDK, without FlagCX,
MPI or GoogleTest. The four controls run as separate bounded processes:

| Control | Purpose |
|---|---|
| `malloc` | Ordinary `cudaMalloc` followed by runtime IPC export |
| `vmm` | Raw VMM allocation/map/access, range query and runtime IPC trial, matching the formal case |
| `vmm-driver` | The same VMM range with driver IPC as a diagnostic comparison |
| `vmm-sync` | The runtime trial after setting SYNC_MEMOPS, isolating that difference from adaptor VMM allocation |

Every control logs API entry/exit, device/context, runtime/driver versions,
symbol library paths and handle sizes. If the range query rejects a VMM
address, the log explicitly says `rejected-before-ipc`: that control did not
exercise IPC. The debugger captures an immediate fault's stack, registers,
instructions and memory mappings. `libraries.log` records actual linked SDK
objects, their hashes and ELF notes/build IDs. Version numbers alone are not
treated as proof that host-mounted driver and image runtime objects match.

The original SymMem error remains the job's exit code even if diagnostics
complete or fail. A zero debugger exit without the program's completion
marker cannot pass a control. These diagnostics collect evidence; adding them
does not establish that the CoreX crash has been fixed.

For a direct comparison in a prepared SDK environment:

```bash
make -C test/unittest/symmem ipc-repro USE_ILUVATAR=1
bash .github/scripts/ci/run_ipc_vmm_diagnostics.sh \
  test/unittest/symmem/build/bin/ipc_vmm_repro /tmp/ipc-vmm-diagnostics
```

Use `USE_NVIDIA=1` to build the same source against CUDA. GDB and GNU timeout
are needed for complete crash diagnostics. No new suite or extra required
GoogleTest case is added to the 19-case independent probe suite.

## Real two-host verification

The Actions job uses the current single-host CoreX runner with two logical
nodes (GPUs 0/1 and 4/5). This is not a physical two-host result.

On a separately prepared two-host environment, set
`FLAGCX_CI_PROBE_HOSTS=hostA,hostB` before running the entry point directly.
Both hosts must have the same binary/library paths, MPI remote launch access,
and a shared writable XML/log directory. Use `FLAGCX_CI_LOG_DIR` to select
that shared directory. Each host needs two accessible CoreX GPUs. The remote
probes use two ranks per host and log real hostnames. The local component
probes still run on the launching host. The container wrapper does not
configure remote hosts or automatically forward this setting.

## Interpretation and current validation limits

The build checks whether the SDK declares all required VMM entry points;
missing declarations leave VMM callbacks unavailable and required probes fail.
Runtime driver symbols are resolved individually and missing symbols report
`NotSupported`. The VA capability bit is a candidate route, not proof of
CoreX VMM peer-memory support. No DMA-BUF capability bit is added.

CPU mapping/ownership tests and CI orchestration tests can run without GPUs.
The provider and probe sources can be syntax-checked with CUDA-compatible
headers, but only compilation against the target CoreX SDK and the real probe
results establish CoreX compatibility. The regular SymMem VMM phases are now
enabled for CI verification; enabling them does not establish a passing CoreX
result. VMM/LSA and VA MR hardware behavior still require target CI results.
