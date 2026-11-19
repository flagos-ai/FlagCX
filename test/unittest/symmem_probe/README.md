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
| `MrFailureAndRetry`, `DeregistrationFailureAndRetry` | `RankLocalMrFailureDoesNotPublishPartialWindow`, `MrRollbackFailureRetainsWindowLeaseUntilRetry`, `CrossNodeCleanupFailureConvergesBeforeRelease` |
| `MultiSegmentSubrange` and the registration controls | Local segmented geometry tests and multi-segment registration/failure-convergence tests; not an unconditional requirement that one MR span all physical segments |

The Iluvatar configuration enables the same SymMem phases as CUDA. Its
VMM+NET auto invocation requires VA routing, and route union exercises the
VA route only and requires successful registration. CUDA retains its existing
DMA-BUF/VA route union. With the default allocation settings, the route-union
case uses one physical segment.
`CrossNodeMixedMultiSegmentVmmConvergesMrFailure` explicitly accepts a converged
registration failure and validates cleanup, while
`SegmentedMapFailureDoesNotPublishNetworkOnlyWindow` validates failure rollback.
A successful whole-range multi-segment MR is therefore not an unconditional
requirement of the existing SymMem suite. The controls diagnose this additional
capability without adding a segmented MR metadata/data-path implementation.

IXCUDA descriptions borrow original create handles, matching CUDA. The source
allocation stays alive until its mappings and bundles have been released;
describing an allocation does not create an export/import reference cycle.

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
