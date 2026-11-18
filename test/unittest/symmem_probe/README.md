# Iluvatar SymMem probes

This diagnostic suite is independent of `adaptor`, `rma`, and `symmem`.
The existing SymMem CI VMM gate remains enabled until hardware results justify
restoring those tests. DMA-BUF is explicitly disabled; POSIX VMM allocation FDs
are still required for local peer mappings.

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
| `MpiVmmProbe.LocalFdPeerMapping` | Two GPU processes, actual FD exchange and peer data, no FlagCX communicator |
| `MpiVmmProbe.LocalArenaWindowReuse` | Communicator arena, concurrent windows, released offset reuse |
| `MpiVmmProbe.MapFailureAndRetry` | Rank-local failure after actual mapping and successful retry |
| `MpiVmmProbe.RemoteWithoutNetRollsBack` | Expected remote registration rejection without NET |
| `MpiVmmProbe.WindowVaGetPut` | Public VMM window, actual VA route, remote GET/PUT payloads |
| `MpiVmmProbe.MrFailureAndRetry` | Rank-local VA MR failure before metadata publication |
| `MpiVmmProbe.DeregistrationFailureAndRetry` | Failed unmap retains the window, then cleanup retries |

The component suite uses topology-local `mlx5` selection through the existing
net adaptor helper. A successful registration alone is not a data-path pass.
The same MR is exercised by READ and WRITE before deregistration. No extra
`deviceSynchronize` is added after RMA completion to hide payload visibility
problems.

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
results establish CoreX compatibility. Until those results exist, VMM/LSA and
VA MR hardware behavior remain unverified and regular SymMem CI stays gated.
