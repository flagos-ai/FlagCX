# MUSA UniRunner kernels

MUSA uses the public Device API's DefaultBackend and the common UniRunner
executor. Build the native device code with:

```bash
make -j USE_MUSA=1 COMPILE_KERNEL=1
make -C test/perf host_api -j USE_MUSA=1 COMPILE_KERNEL=1
```

`DEVICE_HOME` and `CCL_HOME` default to `/usr/local/musa`. The default target
architecture is `MUSA_ARCH=mp_31`. The library retains compiler-managed kernel
image registration and the MUSA fatbin export map.

## Supported execution paths

The port enables dynamic RED/IPC, static RED, combined RED/IPC and Ring DAG
execution. These implement the existing UniRunner AllReduce algorithms,
Reduce, ReduceScatter, AllGather, and float32 AllGatherGemm/GemmReduceScatter.
Algorithm selectors and DAG formats match the other UniRunner platforms.

Reduction supports int8, uint8, int32, uint32, int64, uint64, float32 and
float64, with Sum, Prod, Min, Max and Avg. Float16 and bfloat16 reductions
return `flagcxNotSupported` before publishing work; this executor does not
instantiate MUSA half/bfloat16 reduction specializations. Byte-copy operations
such as AllGather do not have that reduction restriction.

Static execution checks cooperative-launch support, concurrent-kernel support
and per-kernel occupancy. It retains the common residency budget, including
the SM reserved for communication progress. The runtime also probes native
64-bit stream memory operations. Unsupported capabilities are not overridden.
The existing MUSA system release fences and proxy completion policy retain
their Device API defaults.

The launch adaptation follows the native
[MUSA Runtime API](https://docs.mthreads.com/musa-sdk/version-5.1.0/libraries/core_api/runtime_api_reference/)
for cooperative launch, occupancy and kernel launch. Native compilation and
execution remain necessary to validate the installed SDK and driver.

## Device API CI

Both `device_api` and `device_api_unified_ir` build the library and tests with
`USE_MUSA=1 COMPILE_KERNEL=1` in the existing shared Mthreads image. The
`device_api_host` suite retains its host-only configuration. The native suites
print the installed mcc version and fail on build errors instead of returning
a successful capability skip.

The shared unit runner executes all six public programs: the regular Device
API and legacy IR intra/inter programs, plus the unified IR intra/inter
programs. The MUSA configuration enables local P2P with VMM disabled; unified
IR additionally runs the existing forced-NET intra/inter phases. Full sweeps
cover 1 KiB–16 MiB with factor 2; forced-NET sweeps cover 1–2 KiB.

`run_device_api_checked.py` delegates each invocation to the shared MPI timeout
and diagnostic wrapper, then checks its output. K10/K14, legacy IR S10/S14,
and unified IR S16–S25 must pass on every rank for every size. Missing, skipped,
duplicate or failed required results, incomplete size sweeps and missing
summaries fail the invocation. Other reported case failures also fail it.
Each acceptance round saves a transcript and JSON report under the CI log
directory and must pass without an automatic retry.

Inside the configured MUSA container, run the normal four-GPU configuration:

```bash
export GITHUB_WORKSPACE="$PWD"
for suite in device_api device_api_unified_ir; do
  bash .github/scripts/ci/run_hardware_workload.sh mthreads unittest \
    .github/scripts/set_env/mthreads.sh "$suite"
done
```

CI uses exactly four visible GPUs: four intra ranks and two per logical node,
matching the other platforms. Both native Device API suites are required to
execute. Native SDK/compiler failures,
missing required test results and unexpected skips fail their jobs.

The MUSA host suite requires both communicator isolation cases to pass in its
GoogleTest XML: Finalize must complete without draining independent work on a
second communicator. Native MCCL Finalize retains the communicator; rank-local
Destroy releases it only after completed finalization, preserving ownership
when cleanup needs a retry.

## Experimental CI

The MUSA UniRunner experimental job is temporarily disabled after the
four-rank `AllGatherGemm` stall in PR #738, run 45775159982. Ordinary perf,
Torch and Device API CI remain enabled. The workload and kernel implementation
are retained for diagnosis; restoring CI also requires restoring its
required-success completion gate.

The performance jobs are named `perf-test (musa)` and
`perf-test-experimental (musa)`. Both use the single platform configuration
`.github/configs/mthreads.yml`, the shared `set_env/mthreads.sh` environment,
and `run_mthreads_workload.sh`. The internal platform key is `mthreads`;
`platform:musa` and `/test musa` are existing aliases that select that same
configuration. The runner label is `mt-8g-cicd-flagcx`.

The shared CI image is the pinned FlagOS MUSA 5.2.0 runtime (tag 2.3.0).
Its SDK provides native `musa::atomic_ref`, `mcclCommFinalize`, and asynchronous
completion queries. The CI Dockerfile uses the same pinned runtime and adds
build/diagnostic dependencies for local reproduction.

The container starts with `MTHREADS_VISIBLE_DEVICES=0,1,2,3` and
`MTHREADS_DRIVER_CAPABILITIES=compute,utility` so the Mthreads runtime can
inject the host MTML management library. Workloads that initialize native
MCCL load `libmtml.so` before building; a missing library or dependency fails
with loader diagnostics. The host must have a compatible MTML installation
and a runtime utility mapping that includes the library.

Run the same workload inside the configured MUSA environment:

```bash
export GITHUB_WORKSPACE="$PWD"
export FLAGCX_CI_SET_ENV=.github/scripts/set_env/mthreads.sh
bash .github/scripts/ci/run_hardware_workload.sh \
  mthreads perf-test-experimental "$FLAGCX_CI_SET_ENV" ''
```

The workload builds the library and experimental test binaries, then runs
the four-rank collective tests, including cross-process IPC imports and
rebinding.

The four-rank suite forces UniRunner, disables custom operations, and asserts
the expected kernel, stream-value and local P2P capabilities. It covers:

- IpcRingAR, SlicedAR, uneven-count RingAG and GroupedAG.
- Eight reduction data types and five operations, plus Reduce/ReduceScatter.
- AllGatherGemm/GemmReduceScatter and rank-local error convergence.
- IPC buffer rebinding, outer-group behavior, unsupported reduction types,
  repeated calls, alternating streams, and in-place/out-of-place AllReduce.
- IPC Finalize followed by rank-local Destroy, and finalizing grouped
  communicators whose work runs on another communicator's stream.

Perf invocations enable numerical verification. Required GTests must execute
and pass in every rank's XML report; missing or skipped cases fail the job.
The SlicedAR sweep defaults to 64 KiB–64 MiB and can be adjusted through
`FLAGCX_CI_MUSA_SLICED_AR_MAX_BYTES` when running the workload directly.
Performance numbers establish a baseline; CI has no bandwidth threshold.

The platform also participates in ordinary perf, the existing Torch API
suite, and its configured unit suites. Selected MUSA jobs must succeed in the
workflow completion gate.

## Local checks without the SDK

```bash
python3 -m unittest discover -s test/script -p 'test_musa_*.py'
```

These checks compile real reduction dispatch and host launcher functions with
SDK shims, exercise the experimental script with mock builds/launchers, and
validate workflow gates. They do not establish mcc compatibility, device
memory ordering, hardware correctness or performance. Run the native job and
existing MUSA Device API/IR suites before treating the port as hardware-verified.
