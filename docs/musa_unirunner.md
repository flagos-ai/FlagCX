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

## Experimental CI

The performance jobs are named `perf-test (musa)` and
`perf-test-experimental (musa)`. Both use the single platform configuration
`.github/configs/mthreads.yml`, the shared `set_env/mthreads.sh` environment,
and `run_mthreads_workload.sh`. The internal platform key is `mthreads`;
`platform:musa` and `/test musa` are existing aliases that select that same
configuration. The runner label is `mt-8g-cicd-flagcx`.

The existing image `mthreads-dev-fixed-20261022` contains MUSA 4.3.6, MCCL
2.1.6 and MPI. Upstream records that this SDK lacks `musa::atomic_ref` and
keeps the legacy Device API/IR suites disabled. Experimental kernels require
an SDK/image that implements this API. The native preflight compiles and uses
`musa::atomic_ref` before building the library; unsupported SDKs fail the job
rather than report a skipped experimental run as successful. Keep the
platform configuration shared when updating the SDK image.

Run the same workload inside the configured MUSA environment:

```bash
export GITHUB_WORKSPACE="$PWD"
export FLAGCX_CI_SET_ENV=.github/scripts/set_env/mthreads.sh
bash .github/scripts/ci/run_hardware_workload.sh \
  mthreads perf-test-experimental "$FLAGCX_CI_SET_ENV" ''
```

The workload first builds and runs `test/kernel/musa/unirunner_probe.mu` on
four visible GPUs. This checks cooperative launch, occupancy, P2P capability,
IPC allocation export, native system-scope atomics, and an actual stream
wait/write64 sequence. The
subsequent collective tests exercise cross-process IPC imports and rebinding.

The four-rank suite forces UniRunner, disables custom operations, and asserts
the expected kernel, stream-value and local P2P capabilities. It covers:

- IpcRingAR, SlicedAR, uneven-count RingAG and GroupedAG.
- Eight reduction data types and five operations, plus Reduce/ReduceScatter.
- AllGatherGemm/GemmReduceScatter and rank-local error convergence.
- IPC buffer rebinding, outer-group behavior, unsupported reduction types,
  repeated calls, alternating streams, and in-place/out-of-place AllReduce.

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
