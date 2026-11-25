# FlagCX CI

The hardware CI matrix enables nine platforms: CUDA, Hygon, Iluvatar,
MetaX, KunlunXin, Mthreads, PPU, Ascend, and Enflame. PR runs select platforms from changed paths
and PR labels; push and scheduled runs select all enabled platforms. Unit,
performance, and Torch API jobs run for the selected platforms alongside
common checks. See `configs/platforms.yml` for the active registry.

Platform-specific test exclusions are defined under `ignore_test` in each
platform's `configs/*.yml` file. The same configuration drives test filtering
and the plan Summary, where exclusions remain visible as skipped coverage.

`unit-test.yml` is the automatic controller for PRs to `main`/`devops`, pushes
to those branches, and the daily schedule. It runs the selected platforms'
unit, performance, and Torch API tests, plus common checks. Platform-specific
changes use `paths_watch`; any shared or unknown changed path selects every
enabled platform. Renames include both their old and new paths.

PR labels `platform:cuda`, `platform:hygon`, `platform:iluvatar`,
`platform:metax`, `platform:kunlunxin`, `platform:mthreads`, `platform:ppu`,
`platform:ascend`, and `platform:enflame`
override path selection. Multiple platform labels select their union;
`platform:musa` is an alias for `platform:mthreads`; `musa` also works in
workflow dispatch selection and `/test musa` commands.
`platform:all` selects every enabled platform. Invalid platform labels fail
matrix generation. Add/remove label events rerun the controller.

Maintainers with `write`, `maintain`, or `admin` repository permission can post
one of these exact commands on a PR:

```text
/test
/test cuda
/test hygon
/test iluvatar
/test metax
/test kunlunxin
/test ppu
/test ascend
/test enflame
/test all
```

`/test` uses the PR's labels and changed paths. An explicit platform overrides
that selection. Manual dispatch accepts `all` or comma-separated platform names.
Each request tests the resolved PR head repository and immutable commit SHA.
Comment requests have separate concurrency groups and cannot cancel PR checks.
GitHub only enables `issue_comment` workflows present on the default branch;
comment-triggered runs belong to that branch's check suite, so the automatic
PR run remains the merge gate.

## Test Levels

PRs default to L2. Add a `level:L0` through `level:L3` label to choose a level;
when several level labels are present, the highest level wins. Manual dispatch
accepts the same level input. Pushes and the nightly schedule always use L3.

| Level | Workloads | Typical use |
| --- | --- | --- |
| L0 | Common checks and CI plan only | Workflow/configuration-only validation |
| L1 | Selected platforms' unit suites | Fast PR feedback |
| L2 | Unit, performance, and Torch API tests on selected platforms | Default PR coverage |
| L3 | Same test groups as L2 on all enabled platforms | Push and nightly coverage |

The prepare job writes the selected platforms, level, configured suites,
`ignore_test` entries, routing reasons, and blocked-work policy to the GitHub
Summary and archives the machine-readable plan. An unmapped changed path selects
the full platform matrix. If a workload is unavailable, it should be recorded
as blocked while other runnable jobs continue; an unknown path or a partial
plan does not stop test execution. Actual selected test failures still fail
`ci-complete`.

## Required Check

After the first successful run, configure the repository's branch protection
or ruleset to require only **ci-complete** for these test workflows. YAML adds
the check but cannot change repository protection settings. Packaging workflows
keep their existing independent triggers.

The gate fails on matrix/configuration errors, failed or cancelled tests, or
unexpected skips of selected platforms. Empty selections skip hardware jobs
while still running common checks.

Ascend backend limitations, required follow-up, and local verification results
are tracked in [ASCEND_CI_ISSUES.md](ASCEND_CI_ISSUES.md). Ascend uses standard
unit, perf, and Torch API jobs. Its runner launches collectives in separate MPI
processes so one asynchronous failure cannot skip subsequent collectives.
Ordinary SymMem IPC and NET/MR invocations record independent statuses; VMM and
RMA visibility coverage still require the backend contracts listed there.
The first-device selector is restricted to local adaptor/P2P test fixtures;
it does not replace production topology selection or validate APU/NIC proximity.
P2P/RMA/SymMem failures containing SIGSEGV trigger bounded, per-rank GDB
reproduction stored with the attempt artifacts, preserving the suite failure.

Enflame runner tests retain the full host unit suite and the default MPI
collectives except the known S60 Gather/Scatter failures. The platform's
`FLAGCX_CI_RUNNER_MPI_FILTER` is applied alongside existing GTest exclusions.
`FLAGCX_CI_SKIP_RUNNER_HYBRID=1` skips the heterogeneous P2P invocation that
hangs in ECCL. When Enflame preflight reports `FLAGCX_CI_HAS_RDMA=0`, forced
IB NET, multi-QP striping, and PXN invocations are skipped with explicit reasons;
they run when RDMA is available (and PXN is enabled). Host/default MPI failures
still fail the job. Without an active RDMA HCA, Enflame also skips adaptor
IB/GDR loopback cases, the P2P `DevicesReturnsPositive` device probe and WRITE MPI, RMA network/GDR visibility,
and SymMem's local-IPC-to-NET fallback, cross-node IPC+NET, and VMM+NET route
union phases. The Socket fallback cannot register the network MRs these tests
require. Adaptor non-network tests, P2P/RMA unit tests and RMA IPC tests, SymMem
unit/IPC-local tests, and runner default tests remain enabled. P2P, RMA and SymMem
phase exclusions emit workflow warnings and summary entries. With an active
RDMA HCA these tests run normally. Other platforms retain their required RDMA
preflight.

## Retry Logs

KunlunXin job containers retain the Actions-managed bridge for checkout and
artifact steps. RDMA hardware workloads use `run_in_host_network.sh` to enter
the Docker host's network namespace, using the read-only
`/proc:/run/flagcx-host-proc:ro` job-container mount. Only networking changes;
the workload keeps the container filesystem, SDK, environment and working
directory. If the container uses Docker's embedded bridge DNS, a private mount
namespace provides the host resolver for this workload; checkout and artifact
steps retain their original DNS settings. Unit, perf and Torch jobs use the
same entry point and do not launch another container. The image needs util-linux
(`nsenter`, `unshare`, `mount`) and the existing privileged container option.

The entry point verifies the namespace identity and checks that every ACTIVE
mlx5 RoCE port has a GID-associated interface visible through the socket API
before starting the workload. It fails instead of skipping RDMA tests when
the mount, permissions or interfaces are missing. On DinD deployments the
mounted namespace must expose the physical RDMA interfaces; an isolated DinD
network still needs runner-side network provisioning. This check establishes
visibility only; the adaptor Host MR and SendRecv tests verify connectivity.

Mthreads CI uses `mt-8g-cicd-flagcx` and the image configured in
`.github/configs/mthreads.yml`. It runs adaptor, core, device_api_host,
p2p, rma, runner, service, and symmem host unit suites plus
homogeneous and heterogeneous collective benchmarks. The symmem job currently
runs host unit tests only; it does not cover IPC, VMM, or multi-process paths.
The device adaptor suite excludes `DeviceAdaptorTest.StreamWaitEvent`, which
currently fails on MUSA; all other device adaptor tests run.
Device API and unified IR suites are selected but report capability skips because
this repository has no MUSA device-kernel compiler backend. Device adaptor and
topology cases run through the adaptor and core suites respectively. Torch plugin tests are also not
configured for Mthreads. Build its CI image with
`docker build -f docker/mthreads-ci.Dockerfile .`.

CI runs each build and test once by default. Individual commands are bounded
by phase limits in `.github/scripts/ci/timeouts.sh`, and their timeout is capped
by the remaining suite, mode, and workload budgets. Normal unit jobs have a
45-minute job limit (60 minutes with coverage); perf and Torch jobs have a
60-minute limit. Workloads stop earlier to leave time for cleanup and artifacts.
Image pulls retain their existing three attempts.

A manual invocation may set `FLAGCX_CI_TEST_MAX_ATTEMPTS=2` to retry a test once.
Cancellation and missing/non-executable commands do not retry. Recovered tests
emit a flaky warning and Step Summary entry. The `attempts-*` artifacts preserve
invocation logs for seven days, including tests in managed Docker containers.

## Python Coverage

The CUDA, Hygon, MetaX, and PPU Torch jobs run `plugin/torch/tests/test_*.py`
and measure Python code in `src/flagcx` and the Torch build helpers. The suite
currently contains 17 tests. It does not measure C++ or MPI code, or the Torch
API example that runs before coverage starts. Each platform produces a separate
`coverage-<platform>-gpu-torch.json` report and artifact. Coverage runs inside
the configured hardware container, using the built plugin. The report is
persisted through the attempt-log mount and copied to the runner for upload.
Python test failure still fails CI, even when a partial coverage report is generated.

Repository configuration:

- Secret `FLAGCICD_API_TOKEN`: bearer token for the metrics endpoint.
- Endpoint: `https://flagcicd.flagos.net/flagcicd/api/v1/cicd/metrics/`.

The token is explicitly forwarded only for source in the same repository.
Fork PRs, including comment-triggered runs, generate/archive reports but skip
FlagCICD uploads. Missing reports or tokens also skip uploading. Upload errors
and the two-minute upload timeout do not change test conclusions.

The upload action is pinned to verified commit
`5bd0e138a6ae482b1a1cc809ea6705ecfd6eb410`, which removes `user_id` and requires
`api_token`.

Verify an authorized manual run on a runner with access to FlagCICD: the upload
log must show HTTP 2xx and the FlagCICD UI must display that run's Python
coverage. HTTP 401/403 indicates a token issue; curl exit 6 indicates DNS
failure; exit 7/28 indicates connectivity or timeout. Green CI alone does not
prove upload success.

## C++ Coverage

`cpp-coverage.yml` runs the full platform unit-test suite on Saturday UTC and
can be dispatched for selected platforms. It builds FlagCX host sources with
GCC gcov instrumentation and archives per-platform/suite JSON and HTML reports
with line, function, and branch metrics. The matrix summary keeps platforms
separate; the cross-platform denominator is intentionally unset until the
coverage policy is approved. Coverage is informational and does not add a 90%
merge gate.

Reports list production files that did not produce gcov data. This includes
device-code sources until their vendor toolchains have a verified collection
backend; skipped suites include an explicit reason rather than a fabricated
zero-percent result. New supplemental GoogleTest cases use the
`CoverageSupplemental.` suite-name prefix: ordinary PR jobs exclude these
cases, while the weekly coverage workflow runs them.
