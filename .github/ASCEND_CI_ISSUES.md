# Ascend CI Findings and Backend Work

Status is based on the PR review and local Ascend 910 container runs. The
available local host reports `Ascend910`; it is not a substitute for the PR's
`flagcicd-910c` runner. CI setup fixes below improve isolation and diagnostics;
they do not claim unsupported backend behavior as passing coverage.

| Finding | Evidence / current behavior | Required owner and work | CI coverage |
|---|---|---|---|
| HCCL async-error query / Runner | `FlagCXCollTest.AllGather` fails at `coll_allgather.cpp:23`: `synchronizeAndCheckAsyncError()` returns 9 (`flagcxNotSupported`). The query calls `flagcxCommGetAsyncError()`, which dispatches to `hcclAdaptorCommGetAsyncError()`; that callback unconditionally returns `flagcxNotSupported`. Ten isolated default collective cases failed, while `SkewedP2pTeardown` is intentionally skipped in default mode. This is not evidence that `HcclAllGather()` itself returned unsupported. | CANN/HCCL backend owner: implement the async-error query contract or explicitly design a capability-aware interface with the core owner. Do not replace the stub with unconditional success or remove the error check; validate each collective after the query is supported. | `runner` suite; Ascend runner wrapper reports each collective separately. |
| P2P APU-to-NIC route | Recorded CI/local logs show `Could not find any local path from apu 0/1 to net`; `DevicesReturnsPositive` returns selection error and P2P WRITE has failed. The test helper can explicitly select first visible devices for adaptor/P2P interface loopback checks; that does not establish production P2P proximity or make routed WRITE work. | Topology/CANN plus network backend owners: check DCMI BDF mapping for every logical device against `/sys/class/infiniband/*/device`, container visibility, PCI root-complex topology, and the FlagCX `APU -> CPU -> NIC` path. Do not assume NPU index matches NIC index. If no physical path exists, the platform/transport owner must provide a supported route or define P2P-over-RDMA unsupported. | `adaptor` and `p2p` use explicit first-local test selection for their local interface fixtures; the P2P WRITE integration test stays required and reports route failures. Verify on the 910C runner. |
| SymMem proxy SIGSEGV; MPI WRITE/RMA need separate confirmation | With a freshly rebuilt library containing the CANN PCI callback, ordinary-allocation NET SymMem exits 139. GDB reproduces `flagcxProxyService()` at `flagcx/core/proxy.cc:1976`, dereferencing `comm->proxyState->stop`; see the trace below. Both two-rank registration and four-rank logical-node NET reproduction hit this line. | Core proxy/communicator lifecycle owner: inspect service-thread shutdown, communicator/proxy ownership and failed-init cleanup. The crash point is confirmed; null/dangling state ownership is not yet proved. Fix the invalid access and rerun SymMem, then separately reproduce MPI WRITE/RMA; do not attribute all historical crashes to this stack. | Failure-only `run_ascend_diagnostics.sh` attempts separate P2P, RMA, or SymMem MPI reproductions; traces are stored with attempt logs. |
| CANN IPC | `cannAdaptorIpcMemHandleCreate/Get/Open/Close/Free` in `flagcx/adaptor/device/cann_adaptor.cc` return `flagcxNotSupported`; the historical result was 0 passed / 16 skipped. | CANN device-backend owner: implement the supported inter-process device-memory export/import lifecycle (or an explicitly supported equivalent), including ownership, error cleanup, and peer accessibility. If the driver has no such primitive, state the limitation and keep those cases unsupported. | IPC-local SymMem/RMA/adaptor invocations report their own status; they cannot provide coverage until the backend implements the contract. |
| GPU MR and NET registration | Historical adaptor results skipped mandatory GPU MR cases; recorded P2P topology shows no APU-to-NIC path. CANN's absence of CUDA VMM callbacks is a separate limitation. | CANN + RDMA provider owners: validate whether APU allocations can be registered/pinned for the chosen RNIC, provide/import the MR, and support safe deregistration. Do not count host MR or skipped tests as GPU MR support. | Existing GPU coverage gate remains active; NET SymMem tests require `FLAGCX_CI_REQUIRE_NET_MR=1` and explicit IB selection. |
| SymMem VMM/MR routes | CANN Symmetric VMM callbacks (`cannAdaptorSymPhysAlloc/Free`, flat map/unmap and multicast functions) are stubs returning `flagcxNotSupported`. Local CI can exercise ordinary-allocation IPC and NET/MR separately, but VMM routes are unavailable. | CANN backend owner: implement physical symmetric allocation, cross-rank mapping and teardown, plus network-MR routing for VMM-backed allocations; otherwise keep VMM cases explicitly unsupported. | Ascend override separates unit, IPC, local NET fallback, and ordinary NET/MR tests, records each status, and emits this limitation. VMM is not reported as covered. |
| RMA visibility | `cannAdaptorStreamWaitValue64` and `cannAdaptorStreamWriteValue64` return `flagcxNotSupported`; the device-consumer visibility kernel is not built for Ascend. | CANN backend owner: implement acquire/release remote-write visibility semantics with correct ordering and flushing, and add an Ascend device-consumer kernel. Then add capability-aware ordinary and supported-MR route tests. | Current CI runs RMA unit, IPC, and network correctness independently and records the visibility gap. It does not claim CUDA parity for visibility. |
| Perf and Torch API jobs | Ascend jobs are wired in `.github/workflows/test.yml` and `torch-api-test.yml`; job wiring is not evidence the hardware workloads passed. | CI maintainers: authorize the PR run and confirm the 910C runner, image, PyTorch/Torch-NPU, and benchmark topology. | The current `d27fe8d` run was `action_required` with zero jobs. The logged `canghaiX` account had no push permission on upstream `flagos-ai/FlagCX`, so a repository maintainer must approve/enable the run. |

## Local Verification

- The configured debug image pulled successfully and builds for the Ascend
  library plus P2P, SymMem, and Runner suites completed.
- Ascend `AllGather` reproduced twice in isolated 4-rank runs. It failed with
  result 9 (`flagcxNotSupported`) at `synchronizeAndCheckAsyncError()`; the
  HCCL async-query callback is an unimplemented stub, as shown above.
- With `FLAGCX_CI_FORCE_FIRST_LOCAL_DEVICE=1`, the P2P
  `DevicesReturnsPositive` and two P2P loopback cases passed. Without the
  switch, `DevicesReturnsPositive` still fails with selection result 1. This
  proves the local test fallback is effective; the underlying PCI path is not
  fixed.
- An adaptor host-MR loopback case passed with first-device selection. This
  does not validate CANN IPC or GPU MR support.
- RMA unit tests: **54 passed**. SymMem unit tests: **66 passed**.
  Final ordinary-allocation SymMem results: `unit=0 IPC=1 fallback=139 NET=139`.
  Both the local NET fallback and logical-node NET/MR paths crashed. These are
  independent phases, so the IPC failure no longer prevents NET diagnostics.
  Shell syntax, formatting and nine targeted CI regression checks passed;
  the full routing regression requires Ruby, which is absent on this host.
- Full GDB logs are saved locally under
  `/tmp/flagcx-ascend-review-evidence/ascend-symmem-backtrace.NTqYJM/`.
  CI stores the equivalent per-rank files inside the existing attempts artifact.
- This host identifies its device as Ascend910, not Ascend 910C. Therefore
  local results do not replace the required `flagcicd-910c` CI run.

## Confirmed SymMem Crash (2026-10-30)

Image: `harbor.baai.ac.cn/flagos-dev/flagcx:ascend-dev-fixed-20261028-debug`,
digest `sha256:1e4db1717af5caf241c80ba0d756978a7ba787be7114526b4b1278662c5e89ee`.
The library was rebuilt with `make -B -j8 USE_ASCEND=1`; its exported symbols
include `cannAdaptorGetDevicePciBusId`. This trace is from the rebuilt library,
rather than a stale library without the PCI callback.

Reproduce the two-rank case inside the configured Ascend container:

```bash
source .github/scripts/set_env/ascend.sh
export LD_LIBRARY_PATH="$PWD/build/lib:$LD_LIBRARY_PATH"
mpirun --allow-run-as-root -np 2 \
  -x LD_LIBRARY_PATH -x FLAGCX_USE_HETERO_COMM=1 \
  -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 \
  test/unittest/symmem/build/bin/symmem_mpi_tests \
  --gtest_filter=SymMemTest.RegisterSymmetricWindow
```

The diagnostic script also reproduces the four-rank NET case with logical
`FLAGCX_HOSTID=node0/node1`, P2P disabled and required IB MR, using
`SymMemTest.FullMeshRoundFailureConvergesAndRetries`. These are two logical
nodes on one host, not physical multi-host validation.

```text
mpirun noticed that process rank 0 ... exited on signal 11 (Segmentation fault).
Ascend symmem NET MR and fallback attempt 2 failed with exit=139

Thread 19 "symmem_mpi_test" received signal SIGSEGV, Segmentation fault.
#0 flagcxProxyService(args=0xaaaaaf2c7390) at flagcx/core/proxy.cc:1976
   if (!stop && __atomic_load_n(&comm->proxyState->stop, __ATOMIC_ACQUIRE)) {
#1 start_thread(...) at ./nptl/pthread_create.c:447
#2 thread_start() at ../sysdeps/unix/sysv/linux/aarch64/clone3.S:76

Main thread at the time of the crash:
#0 __futex_abstimed_wait_common64(...) at ./nptl/futex-internal.c:57
#1 __futex_abstimed_wait_common(...) at ./nptl/futex-internal.c:87
#2 __GI___futex_abstimed_wait_cancelable64(...) at ./nptl/futex-internal.c:139
#3 __pthread_clockjoin_ex(...) at ./nptl/pthread_join_common.c:102
#4 mmJoinTask() from libmmpa.so
#5 ??() from libruntime_v100.so
#6 __run_exit_handlers(status=1, ...) at ./stdlib/exit.c:108
#7 __GI_exit(...) at ./stdlib/exit.c:138
#8 __libc_start_call_main(...) at ../sysdeps/nptl/libc_start_call_main.h:74
#9 __libc_start_main_impl(...) at ../csu/libc-start.c:360
#10 _start()
```

The captured main thread was already exiting with status 1. This supports
investigating failure/exit cleanup while the proxy thread is still alive; it
is not sufficient to distinguish a null state from use-after-free. Preserve
all-thread backtraces and inspect communicator ownership before adding a guard.
Historical MPI WRITE and RMA failures still require their own stacks.

## Workflow Authorization

At inspection, PR head `d27fe8d` had run `41019034435` in `action_required`
with zero jobs. The logged account cannot approve upstream runs. A repository
maintainer must authorize that run or a new run before the CI changes can be
validated on `flagcicd-910c`. No CUDA extended-perf or extra Device API jobs
were added as part of this work.
