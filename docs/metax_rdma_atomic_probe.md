# MetaX GPU RDMA allocation probe

The RDMA atomic probe is disabled in automatic MetaX CI. Regular MetaX unit
suites are enabled in `.github/configs/metax.yml`; performance and Torch API
hardware jobs follow the normal CI level and platform selection. Existing
platform exclusions remain in effect. Probe code remains available for
manual diagnosis; add `rdma_atomic_probe` to the configured
suite list to enable it in automatic CI again.

The shared MetaX CI environment sets `MACA_VISIBLE_DEVICES=0,2,4,6` inside
the test container before runtime initialization. Probe MPI workers explicitly
inherit this selection. Rank device indices remain logical indices 0–3 in
that visible set; the logged PCI bus IDs identify the GPUs actually used.
The GPU4/NIC2 and GPU6/NIC3 affinities are hypotheses awaiting topology
confirmation. This selection also applies to the other MetaX workloads.
The runner PXN case independently selects
`FLAGCX_CI_PXN_VISIBLE_DEVICES=0,1,4,5` and
`FLAGCX_CI_PXN_HCAS=bnxt_roce0,bnxt_roce2`. Its two logical nodes contain
GPUs [0,1] and [4,5]; SendRecv pairs GPU0 with GPU4 and GPU1 with GPU5.
The intended sender-side relay paths are GPU1 -> GPU0 -> bnxt_roce0 and
GPU5 -> GPU4 -> bnxt_roce2. These require eligible CCI paths between the
GPU pairs and NIC-local relay GPUs, which must be confirmed at runtime.

`rdma_atomic_probe` is an independent MetaX hardware CI suite for diagnosing
the signal-only FetchAdd failure in PR #737. It does not change the production
allocator, enable HOST SIGNAL, call stream wait/write-value operations, or
replace the Device API and RMA suites. All target buffers come from the MACA
allocation APIs or the actual MetaX GDR allocator. Pointer attributes are
recorded rather than assuming all allocation flags have identical properties.

## Allocation coverage

| Mode | Allocation call |
|---|---|
| `mcMalloc` | `mcMalloc` |
| `mcDeviceMallocDefault` | `mcExtMallocWithFlags` |
| `mcDeviceMallocFinegrained` | `mcExtMallocWithFlags` |
| `mcMallocSignalMemory` | `mcExtMallocWithFlags` |
| `mcDeviceMallocWriteCoherence` | `mcExtMallocWithFlags` |
| `mcDeviceMallocMapPcieDefault` | `mcExtMallocWithFlags` |
| `mcDeviceMallocMapPcieCoherence` | `mcExtMallocWithFlags` |
| `mcDeviceMallocFixedMemDefault` | `mcExtMallocWithFlags` |
| `mcDeviceMallocFixedMemCoherence` | `mcExtMallocWithFlags` |
| `gdrMemAlloc` | `macaAdaptor.gdrMemAlloc` with `FLAGCX_VMM_ENABLE=0` |

The GDR case exercises the real interface, including its current finegrained
allocation and `SyncMemops` attribute setting. It is not an alias for the
native finegrained test. Each mode has separate 8-byte and OS-page-size
allocation/copy-round-trip checks on two distinct GPUs. SDK enum availability
is checked during compilation; missing flags remain visible as
`SDK_UNAVAILABLE`, with compiler diagnostics, rather than disappearing from
the matrix. Raw enum values, allocation return codes, pointer attributes and
subsequent stage failures are retained.

## Layers and matrix

The transports use separate executables: `rdma_atomic_native_probe` includes
the system verbs headers and calls libibverbs directly;
`rdma_atomic_adaptor_probe` uses FlagCX's compatible headers and
`flagcxWrapIbv*` queries. Their shared interface contains no verbs types. A
separate bridge calls the real GDR allocator without passing verbs objects
between the two implementations.

1. **Native verbs:** directly establish RC QPs and register the selected GPU
   allocation. Test 8-byte RDMA WRITE, RDMA READ and FetchAdd. Validate both
   the final GPU word and the native atomic return values, including the
   uniqueness of old values under concurrent senders. The initial 64-bit
   value crosses a low-word carry boundary during the default increments.
2. **FlagCX net adaptor:** use the actual IBRC `regMr`, MR metadata exchange,
   lane selection, `iput`, `iget`, and signal-only `iputSignal(size=0)` paths.
   This includes the production MR range handling/cache and atomic return
   scratchpad. The adaptor does not expose FetchAdd old values to its caller;
   its probe validates the target's cumulative result.
3. **Device API barrier:** use the existing Unified IR inter binary with four
   ranks, two logical nodes and four CTAs. A MetaX-only diagnostic environment
   variable stops it after S16 (sync and split barriers), retaining normal
   setup and teardown. Normal Unified IR invocations still execute S16–S25.

For every allocation mode, both transports run READ/WRITE/FetchAdd in both
directions, at both allocation sizes, on the first active HCA/port. Every
additional active HCA/port gets page-sized FetchAdd in both directions for
every mode and transport. The `mcMalloc`, finegrained and GDR paths additionally
vary one setting at a time: offset 8, atomic depth 1 (baseline 16), relaxed
ordering, four QPs, and native `ibv_reg_mr_iova2` (baseline `ibv_reg_mr`).
Every allocation also gets automatic multi-HCA selection and four-rank,
three-sender FetchAdd cases targeting GPUs 2 and 3 (the basic cases target
GPUs 0 and 1). The barrier runs with the first HCA and automatic
HCA selection. The latter lets FlagCX select its actual lanes; native automatic
selection assigns active ports by rank. Logs identify these choices, and the
two are not assumed to produce identical physical routes.

The native worker posts at most one WR per QP per wave, so the depth-1 control
does not deliberately overflow the QP. Adaptor independent ordering domains
address each actual lane, including lanes created for merged NICs. Before
posting, logs record local/remote address and key, QP, PD, lane and device
indices, and atomic return buffer metadata. QP access/depth and HCA atomic
capability are queried, with query return codes. Firmware, SDK/runtime/driver
versions and loaded runtime/provider library paths are also recorded. Error
CQEs report only their guaranteed-valid fields, not inferred opcode/length.

## Isolation, budget and results

Each case has fresh MPI workers, GPU contexts, QPs and registrations. GPU
workers are confined to `localhost`; logical nodes use separate FlagCX host
IDs. An error aborts that case, not the matrix. The supervisor has an external
deadline, terminates the MPI process group and marked descendants (including
reparented workers on Linux), and bounds reaping. Core dumps are disabled.
Internal CQ/connection loops have independent deadlines.

The supervisor independently builds SDK inventory, native, adaptor and barrier
workers, retaining each command, exit status and log. A failed build marks that
worker unavailable; other workers still build and run. Allocation checks prefer
the native worker and use the adaptor worker if needed. A stale executable
left by a failed build is never used. Partial build failure still fails the
overall suite even when independent coverage can complete.

Host inventory is a separate Python process reading sysfs, without loading GPU
or verbs libraries. It saves each discovered port immediately; even this process
has a deadline, because a driver sysfs read can stall. Host provenance is saved
before GPU initialization. An unreadable port remains an explicit inventory
error; successfully tested ports do not hide the missing coverage.

GPU count first uses `rdma_atomic_sdk_inventory`, directly linked only to the
necessary MACA/system libraries, without FlagCX, MPI or libibverbs linkage.
This control calls only `mcGetDeviceCount`. If a control fails and its process
has been reaped, the supervisor tries native/adaptor `--inventory` controls.
They also query only the actual GPU count. A failed startup control remains in
the report and fails CI even if a fallback supplies a valid count and allows
the allocation matrix to finish. No default or inferred count is used.

Both full workers and the SDK control write `PROBE_BOOT main_enter` as their
first main action, before JSON/SDK/MPI work. SDK calls and MPI startup record
enter/return, original codes and elapsed time. Loaded-library paths are logged
before SDK calls. An empty log therefore does not get attributed to a particular
API; it can indicate failure before main. Partial records retain the last
observed stage. Runtime/driver versions run in separate SDK processes after the
matrix, followed independently of the count control; optional information cannot
block allocations from starting. Provider metadata is also deferred until after
the matrix.

An allocation failure blocks only network cases with that allocation mode and
size, naming the prerequisite case. Other allocations and sizes continue.
An unreapable worker blocks remaining cases explicitly. No blocked or absent
case is counted as a pass. Results are saved before launch and after every
case, so failed or interrupted runs retain the complete planned matrix.

Before TERM, after TERM and after KILL, isolated snapshot collectors save PID/TID
state, cmdline, wchan, library maps, syscall and readable stacks. PID start times
guard against reuse. Permission errors, truncation and partial reads are retained;
the collector itself has a 250 ms deadline and 100 ms reap bound. An unreaped
collector also stops further GPU submissions. A failed snapshot never prevents
termination of the original worker. No process environment is dumped.

With four active HCA/ports the default matrix contains **469 cases**. Defaults:

- Per worker group: 10 seconds, plus 1.25 seconds for termination/reap and up to
  1.05 seconds for three snapshot collectors.
- Each barrier control: 180 seconds plus the same cleanup/diagnostic allowance.
- GPU inventory control: 15 seconds, configurable with `--inventory-timeout`;
  the startup timeout has not been increased.
- Supervisor budget: 6,600 seconds (110 minutes), inside the existing 120-minute
  hardware job limit. Project compilation runs before the supervisor; worker
  builds run inside it, each with a 300-second timeout by default.

Before testing, the supervisor checks the full worst-case matrix budget for
available workers, including elapsed build time, cleanup and setup margin.
It also reserves optional metadata time. The four-port matrix reserves 6,108.7
seconds for cases, cleanup and diagnostics; the larger suite budget accounts for
the new evidence collection rather than increasing per-case timeouts.
An insufficient budget produces a setup failure with all remaining entries
explicitly blocked, rather than starting a matrix it cannot finish.
`FLAGCX_CI_RDMA_PROBE_CASE_TIMEOUT` and
`FLAGCX_CI_RDMA_PROBE_SUITE_TIMEOUT` override the supervisor defaults; a larger
budget must still fit the enclosing CI job. No automatic retries replace the
first failure evidence.

Artifacts are collected by the existing always-upload step under
`attempts-unit_test-*-metax-rdma_atomic_probe`. Contents under
`rdma_atomic_probe/` include `results.json`, `summary.md`, `build.json`,
`build/{sdk,native,adaptor,barrier}.log`, SDK feature checks, `host-inventory.json`,
host and per-worker GPU inventory logs, version/provenance
logs and one log per planned case. The summary separates allocation, native
verbs and adaptor outcomes for every allocation mode, followed by every
individual case and its failure stage. A mixed-success diagnostic matrix
returns failure.
Timeout evidence is stored under `<worker-log>.diagnostics/{before-term,after-term,after-kill}/`,
with collector outcomes in `collector.json` and incremental `pid-*.json` snapshots.
`results.json` and the Markdown summary distinguish startup controls from the
actual allocation/transport matrix, including main-entry and last-stage evidence.

Statuses distinguish `BUILD_ERROR`, `ALLOC_ERROR`, `SDK_UNAVAILABLE`, `INIT_ERROR`,
`REG_ERROR`, `POST_ERROR`, `CQ_ERROR`, `VERIFY_ERROR`, `COPY_ERROR`,
`CAPABILITY_ERROR`, `CONNECT_ERROR`, `CLEANUP_ERROR`, `TIMEOUT` and `BLOCKED`.
Startup controls additionally distinguish `STARTUP_TIMEOUT`, `STARTUP_ERROR`,
`SDK_ERROR` and `MISSING_INVENTORY`. An unreapable worker is reported as
`WORKER_STUCK`, with timeout evidence preserved; launch/exit failures and
missing rank results are also explicit failures.
API error records retain their original interface return codes; MACA errors
also include the SDK error string. The GDR interface returns FlagCX codes;
its internal MACA diagnostics remain in the original case log.

## Reading the evidence

| Observation | Next area to investigate |
|---|---|
| Allocation/copy fails | That SDK allocation mode, size or memory attributes |
| Allocation passes, MR registration fails | Peer-memory registration support and MR flags |
| READ/WRITE pass, native FetchAdd fails | GPU-target atomic access, provider/driver/firmware, QP capabilities |
| One allocation mode passes FetchAdd, another fails | Allocation-dependent atomic compatibility; compare reported attributes |
| Native passes, adaptor fails under matching settings | Production registration/range resolution, PD/key or lane pairing |
| Individual HCAs pass, automatic selection fails | Cross-HCA endpoint routing and key pairing |
| Single QP/sender passes, concurrency fails | QP resources, atomic return scratchpad and request lifetime |
| Both transport layers pass, S16 fails | Proxy/context, signal indexing and barrier protocol |

These are diagnostic branches, not automatic root-cause verdicts. A successful
READ/WRITE does not certify remote atomics. Allocation success does not certify
RDMA, and CPU-submitted transport success does not certify IPC kernel polling
visibility. Keep the existing Device API intra acceptance when evaluating any
future production allocation change.

## Running on a MetaX host

The probe is a MetaX-only diagnostic and is not currently selected by the
platform YAML. Its unit-test entry remains callable manually: it builds the
library, then the supervisor builds each worker independently and runs the
complete matrix. The probe Makefile also provides separate
`native` and `adaptor` targets. For an equivalent SDK-equipped local environment:

```bash
bash .github/scripts/ci/run_unit_test.sh .github/scripts/set_env/metax.sh rdma_atomic_probe
```

Hardware-free supervisor coverage lives in
`test/script/test_metax_rdma_probe.py`: complete allocation matrix, disabled
automatic CI selection, GPU-signal enforcement, missing-rank/error
classification, bounded timeout recovery, failure continuation and dependency
reporting. Build-failure
controls verify that each remaining layer runs, all allocation modes remain
covered, build evidence is saved, and failed builds cannot reuse stale binaries.
Startup controls cover independent host discovery, partial/empty output,
reaped-worker fallback, actual-count validation, unreaped-worker stop, deferred
optional metadata, partial snapshots and PID reuse.
