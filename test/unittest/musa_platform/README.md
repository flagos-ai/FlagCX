# MUSA platform migration probe

This standalone probe checks the platform primitives ported from MCCL before
integrating the Host communication backend and full Device API test kernels.
Build and run it on the Linux MUSA server.

```bash
cd /workspace/FlagCX
make -C test/unittest/musa_platform DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
timeout --signal=TERM --kill-after=10s 60s make -C test/unittest/musa_platform run
```

Expected output includes `MUSA host layout PASS` and
`MUSA platform probe PASS (0 failures)`.

The ordinary host compiler checks the public lane mask and cooperative object
layout without including the compiler-specific native atomic headers, matching
MCCL's host/device header boundary. Native order/scope conversion assertions
are compiled by mcc. The device probe checks lane masks, 8-thread tiles,
warp/lane/span and type-erased groups, unsigned 32/64-bit atomics in four
scopes, contended device atomics, and release/acquire publication between two
warps in one block.

## Supported span synchronization

| Span in a one-dimensional CTA | Platform implementation | Validation |
| --- | --- | --- |
| One complete warp at a valid warp index | `__syncwarp(fullMask)` | Two independent warp spans run different numbers of shared-memory exchange rounds |
| All threads in the CTA, starting at warp 0 | `__syncthreads()` | Two warps exchange shared-memory values repeatedly through the span API |
| Multiple warps forming a proper subset of the CTA | Explicitly unsupported | `CoopTileSpan::supportsSync()` returns false, and `sync()` traps |

The supported spans are also exercised through `CoopAny`. The ordinary host
probe checks valid ranges and rejects empty, out-of-range and partial-CTA
multi-warp spans. `Intrin::namedBarrierSync` supports whole-CTA participation;
`CoopTileSpan` additionally recognizes aligned, complete single-warp groups.
A PASS covers these supported mappings. Partial-CTA multi-warp synchronization
remains a capability gap to resolve before claiming complete cooperative-group
support.

The port uses native `<musa/atomic>` APIs following MCCL's scoped access and
fence implementations. Record `MUSART_VERSION`, SDK/compiler versions and the
full build log. A missing `musa::atomic_ref` or native operation identifies an
SDK compatibility gap that needs resolution before Device API integration.
MCCL's old `atomic_ref_wa.h` shim does not distinguish the requested orders.

Cross-block, cross-GPU and CPU/GPU visibility still require separate runtime
and communication integration checks. `CoopOps` and `Clock` extended operations
remain to be ported. Final T1–T4 acceptance uses the common test programs and
their complete cases, sizes and rank topology.

## Counted barrier compiler diagnostic

Repeated `llvm.musa.barrier.sync.cnt` DAG output during compilation points to
lowering of `__barrier_sync_count(1, 64)`, used by MCCL's warp-span group.
Interrupt that build and isolate the compiler operation with the two targets
below. They compile the same 64-thread kernel with either a block barrier or
the counted barrier, using the configured MUSA compiler flags. They include
only `musa_runtime.h`, independently of FlagCX traits and native atomic headers.

```bash
cd /workspace/FlagCX
make -C test/unittest/musa_platform barrier-control
(
  ulimit -f 8192
  timeout --signal=TERM --kill-after=5s 15s \
    make -C test/unittest/musa_platform named-barrier-repro \
    > /tmp/musa_named_barrier_build.log 2>&1
)
repro_status=$?
printf 'named barrier compile exit=%s\n' "$repro_status"
head -c 8192 /tmp/musa_named_barrier_build.log
printf '\n'
/usr/local/musa/bin/mcc --version
```

The shell limits the diagnostic log size and compile duration. Record those
limits with the exit status, since either limit can terminate a stuck compiler.
The repeated DAG diagnostic may have no newline, so display it with a byte
limit (`head -c`), rather than a line limit (`head -n`).
If the control succeeds and the counted operation fails independently, provide
the reproducer, exact command, SDK/compiler version and initial diagnostic to
the MUSA compiler maintainers. If both isolated kernels compile, retain the
complete platform probe as the larger reproducer for the interaction failure.

The raw counted intrinsic is retained in the compiler-only reproducer. Normal
`all` and `run` targets use the supported span mappings described above.
Partial-CTA multi-warp synchronization requires participant-count,
group-isolation and memory visibility semantics. MCCL's software barrier in
`src/device/common.h` relies on shared state initialized by its kernel setup
and requires a corresponding initialization contract before reuse.

The observed mcc 5.1.0 failure and information needed for an SDK report are
recorded in [the compiler issue note](../../../docs/musa_counted_barrier_compiler_issue.md).

## Device abort lowering diagnostic

`Cannot select: ... ExternalSymbol'abort'` while compiling a kernel means a
host-style abort reached device code generation. MUSA device-pass
`FLAGCX_DEVICE_TRAP()` uses the native `__trap()` used by MCCL's
`contrib/mccl_ep/device/device_primitives.cuh`. Host-pass failure paths retain
the host builtin. Range checks and unsupported-operation checks remain active.

An independent compiler-only target checks native trap support with a runtime
condition, preventing optimization from deleting the failure path.

```bash
cd /workspace/FlagCX
timeout --kill-after=5s 60s \
  make -C test/unittest/musa_platform native-trap-repro
```

If this isolated native trap compiles and the full probe still reports an
abort symbol, inspect the device IR or preprocessed SDK assertion path to
identify the remaining caller. The isolated target is diagnostic; the normal
`all` target continues to require both the host layout and full device probe.

## Device function-pointer relocation

On the reported mcc 5.1.0 / mp_31 server, the probe compiled after the native
trap fix but device linking rejected `R_MTGPU_ABS64` relocations for the static
`PlatformCoop` function-pointer table. Both compile and link already used
`-fPIC`; adding that flag again does not address the observed failure.

MUSA now supplies its own 24-byte, 8-byte-aligned `CoopAny`, holding rank,
size, lane mask and a synchronization kind. Calls dispatch directly to the
same warp or block primitive as the original group, without device function
pointers. Invalid or unsupported spans retain a trapping sync path.
MCCL supplies the underlying group primitives; this type-erasure bridge is
FlagCX-specific. Other platforms keep their existing implementation.

The device probe checks conversions and copies of block, warp, tile, lane
and thread groups with shared-memory peer exchanges, plus default construction.
Existing span checks cover whole-CTA and independent single-warp groups.
The revised implementation still requires compile, link and runtime validation
on the Linux MUSA server.

## Explicit device link and image embedding

After removing device function pointers, the server's mcc 5.1.0 link trace
showed two successive device-link invocations. The first produced a 48608-byte
image; the second failed with `invalid buffer: the size (0) is smaller than an
ELF header (64)`. The trace does not identify which buffer was empty. Empty
`.resolution.txt` files alone do not establish the cause.

The standalone probe now builds in four explicit stages, following the
device/host split and image embedding in MCCL's `src/device/Makefile`:

1. Compile the device pass with `--cuda-device-only`, retaining RDC and PIC.
2. Invoke the SDK `lld` once, retaining its plugin argument, target architecture
   and `trap_handler.o`. Use MCCL's final-image link mode and `-plugin-opt=O2`,
   without the `-shared --as-needed` flags from the driver's intermediate link.
3. Compile the host pass with `--cuda-host-only` and embed the linked image
   with `-Xclang -fcuda-include-gpubinary -Xclang <image>`.
4. Link the host object with the host C++ compiler and MUSA runtime libraries.

The CUDA-spelled options are the ones used by MCCL's MUSA build. Device code
is still compiled, linked, registered and executed; no probe checks are omitted.
The SDK resource directory is queried through `mcc -print-resource-dir` to
locate its matching trap handler. The linker and LLVMgold plugin are resolved
relative to the actual compiler executable, following symlinks. On the reported
server the driver passed `/usr/local/musa-4.3.5/lib/LLVMgold.so`, but that file
was also absent. Its first device link nevertheless produced a nonempty image.
The plugin argument is retained to match that invocation, but the path is not
a Makefile file prerequisite. The SDK linker handles the argument; if it
reports a plugin-loading error, that is a separate toolchain dependency to
resolve. Runtime libraries still come from
`DEVICE_LIB`. Paths can be overridden through `MUSA_TOOLCHAIN_ROOT`,
`MUSA_DEVICE_LD`, `MUSA_LTO_PLUGIN`, `MUSA_RESOURCE_DIR` or `MUSA_TRAP_HANDLER`.

This change is limited to the standalone probe Makefile. The four-stage build
and runtime results require validation on the server; it does not establish
support for arbitrary cross-translation-unit kernels in the main library.

## Launch resource validation

The server subsequently ran the executable and reported host layout PASS
(`SIMT=32 mask=8 CoopAny=24 align=8`). The first GPU kernel launch then returned
`too many resources requested for launch`; GPU correctness checks did not run.

All three probe kernels now declare `__launch_bounds__(kThreads)` for the
actual 64-thread block. MCCL's `makefiles/common.mk` documents the relationship
between register budgets and supported thread counts and points to launch
bounds for kernels. This supplies the compiler with the launch contract while
retaining two warps and every existing check. Register pressure is a candidate
cause; the runtime error alone does not prove it.

Before launch the probe queries `musaFuncGetAttributes`, prints each kernel's
thread limit, register count, static shared memory and local memory, and
rejects a thread limit below 64. It also identifies each kernel immediately
before launch. These changes still require server compilation and execution.
If launch fails again, retain the resource report and exact failing kernel
before changing compiler resource limits or splitting the probe.

The next clean rebuild completed all four stages, but preflight reported
`maxThreadsPerBlock=128` for `platformProbe` and `0` for `warpSpanProbe`, both
with `numRegs=256 sharedBytes=268 localBytes=16384`. The preflight rejected
the latter before any kernel launch. These values alone do not prove register
exhaustion. Review found that the embedded image was linked with
`-shared --as-needed` copied from the driver's first, intermediate link.
The recipe is now aligned with MCCL's final image link: neither flag is used,
and `-plugin-opt=O2` is specified. All three kernel attribute queries now run
before evaluating preflight failure. Rebuilding and running on the server
must establish whether the final image mode resolves the resource anomaly;
no resource check or correctness check has been removed.

The subsequent final-image rebuild did not resolve the attribute anomaly:
`platformProbe` reported thread limit 128, 256 registers and 16384 local bytes;
`warpSpanProbe` reported limit 0 with the same register/local-memory values;
`anyGroupProbe` reported limit 1024, 23 registers and zero local bytes. All
reported 268 shared bytes. Preflight stopped before any kernel launch.
Both high-resource kernels retain dynamic trap paths; this is a hypothesis to
isolate, not evidence that native traps are the root cause.

Run the SDK-only controls with the same compile/link/embed pipeline:

```bash
timeout --kill-after=5s 60s \
  make -C test/unittest/musa_platform launch-resources \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

`musa_launch_resources.cu` contains five kernels: plain output, shared-memory
warp synchronization, shared-memory block synchronization, conditional native
trap, and warp synchronization with the same conditional trap. All declare
the same 64-thread launch bound. Only runtime attributes are queried; none of
these kernels is launched. This target neither replaces the platform probe
nor reports GPU correctness PASS. Compare trap versus plain and guarded-warp
versus warp to distinguish the primitive/resource interaction from the full
FlagCX implementation.

The server controls reproduced the anomaly only in kernels with a dynamic
native trap: `plain`, `warp` and `block` reported thread limit 1024, 1–2
registers and no local memory; `trap` and `guarded-warp` reported thread limit
0, 256 registers and 16384 local bytes. This isolates the failing path under
the current build/runtime pipeline; it does not yet identify which SDK layer
causes it. The platform probe remains blocked before GPU correctness checks.
See [the vendor report](../../../docs/musa_trap_resource_issue.md) for exact
results, environment, build details and questions for the MUSA maintainers.

## Re-detect architecture on the Linux server

```bash
DEVICE_HOME=/usr/local/musa bash test/unittest/musa_platform/detect_arch.sh
```

This independent script queries every visible GPU's name, compute capability,
warp size and device thread limit, records runtime/driver version numbers and
actual linked libraries, then compile-checks `mp_31`, `mp_43` and any detected
GPU targets. Extra target names can be supplied as arguments. It does not
enumerate every target supported by the compiler, launch any kernel, or change
the project configuration. A compile failure is reported with its diagnostic
and may have causes other than unsupported architecture. Reports are kept in
a fresh `build/arch_detect.XXXXXX` directory for each invocation.

The server re-detection reported eight S5000 devices, all compute capability
3.1, warp size 32 and device thread limit 1024. Runtime and driver version
queries both returned 40305, with `libmusart.so.4` loaded from
`/usr/local/musa/lib`. The minimal mp_31 compile passed; mp_43 failed with
`No available targets are compatible with triple "mthg-mt-musa"`.
Keep `MUSA_ARCH=mp_31` for this hardware and toolchain. Changing the target to
mp_43 is not a resolution for the reproduced trap resource anomaly.

## Focused GPU validation while dynamic trap is tracked separately

The default `run` target still executes the complete platform probe, including
dynamic spans and their resource preflight. Two additional targets select
independent groups of existing checks:

```bash
make -C test/unittest/musa_platform run-groups DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
make -C test/unittest/musa_platform run-atomics DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

`run-groups` checks block, warp, tile, lane, thread and default groups through
CoopAny conversion/copy and shared-memory peer exchanges. `run-atomics` checks
the same 32/64-bit operations, four scope mappings, contention counters,
device/system fence calls and two-warp release/acquire publication as the full
probe. The atomic checks are shared by both modes. Each selected kernel has
resource preflight, launch-error checks, synchronization and copied-back result
validation. The host layout check runs first in either target.

These targets need no Host DevComm, window or IPC resource setup. Their output
explicitly names the selected suite; a focused PASS does not imply dynamic
Span support, cross-GPU/CPU visibility, full platform PASS or T1–T4 completion.
They leave production guards unchanged. Both focused entry points have now
passed on the Linux S5000 server:

| Entry | Resource report | Runtime result |
| --- | --- | --- |
| groups-only | 23 registers, 268 shared bytes, no local memory, thread limit 1024 | 64 threads; PASS, 0 failures |
| atomics-only | 26 registers, 12 shared bytes, no local memory, thread limit 1024 | 64 threads; PASS, 0 failures |

The host layout check also passed in both invocations. These are positive-path
single-GPU results. The system-scope operations and fence calls executed on
device memory; no CPU/GPU or peer-GPU publication handshake was tested.

## Cross-block device-scope validation

```bash
timeout --kill-after=5s 120s \
  make -C test/unittest/musa_platform run-cross-block \
  DEVICE_HOME=/usr/local/musa MUSA_ARCH=mp_31
```

The additional `--cross-block-only` mode launches two 64-thread CTAs for eight
independently reset rounds. Every thread contends on the same 32/64-bit
device-scope counters. One CTA writes 64 payload words, synchronizes its own
threads, then releases a device-scope flag; a different CTA acquires the flag
and validates all words. Each round changes the payload and checks counters,
observed publication and data validity after kernel completion.

There is no grid barrier or producer wait for the consumer. The consumer loop
is bounded so a missing publication produces a failed result; if that happens,
scheduling and visibility must both be investigated. Resource preflight,
launch errors and synchronization errors remain checked. The Linux S5000
server compiled and ran this mode successfully: PASS, eight rounds, zero
failures. Its resource report was thread limit 1024, 64 registers, 12 shared
bytes and no local memory. This result does not validate cross-GPU, CPU/GPU
or communication backend behavior.
