# Communicator and IPC lifetime

The communication execution path follows `81abe75f`: no per-operation work
tracker, work snapshots or completion events are appended after host callbacks.
Readiness events, proxy copy-stream completion events and Device API semaphore
ordering remain part of execution. Finalize and Destroy run on the caller's
thread; there is no lifecycle recycler thread.

## Caller completion boundary

Before Finalize, finish every user stream, directly launched Device API kernel,
and any application-managed device execution that can access the communicator
or its buffers.
Do not submit concurrently with lifecycle calls, and finish open groups first.
Heterogeneous execution keeps its existing group depth. Native SDK groups have
a separate thread-local nesting count; Barrier and unfinished Finalize reject
an open group before releasing resources. These counters record nesting only,
without per-operation work records or completion events.
C2C planners reject an outer group before allocating scratch, including native
groups opened with a null or different Comm. Their existing staged execution
requires submission before its stream drains.
Finalize drains internal producers and transport owners; it does not discover
all previously accepted user work or synchronize the entire device. This is an
explicit change from the removed work-tracking contract.

## External buffers

A public Window borrows the caller's allocation. The safe order is:

```text
complete local device work and remote accesses
    -> destroy the Window's DevMem views
    -> WindowDeregister on every importing rank
       [close local IPC/VMM mappings and registration handles]
    -> application confirms that all importers have deregistered
    -> caller frees the original allocation
```

WindowDeregister is local. It does not close another rank's imports, free the
user buffer, or supply the application-wide confirmation. It also cannot
revoke raw pointers or application-managed GPU execution. A successful
Window handle is empty and reusable on the same Comm. If cleanup fails or
returns InProgress, retain the handle and retry; completed close/unmap steps
are saved and new views of that handle are rejected.

## Internal buffers

Exported internal allocations have an explicit Comm or transport owner.
Destroying a Default Device API view can retire its IPC generation to the Comm
without closing imports or freeing backing. A subsequent generation gets its
own resources. C2C waits for the user and private streams before releasing
per-call scratch, including after successful native submission, because some
backends free memory without stream ordering. A native GroupEnd that returns
InProgress is polled through its participating native communicator handles
before the next stage's stream drain. Query failures keep the concrete pending
submission and scratch owner for cleanup retry. Failed drains retain the owner and
completed steps for retry. After the user stream drains, cleanup drops its
borrowed wrapper and uses the owned private stream for freeing scratch.
Calls without scratch keep their existing successful
submission behavior; exported persistent allocations remain owned until Destroy.
Releasing a buffer already recorded by the Comm retains that allocation even
before Finalize starts. Only Destroy releases its backing after import closure.

With host-mapped signals enabled, SHM provides shared host pages for local
Barrier and Signal control words. Each name includes the Comm magic, a fresh
root-issued creation generation, and the owning rank. Collective DevComm
creation is serialized; root broadcasts the generation before mappings are
created. Failed creations consume their generation, and retirement never resets
it. Reusing a live stack ID therefore cannot reopen or zero retained storage.
Broadcast participation uses the common barrier/signal requirements. Counter-only
creation allocates no SHM and skips the broadcast on every platform. Participation
does not depend on the local completion word width or whether that rank will
allocate a host-mapped signal plane.

```text
caller completes all user work
    -> Finalize closes submission admission
    -> drain internal producers and pending runner cleanup
    -> retire this Comm's collective queue, including accepted deferred requests
    -> close local views, Windows and registrations
    -> complete relay release ACKs on all ranks
    -> stop/join proxy services and close their imports
    -> close residual IPC-table imports
    -> prepare native-owner catalog and registrations; agree on local results
    -> native SDK Finalize
    -> all phases successful: remember import-closure permission
    -> Destroy: release local backing and remaining owners
```

| Resource | Finalize | Destroy |
| --- | --- | --- |
| DevComm barrier, signal, staging and retired generations | Close peer mappings; retain exporter and cached backing | Free retained allocations/SHM |
| UniRunner ready/ring storage | Close data/ready/ring views | Free Comm-owned backing and private persistent scratch |
| P2P FIFO | Close imports | Free local FIFO/SHM, streams and events |
| NET/PXN relay resources | Deregister and finish existing release ACKs | Free local connections/pools |
| RMA completion counters | Retire workers; close IPC and CPU device mappings | Free device counters and allocated host counters |
| External live Windows | Close this Comm's local imports/registrations | Never free user backing |

The full FlagCX communicator participates in the internal IPC agreement,
including ranks with no local allocation. Heterogeneous initialization declares
this participation before publishing owners. Pure native communicators without
FlagCX IPC resources or collective native-close requirements add no FlagCX
shutdown agreement.

Each phase records local completion separately from common completion. A rank
that succeeds locally retains backing while another rank reports failure. All
ranks retry unfinished Finalize in the same order. A failed or InProgress call
keeps the communicator closed to new submissions and preserves remaining owners.
Bootstrap failures can prevent collective recovery; ownership is retained rather
than treating an unconfirmed close as permission to free. Standalone hetero
Finalize drains rejected group owners before checking transport readiness and
keeps new data submissions closed across failed or InProgress retries.

After successful Finalize, repeated Finalize is a local no-op. Destroy adds no
FlagCX bootstrap exchange, relay RPC or SHM peer-close wait. A local free failure
retains the owner and saved progress; retry Destroy on that rank. Only Success
consumes the Comm pointer. Native SDK Destroy can still impose its own waits.

## Native SDK and integration limits

Every distinct native communicator, including tuner aliases and an automatic
host communicator, is finalized once. Native Finalize returning InProgress is
submitted once and polled through its asynchronous-error API. If another native
owner remains, polling completes the earlier owner before entering the next
collective close. The final owner can return InProgress for subsequent Finalize
calls. Public GetAsyncError also advances a pending Finalize, so the
NCCL wrapper's finalize-once/poll loop can complete. Where close requires peer
agreement, all ranks must continue polling or retrying in the same order.
A synchronous Success does not require an asynchronous poller.
Errors, unsupported operations and missing progress support preserve the owner.
Catalog allocation, local registration release and capability checks complete in
an earlier status-converged phase. A rank-local preparation failure therefore
keeps every participant outside native collective close. Completed preparation
is saved; unrelated registration readers cannot force repeated preparation
inside the subsequent SDK close.
The existing built-in SDK compatibility policy is retained: HCCL, CNCL,
ECCL, PCCL and XCCL use their supported native Destroy operation
at the Finalize boundary. Selection uses the actual built-in adaptor identity;
an unknown plugin does not inherit this behavior from its name. A failed or
InProgress compatibility close retains the native owner and aliases. Only
strict Success clears them. Exported FlagCX backing remains allocated until
public Destroy. Required native collective closure is agreed during init,
separately from the IPC participation bit.

MUSA's pinned 5.2 SDK uses native `mcclCommFinalize` separately from Destroy.
It does not use compatibility Destroy at the Finalize boundary. Native
stream-isolation and IPC cleanup coverage remain required on MUSA.

Initializer failures keep acquired plugin references and partial native owners
for Destroy retry. This path does not call native Finalize on a half-built SDK
communicator. Early local rollback is allowed only before exported Device API
owners exist; later failures follow the agreed import-close phases. Each plugin
reference is released only if the owner acquired it.

Heterogeneous Abort remains unsupported and consumes no native owner. A native
Abort failure retains the public owner; an uncertain Abort must be retried
before native Finalize can complete. Successful native Abort keeps its wrapper
until local destruction. Abort does not enter a hidden peer rendezvous or imply
permission to free exported backing; when agreement is still required it returns
InProgress, and participants must explicitly Finalize together. Diagnostics on a
closed owner use saved status or cached metadata. Only pending Finalize progress
queries the retained native owner; completed, failed, Abort and rollback paths
do not query retired SDK state. Device-number queries use the cached ordinal
whenever it is valid, including before shutdown on SDKs without that query.
Standalone heterogeneous Destroy also selects its saved device on every retry,
including after a cached successful Finalize and partial proxy release.

Torch's asynchronous Work objects, completion handling, and explicit
shutdown/retry APIs are retained. Shutdown completes backend communication
streams before Comm destruction, saving failed-sync owners and completed progress.
They are separate from the removed core
WorkRecord tracker. Existing platform limitations and CI exclusions remain
visible; host mocks cannot establish rank-local behavior inside a vendor SDK.
Ascend SHMEM and MetaX Device API integrations are preserved independently.
The CUDA NCCL wrapper owns stable stream wrappers and completion events for its
asynchronous API contract. It waits for them before core Finalize/Destroy and
retains them across failures. These plugin events do not enter Iluvatar paths.
UniRunner drops its borrowed IPC stream immediately after a successful cleanup
drain, before later fallible cleanup; retries retain only unfinished resources.
Host runner group entry/exit preserves pending native status before changing
group depth or copying staged receives. Only a successful outer GroupEnd makes
those receive buffers available for the existing device-copy step.

Barrier checks allocation, initialization, collective submission and cleanup
statuses. Its private scratch is released after accepted native submission and
the launch stream complete. A failed drain or free retains the buffer and
completed cleanup steps; the next Barrier or Finalize retries cleanup. Keep the
borrowed stream wrapper alive until the stream drain succeeds. Completed drains
drop that reference, and later free retries use no user stream. This adds no work
tracker or callback completion event. Cleanup uses the existing CHECK/GOTO
macros with explicit InProgress branches before consuming an owner.

## Callback and event boundary

The ordinary host-callback group path is:

```text
each operation stream: readiness -> common input-ready wait -> host callback
proxy copy stream:     accepted copy -> copy completion event
```

The default stream participates through its native-null stream wrapper. Every
participating operation stream waits for all inputs before entering a callback;
one callback starts the shared proxy semaphore and the others wait for its
completion. Callbacks are submitted for all Comms before GroupEnd returns. A
sealed notification owner prevents proxy retirement while submission is still
being built, and callback references keep it alive until every callback returns.
Device semaphore slots remain leased until both device completion and host
retirement are confirmed. Their expandable pool uses pthread mutexes.

No core WorkFinish/group-associated completion events are appended to the
callback's launch stream. Existing data-path events remain, including readiness,
proxy copy completion, Device API semaphores, UniRunner DAG dependencies and
Torch Work completion. Subsequent operations, Torch completion handling or the
retained RMA host-function path can still put an
event after an earlier callback. Removing the tracker does not establish that
all SDK callback/event combinations are safe.

Ordinary GroupEnd retains baseline submission behavior. Failed C2C/UniRunner
cleanup retains concrete buffer/stream owners and retries stream-level cleanup;
it creates no per-operation tracker events and adds no device-wide fallback.
A partial notification failure cancels the posted prefix, retains rejected
proxy owners and discards unsubmitted tasks. Cancellation can release a callback
before accepted copy/network requests finish: existing transport retirement
keeps those owners, and a transport-queue count prevents failed runner cleanup
from freeing their private scratch early. Cancelled callbacks retain their own
references until their final access;
proxy retirement does not query borrowed stream wrappers. Device-kernel and
stream-value notifications use their existing owned completion events. If a GPU
notification fails before that event is recorded, the builder tries to record
that same event while the launch stream is still valid. This failure-only proof
is never appended to a host callback. Missing or failed proof retains the owner.
Stream-value cancellation releases its device wait without relying on counters
for unpublished operations.
This count does not collect public API work or record generic stream-tail events.
Finalize and public deregistration wait for concrete transport owners before
closing registrations or mappings, even after user streams finish.
Explicit Flush retains each accepted native request before checking posting or
polling errors. A per-Comm pthread mutex serializes access; cleanup reports
InProgress while another poll owns that mutex. An incomplete request remains
owned for retry. A provider's completion flag retires the request even when it
also reports an error, because the provider may already have freed the request.
MR and connection cleanup wait for these requests; Destroy releases the empty
state before unloading the transport plugin. This state creates no GPU events.
Fused preparation failures retain their specific stream and drain progress with
invocation buffers. The caller keeps this borrowed stream alive until cleanup
succeeds. Cleanup retries that stream before releasing the buffers;
it cannot replace a failed stream/runner drain with device-wide synchronization.
Semaphore cleanup saves successful notification completion before destroying
individual events, so a later cleanup retry never queries an event already freed.
Caller completion before Finalize replaces the removed automatic user-work
drain guarantee. Its performance and overlap effects need hardware measurement.

## Test process cleanup

The Device API correctness binaries keep test resources in a shared test session.
An operation failure is agreed before ranks leave the test body. Cleanup first
drains the user stream, closes views and Window imports on every participant,
then releases caller buffers while their allocator is alive. Comm Finalize and
local Destroy precede stream and device-handle release. Cleanup errors retain
the session's handles and phase; a retry skips successful releases. An exhausted
or terminal cleanup failure aborts the MPI test rather than allowing SDK process
shutdown to invalidate storage still used by retained workers. This is test
process error handling, not a production work tracker or reclamation thread.

## Graph terminology

No complete public graph capture/instantiate/launch API or end-to-end replay
regression was found in the baseline. UniRunner DAG templates are a different
feature and remain supported. The caller-completion rule is a resource-use
contract; it does not claim additional runtime graph-replay support.

## Validation

Host tests inject import-close, backing-free, native asynchronous-progress,
transport and device-selection failures. Real bootstrap tests exercise two and
four ranks, empty ranks, collective retry and concurrent rank entry to Destroy.
These use mocked local device operations and do not certify accelerator IPC.
Hardware runner AllGather, Window/Device API IPC, independent-Comm isolation,
and SDK teardown remain required on the corresponding platform. Iluvatar's
SDK callback/event diagnostic and runner regressions are pending hardware access;
removing work events alone is not evidence of their measured completion.

The standalone native callback/copy probe was on origin/fix-ci-issues, not in
the original 34a81ae3 main snapshot; this change did not remove it from main.
The separate ipc_vmm_repro diagnostic remains. Hardware acceptance is separate
from the host/mock review verdict.
