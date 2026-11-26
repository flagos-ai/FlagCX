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
own resources. Ordinary per-call private scratch keeps its existing execution
cleanup; exported persistent allocations remain owned until Destroy.

```text
caller completes all user work
    -> Finalize closes submission admission
    -> drain internal producers and pending runner cleanup
    -> close local views, Windows and registrations
    -> complete relay release ACKs on all ranks
    -> stop/join proxy services and close their imports
    -> close residual IPC-table imports
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
than treating an unconfirmed close as permission to free.

After successful Finalize, repeated Finalize is a local no-op. Destroy adds no
FlagCX bootstrap exchange, relay RPC or SHM peer-close wait. A local free failure
retains the owner and saved progress; retry Destroy on that rank. Only Success
consumes the Comm pointer. Native SDK Destroy can still impose its own waits.

## Native SDK and integration limits

Every distinct native communicator, including tuner aliases and an automatic
host communicator, is finalized once. Native Finalize returning InProgress is
submitted once and polled through its asynchronous-error API on subsequent
Finalize calls. A synchronous Success does not require an asynchronous poller.
Errors, unsupported operations and missing progress support preserve the owner.
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
closed owner use saved status or cached metadata, without querying retired SDK
state.

Torch's asynchronous Work objects, completion handling, and explicit
shutdown/retry APIs are retained. Shutdown completes backend communication
streams before Comm destruction, saving failed-sync owners and completed progress.
They are separate from the removed core
WorkRecord tracker. Existing platform limitations and CI exclusions remain
visible; host mocks cannot establish rank-local behavior inside a vendor SDK.
Ascend SHMEM and MetaX Device API integrations are preserved independently.

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
from freeing their private scratch early. Notification streams are queried on
failed retirement; an SDK error or InProgress retains the owner for retry.
This count does not collect public API work or record stream-tail events.
Caller completion before Finalize replaces the removed automatic user-work
drain guarantee. Its performance and overlap effects need hardware measurement.

## Graph terminology

No complete public graph capture/instantiate/launch API or end-to-end replay
regression was found in the baseline. UniRunner DAG templates are a different
feature and remain supported. The caller-completion rule is a resource-use
contract; it does not claim additional runtime graph-replay support.

## Validation

Host tests inject import-close, backing-free, native asynchronous-progress,
transport and device-selection failures. Real bootstrap tests exercise two and
four ranks, empty ranks, collective retry and sequential rank-local Destroy.
These use mocked local device operations and do not certify accelerator IPC.
Hardware runner AllGather, Window/Device API IPC, independent-Comm isolation,
and SDK teardown remain required on the corresponding platform. Iluvatar's
SDK callback/event diagnostic and runner regressions are pending hardware access;
removing work events alone is not evidence of their measured completion.

The standalone native callback/copy probe was on origin/fix-ci-issues, not in
the original 34a81ae3 main snapshot; this change did not remove it from main.
The separate ipc_vmm_repro diagnostic remains. Full path/hunk decisions and
preservation hashes are recorded in ipc_preservation_audit.json. Hardware
acceptance is separate from the host/mock review verdict.
