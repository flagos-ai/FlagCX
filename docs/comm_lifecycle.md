# Communicator and IPC lifetime

The communication execution path follows `81abe75f`: no per-operation work
tracker, work snapshots or completion events are appended after host callbacks.
Readiness events, proxy copy-stream completion events and Device API semaphore
ordering remain part of execution. Finalize and Destroy run on the caller's
thread; there is no lifecycle recycler thread.

## Caller completion boundary

Before Finalize, finish every user stream, directly launched Device API kernel,
and captured graph replay that can access the communicator or its buffers.
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
revoke raw pointers or graph replays retained by the application. A successful
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
FlagCX IPC resources add no FlagCX shutdown agreement.

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
There is no core fallback that substitutes native Destroy for Finalize, and no
new platform conditional compilation to simulate support.

Consequently, an adaptor whose Finalize is unsupported or fails cannot satisfy
successful public Finalize/Destroy through this path. Existing SDK-specific
restrictions and CI exclusions remain visible; they are not a guarantee of
rank-local SDK destruction. Ascend SHMEM and MetaX Device API integrations are
preserved independently of those native lifecycle limitations.

This change does not retain the generic initialization-rollback state machine,
Abort owner machinery, or explicit Torch shutdown/retry APIs introduced with the
removed lifecycle series. Those features must not be assumed by callers.

## Validation

Host tests inject import-close, backing-free, native asynchronous-progress,
transport and device-selection failures. Real bootstrap tests exercise two and
four ranks, empty ranks, collective retry and sequential rank-local Destroy.
These use mocked local device operations and do not certify accelerator IPC.
Hardware runner AllGather, Window/Device API IPC, independent-Comm isolation,
and SDK teardown remain required on the corresponding platform. Iluvatar's
native callback/event probe and runner regressions are pending hardware access;
removing work events alone is not evidence of their measured completion.
