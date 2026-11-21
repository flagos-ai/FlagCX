# Communicator finalization and internal IPC ownership

`flagcxCommFinalize` stops new work, drains accepted work, closes imported IPC
mappings and, when the creation protocol declared shared IPC ownership,
confirms closure through the communicator's intra-node bootstrap group. `flagcxCommDestroy` then releases the original exported allocations and
other local resources. A successfully finalized communicator needs no peer to
participate in its destruction.

## Ownership

| Resource | Owner | Release point |
| --- | --- | --- |
| DevComm barrier, signal and completion backing allocations | Comm internal buffer list | Local Destroy after successful Finalize |
| UniRunner IPC ready, scratch and progress backing allocations | Comm internal buffer list | Local Destroy after successful Finalize |
| Device API private caches, IPC pointer arrays and UniRunner private buffers retired during Finalize | Comm internal buffer list after ownership transfer | Local Destroy after successful Finalize |
| IPC imports, device views and registrations | Their DevComm, DevMem, window or proxy connection | Local cleanup during Finalize |
| P2P FIFO and PXN NET relay exporters | Retained proxy connection pool | Local Destroy after successful Finalize |
| Public window backing allocation | Application | Application frees after all users release it |

`flagcxDevCommDestroy` closes the local imports and views immediately. It does
not free a backing allocation that another rank may still import. Repeated
creation and local destruction can therefore retain exported storage until the
parent Comm is finalized and destroyed. Finalize cleans up live Device API
owners automatically; their handles become invalid when cleanup succeeds.

Automatic DevComm/DevMem cleanup reuses the completed communicator Drain phase
instead of issuing another device-wide synchronization. FlagCX-owned private
allocations are transferred to the Comm buffer list before their original owner
fields are cleared. This also avoids implicitly synchronizing SDK allocation
frees during Finalize. Imports and registrations are still closed in their
original stages; native SDK handles follow their own lifecycle protocol.
Allocation-transfer failure retains the original owner, and local Destroy free
failure retains the remaining list. Aliased allocations are recorded once.
Standalone Device API destroy preserves its existing synchronization and private
buffer release behavior. Applications must finish externally launched kernels
and graph replays before either cleanup path and prevent further uses.

RMA Stop rejects further submissions, retires its worker, closes imports and
unmaps CPU views. A stopped `rmaProxy` remains the owner of `doneSeqsDev` and
`readySeqsDev` until local Destroy, for both outer Comm and direct Hetero callers.
Their GDR allocator may fall back to a synchronizing runtime free, so Finalize
never frees these allocations. Local Release frees each pointer only once and
keeps the proxy on failure, including InProgress. Initialization rollback uses
the same Stop/Release sequence without entering the Finalize bootstrap protocol.
Start rejects an existing live or stopped proxy rather than overwriting it.

## Protocol

The C-style lifecycle state holds the common phase, local completion bits and
last result. Creation establishes whether IPC closure agreement is required:

- Native-only communicators have no additional FlagCX IPC teardown handshake.
- Heterogeneous transport creation agrees a node-wide declaration for its lazy
  P2P FIFO/PXN lifetime domain, including ranks with no current connections.
  This declaration is conservative even if a transport is never used.
- Collective DevComm, UniRunner and IPC-window creation agrees the declaration
  before allocations/imports can fail. A peer with no local allocation joins
  the same node protocol. The declaration remains set until Comm destruction.

Finalize uses this stored policy and never performs a collective merely to
probe whether a native-only communicator acquired resources. For a declared
IPC group, each phase first performs its remaining local work and then
exchanges a status with every rank in the node. Empty ranks participate.
A failed phase leaves the common cursor unchanged. Ranks whose local operation
succeeded skip that operation on retry, but participate in the status exchange
again.

1. Drain device work on user streams, fused buffers, device FIFOs and RMA
   producers while services and mappings are alive; agree on drain failures
   before entering any native communicator shutdown.
2. Clean up backend transport metadata.
3. Destroy local Device API owners and registrations, including partial-create
   owners and live windows.
4. Stop the RMA worker, remove cached P2P/NET registrations and close PXN relay
   imports while peer services can still acknowledge their release.
5. Stop and join proxy services and their workers; close P2P FIFO imports and
   UniRunner views while retaining the connection pool and exporters.
6. Close remaining IPC table entries and confirm completion locally in the node.
7. Finalize native communicators after their owners and registrations are closed.
   SDKs without a separate Finalize close their native context here, retaining
   the wrapper with a null native base. Their later Destroy frees only the
   wrapper. These adaptors and legacy plugins use conservative world status
   agreement for SDK shutdown independently of the IPC declaration. SDKs with
   native Finalize add intra-node agreement only for declared FlagCX IPC groups. Initialization agrees the maximum scope across all
   vendors, so mixed-vendor ranks cannot choose different control protocols.

The CCL v1 plugin layout is unchanged. The latest internal descriptor appends
its shutdown scope, and upgraded v1 plugins default to world agreement. No
Finalize callback may free its wrapper. Native SDK errors remain errors; a
failed compatibility close retains its native base for retry.

IPC status exchanges use explicit tags and the existing bootstrap sockets, not
SHM. Existing P2P data synchronization continues to use SHM; FIFO allocation
release no longer waits for a SHM teardown ACK. Cross-node NET relay shutdown
retains its existing rendezvous in the drain phase. Drain errors converge
before that rendezvous so a peer does not enter its barrier after another rank
has returned a local error.

Destroy advances a local cursor through internal buffers, vendor communicators,
heterogeneous resources, host resources, bootstrap and plugins. Each pointer or
handle is cleared only after successful release. A later failure leaves the
cursor and unreleased owners available to the caller. Plugin ownership is
retained if unloading fails.

If heterogeneous initialization fails during RMA startup, cleanup uses a
separate initialization rollback entry. It selects the device, retires the RMA
worker and releases its mappings/allocations before destroying proxy services
and the heterogeneous owner. A failed operation retains the output owner for
retry; the output is cleared only after successful destruction.

The outer Comm records that initialization rollback is pending and rejects
normal work and Finalize. A later CommDestroy resumes this rollback and local
release, without entering the normal Finalize bootstrap protocol or marking
the half-initialized communicator as finalized. Native destroy and plugin
unload failures also retain the remaining outer owner and local release cursor.

## One rank per process

FlagCX supports one rank per process. Finalize and Destroy run on the calling
thread, including native SDK callbacks; no lifecycle worker or same-process
rank reclamation clique is created. Each successful Destroy releases its own
communicator immediately after required finalization.

A process can own multiple independent communicators. Grouped finalizers are
queued once and execute in submission order on the same caller thread when
the outermost group ends. Every process sharing those communicators must use
a compatible shutdown order. The PyTorch backend stores pair communicators
in canonical (low rank, high rank) order, closes all pairs in that order, then
closes the process-group communicator. Pair insertion history cannot change
the teardown order.

Outside a group Finalize blocks. Check every return value in application code.
Do not concurrently submit work or start multiple lifecycle operations on the
same handle. On a local cleanup failure, retry Finalize with the node's ranks;
after successful Finalize, a failed free can be retried by that rank alone.
Proxy cleanup propagates every result other than Success, including InProgress,
through the transport, connection pool, Finalize and Destroy layers. Pending
import closure participates in the current node status exchange but cannot
authorize exporter release. Pending local release retains the pool and resource
descriptors; fields cleared by earlier successful releases are skipped on retry.
Device events, streams, registrations and FIFO allocations follow this rule.

Bootstrap or device failure can prevent progress; owners remain retained while
closure is unconfirmed.

## Shared registration metadata

The process-global registration pool protects its page maps, owner reference
counts, native/NET handles and P2P cache with a pthread mutex. Data paths acquire
an item snapshot and pin its lifetime until their metadata operation completes.
P2P cache lookup, attachment and reader pinning happen in one locked operation.
Final deregistration cannot delete a record while a snapshot remains borrowed.
Repeated registrations retain an owner-specific reference count; another Comm's
registration stays valid when one owner deregisters or is destroyed.

Registration reserves a pending handle before calling the provider. Cleanup
marks a handle as closing, copies its identity and drops the metadata lock before
calling a native SDK or proxy. It then finds the handle again by identity under
the lock, because another registration operation may have compacted the shared vector.
Success removes the handle; failure retains it and its logical reference for
retry, and prevents reuse until closure succeeds. Provider calls and rank
rendezvous never run with the registry mutex held.

Finalize closes native registrations before shutting down their native contexts,
and closes P2P/NET registrations before stopping their proxies. Local Destroy
then drops that owner's remaining logical registration references without any
provider operation. Caller-owned registered buffers remain caller-owned.

## Regression coverage and CI placement

All coverage uses existing suites and platform capability boundaries.

| Existing suite | Coverage |
| --- | --- |
| `adaptor` | Native NCCL wrapper lifetime; compatibility finalization, native-close retry and one-time wrapper release for SDKs without Finalize; v1 descriptor layout and conservative scope |
| `core` | Admission, vendor progress, drain failure, nested finalization groups, caller TLS/FIFO order, independent communicator destruction, local retry, canonical pair shutdown order, P2P raw-base closure, FIFO local free, NET close/free failures and retained proxy pools; concurrent shared registration/cache cleanup, unlocked provider callbacks, snapshot pins, owner references and failed-close retry; retained initialization rollback and local Destroy retries without Finalize |
| `service` | Real bootstrap sockets, sparse rank groups, independent node progress, error convergence/retry, SHM close failure retention and independent mapping lifetime |
| `rma` | Partial CPU mapping ownership, unmap failure retention and device free retry; RMA startup plus rollback failures, retained initialization owners, transport release retry and one-time cleanup prefixes; existing worker drain/stop tests |
| `device_api_host` | Partial DevComm/DevMem cleanup, ownership retention, successful free prefixes and cached pointer retries |
| `runner` | Existing unit tests; real registry/cache cleanup through independent communicator lifetimes; real collective followed by rank-local destruction in normal, P2P, forced NET and PXN invocations |
| `symmem` | Native-only implicit Destroy while peers wait; policy agreement before allocation failure; empty ranks and independent node groups; canonical pair shutdown with real bootstrap; MPI error convergence with empty ranks; local free retry; destroy while peers wait in MPI; real IPC and supported VMM allocator matrices |
| `device_api` | Intra/inter workloads end with explicit Finalize and rank-local destruction; intra leaves a live DevComm for automatic cleanup |
| `device_api_unified_ir` | Unified intra/inter workloads exercise the same cleanup contract |
| Existing CUDA/Hygon experimental UniRunner jobs | Real IPC Ring AllReduce, retained internal exporters, cleared runner owners and rank-local Destroy |

The experimental UniRunner case requires named GoogleTest XML results,
preventing an empty filter or skipped case from passing.
Hardware execution is required to validate GPU IPC and vendor runtime behavior;
host mocks validate ownership transitions and failure retry semantics only.

## Direct Hetero callers

The lower-level API has Finalize and Destroy entry points. Hetero Finalize
stops admission, drains work, closes imports and confirms closure through its
node-local bootstrap group, including ranks with no local connections. It
retains every exporter until that confirmation succeeds. Destroy implicitly
finalizes an active communicator; after successful Finalize, Destroy and any
failed release retry perform only local cleanup. The outer Comm finalizer
uses its existing status protocol and records Hetero completion, so later
Hetero Destroy does not perform a second collective. Merely stopping local
proxy services or invoking a release callback cannot grant FIFO free permission.
Initialization rollback uses its separate local cleanup entry and cannot enter
normal Finalize. Direct callers must finalize shared communicators in compatible
order across their processes.

Direct and outer Finalize share the RMA/cache/PXN import-close function. A
successful rank keeps its service alive until the common relay-close status is
successful. A closed mapping and a completed release ACK have separate state:
a failed ACK can be retried without repeating the mapping close. Proxy Stop
rejects pending relay imports or ACKs before setting its stop flags.

The agreed IPC closure policy has a separate confirmed bit. A required,
confirmed policy is sticky for this communicator and subsequent ready/ring
storage reuse does not repeat its bootstrap declaration. An exchange failure
may conservatively require peer closure, but does not confirm the policy;
creation must still retry agreement before publishing resources. Existing
UniRunner invocation/resource consistency exchanges are unchanged.

## Public device handle cleanup

DeviceHandleInit returns the public function table inside a private owner that
tracks device/CCL plugin references. Free commits each released reference only
on success and retains the handle when cleanup fails. Retry Free on that same
handle; do not resume normal device work after cleanup has started. A successful
Free invalidates the handle. HandlerGroup Free propagates device-handle cleanup
errors and retains the enclosing group and its fields until cleanup succeeds.
The public function-table layout is unchanged. Owner allocation precedes plugin
reference acquisition; an unsuccessful init whose rollback also fails returns a
non-null cleanup owner, which must be released with the matching Free API.

## Graph capture and completion ownership

Ordinary host submissions carry completion events recorded at their actual
submission boundary (or the outermost SDK group end). Native graph capture
carries no additional FlagCX completion event. Capturing defines work rather
than executing it; the application must finish graph replays and prevent future
uses before Finalize, as with externally launched Device API kernels. Captured
markers are never queried, synchronized, or inserted into the event reuse pool
as proof that a graph execution has finished. Their POD records may be reused
without retaining any SDK graph object or recording an event into the graph.

A heterogeneous group establishes one completion stream after all operation
readiness dependencies and its proxy semaphore wait. Every affected communicator
retains its own event recorded at that completion boundary, including work
submitted on other streams. A partially published group without a confirmed
completion boundary retains uncertain records and cannot authorize cleanup.
Empty groups and failures before transport publication keep their original
submission boundaries. No event is appended later at Finalize.

Capture is enabled for native homogeneous operations without tuning. Captured
AllReduce bypasses custom callbacks and uses the native SDK. UniRunner, fused,
and direct Hetero stream submissions reject active capture before issuing work.
Changing capture status across a deferred group boundary retains an uncertain
owner and returns an error. Introspection errors are also propagated before
submission rather than treated as an inactive capture.

The frozen device-plugin v1 table is unchanged. Plugins used for tracked work
must provide the optional `flagcxDeviceAdaptorStreamCaptureStatus_v1` symbol:

```c
flagcxResult_t flagcxDeviceAdaptorStreamCaptureStatus_v1(
    flagcxStream_t stream, flagcxStreamCaptureStatus_t *status);
```

The plugin must unwrap its own opaque stream and return None, Active or
Invalidated. A missing callback returns NotSupported before tracked submissions;
the built-in device adaptor must never inspect a plugin's private stream layout.
Built-in adaptors resolve their runtime's capture introspection entry points;
a runtime exposing capture without introspection is rejected. Runtimes without
stream-capture entry points retain their ordinary stream operation path.

Fused AllGatherGemm and GemmReduceScatter agree on completion-marker preparation
before entering either implementation. Rank-local event creation, capture query
or pending preparation results participate in this bootstrap preflight, so all
ranks return the common failure before any workspace or kernel submission.
Successful markers on a failed preflight are canceled without SDK calls or a
new stream fence. Their records/events remain tracker-owned until local release,
including failed event destruction retries. Cancellation applies only to work
that has not been submitted; an error after entering the implementation still
uses the normal completion fence to retain any submitted prefix.

## FIFO work admitted before Finalize

Hetero host submissions and internal FIFO-derived proxy submissions use separate
completion trackers. Finalize closes host admission first. Accepted device FIFO
work can still submit internal Send/Recv operations while the FIFO drains; these
operations do not reopen host admission. After FIFO completion, Finalize seals
and waits for the proxy tracker, then quiesces RMA. Services stay alive throughout
this drain. Direct Hetero Finalize follows the same order. A terminal producer
error remains an error even when FIFO produced and completed counters are equal.
Both trackers retain event ownership on release failure for local Destroy retry.

## Optional host SDK owners

MPI and Gloo fallback communicators are destroyed whenever their hostComm owner
exists, including automatic fallback for singleton homogeneous clusters without
FLAGCX_USE_HOST_COMM. Staging allocations are associated with that host
communicator and are never reused or released by another communicator. Gloo
releases native unbound buffers before their backing allocations; pending native
operations retain their owner. Finalize can clear the native context while
retaining its wrapper, and Destroy frees the wrapper after successful cleanup.
Host Destroy pending/error results retain the outer communicator and its cursor.
The adaptor suite includes SDK-independent production-helper contracts and a
separate actual-MPI executable with required two-rank XML results.
