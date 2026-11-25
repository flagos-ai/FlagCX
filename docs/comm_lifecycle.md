# Communicator lifecycle

FlagCX supports one process per rank. Finalize and Destroy run on the calling
thread. There is no lifecycle worker or background allocation recycler.

`flagcxCommFinalize` closes submission admission, drains this communicator's
accepted work, closes internal imports and agrees closure through bootstrap.
It then finalizes each distinct native communicator, including tuner aliases
and an automatically created host communicator. Success is cached. New work
and new Device API owners are rejected once finalization begins.

`flagcxCommDestroy` first calls Finalize if needed. After successful Finalize,
FlagCX releases resources locally without another bootstrap exchange, relay RPC
or SHM peer-close wait. A failed free preserves its owner; retrying Destroy does
not repeat completed frees or native Finalize. A failed initialization with a
non-null output also retains an owner: Destroy routes it through initialization
rollback, rather than normal Finalize of a partially initialized communicator.

## Internal IPC resources

| Resource | Import closure | Allocation owner and release |
| --- | --- | --- |
| DevComm barrier, signal and staging buffers | DevComm/view cleanup | Parent Comm, after Finalize |
| UniRunner ready, ring scratch and progress | UniRunner views and IPC table | Parent Comm, local Destroy |
| P2P FIFO | Proxy import-close phase | Receiving transport, local Destroy after bootstrap permission |
| PXN relay pool | Import close plus existing release ACK | Relay transport, local Destroy |
| RMA completion counters | Stop worker/callbacks and CPU mappings | RMA proxy, local Release in Destroy |
| Cached device views and private scratch | Retire views after their recorded work | Deferred Comm allocation or private owner, local Destroy |

IPC participation is declared during initialization or collective resource
creation, before publication. Empty ranks participate in an agreed domain.
Pure native homogeneous communicators without FlagCX IPC resources have no
extra IPC shutdown exchange. IPC status agreement uses the node's rank list;
compatibility native close and host contexts can require agreement over their
FlagCX communicator. Existing cross-node relay protocols remain in place.

Each phase saves local completion separately from common completion. A local
success is not permission to free an exporter while another participant reports
failure. Close failures retain resources and retry the same common phase;
free failures after confirmation only need a local retry. A lost bootstrap
connection preserves ownership but can require job recovery.

Device API handles are local views. Their Destroy does not free the Comm's
internal exported backing. User-provided allocations remain user-owned.

## Native SDK boundaries

NCCL, RCCL, DU, Iluvatar, MetaX, TCCL and PPU retain their separate adaptor
Finalize/Destroy operations. Accepted native asynchronous Finalize is issued
once and progressed through GetAsyncError before Finalize completes.

HCCL, CNCL, ECCL, MUSA MCCL, legacy Sunrise PCCL and XCCL currently report
`flagcxNotSupported` from adaptor Finalize. Core explicitly selects compatibility
close for these built-in adaptors, invokes Destroy in the coordinated vendor
phase, and clears every consumed wrapper alias on strict success. Unknown
plugins and other errors do not trigger this fallback. This records current
adaptor integration, not a claim that every SDK version lacks Finalize.

Abort remains available after a failed Finalize. It serializes owner changes
with Finalize/Destroy and retains every handle on error or InProgress. A
successful native Abort clears the native context and retains its host wrapper;
local Destroy releases the wrapper without calling the SDK again. Registration
and device-view cleanup also skip the retired context. Heterogeneous Abort
remains unsupported.

Abort does not grant permission to free IPC exports. If peer import closure is
still pending, it returns InProgress with the communicator retained; the
participants must complete Finalize before retrying cleanup. Only success from
the public Abort consumes its handle, including through the NCCL wrapper.

Two CI exceptions remain:

- PPU keeps native `pcclCommFinalize` in Finalize and `pcclCommDestroy` in Destroy.
  Its current SDK can wait for peer Destroy even after Finalize. The runner's
  rank-local native Destroy case is excluded until the SDK boundary supports it.
- The current MUSA image lacks `mcclCommFinalize`. Compatibility native Destroy
  may drain another communicator's work. Two native stream-isolation cases are
  excluded; non-VMM RMA isolation and IPC cleanup remain enabled.

The native stream-isolation cases also require a separate native Finalize on
other compatibility adaptors. Ascend and Enflame declare these exclusions in
CI; the cases check the initialized communicator's compatibility policy before
submitting work for local runs and platforms without a CI entry. Public
Finalize/Destroy and mock IPC lifecycle coverage are retained.

Iluvatar CI remains enabled. Its previously reported initialization/communication
timeout is tracked separately and is not hidden with another exclusion.

Enflame's host Device API target remains disabled, including the fresh-process
non-VMM RMA invocation. Kunlunxin excludes only the three host-callback stream
isolation cases; its other host cleanup tests remain enabled. The fresh RMA
invocation respects the effective positive and negative GoogleTest filter.
These exclusions report missing coverage rather than successful execution.

## Work completion and Torch shutdown

Completion events and transport records come from the original submission
boundary. Finalize does not append an event to a user stream or synchronize the
whole device. Applications must complete directly launched Device API kernels
and captured graph replays before destroying their views or communicator.
Graph capture/replay is not tracked by this mechanism.

The Torch backend closes pair communicators in numeric `(min_rank, max_rank)`
order, then the process-group communicator, then events, owned streams and the
device handle. Borrowed Ascend streams keep their ownership rules.
`flagcx.shutdown_backend(backend)` provides an explicit retryable close while
the backend exists. Once shutdown starts, initialization and submission are
rejected, including barrier and coalesced operations, even if cleanup fails.
If destruction encounters a cleanup failure, a retained
cleanup object owns its remaining handles and tensor dependencies;
`flagcx.retry_pending_cleanup()` retries it on the caller's thread. The original
participants must enter any unfinished Finalize in compatible order. Completion
that cannot be proven remains retained and reports failure.
