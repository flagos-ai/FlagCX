# Communicator finalization and internal IPC ownership

`flagcxCommFinalize` stops new work, drains accepted work, closes imported IPC
mappings and confirms closure through the communicator's intra-node bootstrap
group. `flagcxCommDestroy` then releases the original exported allocations and
other local resources. A successfully finalized communicator needs no peer to
participate in its destruction.

## Ownership

| Resource | Owner | Release point |
| --- | --- | --- |
| DevComm barrier, signal and completion backing allocations | Comm internal buffer list | Local Destroy after successful Finalize |
| UniRunner IPC ready, scratch and progress backing allocations | Comm internal buffer list | Local Destroy after successful Finalize |
| IPC imports, device views and registrations | Their DevComm, DevMem, window or proxy connection | Local cleanup during Finalize |
| P2P FIFO and PXN NET relay exporters | Retained proxy connection pool | Local Destroy after successful Finalize |
| Public window backing allocation | Application | Application frees after all users release it |

`flagcxDevCommDestroy` closes the local imports and views immediately. It does
not free a backing allocation that another rank may still import. Repeated
creation and local destruction can therefore retain exported storage until the
parent Comm is finalized and destroyed. Finalize cleans up live Device API
owners automatically; their handles become invalid when cleanup succeeds.

## Protocol

The C-style lifecycle state holds the common phase, local completion bits and
last result. Each phase first performs its remaining local work and then
exchanges a status with every rank in the same node. Empty ranks participate.
A failed phase leaves the common cursor unchanged. Ranks whose local operation
succeeded skip that operation on retry, but participate in the status exchange
again.

1. Drain device work on user streams, fused buffers, device FIFOs and RMA
   producers while services and mappings are alive; then finalize vendor
   communicators. A native finalizer may stop a proxy still needed by a device
   kernel, so vendor finalization follows the device drain.
2. Clean up backend transport metadata.
3. Destroy local Device API owners and registrations, including partial-create
   owners and live windows.
4. Stop the RMA worker, remove cached P2P/NET registrations and close PXN relay
   imports while peer services can still acknowledge their release.
5. Stop and join proxy services and their workers; close P2P FIFO imports and
   UniRunner views while retaining the connection pool and exporters.
6. Close remaining IPC table entries and confirm completion locally in the node.

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

## Multiple ranks in one process

Comm initialization discovers same-process members within each communicator
clique using bootstrap metadata and creates a process-local C struct with a
mutex and an intrusive member list. Independent communicator cliques have
independent counters.

Sequential Destroy calls record one request per member. Earlier successful
calls invalidate their public handles but retain their internal owners. The
last request starts temporary finalization workers for all remaining members,
joins them, releases the other members and finally releases the last caller.
No reclamation thread survives the call. A thread creation gate prevents any
worker from entering a collective unless all workers were created.

If reclamation fails, the last caller remains valid for retry. Successfully
freed members have already been removed from the list. The final successful
Destroy returns only after every member and the clique metadata are released.
Destroy is not allowed inside a FlagCX group.

Explicit finalization of multiple ranks from one thread uses a group:

```cpp
flagcxGroupStart(NULL);
for (int i = 0; i < count; i++)
  flagcxCommFinalize(comms[i]);
flagcxGroupEnd(NULL); // runs and joins the queued finalizers
for (int i = 0; i < count; i++)
  flagcxCommDestroy(comms[i]);
```

Outside a group Finalize blocks. Nested groups defer work until the outermost
end. Check every return value in application code. Do not concurrently submit
work or start multiple lifecycle operations on the same handle. On a local
cleanup failure, retry Finalize with the node's ranks; after successful
Finalize, a failed free can be retried by that rank alone. Bootstrap or device
failure can prevent progress; owners remain retained while closure is
unconfirmed.

## Regression coverage and CI placement

All coverage uses existing suites and platform capability boundaries.

| Existing suite | Coverage |
| --- | --- |
| `core` | Admission, vendor progress, drain failure, nested finalization groups, independent process cliques, sequential requests, last-caller retry, worker creation gate, P2P raw-base closure, FIFO local free, NET close/free failures and retained proxy pools |
| `service` | Real bootstrap sockets, sparse rank groups, independent node progress, error convergence/retry, SHM close failure retention and independent mapping lifetime |
| `rma` | Partial CPU mapping ownership, unmap failure retention and device free retry, plus existing worker drain/stop tests |
| `device_api_host` | Partial DevComm/DevMem cleanup, ownership retention, successful free prefixes and cached pointer retries |
| `runner` | Existing unit tests; dedicated one-process/two-device implicit and grouped lifecycle cases; real collective followed by rank-local destruction in normal, P2P, forced NET and PXN invocations |
| `symmem` | Real bootstrap MPI error convergence with empty ranks; local free retry; destroy while peers wait in MPI; real IPC and supported VMM allocator matrices |
| `device_api` | Intra/inter workloads end with explicit Finalize and rank-local destruction; intra leaves a live DevComm for automatic cleanup |
| `device_api_unified_ir` | Unified intra/inter workloads exercise the same cleanup contract |
| Existing CUDA/Hygon experimental UniRunner jobs | Real IPC Ring AllReduce, retained internal exporters, cleared runner owners and rank-local Destroy |

The dedicated same-process tests and experimental UniRunner case require named
GoogleTest XML results, preventing an empty filter or skipped case from passing.
Hardware execution is required to validate GPU IPC and vendor runtime behavior;
host mocks validate ownership transitions and failure retry semantics only.
