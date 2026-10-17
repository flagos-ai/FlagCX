/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Symmetric memory helpers for the default (non-vendor) path.
 * Called from flagcx.cc when flagcxCommWindowRegister/Deregister/Grow
 * is invoked on the non-homo path with FLAGCX_WIN_COLL_SYMMETRIC.
 ************************************************************************/

#ifndef FLAGCX_SYM_HEAP_H_
#define FLAGCX_SYM_HEAP_H_

#include "comm.h"
#include "flagcx.h"
#include "flagcx_device_adaptor.h"

/* Concrete definition of the opaque flagcxWindow handle.
 * Internal only — external code must treat flagcxWindow_t as opaque. */
struct flagcxWindow {
  flagcxInnerWindow_t vendorBase; // vendor-specific window (NULL if no vendor)
  flagcxSymWindow_t
      defaultBase;        // default symmetric-heap state (NULL on vendor path)
  int isSymmetricDefault; // 1 if using default path, 0 if using vendor path
  int winFlags;           // flags passed at registration time
};

enum flagcxSymWindowState {
  flagcxSymWindowPreparing = 0,
  flagcxSymWindowPublished = 1,
  // Registration failed and rollback did not finish. The returned public
  // handle is a cleanup token and must only be passed to deregistration.
  flagcxSymWindowCleanupRequired = 2,
};

// Selects the callbacks that own physical and flat-mapping resources. This
// does not describe how many physical segments a user MR spans.
enum flagcxSymResourcePath {
  flagcxSymResourceNone = 0,
  flagcxSymResourceLegacyV1 = 1,
  flagcxSymResourceSegments = 2,
  flagcxSymResourceLsa = 3,
};

/* Symmetric window state for the default (non-vendor) path */
struct flagcxSymWindow {
  flagcxWindow_t owner; // owning public handle (for communicator cleanup)
  void *localBase;      // local user allocation backing this window
  void *flatBase;       // flat VA base (NULL if IPC fallback)
  void *mcBase;         // multicast base (NULL if no NVLS)
  size_t mcMapSize;     // multicast VA mapped size (for teardown)
  int mrIndex;          // one-sided MR index (-1 if none)
  uintptr_t mrBase;     // MR base VA
  bool hasNetworkMrRef; // this window retains mrIndex until deregistration
  // An unpublished MR transaction is retained in comm->pendingOneSideCleanup.
  // Keep this window and its allocation lease alive until that MR is released.
  bool hasPendingNetworkCleanup;
  size_t heapSize;      // user-requested size (for bounds info)
  size_t allocSize;     // actual physical allocation size per peer
                        // (granularity-aligned)
  size_t slotStride;    // common VA slot width for segmented flat mappings
  size_t slotAlignment; // common VA base alignment for segmented flat mappings
  // A window owns an offset in each communicator LSA rank slot, rather than
  // the VA reservation that contains all of those slots.
  size_t lsaWindowOffset;
  size_t lsaWindowSpan;
  bool lsaOffsetOwned;
  bool mcLsaBound;       // multicast binding into the communicator LSA object
  size_t mcLsaBindSize;  // provider granularity-aligned binding length
  bool lsaCleanupFailed; // failed mapping teardown may diverge rank allocators
  flagcxSymRankLayout *rankLayouts;
  flagcxSymSegmentMeta *segments;
  int totalSegments;
  int localRanks;   // number of intra-node peers
  int ipcSlot;      // IPC table slot for non-VMM peer mappings (-1 if none)
  void *physHandle; // for cleanup (symPhysFree)
  void *mcHandle;   // this rank's retained multicast handle reference
  // VMM teardown is split into independently retryable provider operations.
  // A successful unmap clears only mapping ownership; the base address stays
  // available until the subsequent VA-free operation succeeds.
  bool flatMappingOwned;
  bool flatVaOwned;
  bool multicastMappingOwned;
  bool multicastVaOwned;
  // Lease held in globalMemAllocRegistry when localBase belongs to
  // flagcxMemAlloc. It prevents callers from freeing memory still referenced
  // by a published window or cleanup token.
  bool hasAllocationLease;
  void *allocationBase;
  // Allocation provenance is independent of whether the optional flat peer
  // mapping succeeded. Network MR routing must use isVmmAllocation; local
  // peer-pointer resolution uses hasFlatMapping/flatBase.
  bool isVmmAllocation;
  flagcxSymResourcePath resourcePath;
  bool hasFlatMapping; // true while a VMM flat mapping is active
  bool published;      // linked into comm->symWindows only after full commit
  flagcxSymWindowState state;
  struct flagcxSymWindow *next; // intrusive link in comm->symWindows
  struct flagcxSymWindow
      *cleanupNext; // unpublished rollback retained for retry
};

// Public window deregistration is collective so every rank reports the same
// MR/cleanup result. Communicator destruction is deliberately rank-local and
// must not enter bootstrap rendezvous while another rank is still live.
enum flagcxSymCleanupMode {
  flagcxSymCleanupCollective = 0,
  flagcxSymCleanupLocal = 1,
};

flagcxResult_t flagcxSymWindowRegister(flagcxHeteroComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags);

// Internal entry point for owners that allocate outside flagcxMemAlloc and
// therefore know the allocation provenance explicitly.
flagcxResult_t flagcxSymWindowRegisterInternal(flagcxHeteroComm_t comm,
                                               void *buff, size_t size,
                                               flagcxWindow_t *win,
                                               int winFlags,
                                               bool isVmmAllocation);

// Legacy IPC export is valid only for ordinary device allocations. A VMM
// allocation whose optional flat mapping failed must use its network MR and
// must never be passed to the legacy IPC exporter.
static inline bool
flagcxSymWindowCanUseLegacyIpc(const struct flagcxSymWindow *window) {
  return window != nullptr && !window->isVmmAllocation &&
         !window->hasFlatMapping;
}

flagcxResult_t flagcxSymWindowDeregister(
    flagcxHeteroComm_t comm, flagcxWindow_t win,
    flagcxSymCleanupMode mode = flagcxSymCleanupCollective);

// Publish only after every collective setup phase, including IPC fallback,
// has converged successfully on all ranks.
flagcxResult_t flagcxSymWindowPublish(flagcxHeteroComm_t comm,
                                      flagcxWindow_t win);
// Non-failing local commit after a communicator-wide route/readiness preflight.
void flagcxSymWindowCommit(flagcxHeteroComm_t comm, flagcxWindow_t win);
flagcxResult_t flagcxSymRetryPendingCleanup(
    flagcxHeteroComm_t comm,
    flagcxSymCleanupMode mode = flagcxSymCleanupCollective);
flagcxResult_t flagcxSymRetainPendingCleanup(flagcxHeteroComm_t comm,
                                             flagcxWindow_t win);
flagcxResult_t flagcxSymConvergeStatus(flagcxHeteroComm_t comm,
                                       flagcxResult_t localStatus,
                                       flagcxResult_t *commonStatus);

// Validate exchanged rank geometry and calculate the common aligned slot
// width. The user range must be fully backed by mapped segments.
flagcxResult_t
flagcxSymValidateLayouts(int nRanks, const flagcxSymRankLayout *layouts,
                         int nSegments, const flagcxSymSegmentMeta *segments,
                         size_t slotAlignment, size_t *slotStride);

// Resolve a user range inside one window of a communicator-wide LSA arena.
// The host and Device API implementations must use the same layout formula.
flagcxResult_t flagcxSymLsaPeerPtr(void *arenaBase, size_t rankStride,
                                   size_t windowOffset, int localRanks,
                                   int peerLocalRank,
                                   const flagcxSymRankLayout *layout,
                                   size_t offset, size_t size, void **ptr);

// Called after all windows and pending cleanup tokens have been drained.
flagcxResult_t flagcxSymLsaDestroy(flagcxHeteroComm_t comm);

// Count the physical mappings touched by the user-visible range within a
// validated rank layout. Mapping the whole allocation may involve more.
flagcxResult_t
flagcxSymCountUserRangeSegments(const flagcxSymRankLayout *layout,
                                const flagcxSymSegmentMeta *rankSegments,
                                int *count);

// Acquire and attach a network MR reference to an unpublished symmetric
// window. The operation is collective and idempotent across retries.
flagcxResult_t flagcxSymWindowEnsureNetworkMr(flagcxHeteroComm_t comm,
                                              flagcxSymWindow_t window);

// A published window must provide a usable path to every peer: flat/IPC/NET
// for local peers when P2P is enabled, and NET for remote peers or whenever
// P2P is disabled.
flagcxResult_t flagcxSymWindowValidateDataRoutes(flagcxHeteroComm_t comm,
                                                 flagcxSymWindow_t window);

// Pure validation seam used by unit tests and by the runtime wrapper above.
// localPeerTransportEnabled controls whether flat/IPC routes are usable.
flagcxResult_t
flagcxSymWindowValidateDataRoutesForMode(flagcxHeteroComm_t comm,
                                         flagcxSymWindow_t window,
                                         bool localPeerTransportEnabled);

// Find the symmetric window containing [ptr, ptr + size) on the local rank.
// Returns NULL when the range is not owned by an active window.
flagcxSymWindow_t flagcxSymWindowFind(flagcxHeteroComm_t comm, const void *ptr,
                                      size_t size, size_t *offset);

// Resolve an intra-node peer pointer using only the window's IPC locator.
// Network MR registration is deliberately not consulted.
flagcxResult_t flagcxSymWindowResolveIpcPeerPtr(flagcxHeteroComm_t comm,
                                                flagcxSymWindow_t window,
                                                int peer, size_t offset,
                                                size_t size, void **ptr);

// Resolve the registered user range for a peer before a window-based RMA
// operation. Segmented local peers use their exchanged layouts; remote peers
// use the collective network MR metadata.
flagcxResult_t flagcxSymWindowPeerUserSize(flagcxHeteroComm_t comm,
                                           flagcxSymWindow_t window, int peer,
                                           size_t *userSize);

#endif // FLAGCX_SYM_HEAP_H_
