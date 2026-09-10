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

/* Concrete definition of the opaque flagcxWindow handle.
 * Internal only — external code must treat flagcxWindow_t as opaque. */
struct flagcxWindow {
  flagcxInnerWindow_t vendorBase; // vendor-specific window (NULL if no vendor)
  flagcxSymWindow_t
      defaultBase;        // default symmetric-heap state (NULL on vendor path)
  int isSymmetricDefault; // 1 if using default path, 0 if using vendor path
  int winFlags;           // flags passed at registration time
};

/* Symmetric window state for the default (non-vendor) path */
struct flagcxSymWindow {
  void *localBase;  // local user allocation backing this window
  void *flatBase;   // flat VA base (NULL if IPC fallback)
  void *mcBase;     // multicast base (NULL if no NVLS)
  size_t mcMapSize; // multicast VA mapped size (for teardown)
  int mrIndex;      // one-sided MR index (-1 if none)
  uintptr_t mrBase; // MR base VA
  size_t heapSize;  // user-requested size (for bounds info)
  size_t allocSize; // actual physical allocation size per peer
                    // (granularity-aligned)
  int localRanks;   // number of intra-node peers
  int ipcSlot;      // IPC table slot for non-VMM peer mappings (-1 if none)
  void *physHandle; // for cleanup (symPhysFree)
  void *mcHandle;   // multicast handle (for cleanup)
  bool isVMM;       // true if VMM path (false = IPC fallback)
  struct flagcxSymWindow *next; // intrusive link in comm->symWindows
};

flagcxResult_t flagcxSymWindowRegister(flagcxHeteroComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags);

flagcxResult_t flagcxSymWindowDeregister(flagcxHeteroComm_t comm,
                                         flagcxWindow_t win);

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

#endif // FLAGCX_SYM_HEAP_H_
