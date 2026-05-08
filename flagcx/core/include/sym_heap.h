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

/* Symmetric window state for the default (non-vendor) path */
struct flagcxSymWindow {
  void *flatBase;   // flat VA base (NULL if IPC fallback)
  void *mcBase;     // multicast base (NULL if no NVLS)
  size_t mcMapSize; // multicast VA mapped size (for teardown)
  int mrIndex;      // one-sided MR index (-1 if none)
  uintptr_t mrBase; // MR base VA
  size_t heapSize;  // user-requested size (for bounds info)
  size_t allocSize; // actual physical allocation size per peer
                    // (granularity-aligned)
  int localRanks;   // number of intra-node peers
  void *physHandle; // for cleanup (symPhysFree)
  void *mcHandle;   // multicast handle (for cleanup)
  bool isVMM;       // true if VMM path (false = IPC fallback)
};

flagcxResult_t flagcxSymWindowRegister(flagcxHeteroComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags);

flagcxResult_t flagcxSymWindowDeregister(flagcxHeteroComm_t comm,
                                         flagcxWindow_t win);

#endif // FLAGCX_SYM_HEAP_H_
