/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Fallback flagcxSymWindow definition for non-vendor (non-NCCL) builds.
 * On the NVIDIA path, nvidia_adaptor.h defines FLAGCX_SYM_WINDOW_DEFINED
 * and provides a richer struct that includes ncclWindow_t.
 * This header must be included AFTER any vendor adaptor header.
 ************************************************************************/

#ifndef FLAGCX_SYM_WINDOW_FALLBACK_H_
#define FLAGCX_SYM_WINDOW_FALLBACK_H_

#ifndef FLAGCX_SYM_WINDOW_DEFINED
struct flagcxSymWindow {
  int winFlags;

  // Symmetric path fields
  void *flatBase;     // flat VA base (NULL if IPC fallback)
  void *mcBase;       // multicast base (NULL if no NVLS)
  void **devPeerPtrs; // device-side peer pointer array
  int mrIndex;        // one-sided MR index (-1 if none)
  uintptr_t mrBase;   // MR base VA
  size_t heapSize;    // currently backed size per peer
  size_t maxHeapSize; // total reserved VA per peer (for growth)
  int localRanks;     // number of intra-node peers
  void *physHandle;   // for cleanup (symPhysFree)
  void *mcHandle;     // multicast handle (for growth + cleanup)

  // Growth tracking
  void **growthPhysHandles;
  int growthCount;
  int growthCapacity;

  // Cleanup state
  bool isSymmetricFallback; // true if non-homo symmetric path
  bool isVMM;               // true if VMM path (false = IPC fallback)
};
#define FLAGCX_SYM_WINDOW_DEFINED
#endif // FLAGCX_SYM_WINDOW_DEFINED

#endif // FLAGCX_SYM_WINDOW_FALLBACK_H_
