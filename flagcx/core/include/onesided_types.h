/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ONESIDED_TYPES_H_
#define FLAGCX_ONESIDED_TYPES_H_

#include <stddef.h>
#include <stdint.h>

struct flagcxSymWindow;
struct flagcxNetMrInfo;

// Transport-neutral one-sided registration metadata. Keep this definition in
// a lightweight header so net adaptors can validate ranges and keys without
// pulling communicator and proxy implementation headers into their common
// transport utilities.
struct flagcxOneSideHandleInfo {
  uintptr_t *baseVas;
  size_t *regionSizes;             // [nRanks]
  struct flagcxNetMrInfo *mrInfos; // [nRanks], including per-NIC keys
  void *localMrHandle;             // local rank's MR handle for deregMr
  void *localRecvComm; // recvComm used for MR registration (PD match)
  // Full-mesh RDMA connections (including self loopback, aligned with NCCL GIN)
  void **fullSendComms; // [nRanks] per-peer sendComm — alias for
                        // contextSendComms[0]
  void **fullRecvComms; // [nRanks] per-peer recvComm — alias for
                        // contextRecvComms[0]
  int nRanks;           // number of ranks (for cleanup iteration)

  // Per-context QP arrays for thread isolation (NCCL GIN pattern).
  // Context 0 = RMA proxy; contexts 1..N = kernel proxy threads.
  // Each context has its own full-mesh of RC QPs so no QP is shared
  // across threads. All contexts share the same MR handles/rkeys (same PD).
  void ***contextSendComms; // [nContexts][nRanks]
  void ***contextRecvComms; // [nContexts][nRanks]
  int nContexts;            // 1 + nKernelProxies

  // Symmetric memory window for intra-node D2D bypass (CE path).
  // NULL if VMM not available or window not registered with
  // FLAGCX_WIN_COLL_SYMMETRIC.
  struct flagcxSymWindow *symWin;
};

#endif // FLAGCX_ONESIDED_TYPES_H_
