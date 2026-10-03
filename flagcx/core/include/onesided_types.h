/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ONESIDED_TYPES_H_
#define FLAGCX_ONESIDED_TYPES_H_

#include <stddef.h>
#include <stdint.h>

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
  void *localRecvComm;       // recvComm used for MR registration (PD match)
  uint8_t registrationRoute; // flagcxVmmMrRoute_t used for this MR
  // Resolved READ/WRITE device-visibility requirements. This is intentionally
  // independent of registrationRoute: a native CQE has the same semantics for
  // ordinary, DMA-BUF, and VMM-backed device memory.
  uint32_t gdrFlushRequirements;
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

  // Ownership is released in dependency order: MR first, then metadata, then
  // connections. A failed deregMr leaves all fields intact for a later retry.
  uint32_t windowRefs; // symmetric windows currently retaining this MR
  uint8_t commOwned;   // public/comm registration retains MR to comm teardown
  uint8_t ownsLocalMr;
  uint8_t ownsConnections;
  struct flagcxOneSideHandleInfo *cleanupNext;
};

#endif // FLAGCX_ONESIDED_TYPES_H_
