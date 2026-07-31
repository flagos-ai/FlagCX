/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Transport-agnostic one-sided handle info and globals.
 * Moved from ib_common.h so that core layer files do not depend on
 * the IB adaptor header.
 ************************************************************************/

#ifndef FLAGCX_ONESIDED_H_
#define FLAGCX_ONESIDED_H_

#include <stdint.h>

#include "comm.h" // for flagcxHeteroComm_t

struct flagcxSymWindow; // forward declaration

struct flagcxOneSideHandleInfo {
  uintptr_t *baseVas;
  size_t regionSize; // size of the registered memory region (bytes)
  uint32_t *rkeys;
  uint32_t *lkeys;
  void *localMrHandle; // local rank's MR handle for deregMr
  void *localRecvComm; // recvComm used for MR registration (PD match)
  // Full-mesh IB connections (including self loopback, aligned with NCCL GIN)
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

  // IPC table slot index for signal buffer D2D bypass (-1 if none).
  int signalIpcSlot;
};

// Internal implementation used by sym_heap and flagcxCommRegister
flagcxResult_t flagcxOneSideRegisterInternal(flagcxHeteroComm_t comm,
                                             void *buff, size_t size);

// Build IPC peer pointer table for a user buffer (intra-node D2D bypass).
// Stores results in comm->ipcTable and returns the table index.
// Returns -1 on failure (IPC not available for this buffer).
struct flagcxComm;
int buildIpcPeerPointers(struct flagcxComm *comm, void *buff, size_t size);

#endif // FLAGCX_ONESIDED_H_
