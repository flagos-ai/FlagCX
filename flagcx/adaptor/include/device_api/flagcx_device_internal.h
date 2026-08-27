/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API Internal — Host-side internal structs.
 *
 * This header contains host-side backing structs (flagcxDevCommInternal,
 * flagcxDevMemInternal) that are populated by host code and passed to
 * device constructors.
 *
 * This header is NOT safe for LLVM bitcode compilation — it pulls in
 * host infrastructure headers (shmutils.h, flagcx_kernel.h).
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_INTERNAL_H_
#define FLAGCX_DEVICE_API_INTERNAL_H_

#include "flagcx_kernel.h"
#include "shmutils.h"
#include <pthread.h>

// Forward declaration for typed vendor device comm handle
struct flagcxInnerDevComm;
typedef struct flagcxInnerDevComm *flagcxInnerDevComm_t;
struct flagcxOneSideHandleInfo;

// ============================================================
// Section 1: flagcxDevCommInternal — Host-Side Opaque Handle
//
#define FLAGCX_MAX_INTER_PEERS 256

// Backing struct for flagcxDevComm_t (declared in flagcx_kernel.h).
// Populated by flagcxDevCommCreate, freed by flagcxDevCommDestroy.
// Unified capability-based design: baseline always populated,
// IPC and Vendor layers added when available.
// ============================================================
struct flagcxDevCommInternal {
  // ---- Baseline (always set) ----
  int rank, nRanks;
  int intraRank, intraSize;
  void *fifoBuffers[FLAGCX_DEVICE_CTA_COUNT]; // Device-accessible FIFOs (one
                                              // per context, from heteroComm,
                                              // may be null)
  // ---- IPC barrier layer (set if IPC barrier setup succeeds, else nullptr)
  // ----
  uint64_t *
      *barrierPeers; // device pointer to array of nLocalRanks device pointers
  uint64_t
      *localBarrierFlags; // this rank's inbox buffer (nLocalRanks × CTA_COUNT)
  uint64_t *epochBuffer;  // Device memory: per-CTA epoch counters
                          // Layout: [CTA_COUNT intra epochs, CTA_COUNT inter
                          // epochs]
  int nBarriers;          // = FLAGCX_DEVICE_CTA_COUNT (needed in kernel)
  // Host-side cleanup bookkeeping (not passed to kernel)
  bool localBarrierFlagsDeviceAllocated; // true only when localBarrierFlags is
                                         // owned device memory, not an SHM
                                         // device-visible alias
  int barrierIpcIndex;  // index into comm->ipcTable (-1 if no IPC barrier)
  int *localRankToRank; // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;
  // flagcxShm barrier path (non-null when FLAGCX_SIGNAL_HOST_ENABLE=1)
  void *localBarrierShmPtr;  // non-owning alias of this rank's entry in
                             // peerBarrierShmPtrs
  void **peerBarrierShmPtrs; // CPU VA array [nLocalRanks] for each peer's shm
                             // mapping; owns each successful host registration
  int registeredBarrierShmCount;     // registered prefix of peerBarrierShmPtrs
  size_t barrierShmSize;             // size in bytes (for hostUnregister)
  uint64_t **barrierDevPeerPtrsRaw;  // standalone device array (deviceFree; shm
                                     // path only)
  flagcxShmHandle_t myShmHandle;     // own shm handle (flagcxShmClose)
  flagcxShmHandle_t *peerShmHandles; // peer shm handles [nLocalRanks]

  // ---- Inter-node signal relay (set if nInterPeers > 0) ----
  int nInterPeers;     // number of inter-node peers (set on ALL ranks)
  int *interPeerRanks; // global ranks of inter-node peers
  // NCCL GIN-style barrier fields (all ranks)
  int teamRank;          // this rank's position in the inter-node team
  int nTeamRanks;        // total number of nodes (team size for barrier)
  int barrierSignalBase; // first signal slot index used for barriers

  // ---- One-sided Default layer (set if interSignalCount/interCounterCount >
  // 0)
  // ----
  uint64_t *signalBuffer; // GPU memory (flagcxMemAlloc), [signalCount] entries
  uint64_t
      *shadowBuffer; // GPU memory (local only, no MR), [signalCount] entries
  uint64_t
      *counterBuffer; // GPU memory (flagcxMemAlloc), [counterCount] entries
  int signalCount;
  int counterCount;
  int contextCount; // = reqs.interContextCount (default 4)
  // Host-only: communicator registrations installed for these buffers.
  // Non-null only when this DevComm created the registration and therefore
  // must deregister it before freeing the backing allocation.
  struct flagcxOneSideHandleInfo *ownedSignalRegistration;
  void *putValueStagingBuffer; // 8 bytes host-pinned, MR registered
  struct flagcxOneSideHandleInfo *ownedStagingRegistration;

  // One-sided transport readiness.  Signal send/wait must use the same
  // communicator-wide path because waits do not identify a sender.
  int netOneSidedReady;
  int netSignalReady;
  int netPutValueReady;
  int useP2pSignals;

  // ---- P2P signal IPC pointers (intra-node direct atomic path) ----
  uint64_t **signalPeerPtrs; // Device array: [localRanks] → peer signal bufs
  int signalIpcSlot; // IPC table slot for signal peer pointers (-1 if not used)

  // ---- Vendor device comm (set if adaptor->devCommCreate succeeds, else NULL)
  // ----
  flagcxInnerDevComm_t devComm; // Typed vendor handle (per-adaptor struct)

  // ---- Device pointer cache (for Triton integration) ----
  void *cachedDevicePtr;      // Lazily allocated by flagcxDevCommGetDevicePtr
  void *cachedNetContextsPtr; // Device memory for pre-allocated flagcxDevNet[]
  void *cachedGridBarrierPtr; // Device memory for grid sync state (2 x uint32)
  pthread_mutex_t cachedPtrMutex; // Protects lazy init of cachedDevicePtr and
                                  // cachedNetContextsPtr
};

// ============================================================
// Section 2: flagcxDevMemInternal — Host-Side Memory Handle
//
// Backing struct for flagcxDevMem_t.
// Created by flagcxDevMemCreate, freed by flagcxDevMemDestroy.
// Unified capability-based design: rawPtr always populated,
// IPC and Window layers added when available.
// Capabilities detected by Window mode:
//   SYMMETRIC + flatBasePtr  → VMM flat VA available
//   ASYMMETRIC + ipcBasePtrs → IPC peer pointers available
//   window != nullptr        → Window available (Vendor or default)
// ============================================================
struct flagcxDevMemInternal {
  // ---- Baseline (always set) ----
  void *rawPtr;   // = buff parameter
  bool hasWindow; // true if any window layer is available (basic or symmetric)
  bool isSymmetric; // true only for FLAGCX_WIN_COLL_SYMMETRIC (enables
                    // one-sided)

  // ---- Per-comm MR layer (set by flagcxDevMemCreate from handle table) ----
  int mrIndex; // index into heteroComm->oneSideHandles (-1 if not registered)
  uintptr_t mrBase; // handles[mrIndex]->baseVas[myRank] (cached for device)

  // ---- IPC layer (set if IPC exchange succeeds, else nullptr) ----
  int ipcIndex;  // index into comm->ipcTable (-1 if no IPC)
  int intraRank; // this rank's local rank index (for IPC local pointer)

  // ---- Window layer (opaque pointer to DeviceAPI::Window) ----
  void *window;    // Points to vendor Window or defaultDeviceImpl::Window
                   // (fallback)
  void *winHandle; // Host-side handle for cleanup

  // ---- Device pointer cache (for Triton integration) ----
  void *cachedDevicePtr; // Lazily allocated by flagcxDevMemGetDevicePtr
  pthread_mutex_t cachedPtrMutex; // Protects lazy init of cachedDevicePtr
};
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

#endif // FLAGCX_DEVICE_API_INTERNAL_H_
