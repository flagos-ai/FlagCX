/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t and flagcxDevMem_t.
 *
 * Capability-based additive design:
 *   Baseline (always): rawPtr + fifoBuffer + rank info
 *   IPC layer:         peer pointers + IPC barriers (if IPC exchange succeeds)
 *   Vendor layer:      vendor DevComm + vendor Window (if vendor supported)
 *
 * Each layer is added when available; lower layers are always present
 * as fallback. Kernel dispatch uses priority: Window > IPC > Raw.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "flagcx_kernel.h"
#include "net.h" // flagcxNetHandle_t
#include "onesided.h"
#include "p2p.h" // flagcxP2pAllocateShareableBuffer, flagcxP2pIpcDesc (+comm.h, transport.h)
#include "shmutils.h" // flagcxShmOpen, flagcxShmClose, flagcxShmUnlink
#include "utils.h"    // flagcxParamSignalHostEnable
#include <algorithm>  // std::min, std::max
#include <new>
#include <sched.h> // sched_yield

// ==========================================================================
// Shared: IPC peer pointer exchange (used by both tiers)
// ==========================================================================

// Build IPC peer pointer table for a user buffer.
// Stores results in comm->ipcTable and returns the table index.
// Returns -1 on failure (IPC not available for this buffer).
static int buildIpcPeerPointers(flagcxComm_t comm, void *buff, size_t size) {

  // Find a free slot in the IPC table
  int slot = -1;
  for (int k = 0; k < FLAGCX_MAX_IPC_ENTRIES; k++) {
    if (comm->ipcTable[k].hostPeerPtrs == nullptr &&
        comm->ipcTable[k].devPeerPtrs == nullptr) {
      slot = k;
      break;
    }
  }
  if (slot < 0) {
    WARN("buildIpcPeerPointers: IPC table full (max %d entries)",
         FLAGCX_MAX_IPC_ENTRIES);
    return -1;
  }

  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int localRanks = comm->localRanks;
  int *localRankToRank = comm->localRankToRank;

  flagcxResult_t res = flagcxSuccess;
  struct flagcxP2pIpcDesc *allDescs = nullptr;
  void **hostPeerPtrs = nullptr;
  void **devPeerPtrs = nullptr;

  // Step 1: Get IPC handle for existing user buffer
  struct flagcxP2pIpcDesc myIpcDesc;
  memset(&myIpcDesc, 0, sizeof(myIpcDesc));
  {
    flagcxIpcMemHandle_t handlePtr = nullptr;
    size_t ipcSize = 0;
    FLAGCXCHECKGOTO(deviceAdaptor->ipcMemHandleCreate(&handlePtr, &ipcSize),
                    res, fail);
    res = deviceAdaptor->ipcMemHandleGet(handlePtr, buff);
    if (res != flagcxSuccess) {
      deviceAdaptor->ipcMemHandleFree(handlePtr);
      goto fail;
    }
    if (ipcSize > sizeof(flagcxIpcHandleData)) {
      deviceAdaptor->ipcMemHandleFree(handlePtr);
      res = flagcxInternalError;
      goto fail;
    }
    memcpy(&myIpcDesc.handleData, handlePtr, ipcSize);
    myIpcDesc.size = size;
    deviceAdaptor->ipcMemHandleFree(handlePtr);
  }

  // Step 2: Exchange IPC handles with all ranks
  allDescs = (struct flagcxP2pIpcDesc *)calloc(nRanks,
                                               sizeof(struct flagcxP2pIpcDesc));
  if (allDescs == nullptr) {
    res = flagcxSystemError;
    goto fail;
  }
  memcpy(&allDescs[myRank], &myIpcDesc, sizeof(struct flagcxP2pIpcDesc));
  FLAGCXCHECKGOTO(bootstrapAllGather(comm->bootstrap, allDescs,
                                     sizeof(struct flagcxP2pIpcDesc)),
                  res, fail);

  // Step 3: Open intra-node peer IPC handles
  hostPeerPtrs = (void **)calloc(localRanks, sizeof(void *));
  if (hostPeerPtrs == nullptr) {
    res = flagcxSystemError;
    goto fail;
  }
  for (int lr = 0; lr < localRanks; lr++) {
    int gr = localRankToRank[lr];
    if (gr == myRank) {
      hostPeerPtrs[lr] = buff;
    } else {
      flagcxIpcMemHandle_t handlePtr =
          (flagcxIpcMemHandle_t)&allDescs[gr].handleData;
      FLAGCXCHECKGOTO(
          deviceAdaptor->ipcMemHandleOpen(handlePtr, &hostPeerPtrs[lr]), res,
          fail);
    }
  }
  free(allDescs);
  allDescs = nullptr;

  // Step 4: Build device peer pointer array
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc((void **)&devPeerPtrs,
                                              localRanks * sizeof(void *),
                                              flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMemcpy(
                      devPeerPtrs, hostPeerPtrs, localRanks * sizeof(void *),
                      flagcxMemcpyHostToDevice, NULL, NULL),
                  res, fail);

  // Store in comm->ipcTable
  comm->ipcTable[slot].hostPeerPtrs = hostPeerPtrs;
  comm->ipcTable[slot].devPeerPtrs = devPeerPtrs;
  comm->ipcTable[slot].nPeers = localRanks;
  comm->ipcTable[slot].basePtr = buff;
  comm->ipcTable[slot].inUse = true;

  INFO(FLAGCX_INIT,
       "buildIpcPeerPointers: rank %d slot %d buff=%p devPeerPtrs=%p", myRank,
       slot, buff, (void *)devPeerPtrs);
  for (int lr = 0; lr < localRanks; lr++) {
    INFO(FLAGCX_INIT, "buildIpcPeerPointers:   hostPeerPtrs[%d]=%p", lr,
         hostPeerPtrs[lr]);
  }
  return slot;

fail:
  free(allDescs);
  // On failure, clean up partially built resources directly
  if (hostPeerPtrs) {
    for (int i = 0; i < localRanks; i++) {
      if (hostPeerPtrs[i] && hostPeerPtrs[i] != buff) {
        deviceAdaptor->ipcMemHandleClose(hostPeerPtrs[i]);
      }
    }
    free(hostPeerPtrs);
  }
  if (devPeerPtrs) {
    deviceAdaptor->deviceFree(devPeerPtrs, flagcxMemDevice, NULL);
  }
  return -1;
}

// ==========================================================================
// Inter-node signal relay: one-sided RDMA atomic setup/teardown
//
// Each CTA writes BarrierSignal entries to the FIFO; the proxy fans out
// iputSignal (RDMA ATOMIC FETCH_AND_ADD) to each inter-node peer,
// directly incrementing the remote peer's interSignalFlagsHost counter.
// The GPU spins on the device pointer of interSignalFlags in
// flagcxInterBarrierSession::wait().
// No recv thread needed — the RDMA NIC atomically increments the counter.
// ==========================================================================

// Setup inter-node signal connections and barrier MR.
// Called from flagcxDevCommCreate when nNodes > 1.
// Lazy-init: on first call, establishes RDMA connections and stores them on
// heteroComm. Subsequent calls just copy the pointers into the new handle.
static flagcxResult_t setupInterNodeSignalRelay(flagcxComm_t comm,
                                                flagcxDevComm_t handle) {
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero == nullptr)
    return flagcxSuccess;

  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int myNode = hetero->node;
  int nNodes = hetero->nNodes;

  // Single-node: nothing to do
  if (nNodes <= 1)
    return flagcxSuccess;

  // Already initialized: just copy pointers into this handle
  if (hetero->relayInitialized) {
    handle->nInterPeers = hetero->nInterPeers;
    handle->isInterLeader = hetero->isInterLeader;
    handle->interPeerRanks = hetero->interPeerRanks;
    handle->interSignalFlags = hetero->interSignalFlags;
    handle->interSignalFlagsHost = hetero->interSignalFlagsHost;
    handle->signalSendComms = hetero->signalSendComms;
    handle->barrierRecvComms = hetero->barrierRecvComms;
    handle->barrierHandleInfo = hetero->barrierHandleInfo;
    handle->netAdaptorPtr = hetero->netAdaptorPtr;
    return flagcxSuccess;
  }

  // First call: establish connections and store on heteroComm.
  int *interPeerRanks = nullptr;
  int nInterPeers = 0;

  // Build list: for each remote node, find the global rank of its localRank 0
  for (int r = 0; r < nRanks; r++) {
    if (hetero->rankToNode[r] != myNode && hetero->rankToLocalRank[r] == 0) {
      nInterPeers++;
    }
  }

  if (nInterPeers == 0) {
    hetero->relayInitialized = true;
    return flagcxSuccess;
  }

  interPeerRanks = (int *)malloc(nInterPeers * sizeof(int));
  if (interPeerRanks == nullptr)
    return flagcxSystemError;

  int idx = 0;
  for (int r = 0; r < nRanks; r++) {
    if (hetero->rankToNode[r] != myNode && hetero->rankToLocalRank[r] == 0) {
      interPeerRanks[idx++] = r;
    }
  }

  // All ranks learn nInterPeers (needed for two-phase barrier logic).
  // Only localRank 0 (the inter leader) manages connections.
  hetero->nInterPeers = nInterPeers;
  hetero->interPeerRanks = interPeerRanks;
  hetero->isInterLeader = (hetero->localRank == 0);

  flagcxResult_t res = flagcxSuccess;
  size_t flagsSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);

  // ---- Leader-only: allocate flags and establish connections ----
  if (hetero->isInterLeader) {
    hetero->netAdaptorPtr = (void *)hetero->netAdaptor;

    // Step 1: Allocate host-mapped signal flags (GPU reads, RDMA NIC writes)
    FLAGCXCHECKGOTO(
        deviceAdaptor->deviceMalloc((void **)&hetero->interSignalFlagsHost,
                                    flagsSize, flagcxMemHost, NULL),
        res, fail);
    memset(hetero->interSignalFlagsHost, 0, flagsSize);
    FLAGCXCHECKGOTO(
        deviceAdaptor->hostGetDevicePointer((void **)&hetero->interSignalFlags,
                                            hetero->interSignalFlagsHost),
        res, fail);

    // Step 2: Establish netAdaptor connections with each inter-node peer.
    // Keep sendComms for iputSignal; keep ALL recvComms alive so that
    // peers' sendComm QPs remain connected (needed for incoming RDMA atomics).
    hetero->signalSendComms = (void **)calloc(nInterPeers, sizeof(void *));
    hetero->barrierRecvComms = (void **)calloc(nInterPeers, sizeof(void *));
    if (!hetero->signalSendComms || !hetero->barrierRecvComms) {
      res = flagcxSystemError;
      goto fail;
    }

    {
      struct bootstrapState *bootstrap = comm->bootstrap;
      int netDev = hetero->netDev;
      struct flagcxNetAdaptor *net = hetero->netAdaptor;
      const int signalTagBase = 2001;

      for (int p = 0; p < nInterPeers; p++) {
        int peer = interPeerRanks[p];
        int pairTag = signalTagBase + std::min(myRank, peer) * nRanks +
                      std::max(myRank, peer);

        // Listen for incoming connection from this peer
        flagcxNetHandle_t listenHandle = {};
        void *listenComm = nullptr;
        FLAGCXCHECKGOTO(net->listen(netDev, &listenHandle, &listenComm), res,
                        fail);

        // Exchange listen handles via bootstrap
        flagcxNetHandle_t peerHandle = {};
        FLAGCXCHECKGOTO(bootstrapSend(bootstrap, peer, pairTag, &listenHandle,
                                      sizeof(flagcxNetHandle_t)),
                        res, fail);
        FLAGCXCHECKGOTO(bootstrapRecv(bootstrap, peer, pairTag, &peerHandle,
                                      sizeof(flagcxNetHandle_t)),
                        res, fail);

        // Non-blocking connect/accept loop
        void *sendComm = nullptr;
        void *recvComm = nullptr;
        while (sendComm == nullptr || recvComm == nullptr) {
          if (sendComm == nullptr) {
            flagcxResult_t r = net->connect(netDev, &peerHandle, &sendComm);
            if (r != flagcxSuccess && r != flagcxInProgress) {
              res = r;
              goto fail;
            }
          }
          if (recvComm == nullptr) {
            flagcxResult_t r = net->accept(listenComm, &recvComm);
            if (r != flagcxSuccess && r != flagcxInProgress) {
              res = r;
              goto fail;
            }
          }
        }
        net->closeListen(listenComm);

        hetero->signalSendComms[p] = sendComm;
        hetero->barrierRecvComms[p] = recvComm;
      }
    }
  }

  // ---- ALL ranks: register barrier MR (collective AllGather inside) ----
  {
    struct flagcxOneSideHandleInfo *barrierInfo = nullptr;
    res = flagcxOneSideBarrierRegister(
        comm, hetero->isInterLeader ? hetero->barrierRecvComms[0] : nullptr,
        hetero->isInterLeader ? hetero->interSignalFlagsHost : nullptr,
        hetero->isInterLeader ? flagsSize : 0, &barrierInfo);
    if (res != flagcxSuccess) {
      WARN("setupInterNodeSignalRelay: barrier MR registration failed (%d)",
           res);
      goto fail;
    }
    if (hetero->isInterLeader) {
      hetero->barrierHandleInfo = barrierInfo;
    } else {
      // Non-leader participated in AllGather but doesn't need the result
      flagcxOneSideBarrierDeregister(comm, barrierInfo);
    }
  }

  hetero->relayInitialized = true;
  INFO(FLAGCX_INIT, "setupInterNodeSignalRelay: rank %d (%s), nInterPeers %d",
       myRank, hetero->isInterLeader ? "leader" : "non-leader", nInterPeers);

  // Copy into handle
  handle->nInterPeers = hetero->nInterPeers;
  handle->isInterLeader = hetero->isInterLeader;
  handle->interPeerRanks = hetero->interPeerRanks;
  handle->interSignalFlags = hetero->interSignalFlags;
  handle->interSignalFlagsHost = hetero->interSignalFlagsHost;
  handle->signalSendComms = hetero->signalSendComms;
  handle->barrierRecvComms = hetero->barrierRecvComms;
  handle->barrierHandleInfo = hetero->barrierHandleInfo;
  handle->netAdaptorPtr = hetero->netAdaptorPtr;
  return flagcxSuccess;

fail:
  // Clean up any partially-allocated relay state so that
  // flagcxCommRelayDestroy (which skips when relayInitialized==false) doesn't
  // leave dangling allocations.
  if (hetero->isInterLeader) {
    struct flagcxNetAdaptor *net = hetero->netAdaptor;
    if (hetero->signalSendComms) {
      for (int p = 0; p < nInterPeers; p++) {
        if (hetero->signalSendComms[p])
          net->closeSend(hetero->signalSendComms[p]);
      }
      free(hetero->signalSendComms);
      hetero->signalSendComms = nullptr;
    }
    if (hetero->barrierRecvComms) {
      for (int p = 0; p < nInterPeers; p++) {
        if (hetero->barrierRecvComms[p])
          net->closeRecv(hetero->barrierRecvComms[p]);
      }
      free(hetero->barrierRecvComms);
      hetero->barrierRecvComms = nullptr;
    }
    if (hetero->interSignalFlagsHost) {
      deviceAdaptor->deviceFree(hetero->interSignalFlagsHost, flagcxMemHost,
                                NULL);
      hetero->interSignalFlagsHost = nullptr;
      hetero->interSignalFlags = nullptr;
    }
  }
  free(hetero->interPeerRanks);
  hetero->interPeerRanks = nullptr;
  hetero->nInterPeers = 0;
  hetero->isInterLeader = false;
  return res;
}

#include "sym_heap.h"

// ==========================================================================
// Platform wrappers for host memory registration.
// Used only when FLAGCX_SIGNAL_HOST_ENABLE=1.
// Delegates to deviceAdaptor->hostRegister / hostUnregister.
// ==========================================================================
static flagcxResult_t shmHostRegister(void *ptr, size_t bytes) {
  if (deviceAdaptor->hostRegister == nullptr) {
    WARN("FLAGCX_SIGNAL_HOST_ENABLE=1: hostRegister not supported on this "
         "platform");
    return flagcxNotSupported;
  }
  return deviceAdaptor->hostRegister(ptr, bytes);
}

static void shmHostUnregister(void *ptr) {
  if (deviceAdaptor->hostUnregister)
    deviceAdaptor->hostUnregister(ptr);
}
// Allocates barrier flags, exchanges pointers with local peers, and builds
// a device-side pointer array. Two paths:
//   Default (FLAGCX_SIGNAL_HOST_ENABLE=0): IPC device memory via ipcTable.
//   Host-pinned (FLAGCX_SIGNAL_HOST_ENABLE=1): Shm + hostRegister.
//     Used on platforms (e.g. Hygon DCU) where GPU L2 cache is not flushed
//     mid-kernel for IPC-mapped peer device memory.
// On failure, partial resources are cleaned up by flagcxDevCommDestroy.
// ==========================================================================
static flagcxResult_t setupIpcBarriers(flagcxComm_t comm,
                                       flagcxDevComm_t handle) {
  int localRanks = comm->localRanks;
  int myRank = comm->rank;
  int myLocalRank = comm->localRank;

  handle->nLocalRanks = localRanks;
  handle->localRankToRank = (int *)malloc(localRanks * sizeof(int));
  if (handle->localRankToRank == nullptr)
    return flagcxSystemError;
  memcpy(handle->localRankToRank, comm->localRankToRank,
         localRanks * sizeof(int));

  size_t barrierSize = localRanks * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);

  if (flagcxParamSignalHostEnable() == 0) {
    // ── IPC device memory path (default) ─────────────────────────────────
    // Allocate device memory, exchange IPC handles, build peer ptr array.
    void *barrierFlags = nullptr;
    FLAGCXCHECK(deviceAdaptor->deviceMalloc(&barrierFlags, barrierSize,
                                            flagcxMemDevice, NULL));
    FLAGCXCHECK(deviceAdaptor->deviceMemset(barrierFlags, 0, barrierSize,
                                            flagcxMemDevice, NULL));
    handle->localBarrierFlags = (uint64_t *)barrierFlags;

    int slot = buildIpcPeerPointers(comm, barrierFlags, barrierSize);
    if (slot < 0) {
      deviceAdaptor->deviceFree(barrierFlags, flagcxMemDevice, NULL);
      handle->localBarrierFlags = nullptr;
      return flagcxSystemError;
    }

    handle->barrierPeers = (uint64_t **)comm->ipcTable[slot].devPeerPtrs;
    handle->barrierIpcIndex = slot;
    handle->localBarrierShmPtr = nullptr;
    handle->peerBarrierShmPtrs = nullptr;
    handle->barrierShmSize = 0;
    handle->barrierDevPeerPtrsRaw = nullptr; // tracked via ipcTable
    handle->nBarriers = FLAGCX_DEVICE_CTA_COUNT;

    INFO(FLAGCX_INIT,
         "setupIpcBarriers(IPC): rank %d slot %d localBarrierFlags=%p "
         "barrierPeers=%p nBarriers=%d",
         myRank, slot, barrierFlags, (void *)handle->barrierPeers,
         handle->nBarriers);

  } else {
    // ── flagcxShm + hipHostRegister path (FLAGCX_SIGNAL_HOST_ENABLE=1) ──
    // Each process creates its own shm segment via flagcxShmOpen, maps all
    // peers, registers them as pinned memory (bypasses GPU L2), then gets
    // per-process device VAs via hostGetDevicePointer.

    flagcxResult_t res = flagcxSuccess;
    void **peerCpuPtrs = nullptr;
    flagcxShmHandle_t *peerShmHandles = nullptr;
    void **devPeerPtrs = nullptr;
    void *myDevPtr = nullptr;
    void **hostDevPtrs = nullptr;

    // Build shm path from magic + global rank for uniqueness.
    char myShmPath[SHM_PATH_MAX];
    snprintf(myShmPath, sizeof(myShmPath), "/dev/shm/flagcx_barrier_%016llx_%d",
             (unsigned long long)comm->magic, myRank);

    // Step 1: Create and map own shm segment.
    flagcxShmHandle_t myShmHandle = nullptr;
    void *myCpuPtr = nullptr;
    FLAGCXCHECKGOTO(flagcxShmOpen(myShmPath, sizeof(myShmPath), barrierSize,
                                  &myCpuPtr, nullptr, localRanks, &myShmHandle),
                    res, fail_own_shm);

    // Step 2: Bootstrap barrier — all ranks have created their shm.
    FLAGCXCHECKGOTO(
        bootstrapBarrier(comm->bootstrap, comm->rank, comm->nranks, 0xBA01),
        res, fail_own_shm);

    // Step 3: Open and map each peer's shm segment.
    peerCpuPtrs = (void **)calloc(localRanks, sizeof(void *));
    peerShmHandles =
        (flagcxShmHandle_t *)calloc(localRanks, sizeof(flagcxShmHandle_t));
    if (peerCpuPtrs == nullptr || peerShmHandles == nullptr) {
      res = flagcxSystemError;
      goto fail_own_shm;
    }
    peerCpuPtrs[myLocalRank] = myCpuPtr;
    for (int lr = 0; lr < localRanks; lr++) {
      if (lr == myLocalRank)
        continue;
      char peerShmPath[SHM_PATH_MAX];
      snprintf(peerShmPath, sizeof(peerShmPath),
               "/dev/shm/flagcx_barrier_%016llx_%d",
               (unsigned long long)comm->magic, comm->localRankToRank[lr]);
      FLAGCXCHECKGOTO(flagcxShmOpen(peerShmPath, sizeof(peerShmPath),
                                    barrierSize, &peerCpuPtrs[lr], nullptr, -1,
                                    &peerShmHandles[lr]),
                      res, fail_peer_shms);
    }

    // Step 4: Bootstrap barrier — all ranks have opened peer shm.
    // Then unlink own shm: safe because peers already have it mapped.
    FLAGCXCHECKGOTO(
        bootstrapBarrier(comm->bootstrap, comm->rank, comm->nranks, 0xBA02),
        res, fail_peer_shms);
    flagcxShmUnlink(myShmHandle);

    // Step 5: Register own CPU mapping as pinned host memory.
    FLAGCXCHECKGOTO(shmHostRegister(myCpuPtr, barrierSize), res,
                    fail_peer_shms);
    FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(&myDevPtr, myCpuPtr),
                    res, fail_unreg_own);
    handle->localBarrierFlags = (uint64_t *)myDevPtr;

    // Step 6: Register each peer's mapping and collect device VAs.
    hostDevPtrs = (void **)calloc(localRanks, sizeof(void *));
    if (hostDevPtrs == nullptr) {
      res = flagcxSystemError;
      goto fail_unreg_own;
    }
    hostDevPtrs[myLocalRank] = myDevPtr;
    for (int lr = 0; lr < localRanks; lr++) {
      if (lr == myLocalRank)
        continue;
      FLAGCXCHECKGOTO(shmHostRegister(peerCpuPtrs[lr], barrierSize), res,
                      fail_unreg_peers);
      FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(&hostDevPtrs[lr],
                                                          peerCpuPtrs[lr]),
                      res, fail_unreg_peers);
    }

    // Step 7: Copy device VA array to device memory for kernel access.
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc((void **)&devPeerPtrs,
                                                localRanks * sizeof(void *),
                                                flagcxMemDevice, NULL),
                    res, fail_unreg_peers);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMemcpy(
                        devPeerPtrs, hostDevPtrs, localRanks * sizeof(void *),
                        flagcxMemcpyHostToDevice, NULL, NULL),
                    res, fail_free_devptrs);
    free(hostDevPtrs);
    hostDevPtrs = nullptr;

    handle->barrierPeers = (uint64_t **)devPeerPtrs;
    handle->barrierIpcIndex = -1;
    handle->localBarrierShmPtr = myCpuPtr;
    handle->peerBarrierShmPtrs = peerCpuPtrs;
    handle->barrierShmSize = barrierSize;
    handle->barrierDevPeerPtrsRaw = (uint64_t **)devPeerPtrs;
    handle->nBarriers = FLAGCX_DEVICE_CTA_COUNT;
    // Store shm handles for cleanup in flagcxDevCommDestroy
    handle->myShmHandle = myShmHandle;
    handle->peerShmHandles = peerShmHandles;

    INFO(FLAGCX_INIT,
         "setupIpcBarriers(shm): rank %d localBarrierFlags=%p "
         "barrierPeers=%p nBarriers=%d",
         myRank, (void *)handle->localBarrierFlags,
         (void *)handle->barrierPeers, handle->nBarriers);
    return flagcxSuccess;

  fail_free_devptrs:
    deviceAdaptor->deviceFree(devPeerPtrs, flagcxMemDevice, NULL);
  fail_unreg_peers:
    free(hostDevPtrs);
    for (int lr = 0; lr < localRanks; lr++) {
      if (lr != myLocalRank && peerCpuPtrs[lr])
        shmHostUnregister(peerCpuPtrs[lr]);
    }
  fail_unreg_own:
    shmHostUnregister(myCpuPtr);
  fail_peer_shms:
    for (int lr = 0; lr < localRanks; lr++) {
      if (lr != myLocalRank && peerShmHandles && peerShmHandles[lr])
        flagcxShmClose(peerShmHandles[lr]);
    }
    free(peerCpuPtrs);
    free(peerShmHandles);
  fail_own_shm:
    flagcxShmClose(myShmHandle);
    return res;
  }

  return flagcxSuccess;
}

// ==========================================================================
// Pre-establish full-mesh connections so the kernel proxy thread never needs
// to trigger lazy connection setup (which may cause hanging issues).
// Called from flagcxDevCommCreate — all ranks call it collectively, so the
// bootstrap rendezvous in flagcxTransportP2pSetup works correctly.
// ==========================================================================
flagcxResult_t preconnectFullMesh(flagcxComm_t comm) {
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero == nullptr)
    return flagcxSuccess;
  if (hetero->proxyState == nullptr || hetero->proxyState->initialized == 0)
    return flagcxSuccess;

  bool needPreconnect = false;
  int channelId = 0;
  for (int peer = 0; peer < hetero->nRanks; peer++) {
    if (peer == hetero->rank)
      continue;
    if (hetero->channels[channelId].peers[peer]->send[0].connected == 0 &&
        hetero->channels[channelId].peers[peer]->send[0].registered == 0) {
      hetero->connectSend[peer] |= (1UL << channelId);
      hetero->channels[channelId].peers[peer]->send[0].registered = 1;
      needPreconnect = true;
    }
    if (hetero->channels[channelId].peers[peer]->recv[0].connected == 0 &&
        hetero->channels[channelId].peers[peer]->recv[0].registered == 0) {
      hetero->connectRecv[peer] |= (1UL << channelId);
      hetero->channels[channelId].peers[peer]->recv[0].registered = 1;
      needPreconnect = true;
    }
  }

  if (needPreconnect) {
    INFO(FLAGCX_INIT, "preconnectFullMesh: rank %d establishing %d-peer mesh",
         hetero->rank, hetero->nRanks - 1);
    FLAGCXCHECK(flagcxTransportP2pSetup(hetero, NULL, 0));
  }
  return flagcxSuccess;
}

// ==========================================================================
// Unified DevComm: Additive capability layers
//   Baseline: rank info + fifoBuffer (always)
//   IPC layer: barrier pointers (if intraBarrierCount > 0)
//   Vendor layer: vendor DevComm (if vendor supported)
// ==========================================================================

flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm) {
  if (comm == nullptr || reqs == nullptr || devComm == nullptr) {
    return flagcxInvalidArgument;
  }

  // Allocate the opaque handle
  flagcxDevComm_t handle =
      (flagcxDevComm_t)malloc(sizeof(struct flagcxDevCommInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevCommInternal));
  handle->barrierIpcIndex = -1;

  // ---- Baseline: always ----
  handle->rank = comm->rank;
  handle->nRanks = comm->nranks;
  handle->intraRank = comm->localRank;
  handle->intraSize = comm->localRanks;
  {
    int ctxCount = (reqs->interContextCount > 0) ? reqs->interContextCount : 1;
    if (comm->heteroComm != nullptr) {
      int available = comm->heteroComm->proxyState->kernelState.contextCount;
      if (available > 0 && ctxCount > available)
        ctxCount = available;
    }
    handle->contextCount = ctxCount;
    for (int i = 0; i < ctxCount; i++) {
      handle->fifoBuffers[i] = (comm->heteroComm != nullptr)
                                   ? comm->heteroComm->fifoBuffers[i]
                                   : nullptr;
    }
  }

  // ---- Vendor path: try devCommCreate via adaptor ----
  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm != nullptr &&
      cclAdaptors[flagcxCCLAdaptorDevice]->devCommCreate != NULL) {
    flagcxInnerDevComm_t innerDevComm = nullptr;
    flagcxResult_t ret = cclAdaptors[flagcxCCLAdaptorDevice]->devCommCreate(
        innerComm, reqs, &innerDevComm);
    if (ret != flagcxSuccess && ret != flagcxNotSupported &&
        ret != flagcxInvalidArgument) {
      WARN("flagcxDevCommCreate: vendor devCommCreate failed (%d)", ret);
      free(handle);
      return ret;
    }
    if (ret == flagcxSuccess)
      handle->devComm = innerDevComm;
  }
  if (handle->devComm != nullptr) {
    int nNodes = 0;
    if (comm->heteroComm != nullptr && comm->heteroComm->nNodes > 0) {
      nNodes = comm->heteroComm->nNodes;
    } else if (handle->intraSize > 0 &&
               handle->nRanks % handle->intraSize == 0) {
      nNodes = handle->nRanks / handle->intraSize;
    }
    if (nNodes > 0)
      handle->nInterPeers = nNodes - 1;
  }
  if (handle->devComm == nullptr) {
    // ---- Default path: IPC barriers + inter-node signal relay + one-sided
    // ----

    // IPC barrier layer: if barriers requested
    if (reqs->intraBarrierCount > 0 || reqs->interBarrierCount > 0) {
      flagcxResult_t res = setupIpcBarriers(comm, handle);
      if (res != flagcxSuccess) {
        WARN("flagcxDevCommCreate: IPC barrier setup failed (%d), "
             "barriers unavailable",
             res);
        free(handle);
        return res;
      }
    }

    // Inter-node signal relay: if multi-node
    {
      flagcxResult_t res = setupInterNodeSignalRelay(comm, handle);
      if (res != flagcxSuccess) {
        WARN("flagcxDevCommCreate: inter-node signal relay setup failed (%d), "
             "falling back to single-node mode",
             res);
        handle->nInterPeers = 0;
        handle->isInterLeader = false;
      }
    }

    // Reset inter-node barrier signal flags so that the new DevComm's
    // interBarrierEpoch (starting at 0) doesn't see stale values from a
    // previous DevComm, which would cause barriers to pass without waiting
    // for the peer.
    if (comm->heteroComm != nullptr &&
        comm->heteroComm->interSignalFlagsHost != nullptr) {
      size_t flagsSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
      memset(comm->heteroComm->interSignalFlagsHost, 0, flagsSize);
    }

    // One-sided Default layer: if signals or counters requested
    if (reqs->interSignalCount > 0 || reqs->interCounterCount > 0) {
      // Use nKernelProxies (max contextCount) for buffer sizing so that
      // the RDMA MR covers all possible context slots across DevComm
      // re-creations with different contextCount values.
      int bufCtxCount =
          (comm->heteroComm != nullptr)
              ? comm->heteroComm->proxyState->kernelState.contextCount
              : handle->contextCount;
      if (bufCtxCount < handle->contextCount)
        bufCtxCount = handle->contextCount;

      // Allocate signal buffer (host-pinned or GDR device memory)
      if (reqs->interSignalCount > 0) {
        handle->signalCount = reqs->interSignalCount;
        size_t sigSize =
            (size_t)handle->signalCount * bufCtxCount * sizeof(uint64_t);
        if (flagcxParamSignalHostEnable()) {
          FLAGCXCHECK(deviceAdaptor->deviceMalloc(
              (void **)&handle->signalBuffer, sigSize, flagcxMemHost, NULL));
          memset(handle->signalBuffer, 0, sigSize);
        } else {
          FLAGCXCHECK(deviceAdaptor->gdrMemAlloc((void **)&handle->signalBuffer,
                                                 sigSize, NULL));
          FLAGCXCHECK(deviceAdaptor->deviceMemset(
              handle->signalBuffer, 0, sigSize, flagcxMemDevice, NULL));
        }
        FLAGCXCHECK(deviceAdaptor->deviceMalloc(
            (void **)&handle->shadowBuffer, sigSize, flagcxMemDevice, NULL));
        FLAGCXCHECK(deviceAdaptor->deviceMemset(
            handle->shadowBuffer, 0, sigSize, flagcxMemDevice, NULL));
      }
      // Allocate counter buffer (host-pinned)
      if (reqs->interCounterCount > 0) {
        handle->counterCount = reqs->interCounterCount;
        size_t cntSize =
            (size_t)handle->counterCount * bufCtxCount * sizeof(uint64_t);
        FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&handle->counterBuffer,
                                                cntSize, flagcxMemHost, NULL));
        memset(handle->counterBuffer, 0, cntSize);
      }
      // PutValue staging buffer (8 bytes host-pinned)
      FLAGCXCHECK(
          deviceAdaptor->deviceMalloc((void **)&handle->putValueStagingBuffer,
                                      sizeof(uint64_t), flagcxMemHost, NULL));
      memset(handle->putValueStagingBuffer, 0, sizeof(uint64_t));

      // Auto-register signal buffer for RDMA one-sided access
      if (handle->signalBuffer) {
        int sigPtrType =
            flagcxParamSignalHostEnable() ? FLAGCX_PTR_HOST : FLAGCX_PTR_CUDA;
        flagcxResult_t regRes = flagcxOneSideSignalRegister(
            comm, handle->signalBuffer,
            (size_t)handle->signalCount * bufCtxCount * sizeof(uint64_t),
            sigPtrType);
        if (regRes != flagcxSuccess) {
          WARN(
              "flagcxDevCommCreate: signal buffer MR registration failed (%d), "
              "one-sided operations will not work",
              regRes);
          return regRes;
        }
      }

      // Auto-register staging buffer for PutValue RDMA source
      if (handle->putValueStagingBuffer) {
        flagcxResult_t regRes = flagcxOneSideStagingRegister(
            comm, handle->putValueStagingBuffer, sizeof(uint64_t));
        if (regRes != flagcxSuccess) {
          WARN("flagcxDevCommCreate: staging buffer MR registration failed "
               "(%d), "
               "putValue will not work",
               regRes);
        }
      }

      INFO(FLAGCX_INIT,
           "flagcxDevCommCreate: one-sided Default buffers allocated "
           "(signals=%d, counters=%d, contexts=%d)",
           handle->signalCount, handle->counterCount, handle->contextCount);
    }
  }

  *devComm = handle;

  // Publish to heteroComm so proxy thread can access this DevComm
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero != nullptr) {
    hetero->devCommHandle = handle;
  }

  INFO(FLAGCX_INIT, "flagcxDevCommCreate: rank %d, layers: baseline%s%s%s%s",
       handle->rank, handle->devComm ? " + vendor devComm" : "",
       handle->barrierPeers ? " + IPC barriers" : "",
       handle->nInterPeers > 0 ? " + inter-node signal relay" : "",
       (handle->signalCount > 0 || handle->counterCount > 0)
           ? " + one-sided Default"
           : "");

  // Pre-establish full-mesh connections from main thread
  FLAGCXCHECK(preconnectFullMesh(comm));

  return flagcxSuccess;
}

flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                    flagcxDevComm_t devComm) {
  if (devComm == nullptr) {
    return flagcxSuccess;
  }

  // Vendor layer cleanup via adaptor
  if (comm != nullptr && devComm->devComm != nullptr) {
    flagcxInnerComm_t innerComm = comm->homoComm;
    if (innerComm != nullptr &&
        cclAdaptors[flagcxCCLAdaptorDevice]->devCommDestroy != NULL) {
      cclAdaptors[flagcxCCLAdaptorDevice]->devCommDestroy(innerComm,
                                                          devComm->devComm);
      devComm->devComm = nullptr;
    }
  }

  // IPC barrier cleanup — mark ipcTable entry as unused.
  // Actual ipcMemHandleClose + deviceFree deferred to flagcxCommCleanupIpcTable
  // (after vendor comm destroy) to avoid implicit device synchronization
  // deadlock.
  if (comm != nullptr && devComm->barrierIpcIndex >= 0 &&
      devComm->barrierIpcIndex < FLAGCX_MAX_IPC_ENTRIES) {
    comm->ipcTable[devComm->barrierIpcIndex].inUse = false;
  }
  // IPC path: localBarrierFlags is device-allocated (tracked via ipcTable).
  // shm path: localBarrierFlags is a device VA alias of localBarrierShmPtr;
  //   do NOT free it — the CPU mapping is freed below via shmHostUnregister.
  if (devComm->localBarrierShmPtr == nullptr && devComm->localBarrierFlags &&
      devComm->barrierIpcIndex < 0) {
    // Standalone device allocation (not tracked via ipcTable): free directly.
    flagcxCommDeferFree(comm, devComm->localBarrierFlags, flagcxMemDevice);
  }
  // flagcxShm path cleanup: unregister and close all shm mappings.
  if (devComm->localBarrierShmPtr) {
    shmHostUnregister(devComm->localBarrierShmPtr);
    if (devComm->myShmHandle)
      flagcxShmClose(devComm->myShmHandle);
  }
  if (devComm->peerBarrierShmPtrs) {
    for (int lr = 0; lr < devComm->nLocalRanks; lr++) {
      void *p = devComm->peerBarrierShmPtrs[lr];
      if (p && p != devComm->localBarrierShmPtr) {
        shmHostUnregister(p);
        if (devComm->peerShmHandles && devComm->peerShmHandles[lr])
          flagcxShmClose(devComm->peerShmHandles[lr]);
      }
    }
    free(devComm->peerBarrierShmPtrs);
    free(devComm->peerShmHandles);
  }
  if (devComm->barrierDevPeerPtrsRaw) {
    // shm path: standalone device pointer array (not in ipcTable).
    flagcxCommDeferFree(comm, devComm->barrierDevPeerPtrsRaw, flagcxMemDevice);
  }

  // Clear heteroComm->devComm + deregister signal buffer
  if (comm != nullptr && comm->heteroComm != nullptr &&
      comm->heteroComm->devCommHandle == devComm) {
    if (devComm->signalBuffer) {
      flagcxOneSideSignalDeregister(comm->heteroComm);
    }
    comm->heteroComm->devCommHandle = nullptr;
  }

  // One-sided Default cleanup
  if (devComm->signalBuffer) {
    flagcxMemType_t sigMt =
        flagcxParamSignalHostEnable() ? flagcxMemHost : flagcxMemDevice;
    flagcxCommDeferFree(comm, devComm->signalBuffer, sigMt);
  }
  if (devComm->shadowBuffer) {
    flagcxCommDeferFree(comm, devComm->shadowBuffer, flagcxMemDevice);
  }
  if (devComm->counterBuffer) {
    flagcxCommDeferFree(comm, devComm->counterBuffer, flagcxMemHost);
  }
  if (devComm->putValueStagingBuffer) {
    flagcxOneSideStagingDeregister(comm);
    flagcxCommDeferFree(comm, devComm->putValueStagingBuffer, flagcxMemHost);
  }

  free(devComm->localRankToRank);
  free(devComm);
  return flagcxSuccess;
}

// ==========================================================================
// Unified DevMem: Additive capability layers
//   Baseline: rawPtr (always)
//   IPC layer: peer pointers (if comm provided and win is null)
//   Window layer: vendor or default Window
// ==========================================================================

flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff, size_t size,
                                  flagcxWindow_t win, flagcxDevMem_t *devMem) {
  if (buff == nullptr || size == 0 || devMem == nullptr) {
    return flagcxInvalidArgument;
  }

  flagcxDevMem_t handle =
      (flagcxDevMem_t)malloc(sizeof(struct flagcxDevMemInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevMemInternal));

  // ---- Baseline: always ----
  handle->rawPtr = buff;
  handle->ipcIndex = -1;

  // ---- Per-comm MR layer: lookup buff in heteroComm->oneSideHandles ----
  handle->mrIndex = -1;
  handle->mrBase = 0;
  if (comm != nullptr && comm->heteroComm != nullptr) {
    struct flagcxHeteroComm *hc = comm->heteroComm;
    for (int i = 0; i < hc->oneSideHandleCount; i++) {
      struct flagcxOneSideHandleInfo *info = hc->oneSideHandles[i];
      if (info != NULL && info->baseVas != NULL) {
        uintptr_t base = info->baseVas[comm->rank];
        if ((uintptr_t)buff == base) {
          handle->mrIndex = i;
          handle->mrBase = base;
          INFO(FLAGCX_INIT,
               "flagcxDevMemCreate: buff %p matched oneSideHandles[%d], "
               "mrBase=0x%lx",
               buff, i, (unsigned long)base);
          break;
        }
      }
    }
  }

  if (comm != nullptr) {
    handle->intraRank = comm->localRank;

    // ---- Symmetric default window ----
    if (win != nullptr && win->isSymmetricDefault) {
      flagcxSymWindow_t d = win->defaultBase;
      handle->hasWindow = true;
      handle->isSymmetric = true;
      handle->winHandle = (void *)win;
      if (d != nullptr && d->mrIndex >= 0) {
        handle->mrIndex = d->mrIndex;
        handle->mrBase = d->mrBase;
      }
      // If VMM failed, fall back to IPC for peer access
      if (d != nullptr && !d->isVMM) {
        int idx = buildIpcPeerPointers(comm, buff, size);
        if (idx >= 0) {
          handle->ipcIndex = idx;
        } else {
          WARN("flagcxDevMemCreate: symmetric window VMM failed and IPC "
               "fallback also failed — no peer access");
        }
      }
      // window pointer will be set in the default block below
      handle->window = nullptr;
    }
    // ---- IPC layer: try if win is null (IPC needs cudaMalloc memory) ----
    else if (win == nullptr) {
      int idx = buildIpcPeerPointers(comm, buff, size);
      if (idx >= 0) {
        handle->ipcIndex = idx;
      } else {
        WARN("flagcxDevMemCreate: IPC peer pointer setup failed, "
             "IPC layer not available");
      }
    }

    // ---- Window layer: if win provided and valid (vendor path) ----
    if (win != nullptr && !win->isSymmetricDefault) {
      handle->hasWindow = true;
      handle->isSymmetric = (win->vendorBase && (win->vendorBase->winFlags &
                                                 FLAGCX_WIN_COLL_SYMMETRIC));
      handle->winHandle = (void *)win;
    }
  }

  // Allocate and populate kernel Window uniformly via traits
  {
    auto *kWin = new (std::nothrow) typename DeviceAPI::Window{};
    if (kWin == nullptr) {
      WARN("flagcxDevMemCreate: failed to allocate DeviceAPI::Window");
      free(handle);
      return flagcxSystemError;
    }
    kWin->populateFromHost(win, handle->rawPtr, handle->intraRank,
                           handle->mrIndex, handle->mrBase, handle->ipcIndex,
                           (handle->ipcIndex >= 0 && comm)
                               ? comm->ipcTable[handle->ipcIndex].devPeerPtrs
                               : nullptr);
    handle->window = kWin;
    handle->hasWindow = kWin->hasAccess() || (handle->rawPtr != nullptr);
  }

  *devMem = handle;
  const char *modeStr = "";
  if (handle->hasWindow) {
    if (handle->isSymmetric) {
      modeStr = " + Window (SYMMETRIC)";
    } else {
      modeStr = " + Window (ASYMMETRIC/IPC)";
    }
  } else if (handle->rawPtr) {
    modeStr = " + Window (raw-only)";
  }
  INFO(FLAGCX_INIT, "flagcxDevMemCreate: ptr %p, layers: rawPtr%s", buff,
       modeStr);
  return flagcxSuccess;
}

flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm, flagcxDevMem_t devMem) {
  if (devMem == nullptr) {
    return flagcxSuccess;
  }

  // Mark IPC table entry as no longer in use (actual cleanup deferred to
  // flagcxCommDestroy.
  if (comm != nullptr && devMem->ipcIndex >= 0 &&
      devMem->ipcIndex < FLAGCX_MAX_IPC_ENTRIES) {
    comm->ipcTable[devMem->ipcIndex].inUse = false;
  }

  // Free window allocation if present
  if (devMem->window != nullptr) {
    delete static_cast<typename DeviceAPI::Window *>(devMem->window);
  }

  free(devMem);
  return flagcxSuccess;
}

// ==========================================================================
// IPC table cleanup — called from flagcxCommDestroy
// ==========================================================================

flagcxResult_t flagcxCommCleanupIpcTable(flagcxComm_t comm) {
  if (comm == nullptr) {
    return flagcxSuccess;
  }

  for (int k = 0; k < FLAGCX_MAX_IPC_ENTRIES; k++) {
    struct flagcxIpcTableEntry *e = &comm->ipcTable[k];
    if (e->hostPeerPtrs == nullptr && e->devPeerPtrs == nullptr) {
      continue; // empty slot
    }

    if (e->inUse) {
      WARN("flagcxCommCleanupIpcTable: entry %d still in use — "
           "flagcxDevMemDestroy should be called before flagcxCommDestroy",
           k);
    }

    // Close IPC handles
    if (e->hostPeerPtrs) {
      for (int i = 0; i < e->nPeers; i++) {
        if (e->hostPeerPtrs[i] && e->hostPeerPtrs[i] != e->basePtr) {
          deviceAdaptor->ipcMemHandleClose(e->hostPeerPtrs[i]);
        }
      }
      free(e->hostPeerPtrs);
      e->hostPeerPtrs = nullptr;
    }

    // Free device memory safely
    if (e->devPeerPtrs) {
      deviceAdaptor->deviceFree(e->devPeerPtrs, flagcxMemDevice, NULL);
      e->devPeerPtrs = nullptr;
    }

    e->inUse = false;
  }

  return flagcxSuccess;
}

// ==========================================================================
// Deferred device/host-pinned memory free.
// ==========================================================================
void flagcxCommDeferFree(flagcxComm_t comm, void *ptr, int memType) {
  if (comm == nullptr || ptr == nullptr)
    return;
  if (comm->deferredFreeCount >= FLAGCX_MAX_DEFERRED_FREES) {
    WARN("flagcxCommDeferFree: deferred free list full (%d), freeing now",
         FLAGCX_MAX_DEFERRED_FREES);
    deviceAdaptor->deviceFree(ptr, (flagcxMemType_t)memType, NULL);
    return;
  }
  comm->deferredFrees[comm->deferredFreeCount].ptr = ptr;
  comm->deferredFrees[comm->deferredFreeCount].memType = memType;
  comm->deferredFreeCount++;
}

flagcxResult_t flagcxCommDrainDeferredFrees(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  for (int i = 0; i < comm->deferredFreeCount; i++) {
    struct flagcxDeferredFree *d = &comm->deferredFrees[i];
    if (d->ptr) {
      deviceAdaptor->deviceFree(d->ptr, (flagcxMemType_t)d->memType, NULL);
      d->ptr = nullptr;
    }
  }
  comm->deferredFreeCount = 0;
  return flagcxSuccess;
}

flagcxResult_t flagcxCommRelayDestroy(flagcxComm_t comm) {
  if (comm == nullptr || comm->heteroComm == nullptr)
    return flagcxSuccess;
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (!hetero->relayInitialized || hetero->nInterPeers == 0)
    return flagcxSuccess;

  // Drain all FIFOs before closing RDMA connections
  if (hetero->proxyState) {
    int ctxCount = hetero->proxyState->kernelState.contextCount;
    for (int i = 0; i < ctxCount; i++) {
      if (hetero->proxyState->kernelState.fifos[i]) {
        volatile uint64_t *buf =
            (volatile uint64_t *)hetero->proxyState->kernelState.fifos[i]
                ->buffer;
        if (buf) {
          while (buf[flagcxFifoIdxConsumed] < buf[flagcxFifoIdxProduced])
            sched_yield();
        }
      }
    }
  }

  // Cross-rank barrier: all ranks drain before any rank closes connections
  bootstrapBarrier(comm->bootstrap, comm->rank, comm->nranks, 0x7f01);

  free(hetero->interPeerRanks);
  hetero->interPeerRanks = nullptr;

  if (!hetero->isInterLeader) {
    hetero->relayInitialized = false;
    return flagcxSuccess;
  }

  struct flagcxNetAdaptor *net =
      (struct flagcxNetAdaptor *)hetero->netAdaptorPtr;

  if (hetero->barrierHandleInfo) {
    flagcxOneSideBarrierDeregister(
        comm, (struct flagcxOneSideHandleInfo *)hetero->barrierHandleInfo);
    hetero->barrierHandleInfo = nullptr;
  }
  if (hetero->signalSendComms) {
    for (int p = 0; p < hetero->nInterPeers; p++)
      if (hetero->signalSendComms[p])
        net->closeSend(hetero->signalSendComms[p]);
    free(hetero->signalSendComms);
    hetero->signalSendComms = nullptr;
  }
  if (hetero->barrierRecvComms) {
    for (int p = 0; p < hetero->nInterPeers; p++)
      if (hetero->barrierRecvComms[p])
        net->closeRecv(hetero->barrierRecvComms[p]);
    free(hetero->barrierRecvComms);
    hetero->barrierRecvComms = nullptr;
  }
  if (hetero->interSignalFlagsHost) {
    flagcxCommDeferFree(comm, hetero->interSignalFlagsHost, flagcxMemHost);
    hetero->interSignalFlagsHost = nullptr;
  }
  hetero->relayInitialized = false;
  return flagcxSuccess;
}

// ==========================================================================
// Communicator property query
// ==========================================================================

flagcxResult_t flagcxCommQueryProperties(flagcxComm_t comm,
                                         flagcxCommProperties_t *props) {
  if (comm == nullptr || props == nullptr) {
    return flagcxInvalidArgument;
  }
  memset(props, 0, sizeof(*props));

  // Baseline fields (always available)
  props->rank = comm->rank;
  props->nRanks = comm->nranks;
  props->deviceId = comm->heteroComm ? comm->heteroComm->cudaDev : -1;

  // Query multicast support via adaptor
  props->deviceApiSupport = true;
  int mcSupported = 0;
  if (deviceAdaptor->symMulticastSupported)
    deviceAdaptor->symMulticastSupported(&mcSupported);
  props->multicastSupport = (mcSupported != 0);
  props->netType = flagcxNetTypeNone;

  return flagcxSuccess;
}

// ==========================================================================
// Barrier requirement stubs (resource-handle model not yet implemented)
// ==========================================================================

flagcxResult_t
flagcxIntraBarrierCreateRequirement(flagcxTeam_t team, int nBarriers,
                                    flagcxIntraBarrierHandle_t *outHandle,
                                    flagcxDevCommRequirements *outReq) {
  (void)team;
  (void)nBarriers;
  (void)outHandle;
  (void)outReq;
  return flagcxNotSupported;
}

flagcxResult_t flagcxInterBarrierCreateRequirement(
    flagcxComm_t comm, flagcxTeam_t team, int nBarriers,
    flagcxInterBarrierHandle_t *outHandle, flagcxDevCommRequirements *outReq) {
  (void)comm;
  (void)team;
  (void)nBarriers;
  (void)outHandle;
  (void)outReq;
  return flagcxNotSupported;
}
