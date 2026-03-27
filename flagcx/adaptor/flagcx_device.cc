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
#include <algorithm> // std::min, std::max

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

  // Compute inter-node peer ranks (one representative per remote node).
  // Use localRank 0 of each remote node as the representative.
  // This keeps the number of connections = nNodes - 1 (not nRanks -
  // localRanks).
  int *interPeerRanks = nullptr;
  int nInterPeers = 0;

  // Build list: for each remote node, find the global rank of its localRank 0
  for (int r = 0; r < nRanks; r++) {
    if (hetero->rankToNode[r] != myNode && hetero->rankToLocalRank[r] == 0) {
      nInterPeers++;
    }
  }

  if (nInterPeers == 0)
    return flagcxSuccess;

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
  handle->nInterPeers = nInterPeers;
  handle->interPeerRanks = interPeerRanks;
  handle->isInterLeader = (hetero->localRank == 0);

  flagcxResult_t res = flagcxSuccess;
  size_t flagsSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);

  // ---- Leader-only: allocate flags and establish connections ----
  if (handle->isInterLeader) {
    handle->netAdaptorPtr = (void *)hetero->netAdaptor;

    // Step 1: Allocate host-mapped signal flags (GPU reads, RDMA NIC writes)
    FLAGCXCHECKGOTO(
        deviceAdaptor->deviceMalloc((void **)&handle->interSignalFlagsHost,
                                    flagsSize, flagcxMemHost, NULL),
        res, fail);
    memset(handle->interSignalFlagsHost, 0, flagsSize);
    FLAGCXCHECKGOTO(
        deviceAdaptor->hostGetDevicePointer((void **)&handle->interSignalFlags,
                                            handle->interSignalFlagsHost),
        res, fail);

    // Step 2: Establish netAdaptor connections with each inter-node peer.
    // Keep sendComms for iputSignal; keep ALL recvComms alive so that
    // peers' sendComm QPs remain connected (needed for incoming RDMA atomics).
    handle->signalSendComms = (void **)calloc(nInterPeers, sizeof(void *));
    handle->barrierRecvComms = (void **)calloc(nInterPeers, sizeof(void *));
    if (!handle->signalSendComms || !handle->barrierRecvComms) {
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

        handle->signalSendComms[p] = sendComm;
        handle->barrierRecvComms[p] = recvComm;
      }
    }
  }

  // ---- ALL ranks: register barrier MR (collective AllGather inside) ----
  {
    struct flagcxOneSideHandleInfo *barrierInfo = nullptr;
    res = flagcxOneSideBarrierRegister(
        comm, handle->isInterLeader ? handle->barrierRecvComms[0] : nullptr,
        handle->isInterLeader ? handle->interSignalFlagsHost : nullptr,
        handle->isInterLeader ? flagsSize : 0, &barrierInfo);
    if (res != flagcxSuccess) {
      WARN("setupInterNodeSignalRelay: barrier MR registration failed (%d)",
           res);
      goto fail;
    }
    if (handle->isInterLeader) {
      handle->barrierHandleInfo = barrierInfo;
    } else {
      // Non-leader participated in AllGather but doesn't need the result
      flagcxOneSideBarrierDeregister(comm, barrierInfo);
    }
  }

  INFO(FLAGCX_INIT, "setupInterNodeSignalRelay: rank %d (%s), nInterPeers %d",
       myRank, handle->isInterLeader ? "leader" : "non-leader", nInterPeers);
  return flagcxSuccess;

fail:
  // Partial cleanup on error (DevCommDestroy will handle the rest)
  return res;
}

// Teardown inter-node signal relay (called from flagcxDevCommDestroy).
static void cleanupInterNodeSignalRelay(flagcxComm_t comm,
                                        flagcxDevComm_t handle) {
  // Step 0: Drain FIFO — wait for proxy thread to finish all pending
  // entries (including BarrierSignal RDMA atomics) before closing connections.
  {
    struct flagcxHeteroComm *hetero = comm->heteroComm;
    if (hetero && hetero->proxyState && hetero->proxyState->kernelState.fifo) {
      volatile uint64_t *buf =
          (volatile uint64_t *)hetero->proxyState->kernelState.fifo->buffer;
      if (buf) {
        while (buf[flagcxFifoIdxConsumed] < buf[flagcxFifoIdxProduced]) {
          sched_yield();
        }
      }
    }
  }

  // Step 1: Cross-rank barrier — all ranks must drain before any rank
  // closes connections, preventing the race where rank A destroys its QP
  // while rank B's proxy is still posting RDMA atomics to rank A.
  bootstrapBarrier(comm->bootstrap, comm->rank, comm->nranks, 0x7f01);

  // Free peer rank list (set on all ranks)
  free(handle->interPeerRanks);
  handle->interPeerRanks = nullptr;

  // Only the leader has connections/flags to clean up
  if (!handle->isInterLeader)
    return;

  struct flagcxNetAdaptor *net =
      (struct flagcxNetAdaptor *)handle->netAdaptorPtr;

  // 1. Deregister barrier MR and free handle info
  if (handle->barrierHandleInfo) {
    flagcxOneSideBarrierDeregister(
        comm, (struct flagcxOneSideHandleInfo *)handle->barrierHandleInfo);
    handle->barrierHandleInfo = nullptr;
  }

  // 2. Close send comms
  if (handle->signalSendComms) {
    for (int p = 0; p < handle->nInterPeers; p++) {
      if (handle->signalSendComms[p]) {
        net->closeSend(handle->signalSendComms[p]);
      }
    }
    free(handle->signalSendComms);
  }

  // 3. Close all barrier recv comms (kept alive for QP connections)
  if (handle->barrierRecvComms) {
    for (int p = 0; p < handle->nInterPeers; p++) {
      if (handle->barrierRecvComms[p]) {
        net->closeRecv(handle->barrierRecvComms[p]);
      }
    }
    free(handle->barrierRecvComms);
  }

  // 4. Defer free of host-mapped signal flags (cudaFreeHost would deadlock
  // on vendor persistent kernels — drain after vendor comm destroy).
  if (handle->interSignalFlagsHost) {
    flagcxCommDeferFree(comm, handle->interSignalFlagsHost, flagcxMemHost);
  }

  // Note: do NOT clear comm->heteroComm->devCommHandle here.
  // flagcxDevCommDestroy needs it to gate signalDeregister.
}

#ifdef FLAGCX_DEVICE_API_VENDOR
#include "nvidia_adaptor.h"
#endif

// ==========================================================================
// IPC barrier setup helper (extracted from old Fallback DevCommCreate)
//
// Allocates IPC-shareable barrier flags, exchanges handles with all ranks,
// and builds a device-side pointer array. On failure, partially-allocated
// resources are cleaned up by flagcxDevCommDestroy (null-safe).
// ==========================================================================
static flagcxResult_t setupIpcBarriers(flagcxComm_t comm,
                                       flagcxDevComm_t handle) {
  int localRanks = comm->localRanks;

  handle->nLocalRanks = localRanks;
  handle->localRankToRank = (int *)malloc(localRanks * sizeof(int));
  if (handle->localRankToRank == nullptr)
    return flagcxSystemError;
  memcpy(handle->localRankToRank, comm->localRankToRank,
         localRanks * sizeof(int));

  // Allocate IPC-shareable barrier flags
  struct flagcxP2pIpcDesc barrierIpcDesc;
  memset(&barrierIpcDesc, 0, sizeof(barrierIpcDesc));
  size_t barrierSize = localRanks * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
  FLAGCXCHECK(flagcxP2pAllocateShareableBuffer(
      barrierSize, 0, &barrierIpcDesc, (void **)&handle->localBarrierFlags));
  FLAGCXCHECK(deviceAdaptor->deviceMemset(handle->localBarrierFlags, 0,
                                          barrierSize, flagcxMemDevice, NULL));

  // Reuse common IPC exchange logic
  int slot = buildIpcPeerPointers(comm, handle->localBarrierFlags, barrierSize);
  if (slot < 0)
    return flagcxInternalError;

  // Store barrier-specific metadata on devComm handle
  handle->barrierPeers = (uint64_t **)comm->ipcTable[slot].devPeerPtrs;
  handle->barrierIpcIndex = slot;
  handle->nBarriers = FLAGCX_DEVICE_CTA_COUNT;

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
  handle->fifoBuffer =
      (comm->heteroComm != nullptr) ? comm->heteroComm->fifoBuffer : nullptr;

  // ---- Vendor path: try devCommCreate via adaptor ----
  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm != nullptr) {
    flagcxInnerDevComm_t innerDevComm = nullptr;
    flagcxResult_t ret = cclAdaptors[flagcxCCLAdaptorDevice]->devCommCreate(
        innerComm, reqs, &innerDevComm);
    if (ret != flagcxSuccess && ret != flagcxNotSupported) {
      WARN("flagcxDevCommCreate: vendor devCommCreate failed (%d)", ret);
      free(handle);
      return ret;
    }
    if (ret == flagcxSuccess)
      handle->devComm = innerDevComm;
  }
  if (handle->devComm == nullptr) {
    // ---- Fallback path: IPC barriers + inter-node signal relay + one-sided
    // ----

    // IPC barrier layer: if barriers requested
    if (reqs->intraBarrierCount > 0) {
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

    // One-sided Fallback layer: if signals or counters requested
    if (reqs->interSignalCount > 0 || reqs->interCounterCount > 0) {
      int ctxCount =
          (reqs->interContextCount > 0) ? reqs->interContextCount : 4;
      handle->contextCount = ctxCount;

      // Allocate signal buffer (GPU, SYNC_MEMOPS via gdrMemAlloc for RDMA)
      if (reqs->interSignalCount > 0) {
        handle->signalCount = reqs->interSignalCount;
        size_t sigSize =
            (size_t)handle->signalCount * ctxCount * sizeof(uint64_t);
        FLAGCXCHECK(deviceAdaptor->gdrMemAlloc((void **)&handle->signalBuffer,
                                               sigSize, NULL));
        FLAGCXCHECK(deviceAdaptor->deviceMemset(
            handle->signalBuffer, 0, sigSize, flagcxMemDevice, NULL));
        FLAGCXCHECK(deviceAdaptor->deviceMalloc(
            (void **)&handle->shadowBuffer, sigSize, flagcxMemDevice, NULL));
        FLAGCXCHECK(deviceAdaptor->deviceMemset(
            handle->shadowBuffer, 0, sigSize, flagcxMemDevice, NULL));
      }
      // Allocate counter buffer (host-pinned)
      if (reqs->interCounterCount > 0) {
        handle->counterCount = reqs->interCounterCount;
        size_t cntSize =
            (size_t)handle->counterCount * ctxCount * sizeof(uint64_t);
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
        flagcxResult_t regRes = flagcxOneSideSignalRegister(
            comm, handle->signalBuffer,
            (size_t)handle->signalCount * handle->contextCount *
                sizeof(uint64_t));
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
           "flagcxDevCommCreate: one-sided Fallback buffers allocated "
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
           ? " + one-sided Fallback"
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

  INFO(FLAGCX_INIT, "flagcxDevCommDestroy: rank %d enter", devComm->rank);

  // Vendor layer cleanup via adaptor
  if (comm != nullptr && devComm->devComm != nullptr) {
    flagcxInnerComm_t innerComm = comm->homoComm;
    if (innerComm != nullptr) {
      cclAdaptors[flagcxCCLAdaptorDevice]->devCommDestroy(innerComm,
                                                          devComm->devComm);
      devComm->devComm = nullptr;
    }
  }

  // Inter-node signal relay cleanup
  cleanupInterNodeSignalRelay(comm, devComm);

  // IPC barrier cleanup — mark ipcTable entry as unused.
  // Actual ipcMemHandleClose + deviceFree deferred to flagcxCommCleanupIpcTable
  // (after vendor comm destroy) to avoid implicit device synchronization
  // deadlock.
  if (comm != nullptr && devComm->barrierIpcIndex >= 0 &&
      devComm->barrierIpcIndex < FLAGCX_MAX_IPC_ENTRIES) {
    comm->ipcTable[devComm->barrierIpcIndex].inUse = false;
  }
  if (devComm->localBarrierFlags) {
    flagcxCommDeferFree(comm, devComm->localBarrierFlags, flagcxMemDevice);
  }

  // Clear heteroComm->devComm + deregister signal buffer
  if (comm != nullptr && comm->heteroComm != nullptr &&
      comm->heteroComm->devCommHandle == devComm) {
    if (devComm->signalBuffer) {
      flagcxOneSideSignalDeregister(comm);
    }
    comm->heteroComm->devCommHandle = nullptr;
  }

  // One-sided Fallback cleanup
  if (devComm->signalBuffer) {
    flagcxCommDeferFree(comm, devComm->signalBuffer, flagcxMemDevice);
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

  INFO(FLAGCX_INIT, "flagcxDevCommDestroy: rank %d done", devComm->rank);
  free(devComm->localRankToRank);
  free(devComm);
  return flagcxSuccess;
}

// ==========================================================================
// Unified DevMem: Additive capability layers
//   Baseline: rawPtr (always)
//   IPC layer: peer pointers (if comm provided and win is null)
//   Window layer: vendor Window (if win provided, Vendor only)
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

  // ---- Per-window MR layer: lookup buff in globalOneSideHandleTable ----
  handle->mrIndex = -1;
  handle->mrBase = 0;
  if (comm != nullptr) {
    for (int i = 0; i < globalOneSideHandleCount; i++) {
      struct flagcxOneSideHandleInfo *info = globalOneSideHandleTable[i];
      if (info != NULL && info->baseVas != NULL) {
        uintptr_t base = info->baseVas[comm->rank];
        if ((uintptr_t)buff == base) {
          handle->mrIndex = i;
          handle->mrBase = base;
          INFO(FLAGCX_INIT,
               "flagcxDevMemCreate: buff %p matched handleTable[%d], "
               "mrBase=0x%lx",
               buff, i, (unsigned long)base);
          break;
        }
      }
    }
  }

  if (comm != nullptr) {
    handle->intraRank = comm->localRank;

#ifndef FLAGCX_DEVICE_API_VENDOR
    if (win != nullptr) {
      WARN("flagcxDevMemCreate: window provided but NCCL device API "
           "unavailable, falling back to IPC");
      win = nullptr;
    }
#endif

    // ---- IPC layer: try if win is null (IPC needs cudaMalloc memory) ----
    if (win == nullptr) {
      int idx = buildIpcPeerPointers(comm, buff, size);
      if (idx >= 0) {
        handle->ipcIndex = idx;
        handle->devPeerPtrs = comm->ipcTable[idx].devPeerPtrs;
      } else {
        WARN("flagcxDevMemCreate: IPC peer pointer setup failed, "
             "IPC layer not available");
        // devPeerPtrs stays nullptr — raw-only mode
      }
    }

    // ---- Window layer: if win provided and valid ----
    if (win != nullptr) {
      handle->hasWindow = true;
#ifdef FLAGCX_DEVICE_API_VENDOR
      handle->isSymmetric = (win->winFlags & FLAGCX_WIN_COLL_SYMMETRIC) != 0;
      // Allocate vendor Window and store in opaque pointer
      ncclWindow_t *ncclWin = (ncclWindow_t *)malloc(sizeof(ncclWindow_t));
      if (ncclWin == nullptr) {
        WARN("flagcxDevMemCreate: failed to allocate ncclWindow_t");
        free(handle);
        return flagcxSystemError;
      }
      *ncclWin = win->base;
      handle->window = ncclWin;
      handle->winHandle = (void *)win;
#else
      handle->window = nullptr;
      handle->winHandle = nullptr;
#endif
    }
  }

#ifndef FLAGCX_DEVICE_API_VENDOR
  // Fallback: always create a Window with baseline/IPC/MR fields so that
  // flagcxDevMem(di) can copy it into _winBase for kernel use.
  if (handle->window == nullptr) {
    auto *fbWin = (typename DeviceAPI::Window *)malloc(
        sizeof(typename DeviceAPI::Window));
    if (fbWin == nullptr) {
      WARN("flagcxDevMemCreate: failed to allocate fallback Window");
      free(handle);
      return flagcxSystemError;
    }
    fbWin->rawPtr = handle->rawPtr;
    fbWin->peerPtrs = handle->devPeerPtrs;
    fbWin->intraRank = handle->intraRank;
    fbWin->mrBase = handle->mrBase;
    fbWin->mrIndex = handle->mrIndex;
    handle->window = fbWin;
    handle->hasWindow =
        (handle->rawPtr != nullptr || handle->devPeerPtrs != nullptr);
  }
#endif

  *devMem = handle;
  INFO(FLAGCX_INIT, "flagcxDevMemCreate: ptr %p, layers: rawPtr%s%s", buff,
       handle->devPeerPtrs ? " + IPC peerPtrs" : "",
       handle->hasWindow ? (handle->isSymmetric ? " + Window (symmetric)"
                                                : " + Window (basic)")
                         : "");
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
    free(devMem->window);
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

  // Vendor-specific fields: fill from vendor if available
#ifdef FLAGCX_DEVICE_API_VENDOR
  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm != nullptr && innerComm->base != nullptr) {
    props->deviceApiSupport = true; // Vendor device API available
    // Vendor CommQueryProperties not yet wired through adaptor — set defaults.
    // Full delegation will be added when the adaptor wrapper is available.
    props->multicastSupport = false;
    props->netType = flagcxNetTypeNone;
  }
#endif

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