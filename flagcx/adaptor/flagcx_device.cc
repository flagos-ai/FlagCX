/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t and flagcxDevMem_t.
 *
 * Capability-based additive design:
 *   Baseline (always): rawPtr + fifoBuffer + rank info
 *   IPC layer:         peer pointers + IPC barriers (if IPC exchange succeeds)
 *   NCCL layer:        ncclDevComm + ncclWindow_t (if NCCL > 2.28)
 *
 * Each layer is added when available; lower layers are always present
 * as fallback. Kernel dispatch uses priority: Window > IPC > Raw.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "flagcx_kernel.h"
#include "net.h"     // flagcxNetHandle_t
#include "p2p.h"     // flagcxP2pAllocateShareableBuffer, flagcxP2pIpcDesc
#include <algorithm> // std::min, std::max
#include <unistd.h>  // usleep

// ==========================================================================
// Shared: IPC peer pointer exchange (used by both tiers)
// ==========================================================================

// Forward declaration (defined below buildIpcPeerPointers).
static void cleanupIpcPeerPointers(void **hostPeerPtrs, void **devPeerPtrs,
                                   int nPeers, void *ownBuff);

// Build IPC peer pointer table for a user buffer.
// Allocates devPeerPtrs (device array) and hostPeerPtrs (host array).
// Caller must free both on cleanup.
static flagcxResult_t buildIpcPeerPointers(flagcxComm_t comm, void *buff,
                                           size_t size, void ***outDevPeerPtrs,
                                           void ***outHostPeerPtrs,
                                           int *outNPeers) {

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

  *outDevPeerPtrs = devPeerPtrs;
  *outHostPeerPtrs = hostPeerPtrs;
  *outNPeers = localRanks;
  return flagcxSuccess;

fail:
  free(allDescs);
  cleanupIpcPeerPointers(hostPeerPtrs, devPeerPtrs, localRanks, buff);
  return res;
}

// Close IPC peer handles and free arrays.
static void cleanupIpcPeerPointers(void **hostPeerPtrs, void **devPeerPtrs,
                                   int nPeers, void *ownBuff) {
  if (hostPeerPtrs) {
    for (int i = 0; i < nPeers; i++) {
      if (hostPeerPtrs[i] && hostPeerPtrs[i] != ownBuff) {
        deviceAdaptor->ipcMemHandleClose(hostPeerPtrs[i]);
      }
    }
    free(hostPeerPtrs);
  }
  if (devPeerPtrs) {
    deviceAdaptor->deviceFree(devPeerPtrs, flagcxMemDevice, NULL);
  }
}

// ==========================================================================
// Inter-node signal relay: recv thread + connection setup/teardown
//
// Each CTA writes Signal entries to the FIFO; the proxy fans out isend.
// The recv thread pre-posts irecv for each inter-node peer, and on
// completion does atomicAdd on the host-mapped interSignalFlags array.
// The GPU spins on the device pointer of interSignalFlags in
// flagcxInterBarrierSession::wait().
// ==========================================================================

// Signal message format: 8 bytes = ctaIndex (4) + reserved (4)
struct flagcxSignalMessage {
  uint32_t ctaIndex;
  uint32_t reserved;
};

// Receiver thread: polls irecv completions and updates interSignalFlags.
static void *flagcxSignalRecvThread(void *arg) {
  flagcxDevComm_t dc = (flagcxDevComm_t)arg;
  int nPeers = dc->nInterPeers;
  struct flagcxNetAdaptor *net = (struct flagcxNetAdaptor *)dc->netAdaptorPtr;

  // Per-peer outstanding request
  void **requests = (void **)calloc(nPeers, sizeof(void *));
  if (requests == nullptr)
    return nullptr;

  int idleIters = 0;
  while (!__atomic_load_n(&dc->signalRecvStop, __ATOMIC_ACQUIRE)) {
    bool progress = false;
    for (int p = 0; p < nPeers; p++) {
      // Post irecv if no outstanding request
      if (requests[p] == nullptr) {
        void *data = &dc->signalRecvBufs[p * sizeof(flagcxSignalMessage)];
        size_t sizes[1] = {sizeof(flagcxSignalMessage)};
        int tags[1] = {0};
        void *mhandles[1] = {dc->signalRecvMrs[p]};
        void *phandles[1] = {nullptr};
        net->irecv(dc->signalRecvComms[p], 1, &data, sizes, tags, mhandles,
                   phandles, &requests[p]);
      }
      // Test for completion
      if (requests[p] != nullptr) {
        int done = 0;
        net->test(requests[p], &done, nullptr);
        if (done) {
          // Extract ctaIndex from received message
          flagcxSignalMessage *msg =
              (flagcxSignalMessage *)&dc
                  ->signalRecvBufs[p * sizeof(flagcxSignalMessage)];
          uint32_t ctaIdx = msg->ctaIndex;
          if (ctaIdx < FLAGCX_DEVICE_CTA_COUNT) {
            // Single writer (this thread), GPU reads via PCIe-mapped pointer.
            // volatile ensures the write reaches memory (not cached in
            // register).
            ((volatile uint64_t *)dc->interSignalFlagsHost)[ctaIdx]++;
          }
          requests[p] = nullptr; // will re-post on next iteration
          progress = true;
        }
      }
    }
    if (progress) {
      idleIters = 0;
    } else {
      idleIters++;
      if (idleIters < 64)
        sched_yield();
      else
        usleep(1); // adaptive backoff: reduce CPU when idle
    }
  }

  free(requests);
  return nullptr;
}

// Setup inter-node signal connections and recv thread.
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
  // Only localRank 0 (the inter leader) manages connections and recv thread.
  handle->nInterPeers = nInterPeers;
  handle->interPeerRanks = interPeerRanks;

  if (hetero->localRank != 0) {
    // Non-leader: knows nInterPeers but has no connections/thread.
    handle->isInterLeader = false;
    INFO(FLAGCX_INIT,
         "setupInterNodeSignalRelay: rank %d (non-leader), nInterPeers %d",
         myRank, nInterPeers);
    return flagcxSuccess;
  }

  // Leader path: allocate connections, signal flags, recv thread.
  handle->isInterLeader = true;
  handle->netAdaptorPtr = (void *)hetero->netAdaptor;

  flagcxResult_t res = flagcxSuccess;

  // Step 1: Allocate host-mapped signal flags (GPU reads, recv thread writes)
  size_t flagsSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMalloc((void **)&handle->interSignalFlagsHost,
                                  flagsSize, flagcxMemHost, NULL),
      res, fail);
  memset(handle->interSignalFlagsHost, 0, flagsSize);
  FLAGCXCHECKGOTO(
      deviceAdaptor->hostGetDevicePointer((void **)&handle->interSignalFlags,
                                          handle->interSignalFlagsHost),
      res, fail);

  // Step 2: Allocate staging buffers for isend/irecv
  handle->signalSendBufs =
      (char *)calloc(nInterPeers, sizeof(flagcxSignalMessage));
  handle->signalRecvBufs =
      (char *)calloc(nInterPeers, sizeof(flagcxSignalMessage));
  if (!handle->signalSendBufs || !handle->signalRecvBufs) {
    res = flagcxSystemError;
    goto fail;
  }

  // Step 3: Establish netAdaptor connections with each inter-node peer.
  // Pattern: each peer pair does listen+exchange+connect/accept.
  handle->signalSendComms = (void **)calloc(nInterPeers, sizeof(void *));
  handle->signalRecvComms = (void **)calloc(nInterPeers, sizeof(void *));
  if (!handle->signalSendComms || !handle->signalRecvComms) {
    res = flagcxSystemError;
    goto fail;
  }

  {
    struct bootstrapState *bootstrap = comm->bootstrap;
    int netDev = hetero->netDev;
    struct flagcxNetAdaptor *net = hetero->netAdaptor;
    // Use a deterministic tag based on both ranks so that the tag is
    // symmetric: rank A sending to rank B uses the same tag as rank B
    // sending to rank A.  signalTag + min(myRank, peer) * nRanks +
    // max(myRank, peer) is unique per pair and order-independent.
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
      handle->signalRecvComms[p] = recvComm;
    }
  }

  // Step 4: Register MR per-peer (each IB connection may have a different PD)
  handle->signalSendMrs = (void **)calloc(nInterPeers, sizeof(void *));
  handle->signalRecvMrs = (void **)calloc(nInterPeers, sizeof(void *));
  if (!handle->signalSendMrs || !handle->signalRecvMrs) {
    res = flagcxSystemError;
    goto fail;
  }
  for (int p = 0; p < nInterPeers; p++) {
    FLAGCXCHECKGOTO(
        hetero->netAdaptor->regMr(
            handle->signalSendComms[p],
            &handle->signalSendBufs[p * sizeof(flagcxSignalMessage)],
            sizeof(flagcxSignalMessage), FLAGCX_PTR_HOST,
            &handle->signalSendMrs[p]),
        res, fail);
    FLAGCXCHECKGOTO(
        hetero->netAdaptor->regMr(
            handle->signalRecvComms[p],
            &handle->signalRecvBufs[p * sizeof(flagcxSignalMessage)],
            sizeof(flagcxSignalMessage), FLAGCX_PTR_HOST,
            &handle->signalRecvMrs[p]),
        res, fail);
  }

  // Step 5: Start receiver thread
  __atomic_store_n(&handle->signalRecvStop, 0, __ATOMIC_RELEASE);
  {
    int err = pthread_create(&handle->signalRecvThread, nullptr,
                             flagcxSignalRecvThread, handle);
    if (err != 0) {
      res = flagcxSystemError;
      goto fail;
    }
  }

  // Publish to heteroComm so proxy can access signal connections
  hetero->signalDevComm = handle;

  INFO(FLAGCX_INIT,
       "setupInterNodeSignalRelay: rank %d (leader), nInterPeers %d, recv "
       "thread started",
       myRank, nInterPeers);
  return flagcxSuccess;

fail:
  // Partial cleanup on error (DevCommDestroy will handle the rest)
  return res;
}

// Teardown inter-node signal relay (called from flagcxDevCommDestroy).
static void cleanupInterNodeSignalRelay(flagcxComm_t comm,
                                        flagcxDevComm_t handle) {
  // Free peer rank list (set on all ranks)
  free(handle->interPeerRanks);
  handle->interPeerRanks = nullptr;

  // Only the leader has connections/thread/flags to clean up
  if (!handle->isInterLeader)
    return;

  struct flagcxNetAdaptor *net =
      (struct flagcxNetAdaptor *)handle->netAdaptorPtr;

  // Stop recv thread
  __atomic_store_n(&handle->signalRecvStop, 1, __ATOMIC_RELEASE);
  if (handle->signalRecvThread) {
    pthread_join(handle->signalRecvThread, nullptr);
  }

  // Deregister MR per-peer
  if (handle->signalSendMrs) {
    for (int p = 0; p < handle->nInterPeers; p++) {
      if (handle->signalSendMrs[p] && handle->signalSendComms &&
          handle->signalSendComms[p]) {
        net->deregMr(handle->signalSendComms[p], handle->signalSendMrs[p]);
      }
    }
    free(handle->signalSendMrs);
  }
  if (handle->signalRecvMrs) {
    for (int p = 0; p < handle->nInterPeers; p++) {
      if (handle->signalRecvMrs[p] && handle->signalRecvComms &&
          handle->signalRecvComms[p]) {
        net->deregMr(handle->signalRecvComms[p], handle->signalRecvMrs[p]);
      }
    }
    free(handle->signalRecvMrs);
  }

  // Close connections
  if (handle->signalSendComms) {
    for (int p = 0; p < handle->nInterPeers; p++) {
      if (handle->signalSendComms[p])
        net->closeSend(handle->signalSendComms[p]);
    }
    free(handle->signalSendComms);
  }
  if (handle->signalRecvComms) {
    for (int p = 0; p < handle->nInterPeers; p++) {
      if (handle->signalRecvComms[p])
        net->closeRecv(handle->signalRecvComms[p]);
    }
    free(handle->signalRecvComms);
  }

  // Free staging buffers
  free(handle->signalSendBufs);
  free(handle->signalRecvBufs);

  // Free host-mapped signal flags
  if (handle->interSignalFlagsHost) {
    deviceAdaptor->deviceFree(handle->interSignalFlagsHost, flagcxMemHost,
                              NULL);
  }

  // Clear heteroComm reference
  if (comm && comm->heteroComm) {
    comm->heteroComm->signalDevComm = nullptr;
  }
}

#ifdef FLAGCX_DEVICE_API_NCCL
#include "nvidia_adaptor.h"
#endif

// ==========================================================================
// IPC barrier setup helper (extracted from old Tier 2 DevCommCreate)
//
// Allocates IPC-shareable barrier flags, exchanges handles with all ranks,
// and builds a device-side pointer array. On failure, partially-allocated
// resources are cleaned up by flagcxDevCommDestroy (null-safe).
// ==========================================================================
static flagcxResult_t setupIpcBarriers(flagcxComm_t comm,
                                       flagcxDevComm_t handle) {
  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int localRanks = comm->localRanks;
  int *lrToR = comm->localRankToRank;

  handle->nLocalRanks = localRanks;
  handle->localRankToRank = (int *)malloc(localRanks * sizeof(int));
  if (handle->localRankToRank == nullptr)
    return flagcxSystemError;
  memcpy(handle->localRankToRank, lrToR, localRanks * sizeof(int));

  // Step 1: Allocate local barrier flags (IPC-shareable device memory)
  struct flagcxP2pIpcDesc barrierIpcDesc;
  memset(&barrierIpcDesc, 0, sizeof(barrierIpcDesc));
  size_t barrierSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
  FLAGCXCHECK(flagcxP2pAllocateShareableBuffer(
      barrierSize, 0, &barrierIpcDesc, (void **)&handle->localBarrierFlags));

  // Zero barrier flags
  FLAGCXCHECK(deviceAdaptor->deviceMemset(handle->localBarrierFlags, 0,
                                          barrierSize, flagcxMemDevice, NULL));

  // Step 2: Exchange barrier IPC handles with all ranks
  struct flagcxP2pIpcDesc *allBarrierDescs = (struct flagcxP2pIpcDesc *)calloc(
      nRanks, sizeof(struct flagcxP2pIpcDesc));
  if (allBarrierDescs == nullptr)
    return flagcxSystemError;
  memcpy(&allBarrierDescs[myRank], &barrierIpcDesc,
         sizeof(struct flagcxP2pIpcDesc));
  FLAGCXCHECK(bootstrapAllGather(comm->bootstrap, allBarrierDescs,
                                 sizeof(struct flagcxP2pIpcDesc)));

  // Step 3: Open intra-node peer barrier IPC handles
  handle->peerBarrierPtrs = (void **)calloc(localRanks, sizeof(void *));
  if (handle->peerBarrierPtrs == nullptr) {
    free(allBarrierDescs);
    return flagcxSystemError;
  }
  for (int lr = 0; lr < localRanks; lr++) {
    int gr = lrToR[lr];
    if (gr == myRank) {
      handle->peerBarrierPtrs[lr] = handle->localBarrierFlags;
    } else {
      flagcxIpcMemHandle_t handlePtr =
          (flagcxIpcMemHandle_t)&allBarrierDescs[gr].handleData;
      FLAGCXCHECK(deviceAdaptor->ipcMemHandleOpen(
          handlePtr, &handle->peerBarrierPtrs[lr]));
    }
  }
  free(allBarrierDescs);

  // Step 4: Build device barrier pointer array
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&handle->barrierPeers,
                                          localRanks * sizeof(uint64_t *),
                                          flagcxMemDevice, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      handle->barrierPeers, handle->peerBarrierPtrs,
      localRanks * sizeof(uint64_t *), flagcxMemcpyHostToDevice, NULL, NULL));

  return flagcxSuccess;
}

// ==========================================================================
// Unified DevComm: Additive capability layers
//   Baseline: rank info + fifoBuffer (always)
//   IPC layer: barrier pointers (if reqs->fields[0] > 0)
//   NCCL layer: ncclDevComm (if NCCL > 2.28)
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

  // ---- Baseline: always ----
  handle->rank = comm->rank;
  handle->nRanks = comm->nranks;
  handle->intraRank = comm->localRank;
  handle->intraSize = comm->localRanks;
  handle->fifoBuffer =
      (comm->heteroComm != nullptr) ? comm->heteroComm->fifoBuffer : nullptr;

  // ---- IPC barrier layer: if barriers requested ----
  if (reqs->fields[0] > 0) {
    flagcxResult_t res = setupIpcBarriers(comm, handle);
    if (res != flagcxSuccess) {
      WARN("flagcxDevCommCreate: IPC barrier setup failed (%d), "
           "barriers unavailable",
           res);
      free(handle);
      return res;
    }
  }

  // ---- Inter-node signal relay: if multi-node ----
  {
    flagcxResult_t res = setupInterNodeSignalRelay(comm, handle);
    if (res != flagcxSuccess) {
      WARN("flagcxDevCommCreate: inter-node signal relay setup failed (%d), "
           "falling back to single-node mode",
           res);
      // Reset so kernel uses single-node barrier path (no inter barrier)
      handle->nInterPeers = 0;
      handle->isInterLeader = false;
    }
  }

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- NCCL layer: try ncclDevCommCreate ----
  {
    flagcxInnerComm_t innerComm = comm->homoComm;
    if (innerComm != nullptr) {
      ncclDevCommRequirements ncclReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      ncclReqs.lsaBarrierCount = reqs->fields[0];
      ncclReqs.lsaMultimem = reqs->fields[1];
      ncclReqs.railGinBarrierCount = reqs->fields[2];
      ncclReqs.ginSignalCount = reqs->fields[3];

      flagcxResult_t ret = ncclAdaptorDevCommCreate(innerComm->base, &ncclReqs,
                                                    &handle->ncclDev);
      if (ret == flagcxSuccess) {
        handle->hasNcclDev = true;
      } else {
        WARN("flagcxDevCommCreate: ncclDevCommCreate failed (%d), "
             "NCCL device layer not available",
             ret);
      }
    }
  }
#endif

  *devComm = handle;
  INFO(FLAGCX_INIT, "flagcxDevCommCreate: rank %d, layers: baseline%s%s%s",
       handle->rank, handle->barrierPeers ? " + IPC barriers" : "",
       handle->nInterPeers > 0 ? " + inter-node signal relay" : "",
       handle->hasNcclDev ? " + ncclDevComm" : "");
  return flagcxSuccess;
}

flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                    flagcxDevComm_t devComm) {
  if (devComm == nullptr) {
    return flagcxSuccess;
  }

#ifdef FLAGCX_DEVICE_API_NCCL
  // NCCL layer cleanup
  if (devComm->hasNcclDev && comm != nullptr) {
    flagcxInnerComm_t innerComm = comm->homoComm;
    if (innerComm != nullptr) {
      ncclAdaptorDevCommDestroy(innerComm->base, &devComm->ncclDev);
    }
  }
#endif

  // Inter-node signal relay cleanup
  cleanupInterNodeSignalRelay(comm, devComm);

  // IPC barrier cleanup
  if (devComm->peerBarrierPtrs) {
    for (int i = 0; i < devComm->nLocalRanks; i++) {
      if (devComm->peerBarrierPtrs[i] &&
          devComm->peerBarrierPtrs[i] != devComm->localBarrierFlags) {
        deviceAdaptor->ipcMemHandleClose(devComm->peerBarrierPtrs[i]);
      }
    }
    free(devComm->peerBarrierPtrs);
  }
  if (devComm->barrierPeers) {
    deviceAdaptor->deviceFree(devComm->barrierPeers, flagcxMemDevice, NULL);
  }
  if (devComm->localBarrierFlags) {
    deviceAdaptor->deviceFree(devComm->localBarrierFlags, flagcxMemDevice,
                              NULL);
  }
  free(devComm->localRankToRank);
  free(devComm);
  return flagcxSuccess;
}

// ==========================================================================
// Unified DevMem: Additive capability layers
//   Baseline: rawPtr (always)
//   IPC layer: peer pointers (if comm provided and win is null)
//   Window layer: ncclWindow_t (if win provided, Tier 1 only)
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

  if (comm != nullptr) {
    handle->intraRank = comm->localRank;

#ifndef FLAGCX_DEVICE_API_NCCL
    if (win != nullptr) {
      WARN("flagcxDevMemCreate: window provided but NCCL device API "
           "unavailable, falling back to IPC");
      win = nullptr;
    }
#endif

    // ---- IPC layer: try if win is null (IPC needs cudaMalloc memory) ----
    if (win == nullptr) {
      handle->basePtr = buff;
      flagcxResult_t res =
          buildIpcPeerPointers(comm, buff, size, &handle->devPeerPtrs,
                               &handle->hostPeerPtrs, &handle->nPeers);
      if (res != flagcxSuccess) {
        WARN("flagcxDevMemCreate: IPC peer pointer setup failed (%d), "
             "IPC layer not available",
             res);
        // devPeerPtrs stays nullptr — raw-only mode
      }
    }

    // ---- Window layer: if win provided and valid ----
    if (win != nullptr) {
      handle->hasWindow = true;
#ifdef FLAGCX_DEVICE_API_NCCL
      handle->isSymmetric = (win->winFlags & FLAGCX_WIN_COLL_SYMMETRIC) != 0;
      handle->ncclWin = win->base;
      handle->winHandle = (void *)win;
#endif
    }
  }

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

  // Clean up IPC peer pointers (if present)
  if (devMem->hostPeerPtrs) {
    cleanupIpcPeerPointers(devMem->hostPeerPtrs, devMem->devPeerPtrs,
                           devMem->nPeers, devMem->basePtr);
  }

  free(devMem);
  return flagcxSuccess;
}