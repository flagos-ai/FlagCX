/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Default Device API backend for flagcxDevComm/DevMem lifecycle.
 * Linked when FLAGCX_COMM_TRAITS_DEFAULT is defined (NCCL < 2.29 or
 * FORCE_DEFAULT_PATH=1).
 *
 * Uses IPC barriers + inter-node signal relay + one-sided buffers.
 * Absorbs all default-path-specific code previously in flagcx_device.cc.
 ************************************************************************/

#include "adaptor.h"
#include "bootstrap.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "net.h"
#include "onesided.h"
#include "p2p.h"
#include "proxy.h"
#include "reg_pool.h"
#include "shmutils.h"
#include "sym_heap.h"
#include "utils.h"
#include <algorithm>
#include <cstddef>
#include <new>
#include <sched.h>

// Host-visible helpers from device_api_host_helpers.cu
#ifdef COMPILE_KERNEL_HOST
extern "C" size_t flagcxDevNetSizeOf();
extern "C" void flagcxDevNetLaunchConstruct(void *devNets, void *devComm,
                                            int count, void *stream);
#else
static size_t flagcxDevNetSizeOf() { return 0; }
static void flagcxDevNetLaunchConstruct(void *, void *, int, void *) {}
#endif

// ==========================================================================
// Platform wrappers for host memory registration
// ==========================================================================

static flagcxResult_t shmHostRegister(void *ptr, size_t bytes) {
  if (deviceAdaptor->hostRegister == nullptr) {
    WARN("FLAGCX_SIGNAL_HOST_ENABLE=1: hostRegister not supported");
    return flagcxNotSupported;
  }
  return deviceAdaptor->hostRegister(ptr, bytes);
}

static void shmHostUnregister(void *ptr) {
  if (deviceAdaptor->hostUnregister)
    deviceAdaptor->hostUnregister(ptr);
}

// ==========================================================================
// Inter-node signal relay setup
// ==========================================================================

static flagcxResult_t setupInterNodeSignalRelay(flagcxComm_t comm,
                                                flagcxDevComm_t devComm) {
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero == nullptr)
    return flagcxSuccess;

  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int myNode = hetero->node;
  int nNodes = hetero->nNodes;

  if (nNodes <= 1)
    return flagcxSuccess;

  // Already initialized: just copy pointers into devComm
  if (hetero->relayInitialized) {
    devComm->nInterPeers = hetero->nInterPeers;
    devComm->isInterLeader = hetero->isInterLeader;
    devComm->interPeerRanks = hetero->interPeerRanks;
    devComm->interSignalFlags = hetero->interSignalFlags;
    devComm->interSignalFlagsHost = hetero->interSignalFlagsHost;
    devComm->signalSendComms = hetero->signalSendComms;
    devComm->barrierRecvComms = hetero->barrierRecvComms;
    devComm->barrierHandleInfo = hetero->barrierHandleInfo;
    devComm->netAdaptorPtr = hetero->netAdaptorPtr;
    return flagcxSuccess;
  }

  // First call: establish connections
  int *interPeerRanks = nullptr;
  int nInterPeers = 0;

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

  hetero->nInterPeers = nInterPeers;
  hetero->interPeerRanks = interPeerRanks;
  hetero->isInterLeader = (hetero->localRank == 0);

  flagcxResult_t res = flagcxSuccess;
  size_t flagsSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);

  if (hetero->isInterLeader) {
    hetero->netAdaptorPtr = (void *)hetero->netAdaptor;

    // Allocate host-pinned interSignalFlagsHost
    FLAGCXCHECKGOTO(
        deviceAdaptor->deviceMalloc((void **)&hetero->interSignalFlagsHost,
                                    flagsSize, flagcxMemHost, NULL),
        res, relay_fail);
    memset(hetero->interSignalFlagsHost, 0, flagsSize);

    // Map to device pointer
    if (deviceAdaptor->hostGetDevicePointer) {
      FLAGCXCHECKGOTO(
          deviceAdaptor->hostGetDevicePointer(
              (void **)&hetero->interSignalFlags, hetero->interSignalFlagsHost),
          res, relay_fail);
    } else {
      hetero->interSignalFlags = (uint64_t *)hetero->interSignalFlagsHost;
    }

    // Establish send/recv connections to peer leaders
    hetero->signalSendComms = (void **)malloc(nInterPeers * sizeof(void *));
    hetero->barrierRecvComms = (void **)malloc(nInterPeers * sizeof(void *));
    if (!hetero->signalSendComms || !hetero->barrierRecvComms) {
      res = flagcxSystemError;
      goto relay_fail;
    }
    memset(hetero->signalSendComms, 0, nInterPeers * sizeof(void *));
    memset(hetero->barrierRecvComms, 0, nInterPeers * sizeof(void *));

    struct flagcxNetAdaptor *net = hetero->netAdaptor;
    int netDev = hetero->netDev;
    struct bootstrapState *bootstrap = comm->bootstrap;
    const int signalTagBase = 2001;

    // Exchange listen handles with each peer leader via point-to-point tagged
    // bootstrap (NOT ring allgather — only leaders participate here, and ring
    // allgather requires all ranks).
    for (int p = 0; p < nInterPeers; p++) {
      int peer = interPeerRanks[p];
      int pairTag = signalTagBase + std::min(myRank, peer) * nRanks +
                    std::max(myRank, peer);

      // Listen for incoming connection from this peer
      flagcxNetHandle_t listenHandle = {};
      void *listenComm = nullptr;
      FLAGCXCHECKGOTO(net->listen(netDev, &listenHandle, &listenComm), res,
                      relay_fail);

      // Exchange listen handles via tagged bootstrap send/recv
      flagcxNetHandle_t peerHandle = {};
      FLAGCXCHECKGOTO(bootstrapSend(bootstrap, peer, pairTag, &listenHandle,
                                    sizeof(flagcxNetHandle_t)),
                      res, relay_fail);
      FLAGCXCHECKGOTO(bootstrapRecv(bootstrap, peer, pairTag, &peerHandle,
                                    sizeof(flagcxNetHandle_t)),
                      res, relay_fail);

      // Non-blocking connect/accept loop
      void *sendComm = nullptr;
      void *recvComm = nullptr;
      while (sendComm == nullptr || recvComm == nullptr) {
        if (sendComm == nullptr) {
          flagcxResult_t r = net->connect(netDev, &peerHandle, &sendComm);
          if (r != flagcxSuccess && r != flagcxInProgress) {
            res = r;
            goto relay_fail;
          }
        }
        if (recvComm == nullptr) {
          flagcxResult_t r = net->accept(listenComm, &recvComm);
          if (r != flagcxSuccess && r != flagcxInProgress) {
            res = r;
            goto relay_fail;
          }
        }
      }
      net->closeListen(listenComm);

      hetero->signalSendComms[p] = sendComm;
      hetero->barrierRecvComms[p] = recvComm;
    }
  }

  // ALL ranks: register barrier MR (collective AllGather inside).
  // Leaders provide recvComm and buffer; non-leaders participate in AllGather.
  {
    struct flagcxOneSideHandleInfo *barrierInfo = nullptr;
    res = flagcxOneSideBarrierRegister(
        comm, hetero->isInterLeader ? hetero->barrierRecvComms[0] : nullptr,
        hetero->isInterLeader ? hetero->interSignalFlagsHost : nullptr,
        hetero->isInterLeader ? flagsSize : 0, &barrierInfo);
    if (res != flagcxSuccess) {
      WARN("setupInterNodeSignalRelay: barrier MR registration failed (%d)",
           res);
      goto relay_fail;
    }
    if (hetero->isInterLeader) {
      hetero->barrierHandleInfo = barrierInfo;
    } else {
      // Non-leader participated in AllGather but doesn't need the result
      flagcxOneSideBarrierDeregister(comm, barrierInfo);
    }
  }

  // All ranks barrier to ensure connections are established
  FLAGCXCHECKGOTO(
      bootstrapCollBarrier(comm->bootstrap, comm->rank, comm->nranks, 0xAB01),
      res, relay_fail);

  hetero->relayInitialized = true;

  // Copy into devComm
  devComm->nInterPeers = hetero->nInterPeers;
  devComm->isInterLeader = hetero->isInterLeader;
  devComm->interPeerRanks = hetero->interPeerRanks;
  devComm->interSignalFlags = hetero->interSignalFlags;
  devComm->interSignalFlagsHost = hetero->interSignalFlagsHost;
  devComm->signalSendComms = hetero->signalSendComms;
  devComm->barrierRecvComms = hetero->barrierRecvComms;
  devComm->barrierHandleInfo = hetero->barrierHandleInfo;
  devComm->netAdaptorPtr = hetero->netAdaptorPtr;

  INFO(FLAGCX_INIT,
       "setupInterNodeSignalRelay: rank %d nInterPeers=%d isLeader=%d", myRank,
       nInterPeers, hetero->isInterLeader);
  return flagcxSuccess;

relay_fail:
  if (hetero->signalSendComms) {
    for (int p = 0; p < nInterPeers; p++) {
      if (hetero->signalSendComms[p])
        hetero->netAdaptor->closeSend(hetero->signalSendComms[p]);
    }
    free(hetero->signalSendComms);
    hetero->signalSendComms = nullptr;
  }
  if (hetero->barrierRecvComms) {
    for (int p = 0; p < nInterPeers; p++) {
      if (hetero->barrierRecvComms[p])
        hetero->netAdaptor->closeRecv(hetero->barrierRecvComms[p]);
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
  free(hetero->interPeerRanks);
  hetero->interPeerRanks = nullptr;
  hetero->nInterPeers = 0;
  hetero->isInterLeader = false;
  return res;
}

// ==========================================================================
// IPC Barrier setup
// ==========================================================================

static flagcxResult_t setupIpcBarriers(flagcxComm_t comm,
                                       flagcxDevComm_t devComm) {
  int localRanks = comm->localRanks;
  int myRank = comm->rank;
  int myLocalRank = comm->localRank;

  devComm->nLocalRanks = localRanks;
  devComm->localRankToRank = (int *)malloc(localRanks * sizeof(int));
  if (devComm->localRankToRank == nullptr)
    return flagcxSystemError;
  memcpy(devComm->localRankToRank, comm->localRankToRank,
         localRanks * sizeof(int));

  size_t barrierSize = localRanks * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);

  if (flagcxParamSignalHostEnable() == 0) {
    // ── IPC device memory path (default) ─────────────────────────────────
    void *barrierFlags = nullptr;
    FLAGCXCHECK(deviceAdaptor->deviceMalloc(&barrierFlags, barrierSize,
                                            flagcxMemDevice, NULL));
    FLAGCXCHECK(deviceAdaptor->deviceMemset(barrierFlags, 0, barrierSize,
                                            flagcxMemDevice, NULL));
    devComm->localBarrierFlags = (uint64_t *)barrierFlags;

    int slot = buildIpcPeerPointers(comm, barrierFlags, barrierSize);
    if (slot < 0) {
      deviceAdaptor->deviceFree(barrierFlags, flagcxMemDevice, NULL);
      devComm->localBarrierFlags = nullptr;
      return flagcxSystemError;
    }

    devComm->barrierPeers = (uint64_t **)comm->ipcTable[slot].devPeerPtrs;
    devComm->barrierIpcIndex = slot;
    devComm->localBarrierShmPtr = nullptr;
    devComm->peerBarrierShmPtrs = nullptr;
    devComm->barrierShmSize = 0;
    devComm->barrierDevPeerPtrsRaw = nullptr;
    devComm->nBarriers = FLAGCX_DEVICE_CTA_COUNT;

    INFO(FLAGCX_INIT,
         "setupIpcBarriers(IPC): rank %d slot %d localBarrierFlags=%p "
         "barrierPeers=%p nBarriers=%d",
         myRank, slot, barrierFlags, (void *)devComm->barrierPeers,
         devComm->nBarriers);

  } else {
    // ── flagcxShm + hipHostRegister path (FLAGCX_SIGNAL_HOST_ENABLE=1) ──
    flagcxResult_t res = flagcxSuccess;
    void **peerCpuPtrs = nullptr;
    flagcxShmHandle_t *peerShmHandles = nullptr;
    void **devPeerPtrs = nullptr;
    void *myDevPtr = nullptr;
    void **hostDevPtrs = nullptr;

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
        bootstrapCollBarrier(comm->bootstrap, comm->rank, comm->nranks, 0xBA01),
        res, fail_own_shm);

    // Step 3: Map peer shm segments.
    peerCpuPtrs = (void **)malloc(localRanks * sizeof(void *));
    peerShmHandles =
        (flagcxShmHandle_t *)malloc(localRanks * sizeof(flagcxShmHandle_t));
    if (!peerCpuPtrs || !peerShmHandles) {
      res = flagcxSystemError;
      goto fail_peer_shm;
    }
    memset(peerCpuPtrs, 0, localRanks * sizeof(void *));
    memset(peerShmHandles, 0, localRanks * sizeof(flagcxShmHandle_t));

    for (int lr = 0; lr < localRanks; lr++) {
      int globalR = comm->localRankToRank[lr];
      if (globalR == myRank) {
        peerCpuPtrs[lr] = myCpuPtr;
        peerShmHandles[lr] = myShmHandle;
      } else {
        char peerPath[SHM_PATH_MAX];
        snprintf(peerPath, sizeof(peerPath),
                 "/dev/shm/flagcx_barrier_%016llx_%d",
                 (unsigned long long)comm->magic, globalR);
        void *peerPtr = nullptr;
        flagcxShmHandle_t peerHandle = nullptr;
        FLAGCXCHECKGOTO(flagcxShmOpen(peerPath, sizeof(peerPath), barrierSize,
                                      &peerPtr, nullptr, 1, &peerHandle),
                        res, fail_peer_shm);
        peerCpuPtrs[lr] = peerPtr;
        peerShmHandles[lr] = peerHandle;
      }
    }

    // Step 4: Register each peer's shm as pinned memory and get device VA.
    hostDevPtrs = (void **)malloc(localRanks * sizeof(void *));
    if (!hostDevPtrs) {
      res = flagcxSystemError;
      goto fail_peer_shm;
    }
    for (int lr = 0; lr < localRanks; lr++) {
      FLAGCXCHECKGOTO(shmHostRegister(peerCpuPtrs[lr], barrierSize), res,
                      fail_peer_shm);
      FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(&hostDevPtrs[lr],
                                                          peerCpuPtrs[lr]),
                      res, fail_peer_shm);
    }

    // Step 5: Copy device VAs to device array.
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc((void **)&devPeerPtrs,
                                                localRanks * sizeof(void *),
                                                flagcxMemDevice, NULL),
                    res, fail_peer_shm);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMemcpy(
                        devPeerPtrs, hostDevPtrs, localRanks * sizeof(void *),
                        flagcxMemcpyHostToDevice, NULL, NULL),
                    res, fail_peer_shm);

    // Get own device pointer
    myDevPtr = hostDevPtrs[myLocalRank];

    devComm->localBarrierFlags = (uint64_t *)myDevPtr;
    devComm->barrierPeers = (uint64_t **)devPeerPtrs;
    devComm->barrierIpcIndex = -1;
    devComm->localBarrierShmPtr = myCpuPtr;
    devComm->peerBarrierShmPtrs = peerCpuPtrs;
    devComm->barrierShmSize = barrierSize;
    devComm->barrierDevPeerPtrsRaw = (uint64_t **)devPeerPtrs;
    devComm->myShmHandle = myShmHandle;
    devComm->peerShmHandles = peerShmHandles;
    devComm->nBarriers = FLAGCX_DEVICE_CTA_COUNT;
    devComm->nLocalRanks = localRanks;

    free(hostDevPtrs);

    INFO(FLAGCX_INIT,
         "setupIpcBarriers(SHM): rank %d localBarrierFlags=%p "
         "barrierPeers=%p nBarriers=%d",
         myRank, myDevPtr, (void *)devPeerPtrs, devComm->nBarriers);
    return flagcxSuccess;

  fail_peer_shm:
    if (hostDevPtrs) {
      for (int lr = 0; lr < localRanks; lr++)
        if (peerCpuPtrs && peerCpuPtrs[lr])
          shmHostUnregister(peerCpuPtrs[lr]);
      free(hostDevPtrs);
    }
    if (devPeerPtrs)
      deviceAdaptor->deviceFree(devPeerPtrs, flagcxMemDevice, NULL);
    if (peerShmHandles) {
      for (int lr = 0; lr < localRanks; lr++)
        if (peerShmHandles[lr] && peerShmHandles[lr] != myShmHandle)
          flagcxShmClose(peerShmHandles[lr]);
      free(peerShmHandles);
    }
    free(peerCpuPtrs);
  fail_own_shm:
    if (myShmHandle)
      flagcxShmClose(myShmHandle);
    return res;
  }

  return flagcxSuccess;
}

// ==========================================================================
// Pre-establish full-mesh proxy connections
// ==========================================================================

static flagcxResult_t preconnectFullMesh(flagcxComm_t comm) {
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero == nullptr)
    return flagcxSuccess;

  bool needPreconnect = false;
  for (int peer = 0; peer < hetero->nRanks; peer++) {
    if (peer == hetero->rank)
      continue;
    for (int channelId = 0; channelId < hetero->nChannels; channelId++) {
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
  }

  if (needPreconnect) {
    INFO(FLAGCX_INIT, "preconnectFullMesh: rank %d establishing %d-peer mesh",
         hetero->rank, hetero->nRanks - 1);
    FLAGCXCHECK(flagcxTransportP2pSetup(hetero, NULL, 0));
  }
  return flagcxSuccess;
}

// ==========================================================================
// DevComm Create — full default backend implementation
// ==========================================================================

static flagcxResult_t
defaultDevApiCommCreate(flagcxComm_t comm,
                        const struct flagcxDevCommRequirements *reqs,
                        flagcxDevComm_t devComm) {
  // IPC barrier layer
  if (reqs->intraBarrierCount > 0 || reqs->interBarrierCount > 0) {
    flagcxResult_t res = setupIpcBarriers(comm, devComm);
    if (res != flagcxSuccess) {
      return res;
    }
  }

  // Inter-node signal relay
  {
    flagcxResult_t res = setupInterNodeSignalRelay(comm, devComm);
    if (res != flagcxSuccess) {
      devComm->nInterPeers = 0;
      devComm->isInterLeader = false;
    }
  }

  // Reset inter-node barrier signal flags
  if (comm->heteroComm != nullptr &&
      comm->heteroComm->interSignalFlagsHost != nullptr) {
    size_t flagsSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
    memset(comm->heteroComm->interSignalFlagsHost, 0, flagsSize);
  }

  // Allocate epoch buffer
  {
    size_t epochBufSize = 2 * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
    INFO(FLAGCX_INIT,
         "defaultDevApiCommCreate: allocating epochBuffer size=%zu",
         epochBufSize);
    flagcxResult_t res = deviceAdaptor->deviceMalloc(
        (void **)&devComm->epochBuffer, epochBufSize, flagcxMemDevice, NULL);
    if (res != flagcxSuccess) {
      WARN("defaultDevApiCommCreate: epochBuffer malloc failed (%d)", res);
      return res;
    }
    res = deviceAdaptor->deviceMemset(devComm->epochBuffer, 0, epochBufSize,
                                      flagcxMemDevice, NULL);
    if (res != flagcxSuccess) {
      WARN("defaultDevApiCommCreate: epochBuffer memset failed (%d)", res);
      return res;
    }
    INFO(FLAGCX_INIT, "defaultDevApiCommCreate: epochBuffer OK");
  }

  // One-sided buffers: signals, counters, staging
  INFO(FLAGCX_INIT,
       "defaultDevApiCommCreate: nInterPeers=%d interSignalCount=%d "
       "interCounterCount=%d",
       devComm->nInterPeers, reqs->interSignalCount, reqs->interCounterCount);
  if (devComm->nInterPeers > 0 &&
      (reqs->interSignalCount > 0 || reqs->interCounterCount > 0)) {
    int bufCtxCount =
        (comm->heteroComm != nullptr)
            ? comm->heteroComm->proxyState->kernelState.contextCount
            : devComm->contextCount;
    if (bufCtxCount < devComm->contextCount)
      bufCtxCount = devComm->contextCount;
    INFO(FLAGCX_INIT, "defaultDevApiCommCreate: bufCtxCount=%d contextCount=%d",
         bufCtxCount, devComm->contextCount);

    flagcxResult_t res;

    // Signal buffer (host-pinned or GDR device memory)
    if (reqs->interSignalCount > 0) {
      devComm->signalCount = reqs->interSignalCount;
      size_t sigSize =
          (size_t)devComm->signalCount * bufCtxCount * sizeof(uint64_t);
      INFO(
          FLAGCX_INIT,
          "defaultDevApiCommCreate: signalBuffer sigSize=%zu (count=%d ctx=%d)",
          sigSize, devComm->signalCount, bufCtxCount);
      if (flagcxParamSignalHostEnable()) {
        res = deviceAdaptor->deviceMalloc((void **)&devComm->signalBuffer,
                                          sigSize, flagcxMemHost, NULL);
        if (res != flagcxSuccess) {
          WARN("defaultDevApiCommCreate: signalBuffer host malloc failed (%d)",
               res);
          return res;
        }
        memset(devComm->signalBuffer, 0, sigSize);
      } else {
        res = deviceAdaptor->gdrMemAlloc((void **)&devComm->signalBuffer,
                                         sigSize, NULL);
        if (res != flagcxSuccess) {
          WARN("defaultDevApiCommCreate: signalBuffer gdrMemAlloc failed (%d)",
               res);
          return res;
        }
        res = deviceAdaptor->deviceMemset(devComm->signalBuffer, 0, sigSize,
                                          flagcxMemDevice, NULL);
        if (res != flagcxSuccess) {
          WARN("defaultDevApiCommCreate: signalBuffer memset failed (%d)", res);
          return res;
        }
      }
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: signalBuffer OK at %p",
           devComm->signalBuffer);
      res = deviceAdaptor->deviceMalloc((void **)&devComm->shadowBuffer,
                                        sigSize, flagcxMemDevice, NULL);
      if (res != flagcxSuccess) {
        WARN("defaultDevApiCommCreate: shadowBuffer malloc failed (%d)", res);
        return res;
      }
      res = deviceAdaptor->deviceMemset(devComm->shadowBuffer, 0, sigSize,
                                        flagcxMemDevice, NULL);
      if (res != flagcxSuccess) {
        WARN("defaultDevApiCommCreate: shadowBuffer memset failed (%d)", res);
        return res;
      }
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: shadowBuffer OK");
    }

    // Counter buffer (host-pinned)
    if (reqs->interCounterCount > 0) {
      devComm->counterCount = reqs->interCounterCount;
      size_t cntSize =
          (size_t)devComm->counterCount * bufCtxCount * sizeof(uint64_t);
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: counterBuffer cntSize=%zu",
           cntSize);
      res = deviceAdaptor->deviceMalloc((void **)&devComm->counterBuffer,
                                        cntSize, flagcxMemHost, NULL);
      if (res != flagcxSuccess) {
        WARN("defaultDevApiCommCreate: counterBuffer malloc failed (%d)", res);
        return res;
      }
      memset(devComm->counterBuffer, 0, cntSize);
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: counterBuffer OK");
    }

    // PutValue staging buffer
    size_t stagingSize = (size_t)comm->heteroComm->nRanks * sizeof(uint64_t);
    INFO(FLAGCX_INIT, "defaultDevApiCommCreate: stagingBuffer size=%zu",
         stagingSize);
    res = deviceAdaptor->deviceMalloc((void **)&devComm->putValueStagingBuffer,
                                      stagingSize, flagcxMemHost, NULL);
    if (res != flagcxSuccess) {
      WARN("defaultDevApiCommCreate: stagingBuffer malloc failed (%d)", res);
      return res;
    }
    memset(devComm->putValueStagingBuffer, 0, stagingSize);
    INFO(FLAGCX_INIT, "defaultDevApiCommCreate: stagingBuffer OK");

    // Register signal buffer for RDMA one-sided access
    if (devComm->signalBuffer) {
      int sigPtrType =
          flagcxParamSignalHostEnable() ? FLAGCX_PTR_HOST : FLAGCX_PTR_CUDA;
      INFO(FLAGCX_INIT,
           "defaultDevApiCommCreate: registering signalBuffer (ptrType=%d)",
           sigPtrType);
      res = flagcxOneSideSignalRegister(comm, devComm->signalBuffer,
                                        (size_t)devComm->signalCount *
                                            bufCtxCount * sizeof(uint64_t),
                                        sigPtrType);
      if (res != flagcxSuccess) {
        WARN("defaultDevApiCommCreate: flagcxOneSideSignalRegister failed (%d)",
             res);
        return res;
      }
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: signalRegister OK");
    }

    // Register staging buffer for PutValue RDMA source
    if (devComm->putValueStagingBuffer) {
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: registering stagingBuffer");
      res = flagcxOneSideStagingRegister(comm, devComm->putValueStagingBuffer,
                                         stagingSize);
      if (res != flagcxSuccess) {
        WARN(
            "defaultDevApiCommCreate: flagcxOneSideStagingRegister failed (%d)",
            res);
        return res;
      }
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: stagingRegister OK");
    }

    INFO(FLAGCX_INIT,
         "defaultDevApiCommCreate: one-sided buffers allocated "
         "(signals=%d, counters=%d, contexts=%d)",
         devComm->signalCount, devComm->counterCount, devComm->contextCount);
  }

  // Pre-establish full-mesh connections from main thread
  INFO(FLAGCX_INIT, "defaultDevApiCommCreate: calling preconnectFullMesh");
  {
    flagcxResult_t res = preconnectFullMesh(comm);
    if (res != flagcxSuccess) {
      WARN("defaultDevApiCommCreate: preconnectFullMesh failed (%d)", res);
      return res;
    }
  }
  INFO(FLAGCX_INIT, "defaultDevApiCommCreate: preconnectFullMesh OK");

  return flagcxSuccess;
}

// ==========================================================================
// DevComm Destroy — immediate cleanup, no deferral
// ==========================================================================

static flagcxResult_t defaultDevApiCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
  // ── IPC slot: immediate full cleanup ──────────────────────────────────
  if (comm != nullptr && devComm->barrierIpcIndex >= 0 &&
      devComm->barrierIpcIndex < FLAGCX_MAX_IPC_ENTRIES) {
    struct flagcxIpcTableEntry *e = &comm->ipcTable[devComm->barrierIpcIndex];
    if (e->hostPeerPtrs) {
      for (int i = 0; i < e->nPeers; i++) {
        if (e->hostPeerPtrs[i] && e->hostPeerPtrs[i] != e->basePtr)
          deviceAdaptor->ipcMemHandleClose(e->hostPeerPtrs[i]);
      }
      free(e->hostPeerPtrs);
      e->hostPeerPtrs = nullptr;
    }
    if (e->devPeerPtrs) {
      deviceAdaptor->deviceFree(e->devPeerPtrs, flagcxMemDevice, NULL);
      e->devPeerPtrs = nullptr;
    }
    e->inUse = false;
  }

  // ── Shm path cleanup (FLAGCX_SIGNAL_HOST_ENABLE=1 only) ──────────────
  if (devComm->peerBarrierShmPtrs) {
    for (int lr = 0; lr < devComm->nLocalRanks; lr++) {
      if (devComm->peerBarrierShmPtrs[lr])
        shmHostUnregister(devComm->peerBarrierShmPtrs[lr]);
    }
    free(devComm->peerBarrierShmPtrs);
    devComm->peerBarrierShmPtrs = nullptr;
  }
  if (devComm->localBarrierShmPtr) {
    shmHostUnregister(devComm->localBarrierShmPtr);
    // Close peer shm handles (skip own slot which equals myShmHandle)
    if (devComm->peerShmHandles) {
      for (int lr = 0; lr < devComm->nLocalRanks; lr++) {
        if (devComm->peerShmHandles[lr] &&
            devComm->peerShmHandles[lr] != devComm->myShmHandle)
          flagcxShmClose(devComm->peerShmHandles[lr]);
      }
      free(devComm->peerShmHandles);
      devComm->peerShmHandles = nullptr;
    }
    flagcxShmClose(devComm->myShmHandle);
    devComm->localBarrierShmPtr = nullptr;
  }
  if (devComm->barrierDevPeerPtrsRaw) {
    deviceAdaptor->deviceFree(devComm->barrierDevPeerPtrsRaw, flagcxMemDevice,
                              NULL);
    devComm->barrierDevPeerPtrsRaw = nullptr;
  }

  // ── MR deregistration ────────────────────────────────────────────────
  // (signal and staging MR are deregistered at comm level in commCleanup)

  // ── Free one-sided buffers immediately ────────────────────────────────
  if (devComm->localBarrierFlags) {
    deviceAdaptor->deviceFree(devComm->localBarrierFlags, flagcxMemDevice,
                              NULL);
    devComm->localBarrierFlags = nullptr;
  }
  if (devComm->epochBuffer) {
    deviceAdaptor->deviceFree(devComm->epochBuffer, flagcxMemDevice, NULL);
    devComm->epochBuffer = nullptr;
  }
  if (devComm->signalBuffer) {
    if (flagcxParamSignalHostEnable())
      deviceAdaptor->deviceFree(devComm->signalBuffer, flagcxMemHost, NULL);
    else
      deviceAdaptor->gdrMemFree(devComm->signalBuffer, NULL);
    devComm->signalBuffer = nullptr;
  }
  if (devComm->shadowBuffer) {
    deviceAdaptor->deviceFree(devComm->shadowBuffer, flagcxMemDevice, NULL);
    devComm->shadowBuffer = nullptr;
  }
  if (devComm->counterBuffer) {
    deviceAdaptor->deviceFree(devComm->counterBuffer, flagcxMemHost, NULL);
    devComm->counterBuffer = nullptr;
  }
  if (devComm->putValueStagingBuffer) {
    deviceAdaptor->deviceFree(devComm->putValueStagingBuffer, flagcxMemHost,
                              NULL);
    devComm->putValueStagingBuffer = nullptr;
  }
  free(devComm->localRankToRank);
  devComm->localRankToRank = nullptr;

  // ── Free cached device pointers immediately ───────────────────────────
  if (devComm->cachedDevicePtr) {
    deviceAdaptor->deviceFree(devComm->cachedDevicePtr, flagcxMemDevice, NULL);
    devComm->cachedDevicePtr = nullptr;
  }
  if (devComm->cachedNetContextsPtr) {
    deviceAdaptor->deviceFree(devComm->cachedNetContextsPtr, flagcxMemDevice,
                              NULL);
    devComm->cachedNetContextsPtr = nullptr;
  }

  return flagcxSuccess;
}

// ==========================================================================
// DevMem Create
// ==========================================================================

static flagcxResult_t defaultDevApiMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t devMem) {
  // ---- Per-comm MR layer: lookup buff in heteroComm->oneSideHandles ----
  devMem->mrIndex = -1;
  devMem->mrBase = 0;
  if (comm != nullptr && comm->heteroComm != nullptr) {
    struct flagcxHeteroComm *hc = comm->heteroComm;
    for (int i = 0; i < hc->oneSideHandleCount; i++) {
      struct flagcxOneSideHandleInfo *info = hc->oneSideHandles[i];
      if (info != NULL && info->baseVas != NULL) {
        uintptr_t base = info->baseVas[comm->rank];
        if ((uintptr_t)buff == base) {
          devMem->mrIndex = i;
          devMem->mrBase = base;
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
    devMem->intraRank = comm->localRank;

    // ---- Priority 1 & 2: Symmetric default window (VMM or IPC fallback) ----
    if (win != nullptr && win->isSymmetricDefault) {
      flagcxSymWindow_t d = win->defaultBase;
      devMem->hasWindow = true;
      devMem->isSymmetric = true;
      devMem->winHandle = (void *)win;
      if (d != nullptr && d->mrIndex >= 0) {
        devMem->mrIndex = d->mrIndex;
        devMem->mrBase = d->mrBase;
      }
      if (d == nullptr || !d->isVMM || !d->flatBase) {
        // Priority 2: Symmetric IPC fallback (VMM not available)
        int idx = buildIpcPeerPointers(comm, buff, size);
        if (idx >= 0) {
          devMem->ipcIndex = idx;
        } else {
          WARN("flagcxDevMemCreate: symmetric window VMM failed and IPC "
               "fallback also failed — no peer access");
        }
      }
      devMem->window = nullptr;
    }
    // ---- Priority 3: Vendor native window ----
    else if (win != nullptr && !win->isSymmetricDefault) {
      devMem->hasWindow = true;
      devMem->isSymmetric = (win->winFlags & FLAGCX_WIN_COLL_SYMMETRIC);
      devMem->winHandle = (void *)win;
    }
    // ---- Priority 4 & 5: No window — IPC ----
    else if (win == nullptr) {
      // Check if buffer supports IPC (VMM-allocated buffers do not support
      // cudaIpcGetMemHandle). Probe via ipcMemHandleGet before entering the
      // collective buildIpcPeerPointers.
      bool ipcCapable = false;
      if (deviceAdaptor->ipcMemHandleCreate && deviceAdaptor->ipcMemHandleGet &&
          deviceAdaptor->ipcMemHandleFree) {
        flagcxIpcMemHandle_t probe = NULL;
        size_t probeSize = 0;
        if (deviceAdaptor->ipcMemHandleCreate(&probe, &probeSize) ==
            flagcxSuccess) {
          if (deviceAdaptor->ipcMemHandleGet(probe, buff) == flagcxSuccess) {
            ipcCapable = true;
          }
          deviceAdaptor->ipcMemHandleFree(probe);
        }
      }

      if (!ipcCapable) {
        INFO(FLAGCX_INIT,
             "flagcxDevMemCreate: buff %p not IPC-capable, skipping IPC", buff);
      } else {
        int existingIdx = -1;
        for (int i = 0; i < FLAGCX_MAX_IPC_ENTRIES; i++) {
          if (comm->ipcTable[i].inUse && comm->ipcTable[i].basePtr == buff) {
            existingIdx = i;
            break;
          }
        }
        if (existingIdx >= 0) {
          devMem->ipcIndex = existingIdx;
        } else {
          int idx = buildIpcPeerPointers(comm, buff, size);
          if (idx >= 0) {
            devMem->ipcIndex = idx;
          } else {
            WARN("flagcxDevMemCreate: IPC peer pointer setup failed, "
                 "IPC layer not available");
          }
        }
      }
    }
  }

  // Allocate and populate kernel Window uniformly via traits
  {
    auto *kWin = new (std::nothrow) typename DeviceAPI::Window{};
    if (kWin == nullptr) {
      WARN("flagcxDevMemCreate: failed to allocate DeviceAPI::Window");
      return flagcxSystemError;
    }
    kWin->populateFromHost(win, devMem->rawPtr, devMem->intraRank,
                           devMem->mrIndex, devMem->mrBase, devMem->ipcIndex,
                           (devMem->ipcIndex >= 0 && comm)
                               ? comm->ipcTable[devMem->ipcIndex].devPeerPtrs
                               : nullptr);
    devMem->window = kWin;
    devMem->hasWindow = kWin->hasAccess();

    if (!devMem->hasWindow && win != nullptr && win->isSymmetricDefault) {
      flagcxSymWindow_t d = win->defaultBase;
      WARN("flagcxDevMemCreate: kWin->hasAccess() returned false for symmetric "
           "default window. ipcIndex=%d, intraRank=%d, "
           "defaultBase=%p, isVMM=%d, flatBase=%p, ipcDevPeerPtrs=%p",
           devMem->ipcIndex, devMem->intraRank, (void *)d, (d ? d->isVMM : -1),
           (d ? d->flatBase : nullptr),
           (devMem->ipcIndex >= 0 && comm)
               ? (void *)comm->ipcTable[devMem->ipcIndex].devPeerPtrs
               : nullptr);
      delete kWin;
      return flagcxInvalidUsage;
    }
  }

  return flagcxSuccess;
}

// ==========================================================================
// DevMem Destroy
// ==========================================================================

static flagcxResult_t defaultDevApiMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  // Mark IPC table entry as no longer in use
  if (comm != nullptr && devMem->ipcIndex >= 0 &&
      devMem->ipcIndex < FLAGCX_MAX_IPC_ENTRIES) {
    comm->ipcTable[devMem->ipcIndex].inUse = false;
  }

  // Free window allocation if present
  if (devMem->window != nullptr) {
    delete static_cast<typename DeviceAPI::Window *>(devMem->window);
  }

  // Free cached device pointer immediately
  if (devMem->cachedDevicePtr) {
    deviceAdaptor->deviceFree(devMem->cachedDevicePtr, flagcxMemDevice, NULL);
  }

  return flagcxSuccess;
}

// ==========================================================================
// Device Pointer API — for Triton integration
// ==========================================================================

static flagcxResult_t defaultDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                    void **devPtr) {
  if (!devComm || !devPtr)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devComm->cachedPtrMutex);

  if (devComm->cachedDevicePtr) {
    *devPtr = devComm->cachedDevicePtr;
    pthread_mutex_unlock(&devComm->cachedPtrMutex);
    return flagcxSuccess;
  }

  // Construct value struct on host stack
  flagcxDevComm hostCopy(*devComm);
  hostCopy._netContexts = nullptr;

  // Step 1: Copy flagcxDevComm to device
  void *dPtr = nullptr;
  void *netDevPtr = nullptr;
  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&dPtr, sizeof(flagcxDevComm),
                                              flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMemcpy(dPtr, &hostCopy, sizeof(flagcxDevComm),
                                  flagcxMemcpyHostToDevice, NULL, NULL),
      res, fail);

  // Step 2: Allocate + construct net array on device
  if (hostCopy._contextCount > 0 && flagcxDevNetSizeOf() > 0) {
    size_t netArraySize = hostCopy._contextCount * flagcxDevNetSizeOf();
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&netDevPtr, netArraySize,
                                                flagcxMemDevice, NULL),
                    res, fail);
    flagcxDevNetLaunchConstruct(netDevPtr, dPtr, hostCopy._contextCount,
                                nullptr);
    void *netCtxField = (char *)dPtr + offsetof(flagcxDevComm, _netContexts);
    FLAGCXCHECKGOTO(
        deviceAdaptor->deviceMemcpy(netCtxField, &netDevPtr, sizeof(void *),
                                    flagcxMemcpyHostToDevice, NULL, NULL),
        res, fail);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceSynchronize(), res, fail);
  }

  devComm->cachedDevicePtr = dPtr;
  devComm->cachedNetContextsPtr = netDevPtr;
  *devPtr = dPtr;
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  return flagcxSuccess;

fail:
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  if (netDevPtr) {
    deviceAdaptor->deviceFree(netDevPtr, flagcxMemDevice, NULL);
  }
  if (dPtr) {
    deviceAdaptor->deviceFree(dPtr, flagcxMemDevice, NULL);
  }
  return res;
}

static flagcxResult_t defaultDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  if (!devComm)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devComm->cachedPtrMutex);
  void *ptr = devComm->cachedDevicePtr;
  void *netPtr = devComm->cachedNetContextsPtr;
  devComm->cachedDevicePtr = nullptr;
  devComm->cachedNetContextsPtr = nullptr;
  pthread_mutex_unlock(&devComm->cachedPtrMutex);

  if (netPtr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(netPtr, flagcxMemDevice, NULL));
  }
  if (ptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(ptr, flagcxMemDevice, NULL));
  }
  return flagcxSuccess;
}

static flagcxResult_t defaultDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
                                                   void **devPtr) {
  if (!devMem || !devPtr)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devMem->cachedPtrMutex);

  if (devMem->cachedDevicePtr) {
    *devPtr = devMem->cachedDevicePtr;
    pthread_mutex_unlock(&devMem->cachedPtrMutex);
    return flagcxSuccess;
  }

  flagcxDevMem hostCopy(*devMem);

  void *dPtr = nullptr;
  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&dPtr, sizeof(flagcxDevMem),
                                              flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMemcpy(dPtr, &hostCopy, sizeof(flagcxDevMem),
                                  flagcxMemcpyHostToDevice, NULL, NULL),
      res, fail);

  devMem->cachedDevicePtr = dPtr;
  *devPtr = dPtr;
  pthread_mutex_unlock(&devMem->cachedPtrMutex);
  return flagcxSuccess;

fail:
  pthread_mutex_unlock(&devMem->cachedPtrMutex);
  if (dPtr) {
    deviceAdaptor->deviceFree(dPtr, flagcxMemDevice, NULL);
  }
  return res;
}

static flagcxResult_t defaultDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
  if (!devMem)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devMem->cachedPtrMutex);
  void *ptr = devMem->cachedDevicePtr;
  devMem->cachedDevicePtr = nullptr;
  pthread_mutex_unlock(&devMem->cachedPtrMutex);

  if (ptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(ptr, flagcxMemDevice, NULL));
  }
  return flagcxSuccess;
}

// ==========================================================================
// Comm-level cleanup — relay teardown + IPC table
// ==========================================================================

static flagcxResult_t defaultCommCleanup(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;

  // Tear down inter-node signal relay
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero != nullptr && hetero->relayInitialized &&
      hetero->nInterPeers > 0) {
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
    bootstrapCollBarrier(comm->bootstrap, comm->rank, comm->nranks, 0x7f01);

    free(hetero->interPeerRanks);
    hetero->interPeerRanks = nullptr;

    if (hetero->isInterLeader) {
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
        deviceAdaptor->deviceFree(hetero->interSignalFlagsHost, flagcxMemHost,
                                  NULL);
        hetero->interSignalFlagsHost = nullptr;
      }
    }
    hetero->relayInitialized = false;
  }

  return flagcxSuccess;
}

// ==========================================================================
// Backend vtable
// ==========================================================================

static struct flagcxDevApiBackend defaultBackend = {
    .name = "default",
    .devCommCreate = defaultDevApiCommCreate,
    .devCommDestroy = defaultDevApiCommDestroy,
    .devMemCreate = defaultDevApiMemCreate,
    .devMemDestroy = defaultDevApiMemDestroy,
    .devCommGetDevicePtr = defaultDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = defaultDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = defaultDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = defaultDevApiMemFreeDevicePtr,
    .commCleanup = defaultCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &defaultBackend;
