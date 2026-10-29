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
#include "alloc.h"
#include "bootstrap.h"
#include "dev_api_backend.h"
#include "device_api/completion_word.h"
#include "device_api/flagcx_device.h"
#include "flagcx_hetero.h"
#include "flagcx_kernel_internal.h"
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
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <sched.h>
#include <unistd.h>

using DefaultCompletionWord = typename DeviceAPI::CompletionWord;

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
    WARN("host-mapped device API: hostRegister not supported");
    return flagcxNotSupported;
  }
  return deviceAdaptor->hostRegister(ptr, bytes);
}

static void shmHostUnregister(void *ptr) {
  if (deviceAdaptor->hostUnregister)
    deviceAdaptor->hostUnregister(ptr);
}

static bool
registrationMatchesBuffer(flagcxComm_t comm,
                          const struct flagcxOneSideHandleInfo *registration,
                          const void *buffer) {
  if (comm == nullptr || comm->heteroComm == nullptr ||
      registration == nullptr || registration->baseVas == nullptr ||
      comm->rank < 0 || comm->rank >= comm->heteroComm->nRanks) {
    return false;
  }
  return registration->baseVas[comm->rank] == (uintptr_t)buffer;
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

  if (nNodes <= 1) {
    devComm->nInterPeers = 0;
    return flagcxSuccess;
  }

  // Already initialized: just copy pointers into devComm
  if (hetero->relayInitialized) {
    devComm->nInterPeers = hetero->nInterPeers;
    devComm->interPeerRanks = hetero->interPeerRanks;
    devComm->teamRank = hetero->node;
    devComm->nTeamRanks = hetero->nNodes;
    devComm->barrierSignalBase = 0;
    return flagcxSuccess;
  }

  // First call: compute inter-node peer info
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

  // All ranks barrier to ensure inter-node peer info is consistent
  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECKGOTO(
      bootstrapCollBarrier(comm->bootstrap, comm->rank, comm->nranks, 0xAB01),
      res, relay_fail);

  hetero->relayInitialized = true;

  // Copy into devComm
  devComm->nInterPeers = hetero->nInterPeers;
  devComm->interPeerRanks = hetero->interPeerRanks;
  devComm->teamRank = hetero->node;
  devComm->nTeamRanks = hetero->nNodes;
  devComm->barrierSignalBase = 0;

  INFO(FLAGCX_INIT,
       "setupInterNodeSignalRelay: rank %d nInterPeers=%d teamRank=%d "
       "nTeamRanks=%d",
       myRank, nInterPeers, hetero->node, nNodes);
  return flagcxSuccess;

relay_fail:
  free(hetero->interPeerRanks);
  hetero->interPeerRanks = nullptr;
  hetero->nInterPeers = 0;
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
    flagcxResult_t res =
        deviceAdaptor->deviceMalloc((void **)&devComm->localBarrierFlags,
                                    barrierSize, flagcxMemDevice, NULL);
    if (res != flagcxSuccess)
      return res;
    devComm->localBarrierFlagsDeviceAllocated = true;
    res = deviceAdaptor->deviceMemset(devComm->localBarrierFlags, 0,
                                      barrierSize, flagcxMemDevice, NULL);
    if (res != flagcxSuccess)
      return res;

    int slot =
        buildIpcPeerPointers(comm, devComm->localBarrierFlags, barrierSize);
    if (slot < 0)
      return flagcxSystemError;

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
         myRank, slot, (void *)devComm->localBarrierFlags,
         (void *)devComm->barrierPeers, devComm->nBarriers);

  } else {
    // ── flagcxShm + mapped host-memory barrier path ──
    flagcxResult_t res = flagcxSuccess;
    void **peerCpuPtrs = nullptr;
    flagcxShmHandle_t *peerShmHandles = nullptr;
    void **devPeerPtrs = nullptr;
    void *myDevPtr = nullptr;
    void **hostDevPtrs = nullptr;
    int registeredPeerCount = 0;

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
    peerCpuPtrs = (void **)calloc(localRanks, sizeof(void *));
    peerShmHandles =
        (flagcxShmHandle_t *)calloc(localRanks, sizeof(flagcxShmHandle_t));
    if (!peerCpuPtrs || !peerShmHandles) {
      res = flagcxSystemError;
      goto fail_peer_shm;
    }

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
                                      &peerPtr, nullptr, -1, &peerHandle),
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
      registeredPeerCount++;
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
    devComm->registeredBarrierShmCount = registeredPeerCount;
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
    for (int lr = 0; lr < registeredPeerCount; lr++)
      shmHostUnregister(peerCpuPtrs[lr]);
    free(hostDevPtrs);
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

// Each rank owns one SHM segment with disjoint local-GPU and network signal
// planes. Local peers register that segment and use its GPU alias for direct
// atomics; the NIC registers the owning rank's network plane through its CPU
// address. Keeping the planes separate avoids mixing GPU and NIC atomics on
// the same word when a WORLD communicator has both local and remote senders.
static flagcxResult_t setupHostSignals(flagcxComm_t comm,
                                       flagcxDevComm_t devComm,
                                       size_t planeBytes) {
  // Keep GPU and NIC atomics on different pages, including for small signal
  // counts whose unpadded planes would otherwise share a cache line.
  const long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0)
    return flagcxSystemError;
  const size_t planeAlignment = (size_t)pageSize;
  if (planeBytes == 0 || planeBytes > SIZE_MAX - (planeAlignment - 1) ||
      comm->localRanks <= 0 || comm->localRank < 0 ||
      comm->localRank >= comm->localRanks)
    return flagcxInvalidArgument;
  if (deviceAdaptor->hostGetDevicePointer == nullptr) {
    WARN("host-mapped signals: hostGetDevicePointer not supported");
    return flagcxNotSupported;
  }

  const size_t planeStride =
      ((planeBytes + planeAlignment - 1) / planeAlignment) * planeAlignment;
  if (planeStride > (SIZE_MAX - sizeof(int)) / 2)
    return flagcxInvalidArgument;
  const size_t shmBytes = 2 * planeStride;
  const int localRanks = comm->localRanks;
  devComm->nLocalRanks = localRanks;
  flagcxResult_t res = flagcxSuccess;
  void **hostDevicePtrs = nullptr;
  void *myCpuPtr = nullptr;
  char myPath[SHM_PATH_MAX];
  snprintf(myPath, sizeof(myPath), "/dev/shm/flagcx_signal_%016llx_%d",
           (unsigned long long)comm->magic, comm->rank);

  FLAGCXCHECKGOTO(flagcxShmOpen(myPath, sizeof(myPath), shmBytes, &myCpuPtr,
                                nullptr, localRanks,
                                &devComm->mySignalShmHandle),
                  res, fail);
  devComm->signalHostBuffer = myCpuPtr;
  FLAGCXCHECKGOTO(
      bootstrapCollBarrier(comm->bootstrap, comm->rank, comm->nranks, 0x5A01),
      res, fail);

  devComm->peerSignalShmPtrs = (void **)calloc(localRanks, sizeof(void *));
  devComm->peerSignalShmHandles =
      (flagcxShmHandle_t *)calloc(localRanks, sizeof(flagcxShmHandle_t));
  hostDevicePtrs = (void **)malloc(localRanks * sizeof(void *));
  if (!devComm->peerSignalShmPtrs || !devComm->peerSignalShmHandles ||
      !hostDevicePtrs) {
    res = flagcxSystemError;
    goto fail;
  }

  for (int lr = 0; lr < localRanks; lr++) {
    int rank = comm->localRankToRank[lr];
    if (rank == comm->rank) {
      devComm->peerSignalShmPtrs[lr] = myCpuPtr;
      devComm->peerSignalShmHandles[lr] = devComm->mySignalShmHandle;
    } else {
      char peerPath[SHM_PATH_MAX];
      snprintf(peerPath, sizeof(peerPath), "/dev/shm/flagcx_signal_%016llx_%d",
               (unsigned long long)comm->magic, rank);
      FLAGCXCHECKGOTO(flagcxShmOpen(peerPath, sizeof(peerPath), shmBytes,
                                    &devComm->peerSignalShmPtrs[lr], nullptr,
                                    -1, &devComm->peerSignalShmHandles[lr]),
                      res, fail);
    }
  }

  for (int lr = 0; lr < localRanks; lr++) {
    FLAGCXCHECKGOTO(shmHostRegister(devComm->peerSignalShmPtrs[lr], shmBytes),
                    res, fail);
    devComm->registeredSignalShmCount++;
    FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(
                        &hostDevicePtrs[lr], devComm->peerSignalShmPtrs[lr]),
                    res, fail);
  }

  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(
                      (void **)&devComm->signalDevPeerPtrsRaw,
                      localRanks * sizeof(void *), flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMemcpy(devComm->signalDevPeerPtrsRaw, hostDevicePtrs,
                                  localRanks * sizeof(void *),
                                  flagcxMemcpyHostToDevice, NULL, NULL),
      res, fail);

  devComm->signalPeerPtrs = devComm->signalDevPeerPtrsRaw;
  devComm->completionSignalPeerPtrs = (void **)devComm->signalDevPeerPtrsRaw;
  devComm->localSignalBuffer = (uint64_t *)hostDevicePtrs[comm->localRank];
  devComm->signalBuffer =
      devComm->localSignalBuffer + planeStride / sizeof(uint64_t);
  devComm->signalMrHostPtr =
      (uint64_t *)myCpuPtr + planeStride / sizeof(uint64_t);
  INFO(FLAGCX_INIT,
       "setupHostSignals(SHM): rank %d local=%p network=%p mrHost=%p peers=%d",
       comm->rank, (void *)devComm->localSignalBuffer,
       (void *)devComm->signalBuffer, devComm->signalMrHostPtr, localRanks);
  free(hostDevicePtrs);
  return flagcxSuccess;

fail:
  free(hostDevicePtrs);
  // The caller invokes defaultDevApiCommDestroy on a failed create. All
  // successfully acquired SHM mappings and registrations remain tracked there.
  return res;
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
                        size_t reqsSize, flagcxDevComm_t devComm) {
  // The default backend consumes only fields in the frozen prefix.
  (void)reqsSize;
  const bool forceOneSidedNet = flagcxParamDeviceOneSidedForceNet() != 0;
  const bool hostMappedSignals =
      flagcxParamSignalHostEnable() != 0 &&
      sizeof(DefaultCompletionWord) == sizeof(uint64_t);
  const bool oneSidedRequested =
      reqs->interSignalCount > 0 || reqs->interCounterCount > 0;
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
    }
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
  // Allocated when signal/counter slots are requested — needed for both
  // inter-node (Net FIFO) and intra-node (P2P atomic) paths.
  INFO(FLAGCX_INIT,
       "defaultDevApiCommCreate: nInterPeers=%d interSignalCount=%d "
       "interCounterCount=%d",
       devComm->nInterPeers, reqs->interSignalCount, reqs->interCounterCount);
  if (reqs->interSignalCount > 0 || reqs->interCounterCount > 0) {
    int bufCtxCount =
        (comm->heteroComm != nullptr && comm->heteroComm->proxyState != nullptr)
            ? comm->heteroComm->proxyState->kernelState.contextCount
            : devComm->contextCount;
    if (bufCtxCount < devComm->contextCount)
      bufCtxCount = devComm->contextCount;
    INFO(FLAGCX_INIT, "defaultDevApiCommCreate: bufCtxCount=%d contextCount=%d",
         bufCtxCount, devComm->contextCount);

    flagcxResult_t res;

    // Signal buffer (host-pinned or GDR device memory)
    if (reqs->interSignalCount > 0) {
      // Expand signalCount to include barrier slots:
      // barrier needs nTeamRanks slots per barrier instance (one per CTA)
      int userSignals = reqs->interSignalCount;
      if (devComm->nTeamRanks < 0 || reqs->interBarrierCount < 0 ||
          (reqs->interBarrierCount > 0 &&
           devComm->nTeamRanks >
               std::numeric_limits<int>::max() / reqs->interBarrierCount)) {
        WARN("barrierSignalBase overflow: nTeamRanks=%d interBarrierCount=%d",
             devComm->nTeamRanks, reqs->interBarrierCount);
        return flagcxInternalError;
      }
      int barrierSlots = devComm->nTeamRanks * reqs->interBarrierCount;
      if (userSignals < 0 || barrierSlots < 0 ||
          userSignals > std::numeric_limits<int>::max() - barrierSlots ||
          bufCtxCount <= 0 ||
          (size_t)(userSignals + barrierSlots) >
              SIZE_MAX / (size_t)bufCtxCount / sizeof(uint64_t)) {
        WARN("defaultDevApiCommCreate: signal buffer size overflow");
        return flagcxInvalidArgument;
      }
      devComm->signalCount = userSignals + barrierSlots;
      devComm->barrierSignalBase = userSignals;
      size_t sigSize =
          (size_t)devComm->signalCount * bufCtxCount * sizeof(uint64_t);
      devComm->signalBufferIsVmm = false;
      INFO(
          FLAGCX_INIT,
          "defaultDevApiCommCreate: signalBuffer sigSize=%zu (count=%d ctx=%d)",
          sigSize, devComm->signalCount, bufCtxCount);
      if (hostMappedSignals) {
        res = setupHostSignals(comm, devComm, sigSize);
        if (res != flagcxSuccess) {
          WARN("defaultDevApiCommCreate: host signal SHM setup failed (%d)",
               res);
          return res;
        }
      } else if (flagcxParamSignalHostEnable()) {
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
        devComm->signalBufferIsVmm = flagcxDeviceAdaptorNativeAllocIsVmm(
            deviceAdaptor, flagcxParamVmmEnable());
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

      if (sizeof(DefaultCompletionWord) == sizeof(uint32_t)) {
        size_t directSigSize = (size_t)devComm->signalCount * bufCtxCount *
                               sizeof(DefaultCompletionWord);
        res = deviceAdaptor->deviceMalloc(&devComm->completionSignalBuffer,
                                          directSigSize, flagcxMemDevice, NULL);
        if (res != flagcxSuccess) {
          WARN("defaultDevApiCommCreate: 32-bit completion signal allocation "
               "failed (%d)",
               res);
          return res;
        }
        res = deviceAdaptor->deviceMemset(devComm->completionSignalBuffer, 0,
                                          directSigSize, flagcxMemDevice, NULL);
        if (res != flagcxSuccess)
          return res;
      } else {
        devComm->completionSignalBuffer = devComm->signalBuffer;
      }
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
      if (hostMappedSignals) {
        if (deviceAdaptor->hostGetDevicePointer == nullptr)
          return flagcxNotSupported;
        res = deviceAdaptor->hostGetDevicePointer(
            (void **)&devComm->counterDeviceBuffer, devComm->counterBuffer);
        if (res != flagcxSuccess)
          return res;
      }
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: counterBuffer OK");
      if (sizeof(DefaultCompletionWord) == sizeof(uint32_t)) {
        size_t directCntSize = (size_t)devComm->counterCount * bufCtxCount *
                               sizeof(DefaultCompletionWord);
        res = deviceAdaptor->deviceMalloc(&devComm->completionCounterBuffer,
                                          directCntSize, flagcxMemDevice, NULL);
        if (res != flagcxSuccess) {
          WARN("defaultDevApiCommCreate: 32-bit completion counter allocation "
               "failed (%d)",
               res);
          return res;
        }
        res = deviceAdaptor->deviceMemset(devComm->completionCounterBuffer, 0,
                                          directCntSize, flagcxMemDevice, NULL);
        if (res != flagcxSuccess)
          return res;
      } else {
        devComm->completionCounterBuffer = hostMappedSignals
                                               ? devComm->counterDeviceBuffer
                                               : devComm->counterBuffer;
      }
    }

    // Prepare the Net fallback even on a single node.  IPC setup can fail for
    // an otherwise valid local allocation, and PutValue then needs this
    // registered staging source to make the fallback complete.
    if (comm->heteroComm != nullptr) {
      devComm->putValueStagingSlotCount = FLAGCX_KERNEL_PROXY_PUT_VALUE_SLOTS;
      devComm->putValueStagingContextCount = bufCtxCount;
      size_t stagingSize = 0;
      if (!flagcxPutValueStagingLayout(comm->heteroComm->nRanks, bufCtxCount,
                                       devComm->putValueStagingSlotCount,
                                       &devComm->putValueStagingKernelBaseSlot,
                                       &devComm->putValueStagingTotalSlotCount,
                                       &stagingSize)) {
        WARN("defaultDevApiCommCreate: PutValue staging layout overflow "
             "(ranks=%d contexts=%d slots=%d)",
             comm->heteroComm->nRanks, bufCtxCount,
             devComm->putValueStagingSlotCount);
        return flagcxInvalidArgument;
      }
      INFO(FLAGCX_INIT, "defaultDevApiCommCreate: stagingBuffer size=%zu",
           stagingSize);
      res =
          deviceAdaptor->deviceMalloc((void **)&devComm->putValueStagingBuffer,
                                      stagingSize, flagcxMemHost, NULL);
      if (res != flagcxSuccess) {
        WARN("defaultDevApiCommCreate: stagingBuffer malloc failed (%d)", res);
        devComm->putValueStagingBuffer = nullptr;
      } else {
        memset(devComm->putValueStagingBuffer, 0, stagingSize);
        INFO(FLAGCX_INIT, "defaultDevApiCommCreate: stagingBuffer OK");
      }

      // Register signal buffer for RDMA one-sided access
      if (devComm->signalBuffer) {
        void *signalRegistrationBuffer = hostMappedSignals
                                             ? devComm->signalMrHostPtr
                                             : (void *)devComm->signalBuffer;
        void *previousSignalBase = comm->heteroComm->rmaSignalBase;
        struct flagcxOneSideHandleInfo *previousRegistration =
            comm->heteroComm->signalHandle;
        int sigPtrType =
            flagcxParamSignalHostEnable() ? FLAGCX_PTR_HOST : FLAGCX_PTR_CUDA;
        INFO(FLAGCX_INIT,
             "defaultDevApiCommCreate: registering signalBuffer (ptrType=%d)",
             sigPtrType);
        res = flagcxOneSideSignalRegisterInternal(
            comm, signalRegistrationBuffer,
            (size_t)devComm->signalCount * bufCtxCount * sizeof(uint64_t),
            sigPtrType, devComm->signalBufferIsVmm);
        struct flagcxOneSideHandleInfo *registration =
            comm->heteroComm->signalHandle;
        bool signalRegistered =
            res == flagcxSuccess &&
            comm->heteroComm->rmaSignalBase == signalRegistrationBuffer;
        bool networkSignalRegistered = registrationMatchesBuffer(
            comm, registration, signalRegistrationBuffer);
        if (!signalRegistered) {
          WARN("defaultDevApiCommCreate: flagcxOneSideSignalRegister failed "
               "or retained another buffer (%d, handle=%p, buffer=%p)",
               res, (void *)registration, signalRegistrationBuffer);
        } else {
          if (previousSignalBase == nullptr)
            devComm->ownedSignalBuffer = (uint64_t *)signalRegistrationBuffer;
          if (networkSignalRegistered && previousRegistration == nullptr)
            devComm->ownedSignalRegistration = registration;
          devComm->netSignalReady = networkSignalRegistered ? 1 : 0;
          INFO(FLAGCX_INIT,
               "defaultDevApiCommCreate: signalRegister OK (ipcSlot=%d, "
               "network=%d)",
               comm->heteroComm->rmaSignalIpcSlot,
               networkSignalRegistered ? 1 : 0);
        }
      }

      // Register staging buffer for PutValue RDMA source
      if (devComm->putValueStagingBuffer) {
        struct flagcxOneSideHandleInfo *previousRegistration =
            comm->heteroComm->stagingHandle;
        INFO(FLAGCX_INIT, "defaultDevApiCommCreate: registering stagingBuffer");
        res = flagcxOneSideStagingRegister(comm, devComm->putValueStagingBuffer,
                                           stagingSize);
        struct flagcxOneSideHandleInfo *registration =
            comm->heteroComm->stagingHandle;
        if (res != flagcxSuccess ||
            !registrationMatchesBuffer(comm, registration,
                                       devComm->putValueStagingBuffer)) {
          WARN("defaultDevApiCommCreate: flagcxOneSideStagingRegister failed "
               "or returned a registration for another buffer "
               "(%d, handle=%p, buffer=%p)",
               res, (void *)registration,
               (void *)devComm->putValueStagingBuffer);
        } else {
          if (previousRegistration == nullptr)
            devComm->ownedStagingRegistration = registration;
          devComm->netPutValueReady = 1;
          INFO(FLAGCX_INIT, "defaultDevApiCommCreate: stagingRegister OK");
        }
      }
    }

    INFO(FLAGCX_INIT,
         "defaultDevApiCommCreate: one-sided buffers allocated "
         "(signals=%d, counters=%d, contexts=%d)",
         devComm->signalCount, devComm->counterCount, devComm->contextCount);
  }

  // ==========================================================================
  // P2P signal/counter IPC setup (intra-node direct atomic fast path)
  // The direct completion plane is ordinary device memory so it can be
  // exported through the platform IPC mechanism.
  // ==========================================================================
  if (devComm->completionSignalBuffer && !forceOneSidedNet &&
      (sizeof(DefaultCompletionWord) == sizeof(uint32_t) ||
       !flagcxParamSignalHostEnable())) {
    size_t sigSize = (size_t)devComm->signalCount * devComm->contextCount *
                     sizeof(DefaultCompletionWord);
    int slot =
        buildIpcPeerPointers(comm, devComm->completionSignalBuffer, sigSize);
    if (slot >= 0) {
      devComm->completionSignalPeerPtrs = comm->ipcTable[slot].devPeerPtrs;
      devComm->signalIpcSlot = slot;
      if (sizeof(DefaultCompletionWord) == sizeof(uint64_t))
        devComm->signalPeerPtrs =
            (uint64_t **)devComm->completionSignalPeerPtrs;
      INFO(FLAGCX_INIT,
           "defaultDevApiCommCreate: completionSignalPeerPtrs IPC slot=%d "
           "ptr=%p wordBytes=%zu",
           slot, (void *)devComm->completionSignalPeerPtrs,
           sizeof(DefaultCompletionWord));
    } else {
      WARN("defaultDevApiCommCreate: signalPeerPtrs IPC exchange failed");
      // Non-fatal: falls back to Net FIFO path
    }
  }

  if (sizeof(DefaultCompletionWord) == sizeof(uint32_t)) {
    // The direct and proxy planes are independent, so local IPC and remote
    // Net signals may coexist in one communicator.
    devComm->useP2pSignals = (devComm->signalIpcSlot >= 0) ? 1 : 0;
  } else {
    // Local SHM signals and remote Net signals can coexist. GPU-memory
    // signals keep the existing whole-communicator transport choice.
    devComm->useP2pSignals =
        hostMappedSignals
            ? (!forceOneSidedNet && devComm->signalDevPeerPtrsRaw != nullptr)
            : (devComm->nInterPeers == 0 && devComm->signalIpcSlot >= 0);
  }

  if (devComm->signalBuffer &&
      (devComm->nInterPeers > 0 || !devComm->useP2pSignals ||
       forceOneSidedNet) &&
      !devComm->netSignalReady) {
    WARN("defaultDevApiCommCreate: neither signal IPC nor Net fallback is "
         "available");
    return flagcxNotSupported;
  }
  if (oneSidedRequested && forceOneSidedNet && !devComm->netPutValueReady) {
    WARN("defaultDevApiCommCreate: forced Net fallback requested but "
         "PutValue staging is not registered");
    return flagcxNotSupported;
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
  if (comm->heteroComm != nullptr && comm->heteroComm->oneSideHandleCount > 0 &&
      comm->heteroComm->oneSideHandles[0] != nullptr &&
      comm->heteroComm->oneSideHandles[0]->fullSendComms != nullptr) {
    devComm->netOneSidedReady = 1;
    for (int i = 0; i < devComm->contextCount; i++) {
      if (devComm->fifoBuffers[i] == nullptr) {
        devComm->netOneSidedReady = 0;
        break;
      }
    }
  }
  if (oneSidedRequested && forceOneSidedNet && !devComm->netOneSidedReady) {
    WARN("defaultDevApiCommCreate: Net one-sided full mesh/FIFO is not ready");
    return flagcxNotSupported;
  }

  INFO(FLAGCX_INIT,
       "defaultDevApiCommCreate: one-sided paths p2pSignal=%d net=%d "
       "netSignal=%d netPutValue=%d forceNet=%d",
       devComm->useP2pSignals, devComm->netOneSidedReady,
       devComm->netSignalReady, devComm->netPutValueReady,
       forceOneSidedNet ? 1 : 0);

  return flagcxSuccess;
}

// ==========================================================================
// DevComm Destroy — immediate cleanup, no deferral
// ==========================================================================

static flagcxResult_t defaultDevApiCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
  const bool hostSignalShm = devComm->signalHostBuffer != nullptr;
  // Deregister communicator-level MRs before freeing their backing buffers.
  // Check handle identity so rollback cannot remove a registration owned by
  // another DevComm.
  if (devComm->ownedStagingRegistration) {
    if (comm != nullptr && comm->heteroComm != nullptr &&
        comm->heteroComm->stagingHandle == devComm->ownedStagingRegistration) {
      flagcxResult_t result = flagcxOneSideStagingDeregister(comm);
      if (result != flagcxSuccess)
        return result;
    }
    devComm->ownedStagingRegistration = nullptr;
  }
  if (devComm->ownedSignalBuffer) {
    if (comm != nullptr && comm->heteroComm != nullptr &&
        comm->heteroComm->rmaSignalBase == devComm->ownedSignalBuffer) {
      flagcxResult_t result = flagcxOneSideSignalDeregister(comm);
      if (result != flagcxSuccess)
        return result;
    }
    devComm->ownedSignalBuffer = nullptr;
    devComm->ownedSignalRegistration = nullptr;
  }

  // ── IPC slot: immediate full cleanup ──────────────────────────────────
  if (comm != nullptr && devComm->barrierIpcIndex >= 0 &&
      devComm->barrierIpcIndex < FLAGCX_MAX_IPC_ENTRIES) {
    struct flagcxIpcTableEntry *e = &comm->ipcTable[devComm->barrierIpcIndex];
    if (e->hostPeerBasePtrs) {
      for (int i = 0; i < e->nPeers; i++) {
        if (e->hostPeerBasePtrs[i])
          deviceAdaptor->ipcMemHandleClose(e->hostPeerBasePtrs[i]);
      }
      free(e->hostPeerBasePtrs);
      e->hostPeerBasePtrs = nullptr;
    }
    if (e->hostPeerPtrs) {
      free(e->hostPeerPtrs);
      e->hostPeerPtrs = nullptr;
    }
    if (e->devPeerPtrs) {
      deviceAdaptor->deviceFree(e->devPeerPtrs, flagcxMemDevice, NULL);
      e->devPeerPtrs = nullptr;
    }
    e->inUse = false;
  }
  devComm->barrierIpcIndex = -1;

  // ── P2P signal/counter IPC slot cleanup ─────────────────────────────────
  auto cleanupIpcSlot = [&](int slot) {
    if (comm == nullptr || slot < 0 || slot >= FLAGCX_MAX_IPC_ENTRIES)
      return;
    struct flagcxIpcTableEntry *e = &comm->ipcTable[slot];
    if (e->hostPeerBasePtrs) {
      for (int i = 0; i < e->nPeers; i++) {
        if (e->hostPeerBasePtrs[i])
          deviceAdaptor->ipcMemHandleClose(e->hostPeerBasePtrs[i]);
      }
      free(e->hostPeerBasePtrs);
      e->hostPeerBasePtrs = nullptr;
    }
    if (e->hostPeerPtrs) {
      free(e->hostPeerPtrs);
      e->hostPeerPtrs = nullptr;
    }
    if (e->devPeerPtrs) {
      deviceAdaptor->deviceFree(e->devPeerPtrs, flagcxMemDevice, NULL);
      e->devPeerPtrs = nullptr;
    }
    e->inUse = false;
  };
  cleanupIpcSlot(devComm->signalIpcSlot);
  devComm->signalIpcSlot = -1;
  devComm->completionSignalPeerPtrs = nullptr;
  devComm->signalPeerPtrs = nullptr;

  // ── Shm barrier path cleanup ──────────────────────────────────────────
  if (devComm->peerBarrierShmPtrs) {
    for (int lr = 0; lr < devComm->registeredBarrierShmCount; lr++) {
      if (devComm->peerBarrierShmPtrs[lr])
        shmHostUnregister(devComm->peerBarrierShmPtrs[lr]);
    }
    free(devComm->peerBarrierShmPtrs);
    devComm->peerBarrierShmPtrs = nullptr;
  }
  devComm->registeredBarrierShmCount = 0;
  // Close peer shm handles (skip own slot which aliases myShmHandle).
  if (devComm->peerShmHandles) {
    for (int lr = 0; lr < devComm->nLocalRanks; lr++) {
      if (devComm->peerShmHandles[lr] &&
          devComm->peerShmHandles[lr] != devComm->myShmHandle) {
        flagcxShmClose(devComm->peerShmHandles[lr]);
      }
    }
    free(devComm->peerShmHandles);
    devComm->peerShmHandles = nullptr;
  }
  if (devComm->myShmHandle) {
    flagcxShmClose(devComm->myShmHandle);
    devComm->myShmHandle = nullptr;
  }
  devComm->localBarrierShmPtr = nullptr;
  devComm->barrierShmSize = 0;
  if (devComm->barrierDevPeerPtrsRaw) {
    deviceAdaptor->deviceFree(devComm->barrierDevPeerPtrsRaw, flagcxMemDevice,
                              NULL);
    devComm->barrierDevPeerPtrsRaw = nullptr;
  }
  devComm->barrierPeers = nullptr;

  // Host signal SHM owns CPU mappings and registrations. The GPU pointers in
  // signalBuffer and signalPeerPtrs are aliases, never allocations to free.
  if (devComm->peerSignalShmPtrs) {
    for (int lr = 0; lr < devComm->registeredSignalShmCount; lr++) {
      if (devComm->peerSignalShmPtrs[lr])
        shmHostUnregister(devComm->peerSignalShmPtrs[lr]);
    }
    free(devComm->peerSignalShmPtrs);
    devComm->peerSignalShmPtrs = nullptr;
  }
  devComm->registeredSignalShmCount = 0;
  if (devComm->peerSignalShmHandles) {
    for (int lr = 0; lr < devComm->nLocalRanks; lr++) {
      if (devComm->peerSignalShmHandles[lr] &&
          devComm->peerSignalShmHandles[lr] != devComm->mySignalShmHandle)
        flagcxShmClose(devComm->peerSignalShmHandles[lr]);
    }
    free(devComm->peerSignalShmHandles);
    devComm->peerSignalShmHandles = nullptr;
  }
  if (devComm->mySignalShmHandle) {
    flagcxShmClose(devComm->mySignalShmHandle);
    devComm->mySignalShmHandle = nullptr;
  }
  devComm->signalHostBuffer = nullptr;
  devComm->signalMrHostPtr = nullptr;
  devComm->localSignalBuffer = nullptr;
  if (devComm->signalDevPeerPtrsRaw) {
    deviceAdaptor->deviceFree(devComm->signalDevPeerPtrsRaw, flagcxMemDevice,
                              NULL);
    devComm->signalDevPeerPtrsRaw = nullptr;
  }

  // ── Free one-sided buffers immediately ────────────────────────────────
  if (devComm->localBarrierFlags) {
    if (devComm->localBarrierFlagsDeviceAllocated) {
      deviceAdaptor->deviceFree(devComm->localBarrierFlags, flagcxMemDevice,
                                NULL);
    }
    devComm->localBarrierFlags = nullptr;
  }
  devComm->localBarrierFlagsDeviceAllocated = false;
  if (devComm->epochBuffer) {
    deviceAdaptor->deviceFree(devComm->epochBuffer, flagcxMemDevice, NULL);
    devComm->epochBuffer = nullptr;
  }
  if (devComm->signalBuffer && !hostSignalShm) {
    flagcxResult_t result = flagcxSuccess;
    if (flagcxParamSignalHostEnable())
      result =
          deviceAdaptor->deviceFree(devComm->signalBuffer, flagcxMemHost, NULL);
    else
      result = deviceAdaptor->gdrMemFree(devComm->signalBuffer, NULL);
    // Preserve the buffer ownership record when provider teardown fails so a
    // subsequent communicator-destroy attempt can retry it.
    if (result != flagcxSuccess)
      return result;
  }
  devComm->signalBuffer = nullptr;
  devComm->signalBufferIsVmm = false;
  if (sizeof(DefaultCompletionWord) == sizeof(uint32_t)) {
    if (devComm->completionSignalBuffer) {
      deviceAdaptor->deviceFree(devComm->completionSignalBuffer,
                                flagcxMemDevice, NULL);
      devComm->completionSignalBuffer = nullptr;
    }
  } else {
    devComm->completionSignalBuffer = nullptr;
  }
  if (devComm->shadowBuffer) {
    deviceAdaptor->deviceFree(devComm->shadowBuffer, flagcxMemDevice, NULL);
    devComm->shadowBuffer = nullptr;
  }
  if (devComm->counterBuffer) {
    deviceAdaptor->deviceFree(devComm->counterBuffer, flagcxMemHost, NULL);
    devComm->counterBuffer = nullptr;
  }
  if (sizeof(DefaultCompletionWord) == sizeof(uint32_t)) {
    if (devComm->completionCounterBuffer) {
      deviceAdaptor->deviceFree(devComm->completionCounterBuffer,
                                flagcxMemDevice, NULL);
      devComm->completionCounterBuffer = nullptr;
    }
  } else {
    devComm->completionCounterBuffer = nullptr;
  }
  if (devComm->putValueStagingBuffer) {
    deviceAdaptor->deviceFree(devComm->putValueStagingBuffer, flagcxMemHost,
                              NULL);
    devComm->putValueStagingBuffer = nullptr;
  }
  devComm->putValueStagingSlotCount = 0;
  devComm->putValueStagingContextCount = 0;
  devComm->putValueStagingKernelBaseSlot = 0;
  devComm->putValueStagingTotalSlotCount = 0;
  free(devComm->localRankToRank);
  devComm->localRankToRank = nullptr;

  // Cached device state remains owned by the outer DevComm until the
  // communicator's deferred-free drain.
  return flagcxSuccess;
}

// ==========================================================================
// DevMem Create
// ==========================================================================

struct uniRunnerIpcReuseChoice {
  int canReuse;
  int canBuild;
};

static flagcxResult_t agreeUniRunnerIpcReuse(flagcxComm_t comm,
                                             bool localCanReuse,
                                             bool localCanBuild,
                                             bool *allCanReuse,
                                             bool *allCanBuild) {
  if (comm == nullptr || comm->bootstrap == nullptr || allCanReuse == nullptr ||
      allCanBuild == nullptr || comm->nranks <= 0 || comm->rank < 0 ||
      comm->rank >= comm->nranks)
    return flagcxInvalidArgument;
  *allCanReuse = false;
  *allCanBuild = false;
  uniRunnerIpcReuseChoice *choices = nullptr;
  flagcxResult_t allocationResult =
      flagcxCalloc(&choices, static_cast<size_t>(comm->nranks));
  flagcxResult_t commonPrepare = flagcxSuccess;
  flagcxResult_t result =
      flagcxConvergeBootstrapStatus(comm, allocationResult, &commonPrepare);
  if (result != flagcxSuccess || commonPrepare != flagcxSuccess) {
    free(choices);
    return result != flagcxSuccess ? result : commonPrepare;
  }
  choices[comm->rank] = {localCanReuse ? 1 : 0, localCanBuild ? 1 : 0};
  result = bootstrapCollAllGather(comm->bootstrap, choices,
                                  sizeof(uniRunnerIpcReuseChoice));
  if (result == flagcxSuccess) {
    *allCanReuse = true;
    *allCanBuild = true;
    for (int rank = 0; rank < comm->nranks; ++rank) {
      if (choices[rank].canReuse == 0)
        *allCanReuse = false;
      if (choices[rank].canBuild == 0)
        *allCanBuild = false;
    }
  }
  free(choices);
  return result;
}

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
      if ((d == nullptr || flagcxSymWindowCanUseLegacyIpc(d)) &&
          !flagcxParamDeviceOneSidedForceNet()) {
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
    // ---- Priority 4 & 5: No window — uniRunner IPC Ring AR ----
    else if (win == nullptr && devMem->uniRunnerIpc != nullptr) {
      int existingIdx = -1;
      uint64_t newestGeneration = 0;
      const bool localCanBuild = !flagcxParamDeviceOneSidedForceNet();
      if (localCanBuild) {
        for (int i = 0; i < FLAGCX_MAX_IPC_ENTRIES; ++i) {
          const struct flagcxIpcTableEntry &entry = comm->ipcTable[i];
          if (entry.inUse && entry.basePtr == buff && entry.size >= size &&
              entry.hostPeerPtrs != nullptr && entry.devPeerPtrs != nullptr &&
              entry.refCount > 0 &&
              entry.refCount < std::numeric_limits<size_t>::max() &&
              entry.bindingGeneration > newestGeneration) {
            existingIdx = i;
            newestGeneration = entry.bindingGeneration;
          }
        }
      }
      bool allCanReuse = false;
      bool allCanBuild = false;
      FLAGCXCHECK(agreeUniRunnerIpcReuse(comm, existingIdx >= 0, localCanBuild,
                                         &allCanReuse, &allCanBuild));
      if (allCanReuse) {
        struct flagcxIpcTableEntry &entry = comm->ipcTable[existingIdx];
        ++entry.refCount;
        devMem->ipcIndex = existingIdx;
      } else if (allCanBuild) {
        int idx = buildIpcPeerPointers(comm, buff, size);
        if (idx >= 0) {
          struct flagcxIpcTableEntry &entry = comm->ipcTable[idx];
          entry.size = size;
          entry.refCount = 1;
          entry.bindingGeneration = ++comm->ipcTableGeneration;
          if (entry.bindingGeneration == 0)
            entry.bindingGeneration = ++comm->ipcTableGeneration;
          devMem->ipcIndex = idx;
        } else {
          WARN("flagcxDevMemCreate: uniRunner IPC peer setup failed");
        }
      }
      if (devMem->ipcIndex >= 0) {
        const struct flagcxIpcTableEntry &entry =
            comm->ipcTable[devMem->ipcIndex];
        devMem->uniRunnerIpc->slot = devMem->ipcIndex;
        devMem->uniRunnerIpc->generation = entry.bindingGeneration;
        devMem->uniRunnerIpc->boundSize = entry.size;
        devMem->uniRunnerIpc->ownsTableReference = true;
      }
    }
    // ---- Priority 4 & 5: No window — ordinary Device API IPC ----
    else if (win == nullptr && !flagcxParamDeviceOneSidedForceNet()) {
      int existingIdx = -1;
      for (int i = 0; i < FLAGCX_MAX_IPC_ENTRIES; i++) {
        if (comm->ipcTable[i].inUse && comm->ipcTable[i].basePtr == buff &&
            comm->ipcTable[i].bindingGeneration == 0) {
          existingIdx = i;
          break;
        }
      }
      if (existingIdx >= 0) {
        devMem->ipcIndex = existingIdx;
      } else {
        // IPC capability is a collective property of the local group. Let
        // buildIpcPeerPointers exchange per-rank export status instead of
        // making an independent capability decision on each rank.
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

  // Allocate and populate kernel Window uniformly via traits
  {
    auto *kWin = new (std::nothrow) typename DeviceAPI::Window{};
    if (kWin == nullptr) {
      WARN("flagcxDevMemCreate: failed to allocate DeviceAPI::Window");
      return flagcxSystemError;
    }
    kWin->populateFromHost(win, devMem->rawPtr, devMem->intraRank,
                           comm ? comm->localRanks : 1, devMem->mrIndex,
                           devMem->mrBase, devMem->ipcIndex,
                           (devMem->ipcIndex >= 0 && comm)
                               ? comm->ipcTable[devMem->ipcIndex].devPeerPtrs
                               : nullptr);
    if (flagcxParamDeviceOneSidedForceNet()) {
      kWin->mode = DeviceAPI::ASYMMETRIC;
      kWin->flatBasePtr = nullptr;
      kWin->ipcBasePtrs = nullptr;
    }

    bool hasP2pAccess =
        (kWin->mode == DeviceAPI::SYMMETRIC && kWin->flatBasePtr != nullptr) ||
        (kWin->mode == DeviceAPI::ASYMMETRIC && kWin->ipcBasePtrs != nullptr);
    flagcxDevComm_t dc = (comm != nullptr && comm->heteroComm != nullptr)
                             ? comm->heteroComm->devCommHandle
                             : nullptr;
    bool hasRemotePeers = comm != nullptr && comm->heteroComm != nullptr &&
                          comm->heteroComm->nNodes > 1;
    bool needsNetFallback = !hasP2pAccess || hasRemotePeers;
    if (win != nullptr && win->isSymmetricDefault && needsNetFallback) {
      bool netFallbackReady = dc != nullptr && dc->netOneSidedReady &&
                              dc->netPutValueReady && devMem->mrIndex >= 0;
      if (!netFallbackReady) {
        WARN("flagcxDevMemCreate: Net fallback is required but incomplete "
             "(mrIndex=%d, net=%d, putValue=%d)",
             devMem->mrIndex, dc ? dc->netOneSidedReady : 0,
             dc ? dc->netPutValueReady : 0);
        delete kWin;
        return flagcxInvalidUsage;
      }
    }
    devMem->window = kWin;
    devMem->hasWindow = kWin->hasAccess();

    if (!devMem->hasWindow && win != nullptr && win->isSymmetricDefault) {
      flagcxSymWindow_t d = win->defaultBase;
      WARN("flagcxDevMemCreate: kWin->hasAccess() returned false for symmetric "
           "default window. ipcIndex=%d, intraRank=%d, "
           "defaultBase=%p, hasFlatMapping=%d, flatBase=%p, ipcDevPeerPtrs=%p",
           devMem->ipcIndex, devMem->intraRank, (void *)d,
           (d ? d->hasFlatMapping : -1), (d ? d->flatBase : nullptr),
           (devMem->ipcIndex >= 0 && comm)
               ? (void *)comm->ipcTable[devMem->ipcIndex].devPeerPtrs
               : nullptr);
      delete kWin;
      devMem->window = nullptr;
      return flagcxInvalidUsage;
    }

    if (flagcxParamDeviceOneSidedForceNet()) {
      if (devMem->mrIndex < 0 || dc == nullptr || !dc->netOneSidedReady ||
          !dc->netPutValueReady) {
        WARN("flagcxDevMemCreate: forced Net fallback is incomplete for buff "
             "%p (mrIndex=%d, net=%d, putValue=%d)",
             buff, devMem->mrIndex, dc ? dc->netOneSidedReady : 0,
             dc ? dc->netPutValueReady : 0);
        delete kWin;
        devMem->window = nullptr;
        return flagcxInvalidUsage;
      }
    }

    flagcxSymWindow_t d = (win != nullptr && win->isSymmetricDefault)
                              ? win->defaultBase
                              : nullptr;
    if (d != nullptr && d->rankLayouts != nullptr &&
        kWin->mode == DeviceAPI::SYMMETRIC) {
      const int nPeers = comm->localRanks;
      if (d->rankLayouts == nullptr || nPeers <= 0 || d->slotStride == 0 ||
          (size_t)nPeers > std::numeric_limits<size_t>::max() /
                               sizeof(typename DeviceAPI::SymPeerLayout)) {
        delete kWin;
        devMem->window = nullptr;
        return flagcxInvalidArgument;
      }
      const size_t bytes =
          (size_t)nPeers * sizeof(typename DeviceAPI::SymPeerLayout);
      auto *hostLayouts = (typename DeviceAPI::SymPeerLayout *)malloc(bytes);
      if (hostLayouts == nullptr) {
        delete kWin;
        devMem->window = nullptr;
        return flagcxSystemError;
      }
      for (int peer = 0; peer < nPeers; peer++) {
        const flagcxSymRankLayout &layout = d->rankLayouts[peer];
        hostLayouts[peer] = {layout.userOffset, layout.userSize};
      }
      void *deviceLayouts = nullptr;
      flagcxResult_t result = deviceAdaptor->deviceMalloc(
          &deviceLayouts, bytes, flagcxMemDevice, NULL);
      if (result == flagcxSuccess)
        result =
            deviceAdaptor->deviceMemcpy(deviceLayouts, hostLayouts, bytes,
                                        flagcxMemcpyHostToDevice, NULL, NULL);
      free(hostLayouts);
      if (result != flagcxSuccess || deviceLayouts == nullptr) {
        if (deviceLayouts != nullptr)
          deviceAdaptor->deviceFree(deviceLayouts, flagcxMemDevice, NULL);
        delete kWin;
        devMem->window = nullptr;
        return result == flagcxSuccess ? flagcxInternalError : result;
      }
      kWin->symPeerLayouts = (typename DeviceAPI::SymPeerLayout *)deviceLayouts;
      devMem->segmentedPeerLayouts = deviceLayouts;
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
  if (comm != nullptr && devMem->uniRunnerIpc == nullptr &&
      devMem->ipcIndex >= 0 && devMem->ipcIndex < FLAGCX_MAX_IPC_ENTRIES) {
    comm->ipcTable[devMem->ipcIndex].inUse = false;
  }

  // Free window allocation if present
  if (devMem->window != nullptr) {
    delete static_cast<typename DeviceAPI::Window *>(devMem->window);
    devMem->window = nullptr;
  }

  // The outer DevMem lifecycle defers cached device state and segmented peer
  // layouts together, so a queued kernel cannot outlive either allocation.

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

  // Step 1: Allocate grid sync state (2 x unsigned int, zero-initialized)
  void *dPtr = nullptr;
  void *netDevPtr = nullptr;
  void *gridSyncPtr = nullptr;
  flagcxResult_t res = flagcxSuccess;
  {
    size_t gsSize = 2 * sizeof(unsigned int);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&gridSyncPtr, gsSize,
                                                flagcxMemDevice, NULL),
                    res, fail);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMemset(gridSyncPtr, 0, gsSize,
                                                flagcxMemDevice, NULL),
                    res, fail);
  }
  hostCopy._gridBarrierState = (unsigned int *)gridSyncPtr;

  // Step 2: Copy flagcxDevComm to device
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&dPtr, sizeof(flagcxDevComm),
                                              flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMemcpy(dPtr, &hostCopy, sizeof(flagcxDevComm),
                                  flagcxMemcpyHostToDevice, NULL, NULL),
      res, fail);

  // Step 3: Allocate + construct net array on device
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
  devComm->cachedGridBarrierPtr = gridSyncPtr;
  *devPtr = dPtr;
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  return flagcxSuccess;

fail:
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  if (gridSyncPtr) {
    deviceAdaptor->deviceFree(gridSyncPtr, flagcxMemDevice, NULL);
  }
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
  void *gridPtr = devComm->cachedGridBarrierPtr;
  devComm->cachedDevicePtr = nullptr;
  devComm->cachedNetContextsPtr = nullptr;
  devComm->cachedGridBarrierPtr = nullptr;
  pthread_mutex_unlock(&devComm->cachedPtrMutex);

  if (gridPtr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(gridPtr, flagcxMemDevice, NULL));
  }
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
// Comm-level quiesce and cleanup
// ==========================================================================

static flagcxResult_t defaultCommQuiesce(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;

  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero == nullptr || hetero->oneSideQuiesced)
    return flagcxSuccess;

  // Drain every device producer and its native requests before MR
  // deregistration is allowed to close an RDMA connection reachable from a
  // FIFO descriptor. consumed only means the proxy posted the descriptor;
  // completed is advanced after the corresponding native request retires.
  if (hetero->proxyState) {
    int ctxCount = hetero->proxyState->kernelState.contextCount;
    for (int i = 0; i < ctxCount; i++) {
      if (hetero->proxyState->kernelState.fifos[i]) {
        volatile uint64_t *buf =
            (volatile uint64_t *)hetero->proxyState->kernelState.fifos[i]
                ->buffer;
        if (buf) {
          while (
              __atomic_load_n(flagcxFifoControlPtr(buf, flagcxFifoIdxCompleted),
                              __ATOMIC_ACQUIRE) !=
              __atomic_load_n(flagcxFifoControlPtr(buf, flagcxFifoIdxProduced),
                              __ATOMIC_ACQUIRE)) {
            flagcxResult_t terminal =
                __atomic_load_n(&hetero->proxyState->kernelState.terminalResult,
                                __ATOMIC_ACQUIRE);
            if (terminal != flagcxSuccess)
              return terminal;
            if (hetero->rmaProxy != nullptr &&
                __atomic_load_n(&hetero->rmaProxy->rmaError, __ATOMIC_ACQUIRE))
              return flagcxRemoteError;
            flagcxResult_t asyncResult = __atomic_load_n(
                &hetero->proxyState->asyncResult, __ATOMIC_ACQUIRE);
            if (asyncResult != flagcxSuccess && asyncResult != flagcxInProgress)
              return asyncResult;
            sched_yield();
          }
        }
      }
    }
  }

  // The common dispatcher quiesces the RMA proxy after every backend has
  // drained its local producers.
  return flagcxSuccess;
}

static flagcxResult_t defaultCommCleanup(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;

  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero != nullptr && hetero->relayInitialized &&
      hetero->nInterPeers > 0) {
    // Preserve the existing relay shutdown rendezvous. The local quiesce phase
    // has already retired FIFO-native requests before reaching this point.
    FLAGCXCHECK(bootstrapCollBarrier(comm->bootstrap, comm->rank, comm->nranks,
                                     0x7f01));
    free(hetero->interPeerRanks);
    hetero->interPeerRanks = nullptr;
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
    .commQuiesce = defaultCommQuiesce,
    .commCleanup = defaultCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &defaultBackend;
