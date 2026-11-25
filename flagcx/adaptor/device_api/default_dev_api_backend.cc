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
#include "kernel_proxy_transport.h"
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

static flagcxResult_t shmHostUnregister(void *ptr) {
  if (deviceAdaptor->hostUnregister == NULL)
    return flagcxNotSupported;
  return deviceAdaptor->hostUnregister(ptr);
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
  if (hetero == nullptr) {
    devComm->teamRank = 0;
    devComm->nTeamRanks = 1;
    return flagcxSuccess;
  }

  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int myNode = hetero->node;
  int nNodes = hetero->nNodes;

  if (nNodes <= 1) {
    // A singleton Inter team has one rank, even though it has no remote peers.
    devComm->teamRank = 0;
    devComm->nTeamRanks = 1;
    devComm->nInterPeers = 0;
    return flagcxSuccess;
  }

  // Collective relay creation also declares its communicator-wide teardown.
  // Keep the scope through partial creation and cleanup retries.
  comm->nativeFinalizeScope = flagcxNativeFinalizeComm;

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
    flagcxResult_t res = flagcxCommAllocInternalBuffer(
        comm, (void **)&devComm->localBarrierFlags, barrierSize,
        flagcxInternalBufferDevice);
    if (res == flagcxSuccess) {
      devComm->localBarrierFlagsDeviceAllocated = true;
      res = deviceAdaptor->deviceMemset(devComm->localBarrierFlags, 0,
                                        barrierSize, flagcxMemDevice, NULL);
    }
    flagcxResult_t common = flagcxSuccess;
    FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, res, &common));
    if (common != flagcxSuccess)
      return common;

    auto ipc = flagcxCommWindowRegisterIpcInternal(
        comm, devComm->localBarrierFlags, barrierSize, false);
    devComm->barrierIpcIndex = ipc.slot;
    FLAGCXCHECK(ipc.result());
    devComm->barrierPeers = (uint64_t **)comm->ipcTable[ipc.slot].devPeerPtrs;
    devComm->localBarrierShmPtr = nullptr;
    devComm->peerBarrierShmPtrs = nullptr;
    devComm->barrierShmSize = 0;
    devComm->barrierDevPeerPtrsRaw = nullptr;
    devComm->nBarriers = FLAGCX_DEVICE_CTA_COUNT;

    INFO(FLAGCX_INIT,
         "setupIpcBarriers(IPC): rank %d slot %d localBarrierFlags=%p "
         "barrierPeers=%p nBarriers=%d",
         myRank, ipc.slot, (void *)devComm->localBarrierFlags,
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
        FLAGCXCHECKGOTO(flagcxShmOpen(peerPath, sizeof(peerPath), barrierSize,
                                      &peerCpuPtrs[lr], nullptr, -1,
                                      &peerShmHandles[lr]),
                        res, fail_peer_shm);
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
  fail_own_shm:
    // Creation rollback uses the same persistent owner as normal destroy.
    devComm->nLocalRanks = localRanks;
    devComm->myShmHandle = myShmHandle;
    devComm->localBarrierShmPtr = myCpuPtr;
    devComm->peerShmHandles = peerShmHandles;
    devComm->peerBarrierShmPtrs = peerCpuPtrs;
    devComm->registeredBarrierShmCount = registeredPeerCount;
    devComm->barrierDevPeerPtrsRaw = (uint64_t **)devPeerPtrs;
    free(hostDevPtrs);
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
  if (reqs->interBarrierCount < 0)
    return flagcxInvalidArgument;
  devComm->interBarrierCount = reqs->interBarrierCount;
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
      if (reqs->interBarrierCount > 0)
        return res;
    }
  }
  if (reqs->interBarrierCount > 0 && comm->heteroComm != nullptr &&
      comm->heteroComm->nNodes > 1 &&
      (devComm->nInterPeers <= 0 || devComm->nTeamRanks <= 1)) {
    WARN("defaultDevApiCommCreate: Inter barrier requested without a valid "
         "inter-node team");
    return flagcxInternalError;
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

  // One-sided buffers: signals, counters, staging. Inter barriers need signal
  // slots even when no user-visible signals were requested.
  INFO(FLAGCX_INIT,
       "defaultDevApiCommCreate: nInterPeers=%d interSignalCount=%d "
       "interCounterCount=%d",
       devComm->nInterPeers, reqs->interSignalCount, reqs->interCounterCount);
  if (reqs->interSignalCount > 0 || reqs->interCounterCount > 0 ||
      reqs->interBarrierCount > 0) {
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
    if (reqs->interSignalCount > 0 || reqs->interBarrierCount > 0) {
      // Expand signalCount to include barrier slots:
      // Each barrier channel reserves one signal slot per Inter team rank.
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
        res =
            flagcxCommAllocInternalBuffer(comm, (void **)&devComm->signalBuffer,
                                          sigSize, flagcxInternalBufferHost);
        if (res != flagcxSuccess) {
          WARN("defaultDevApiCommCreate: signalBuffer host malloc failed (%d)",
               res);
          return res;
        }
        memset(devComm->signalBuffer, 0, sigSize);
      } else {
        res =
            flagcxCommAllocInternalBuffer(comm, (void **)&devComm->signalBuffer,
                                          sigSize, flagcxInternalBufferGdr);
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
        res = flagcxCommAllocInternalBuffer(
            comm, &devComm->completionSignalBuffer, directSigSize,
            flagcxInternalBufferDevice);
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
      res = flagcxCommAllocInternalBuffer(
          comm, (void **)&devComm->putValueStagingBuffer, stagingSize,
          flagcxInternalBufferHost);
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
        flagcxResult_t ipcCleanup = flagcxSuccess;
        res = flagcxOneSideSignalRegisterInternal(
            comm, signalRegistrationBuffer,
            (size_t)devComm->signalCount * bufCtxCount * sizeof(uint64_t),
            sigPtrType, devComm->signalBufferIsVmm, &ipcCleanup);
        if (previousSignalBase == nullptr &&
            comm->heteroComm->rmaSignalBase == signalRegistrationBuffer)
          devComm->ownedSignalBuffer = (uint64_t *)signalRegistrationBuffer;
        if (ipcCleanup != flagcxSuccess)
          return ipcCleanup;
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
    auto ipc = flagcxCommWindowRegisterIpcInternal(
        comm, devComm->completionSignalBuffer, sigSize, false);
    const int slot = ipc.slot;
    devComm->signalIpcSlot = slot;
    if (ipc.cleanupStatus != flagcxSuccess)
      return ipc.cleanupStatus;
    if (ipc.buildStatus == flagcxSuccess) {
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
  if (reqs->interBarrierCount > 0 && devComm->nInterPeers > 0 &&
      !devComm->netOneSidedReady) {
    WARN("defaultDevApiCommCreate: Inter barrier Proxy FIFO is not ready");
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
// DevComm Destroy — close local imports; Comm owns exporter allocations
// ==========================================================================

static flagcxResult_t defaultDevApiCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
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
  if (comm != nullptr && devComm->barrierIpcIndex >= 0)
    do {
      flagcxResult_t result =
          releaseIpcTableSlot(comm, devComm->barrierIpcIndex);
      if (result != flagcxSuccess)
        return result;
    } while (0);
  devComm->barrierIpcIndex = -1;

  // ── P2P signal/counter IPC slot cleanup ─────────────────────────────────
  if (comm != nullptr && devComm->signalIpcSlot >= 0)
    do {
      flagcxResult_t result = releaseIpcTableSlot(comm, devComm->signalIpcSlot);
      if (result != flagcxSuccess)
        return result;
    } while (0);
  devComm->signalIpcSlot = -1;
  devComm->completionSignalPeerPtrs = nullptr;
  devComm->signalPeerPtrs = nullptr;

  // Unregister CPU mappings before unmapping; successful prefixes are kept.
  while (devComm->registeredBarrierShmCount > 0) {
    int lr = devComm->registeredBarrierShmCount - 1;
    flagcxResult_t result = shmHostUnregister(devComm->peerBarrierShmPtrs[lr]);
    if (result != flagcxSuccess)
      return result;
    --devComm->registeredBarrierShmCount;
  }
  free(devComm->peerBarrierShmPtrs);
  devComm->peerBarrierShmPtrs = NULL;
  if (devComm->peerShmHandles != NULL) {
    for (int lr = 0; lr < devComm->nLocalRanks; ++lr) {
      if (devComm->peerShmHandles[lr] == devComm->myShmHandle)
        devComm->peerShmHandles[lr] = NULL;
      if (devComm->peerShmHandles[lr] == NULL)
        continue;
      FLAGCXCHECK(flagcxShmClose(devComm->peerShmHandles[lr]));
      devComm->peerShmHandles[lr] = NULL;
    }
    free(devComm->peerShmHandles);
    devComm->peerShmHandles = NULL;
  }
  if (devComm->myShmHandle != NULL) {
    FLAGCXCHECK(flagcxShmClose(devComm->myShmHandle));
    devComm->myShmHandle = NULL;
  }
  devComm->localBarrierShmPtr = NULL;
  devComm->barrierShmSize = 0;
  if (devComm->barrierDevPeerPtrsRaw != NULL) {
    flagcxResult_t result = flagcxCommReleaseBuffer(
        comm, devComm->barrierDevPeerPtrsRaw, flagcxInternalBufferDevice);
    if (result != flagcxSuccess)
      return result;
    devComm->barrierDevPeerPtrsRaw = NULL;
  }
  devComm->barrierPeers = NULL;
  while (devComm->registeredSignalShmCount > 0) {
    int lr = devComm->registeredSignalShmCount - 1;
    flagcxResult_t result = shmHostUnregister(devComm->peerSignalShmPtrs[lr]);
    if (result != flagcxSuccess)
      return result;
    --devComm->registeredSignalShmCount;
  }
  free(devComm->peerSignalShmPtrs);
  devComm->peerSignalShmPtrs = NULL;
  if (devComm->peerSignalShmHandles != NULL) {
    for (int lr = 0; lr < devComm->nLocalRanks; ++lr) {
      if (devComm->peerSignalShmHandles[lr] == devComm->mySignalShmHandle)
        devComm->peerSignalShmHandles[lr] = NULL;
      if (devComm->peerSignalShmHandles[lr] == NULL)
        continue;
      FLAGCXCHECK(flagcxShmClose(devComm->peerSignalShmHandles[lr]));
      devComm->peerSignalShmHandles[lr] = NULL;
    }
    free(devComm->peerSignalShmHandles);
    devComm->peerSignalShmHandles = NULL;
  }
  if (devComm->mySignalShmHandle != NULL) {
    FLAGCXCHECK(flagcxShmClose(devComm->mySignalShmHandle));
    devComm->mySignalShmHandle = NULL;
  }
  devComm->signalHostBuffer = NULL;
  devComm->signalMrHostPtr = NULL;
  devComm->localSignalBuffer = NULL;
  if (devComm->signalDevPeerPtrsRaw != NULL) {
    flagcxResult_t result = flagcxCommReleaseBuffer(
        comm, devComm->signalDevPeerPtrsRaw, flagcxInternalBufferDevice);
    if (result != flagcxSuccess)
      return result;
    devComm->signalDevPeerPtrsRaw = NULL;
  }

  // Exported storage belongs to Comm, including partial-create allocations.
  // Peers may still hold IPC imports after this local DevCommDestroy.
  devComm->localBarrierFlags = nullptr;
  devComm->localBarrierFlagsDeviceAllocated = false;
  if (devComm->epochBuffer) {
    flagcxResult_t result = flagcxCommReleaseBuffer(comm, devComm->epochBuffer,
                                                    flagcxInternalBufferDevice);
    if (result != flagcxSuccess)
      return result;
    devComm->epochBuffer = nullptr;
  }
  devComm->signalBuffer = nullptr;
  devComm->signalBufferIsVmm = false;
  // In the 64-bit path this aliases signalBuffer; Comm records it only once.
  devComm->completionSignalBuffer = nullptr;
  if (devComm->shadowBuffer) {
    flagcxResult_t result = flagcxCommReleaseBuffer(comm, devComm->shadowBuffer,
                                                    flagcxInternalBufferDevice);
    if (result != flagcxSuccess)
      return result;
    devComm->shadowBuffer = nullptr;
  }
  if (devComm->counterBuffer) {
    flagcxResult_t result = flagcxCommReleaseBuffer(
        comm, devComm->counterBuffer, flagcxInternalBufferHost);
    if (result != flagcxSuccess)
      return result;
    devComm->counterBuffer = nullptr;
  }
  if (sizeof(DefaultCompletionWord) == sizeof(uint32_t)) {
    if (devComm->completionCounterBuffer) {
      flagcxResult_t result = flagcxCommReleaseBuffer(
          comm, devComm->completionCounterBuffer, flagcxInternalBufferDevice);
      if (result != flagcxSuccess)
        return result;
      devComm->completionCounterBuffer = nullptr;
    }
  } else {
    devComm->completionCounterBuffer = nullptr;
  }
  devComm->putValueStagingBuffer = nullptr;
  devComm->putValueStagingSlotCount = 0;
  devComm->putValueStagingContextCount = 0;
  devComm->putValueStagingKernelBaseSlot = 0;
  devComm->putValueStagingTotalSlotCount = 0;
  free(devComm->localRankToRank);
  devComm->localRankToRank = nullptr;

  // The outer DevComm frees its cached device state after backend teardown.
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
      devMem->isSymmetric = win->winFlags == FLAGCX_WIN_SYMMETRIC;
      devMem->winHandle = (void *)win;
      if (d != nullptr && d->mrIndex >= 0) {
        devMem->mrIndex = d->mrIndex;
        devMem->mrBase = d->mrBase;
      }
      if ((d == nullptr || flagcxSymWindowCanUseLegacyIpc(d)) &&
          !flagcxParamDeviceOneSidedForceNet()) {
        // Window registration already exchanged the IPC handles. Borrow its
        // mapping without another cross-rank operation.
        if (d != nullptr && d->ipcSlot >= 0) {
          auto &entry = comm->ipcTable[d->ipcSlot];
          const uintptr_t base = reinterpret_cast<uintptr_t>(entry.basePtr);
          const uintptr_t view = reinterpret_cast<uintptr_t>(buff);
          if (!entry.inUse || !entry.ownerHeld ||
              entry.basePtr != d->localBase || entry.refCount == SIZE_MAX ||
              view < base || view - base > entry.size ||
              size > entry.size - (view - base))
            return flagcxInvalidUsage;
          ++entry.refCount;
          devMem->ipcIndex = d->ipcSlot;
        }
      }
      devMem->window = nullptr;
    }
    // ---- Priority 3: Vendor native window ----
    else if (win != nullptr && !win->isSymmetricDefault) {
      devMem->hasWindow = true;
      devMem->isSymmetric = (win->winFlags & FLAGCX_WIN_SYMMETRIC);
      devMem->winHandle = (void *)win;
    }
    // A prepared IPC slot already owns one reference. Otherwise, attach only
    // to a previously registered owned IPC mapping.
    else if (win == nullptr && devMem->ipcIndex < 0 &&
             !flagcxParamDeviceOneSidedForceNet()) {
      int existingIdx = -1;
      for (int i = 0; i < comm->ipcTableCapacity; i++) {
        if (comm->ipcTable[i].inUse && comm->ipcTable[i].basePtr == buff &&
            comm->ipcTable[i].size >= size && comm->ipcTable[i].ownerHeld &&
            comm->ipcTable[i].hostPeerPtrs != nullptr &&
            comm->ipcTable[i].devPeerPtrs != nullptr) {
          existingIdx = i;
          break;
        }
      }
      if (existingIdx >= 0) {
        if (comm->ipcTable[existingIdx].refCount == SIZE_MAX)
          return flagcxInvalidUsage;
        ++comm->ipcTable[existingIdx].refCount;
        devMem->ipcIndex = existingIdx;
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
  // Free window allocation if present
  if (devMem->window != nullptr) {
    delete static_cast<typename DeviceAPI::Window *>(devMem->window);
    devMem->window = nullptr;
  }

  // The outer DevMem lifecycle waits for local device work and frees cached
  // device state and segmented layouts before returning.

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

  // Allocate the net array before copying the communicator so its pointer is
  // initialized without computing an offset into a non-standard-layout type.
  if (hostCopy._contextCount > 0 && flagcxDevNetSizeOf() > 0) {
    size_t netArraySize = hostCopy._contextCount * flagcxDevNetSizeOf();
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&netDevPtr, netArraySize,
                                                flagcxMemDevice, NULL),
                    res, fail);
    hostCopy._netContexts = netDevPtr;
  }

  // Step 2: Copy the fully initialized flagcxDevComm to device.
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&dPtr, sizeof(flagcxDevComm),
                                              flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMemcpy(dPtr, &hostCopy, sizeof(flagcxDevComm),
                                  flagcxMemcpyHostToDevice, NULL, NULL),
      res, fail);

  // Step 3: Construct net contexts using the initialized device communicator.
  if (netDevPtr != nullptr) {
    flagcxDevNetLaunchConstruct(netDevPtr, dPtr, hostCopy._contextCount,
                                nullptr);
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
          flagcxResult_t gate = flagcxKernelProxyCloseFifoProducerGate(
              hetero->proxyState->kernelState.fifos[i]->buffer);
          if (gate != flagcxSuccess)
            return gate;
          const flagcxCompletionWord_t target =
              __atomic_load_n(flagcxFifoControlPtr(buf, flagcxFifoIdxProduced),
                              __ATOMIC_ACQUIRE);
          while (
              __atomic_load_n(flagcxFifoControlPtr(buf, flagcxFifoIdxCompleted),
                              __ATOMIC_ACQUIRE) != target) {
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
