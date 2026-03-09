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
#include "p2p.h" // flagcxP2pAllocateShareableBuffer, flagcxP2pIpcDesc

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
  size_t barrierSize = FLAGCX_DEVICE_CTA_COUNT * sizeof(uint32_t);
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
                                          localRanks * sizeof(uint32_t *),
                                          flagcxMemDevice, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      handle->barrierPeers, handle->peerBarrierPtrs,
      localRanks * sizeof(uint32_t *), flagcxMemcpyHostToDevice, NULL, NULL));

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

  // ---- Grid sync counter (for multi-block two-sided kernels) ----
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&handle->gridDoneCounter,
                                          sizeof(unsigned int), flagcxMemDevice,
                                          NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemset(
      handle->gridDoneCounter, 0, sizeof(unsigned int), flagcxMemDevice, NULL));

  // ---- IPC barrier layer: if barriers requested ----
  if (reqs->fields[0] > 0) {
    flagcxResult_t res = setupIpcBarriers(comm, handle);
    if (res != flagcxSuccess) {
      WARN("flagcxDevCommCreate: IPC barrier setup failed (%d), "
           "barriers unavailable",
           res);
      deviceAdaptor->deviceFree(handle->gridDoneCounter, flagcxMemDevice, NULL);
      free(handle);
      return res;
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
  INFO(FLAGCX_INIT, "flagcxDevCommCreate: rank %d, layers: baseline%s%s",
       handle->rank, handle->barrierPeers ? " + IPC barriers" : "",
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
  if (devComm->gridDoneCounter) {
    deviceAdaptor->deviceFree(devComm->gridDoneCounter, flagcxMemDevice, NULL);
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