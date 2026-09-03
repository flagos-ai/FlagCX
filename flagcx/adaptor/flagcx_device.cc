/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t and flagcxDevMem_t.
 * Thin dispatcher: delegates all backend-specific logic via devApiBackend.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "comm.h"
#include "flagcx_kernel.h"
#include "mem_alloc_registry.h"
#include "p2p.h"
#include "proxy.h"
#include "reg_pool.h"
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>

#include "dev_api_backend.h"

// ==========================================================================
// DevComm lifecycle
// ==========================================================================

extern "C" flagcxResult_t
flagcxDevCommCreate(flagcxComm_t comm, const flagcxDevCommRequirements *reqs,
                    flagcxDevComm_t *devComm) {
  if (comm == nullptr || reqs == nullptr || devComm == nullptr) {
    return flagcxInvalidArgument;
  }
  *devComm = nullptr;

  flagcxDevComm_t handle =
      (flagcxDevComm_t)malloc(sizeof(struct flagcxDevCommInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevCommInternal));
  pthread_mutex_init(&handle->cachedPtrMutex, NULL);
  handle->barrierIpcIndex = -1;
  handle->signalIpcSlot = -1;

  // Baseline: always
  handle->rank = comm->rank;
  handle->nRanks = comm->nranks;
  handle->intraRank = comm->localRank;
  handle->intraSize = comm->localRanks;
  {
    int ctxCount = (reqs->interContextCount > 0) ? reqs->interContextCount : 1;
    if (comm->heteroComm != nullptr &&
        comm->heteroComm->proxyState != nullptr) {
      int available = comm->heteroComm->proxyState->kernelState.contextCount;
      if (available > 0 && ctxCount > available)
        ctxCount = available;
    }
    if (ctxCount > FLAGCX_DEVICE_CTA_COUNT)
      ctxCount = FLAGCX_DEVICE_CTA_COUNT;
    handle->contextCount = ctxCount;
    for (int i = 0; i < ctxCount; i++) {
      handle->fifoBuffers[i] = (comm->heteroComm != nullptr)
                                   ? comm->heteroComm->fifoBuffers[i]
                                   : nullptr;
    }
  }

  // Backend-specific creation
  {
    flagcxResult_t ret = devApiBackend->devCommCreate(comm, reqs, handle);
    if (ret != flagcxSuccess) {
      WARN("flagcxDevCommCreate: %s backend failed (%d)", devApiBackend->name,
           ret);
      flagcxResult_t cleanupRet = devApiBackend->devCommDestroy(comm, handle);
      if (cleanupRet != flagcxSuccess) {
        WARN("flagcxDevCommCreate: %s backend rollback failed (%d)",
             devApiBackend->name, cleanupRet);
      }
      pthread_mutex_destroy(&handle->cachedPtrMutex);
      free(handle);
      return ret;
    }
  }

  *devComm = handle;

  // Publish to heteroComm so proxy thread can access this DevComm
  struct flagcxHeteroComm *hetero = comm->heteroComm;
  if (hetero != nullptr) {
    hetero->devCommHandle = handle;
  }

  return flagcxSuccess;
}

extern "C" flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
  if (devComm == nullptr) {
    return flagcxSuccess;
  }

  // The proxy owns no reference to DevComm. Stop publishing this handle before
  // backend resources are released instead of leaving a stale pointer behind.
  if (comm != nullptr && comm->heteroComm != nullptr &&
      comm->heteroComm->devCommHandle == devComm) {
    comm->heteroComm->devCommHandle = nullptr;
  }

  devApiBackend->devCommDestroy(comm, devComm);

  // Free cached device pointers (thin-layer responsibility)
  if (devComm->cachedNetContextsPtr) {
    flagcxCommDeferFree(comm, devComm->cachedNetContextsPtr, flagcxMemDevice);
  }
  if (devComm->cachedDevicePtr) {
    flagcxCommDeferFree(comm, devComm->cachedDevicePtr, flagcxMemDevice);
  }

  pthread_mutex_destroy(&devComm->cachedPtrMutex);
  free(devComm);
  return flagcxSuccess;
}

// ==========================================================================
// DevMem lifecycle
// ==========================================================================

extern "C" flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t *devMem) {
  if (comm == nullptr || buff == nullptr || size == 0 || devMem == nullptr) {
    return flagcxInvalidArgument;
  }
  *devMem = nullptr;

  flagcxDevMem_t handle =
      (flagcxDevMem_t)malloc(sizeof(struct flagcxDevMemInternal));
  if (handle == nullptr) {
    return flagcxSystemError;
  }
  memset(handle, 0, sizeof(struct flagcxDevMemInternal));
  pthread_mutex_init(&handle->cachedPtrMutex, NULL);

  // Baseline: always
  handle->rawPtr = buff;
  handle->ipcIndex = -1;

  // A subrange is valid only when its complete byte range lies inside the
  // exact flagcxMemAlloc allocation that contains its first byte. Unknown
  // pointers remain supported for the default/CCL backend.
  flagcxMemAllocationInfo allocation;
  flagcxResult_t allocationRes =
      globalMemAllocRegistry.findRange(buff, 1, &allocation);
  if (allocationRes == flagcxSuccess) {
    if (globalMemAllocRegistry.findRange(buff, size, &allocation) !=
        flagcxSuccess) {
      pthread_mutex_destroy(&handle->cachedPtrMutex);
      free(handle);
      return flagcxInvalidUsage;
    }
    handle->allocationTracked = true;
    handle->allocationBase = allocation.base;
    handle->allocationSize = allocation.size;
    handle->allocator = allocation.allocator;
    handle->allocBackend = allocation.backend;
  } else if (allocationRes != flagcxInvalidUsage) {
    pthread_mutex_destroy(&handle->cachedPtrMutex);
    free(handle);
    return allocationRes;
  }

  // Backend-specific creation
  {
    flagcxResult_t ret =
        devApiBackend->devMemCreate(comm, buff, size, win, handle);
    if (ret != flagcxSuccess) {
      WARN("flagcxDevMemCreate: %s backend failed (%d)", devApiBackend->name,
           ret);
      pthread_mutex_destroy(&handle->cachedPtrMutex);
      free(handle);
      return ret;
    }
  }

  *devMem = handle;
  return flagcxSuccess;
}

extern "C" flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  if (devMem == nullptr) {
    return flagcxSuccess;
  }

  // Release IPC table slot (resources moved to deferred queue)
  if (devMem->ipcIndex >= 0) {
    releaseIpcTableSlot(comm, devMem->ipcIndex);
  }

  devApiBackend->devMemDestroy(comm, devMem);

  // Free cached device pointer
  if (devMem->cachedDevicePtr) {
    flagcxCommDeferFree(comm, devMem->cachedDevicePtr, flagcxMemDevice);
  }

  pthread_mutex_destroy(&devMem->cachedPtrMutex);
  free(devMem);
  return flagcxSuccess;
}

// ==========================================================================
// Device Pointer API
// ==========================================================================

extern "C" flagcxResult_t flagcxDevCommGetDevicePtr(flagcxDevComm_t devComm,
                                                    void **devPtr) {
  return devApiBackend->devCommGetDevicePtr(devComm, devPtr);
}

extern "C" flagcxResult_t flagcxDevCommFreeDevicePtr(flagcxDevComm_t devComm) {
  return devApiBackend->devCommFreeDevicePtr(devComm);
}

extern "C" flagcxResult_t flagcxDevMemGetDevicePtr(flagcxDevMem_t devMem,
                                                   void **devPtr) {
  return devApiBackend->devMemGetDevicePtr(devMem, devPtr);
}

extern "C" flagcxResult_t flagcxDevMemFreeDevicePtr(flagcxDevMem_t devMem) {
  return devApiBackend->devMemFreeDevicePtr(devMem);
}

// ==========================================================================
// Comm-level cleanup — called from flagcxCommDestroy in flagcx.cc
// ==========================================================================

extern "C" flagcxResult_t flagcxCommCleanup(flagcxComm_t comm) {
  return devApiBackend->commCleanup(comm);
}

// ==========================================================================
// IPC table cleanup — called from flagcxCommDestroy after heteroComm destroy
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
// Deferred IPC table slot release.
// ==========================================================================
void releaseIpcTableSlot(flagcxComm_t comm, int slot) {
  if (comm == nullptr || slot < 0 || slot >= FLAGCX_MAX_IPC_ENTRIES) {
    return;
  }
  struct flagcxIpcTableEntry *e = &comm->ipcTable[slot];
  if (e->hostPeerPtrs == nullptr && e->devPeerPtrs == nullptr) {
    e->inUse = false;
    return;
  }

  // Move resources to deferred linked list for cleanup at comm destroy
  struct flagcxDeferredIpcEntry *d =
      (struct flagcxDeferredIpcEntry *)malloc(sizeof(*d));
  if (d == nullptr) {
    // OOM: leave slot occupied so flagcxCommCleanupIpcTable handles it at
    // destroy. The slot won't be reusable, but resources are still safe.
    WARN(
        "releaseIpcTableSlot: OOM, keeping slot %d occupied until comm destroy",
        slot);
    e->inUse = false;
    return;
  }
  d->hostPeerPtrs = e->hostPeerPtrs;
  d->devPeerPtrs = e->devPeerPtrs;
  d->nPeers = e->nPeers;
  d->basePtr = e->basePtr;
  d->next = nullptr;
  flagcxIntruQueueEnqueue(&comm->deferredIpcQueue, d);

  // Clear slot — now reusable by buildIpcPeerPointers
  e->hostPeerPtrs = nullptr;
  e->devPeerPtrs = nullptr;
  e->nPeers = 0;
  e->basePtr = nullptr;
  e->inUse = false;
}

// ==========================================================================
// IPC peer pointer exchange
// ==========================================================================

int buildIpcPeerPointers(flagcxComm_t comm, void *buff, size_t size) {
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

  // Step 1: Get IPC handle for existing user buffer.
  struct flagcxP2pIpcDesc myIpcDesc;
  memset(&myIpcDesc, 0, sizeof(myIpcDesc));

  // Check globalRegPool for pre-registered handle
  {
    flagcxRegItem *item = globalRegPool.getItem(nullptr, buff);
    if (item && item->localIpcHandleData.reserved[0] != 0) {
      memcpy(&myIpcDesc.handleData, &item->localIpcHandleData,
             sizeof(myIpcDesc.handleData));
      myIpcDesc.size = size;
    } else {
      // Create IPC handle directly on the existing buffer (do NOT allocate new)
      size_t ipcSize = 0;
      flagcxIpcMemHandle_t handlePtr = NULL;
      res = deviceAdaptor->ipcMemHandleCreate(&handlePtr, &ipcSize);
      if (res != flagcxSuccess) {
        WARN("buildIpcPeerPointers: ipcMemHandleCreate failed");
        return -1;
      }
      res = deviceAdaptor->ipcMemHandleGet(handlePtr, buff);
      if (res != flagcxSuccess) {
        WARN("buildIpcPeerPointers: ipcMemHandleGet failed for buff %p", buff);
        deviceAdaptor->ipcMemHandleFree(handlePtr);
        return -1;
      }
      memcpy(&myIpcDesc.handleData, handlePtr, sizeof(flagcxIpcHandleData));
      myIpcDesc.size = size;
      deviceAdaptor->ipcMemHandleFree(handlePtr);
    }
  }

  // Step 2: All-gather IPC descriptors across local ranks
  allDescs = (struct flagcxP2pIpcDesc *)malloc(nRanks *
                                               sizeof(struct flagcxP2pIpcDesc));
  if (!allDescs)
    return -1;
  memset(allDescs, 0, nRanks * sizeof(struct flagcxP2pIpcDesc));
  allDescs[myRank] = myIpcDesc;

  FLAGCXCHECKGOTO(bootstrapCollAllGather(comm->bootstrap, allDescs,
                                         sizeof(struct flagcxP2pIpcDesc)),
                  res, fail);

  // Step 3: Open peer IPC handles
  hostPeerPtrs = (void **)malloc(localRanks * sizeof(void *));
  if (!hostPeerPtrs) {
    res = flagcxSystemError;
    goto fail;
  }
  memset(hostPeerPtrs, 0, localRanks * sizeof(void *));

  for (int lr = 0; lr < localRanks; lr++) {
    int globalR = localRankToRank[lr];
    if (globalR == myRank) {
      hostPeerPtrs[lr] = buff;
    } else {
      struct flagcxP2pIpcDesc *pd = &allDescs[globalR];
      if (pd->size > 0 && deviceAdaptor->ipcMemHandleOpen) {
        void *peerPtr = nullptr;
        flagcxIpcMemHandle_t handlePtr = (flagcxIpcMemHandle_t)&pd->handleData;
        res = deviceAdaptor->ipcMemHandleOpen(handlePtr, &peerPtr);
        if (res != flagcxSuccess) {
          WARN("buildIpcPeerPointers: ipcMemHandleOpen failed for rank %d",
               globalR);
          goto fail;
        }
        hostPeerPtrs[lr] = peerPtr;
      } else {
        hostPeerPtrs[lr] = nullptr;
      }
    }
  }

  free(allDescs);
  allDescs = nullptr;

  // Step 4: Build device-side peer pointer array
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

flagcxResult_t flagcxCommDrainDeferredIpc(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  while (!flagcxIntruQueueEmpty(&comm->deferredIpcQueue)) {
    struct flagcxDeferredIpcEntry *d =
        flagcxIntruQueueDequeue(&comm->deferredIpcQueue);
    if (d->hostPeerPtrs) {
      for (int j = 0; j < d->nPeers; j++) {
        if (d->hostPeerPtrs[j] && d->hostPeerPtrs[j] != d->basePtr)
          deviceAdaptor->ipcMemHandleClose(d->hostPeerPtrs[j]);
      }
      free(d->hostPeerPtrs);
    }
    if (d->devPeerPtrs)
      deviceAdaptor->deviceFree(d->devPeerPtrs, flagcxMemDevice, NULL);
    free(d);
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

flagcxResult_t flagcxCommDrainDeferredBuffers(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  while (!flagcxIntruQueueEmpty(&comm->deferredBufferQueue)) {
    struct flagcxDevCommBufferHandle *h =
        flagcxIntruQueueDequeue(&comm->deferredBufferQueue);
    if (h->localBarrierFlags)
      deviceAdaptor->deviceFree(h->localBarrierFlags, flagcxMemDevice, NULL);
    if (h->epochBuffer)
      deviceAdaptor->deviceFree(h->epochBuffer, flagcxMemDevice, NULL);
    if (h->signalBuffer) {
      if (h->signalHostEnable)
        deviceAdaptor->deviceFree(h->signalBuffer, flagcxMemHost, NULL);
      else
        deviceAdaptor->gdrMemFree(h->signalBuffer, NULL);
    }
    if (h->shadowBuffer)
      deviceAdaptor->deviceFree(h->shadowBuffer, flagcxMemDevice, NULL);
    if (h->counterBuffer)
      deviceAdaptor->deviceFree(h->counterBuffer, flagcxMemHost, NULL);
    if (h->putValueStagingBuffer)
      deviceAdaptor->deviceFree(h->putValueStagingBuffer, flagcxMemHost, NULL);
    free(h);
  }
  comm->deferredBufferCount = 0;
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
#ifdef FLAGCX_DEVICE_API_VENDOR
  props->vendorDeviceApiSupport = true;
#else
  props->vendorDeviceApiSupport = false;
#endif
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
