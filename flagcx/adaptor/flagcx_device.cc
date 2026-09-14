/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t and flagcxDevMem_t.
 * Thin dispatcher: delegates all backend-specific logic via devApiBackend.
 ************************************************************************/

#define FLAGCX_DISABLE_DEV_COMM_CREATE_SIZE_DISPATCH
#include "device_api/flagcx_device.h"
#include "comm.h"
#include "flagcx_kernel_internal.h"
#include "mem_alloc_registry.h"
#include "p2p.h"
#include "proxy.h"
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>

#include "dev_api_backend.h"
#include <cstdint>

// ==========================================================================
// DevComm lifecycle
// ==========================================================================

static flagcxResult_t
flagcxDevCommCreateInternal(flagcxComm_t comm,
                            const struct flagcxDevCommRequirements *reqs,
                            size_t reqsSize, flagcxDevComm_t *devComm) {
  if (comm == nullptr || reqs == nullptr || devComm == nullptr ||
      reqsSize < FLAGCX_DEV_COMM_REQUIREMENTS_LEGACY_SIZE)
    return flagcxInvalidArgument;
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
    flagcxResult_t ret =
        devApiBackend->devCommCreate(comm, reqs, reqsSize, handle);
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

extern "C" flagcxResult_t
flagcxDevCommCreate(flagcxComm_t comm,
                    const struct flagcxDevCommRequirements *reqs,
                    flagcxDevComm_t *devComm) {
  if (devComm != nullptr)
    *devComm = nullptr;
  return flagcxDevCommCreateInternal(
      comm, reqs, FLAGCX_DEV_COMM_REQUIREMENTS_LEGACY_SIZE, devComm);
}

extern "C" flagcxResult_t
flagcxDevCommCreateSized(flagcxComm_t comm,
                         const struct flagcxDevCommRequirements *reqs,
                         size_t reqsSize, flagcxDevComm_t *devComm) {
  if (devComm != nullptr)
    *devComm = nullptr;
  return flagcxDevCommCreateInternal(comm, reqs, reqsSize, devComm);
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
  if (devMem == nullptr) {
    return flagcxInvalidArgument;
  }
  *devMem = nullptr;
  if (comm == nullptr || buff == nullptr || size == 0) {
    return flagcxInvalidArgument;
  }

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
    if (e->hostPeerPtrs == nullptr && e->hostPeerBasePtrs == nullptr &&
        e->devPeerPtrs == nullptr) {
      continue; // empty slot
    }

    if (e->inUse) {
      WARN("flagcxCommCleanupIpcTable: entry %d still in use — "
           "flagcxDevMemDestroy should be called before flagcxCommDestroy",
           k);
    }

    // Close IPC handles
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
  if (e->hostPeerPtrs == nullptr && e->hostPeerBasePtrs == nullptr &&
      e->devPeerPtrs == nullptr) {
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
  d->hostPeerBasePtrs = e->hostPeerBasePtrs;
  d->devPeerPtrs = e->devPeerPtrs;
  d->nPeers = e->nPeers;
  d->basePtr = e->basePtr;
  d->next = nullptr;
  flagcxIntruQueueEnqueue(&comm->deferredIpcQueue, d);

  // Clear slot — now reusable by buildIpcPeerPointers
  e->hostPeerPtrs = nullptr;
  e->hostPeerBasePtrs = nullptr;
  e->devPeerPtrs = nullptr;
  e->nPeers = 0;
  e->basePtr = nullptr;
  e->inUse = false;
}

// ==========================================================================
// IPC peer pointer exchange
// ==========================================================================

struct flagcxIpcPeerDesc {
  flagcxIpcHandleData handleData;
  size_t handleSize;
  size_t allocationSize;
  size_t userOffset;
  size_t userSize;
  bool valid;
};

flagcxResult_t flagcxGetIpcExportRange(const void *buff, size_t size,
                                       void **exportBase,
                                       size_t *allocationSize,
                                       size_t *userOffset) {
  if (buff == nullptr || size == 0 || exportBase == nullptr ||
      allocationSize == nullptr || userOffset == nullptr)
    return flagcxInvalidArgument;

  *exportBase = const_cast<void *>(buff);
  *allocationSize = size;
  *userOffset = 0;
  if (deviceAdaptor->getAddressRange == nullptr)
    return flagcxSuccess;

  void *allocationBase = nullptr;
  size_t queriedSize = 0;
  flagcxResult_t res =
      deviceAdaptor->getAddressRange(buff, &allocationBase, &queriedSize);
  if (res == flagcxNotSupported)
    return flagcxSuccess;
  if (res != flagcxSuccess)
    return res;

  uintptr_t userAddress = reinterpret_cast<uintptr_t>(buff);
  uintptr_t baseAddress = reinterpret_cast<uintptr_t>(allocationBase);
  if (allocationBase == nullptr || userAddress < baseAddress) {
    WARN("IPC allocation range does not contain buffer %p (base %p)", buff,
         allocationBase);
    return flagcxInvalidUsage;
  }
  size_t offset = userAddress - baseAddress;
  if (offset > queriedSize || size > queriedSize - offset) {
    WARN("IPC buffer %p size %zu exceeds allocation %p size %zu", buff, size,
         allocationBase, queriedSize);
    return flagcxInvalidUsage;
  }

  *exportBase = allocationBase;
  *allocationSize = queriedSize;
  *userOffset = offset;
  return flagcxSuccess;
}

flagcxResult_t flagcxResolveIpcPeerAddress(void *importedBase,
                                           size_t allocationSize,
                                           size_t userOffset, size_t userSize,
                                           void **peerPtr) {
  if (importedBase == nullptr || allocationSize == 0 || userSize == 0 ||
      peerPtr == nullptr)
    return flagcxInvalidArgument;
  *peerPtr = nullptr;

  if (userOffset > allocationSize || userSize > allocationSize - userOffset) {
    WARN("IPC user range offset %zu size %zu exceeds allocation size %zu",
         userOffset, userSize, allocationSize);
    return flagcxInvalidUsage;
  }
  uintptr_t mappingBase = reinterpret_cast<uintptr_t>(importedBase);
  if (userOffset > UINTPTR_MAX - mappingBase) {
    WARN("IPC peer mapping address overflow for base %p offset %zu",
         importedBase, userOffset);
    return flagcxInvalidUsage;
  }

  *peerPtr = reinterpret_cast<void *>(mappingBase + userOffset);
  return flagcxSuccess;
}

int buildIpcPeerPointers(flagcxComm_t comm, void *buff, size_t size) {
  int slot = -1;
  for (int k = 0; k < FLAGCX_MAX_IPC_ENTRIES; k++) {
    if (comm->ipcTable[k].hostPeerPtrs == nullptr &&
        comm->ipcTable[k].hostPeerBasePtrs == nullptr &&
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
  struct flagcxIpcPeerDesc *allDescs = nullptr;
  void **hostPeerPtrs = nullptr;
  void **hostPeerBasePtrs = nullptr;
  void **devPeerPtrs = nullptr;

  // Step 1: Get IPC handle for existing user buffer.
  struct flagcxIpcPeerDesc myIpcDesc;
  memset(&myIpcDesc, 0, sizeof(myIpcDesc));
  void *exportBase = buff;
  res =
      flagcxGetIpcExportRange(buff, size, &exportBase,
                              &myIpcDesc.allocationSize, &myIpcDesc.userOffset);
  if (res != flagcxSuccess) {
    WARN("buildIpcPeerPointers: cannot resolve allocation range for buff %p",
         buff);
  } else {
    INFO(FLAGCX_INIT,
         "buildIpcPeerPointers: buff=%p size=%zu exportBase=%p "
         "allocationSize=%zu userOffset=%zu",
         buff, size, exportBase, myIpcDesc.allocationSize,
         myIpcDesc.userOffset);
  }

  // Export lazily from the allocation base. globalRegPool entries describe
  // page-aligned registration ranges and may span more than one allocation,
  // so they cannot safely cache a single IPC handle.
  if (res == flagcxSuccess) {
    size_t ipcSize = 0;
    flagcxIpcMemHandle_t handlePtr = NULL;
    res = deviceAdaptor->ipcMemHandleCreate(&handlePtr, &ipcSize);
    if (res != flagcxSuccess) {
      WARN("buildIpcPeerPointers: ipcMemHandleCreate failed");
      if (handlePtr != NULL)
        deviceAdaptor->ipcMemHandleFree(handlePtr);
    } else if (handlePtr == NULL || ipcSize == 0 ||
               ipcSize > sizeof(myIpcDesc.handleData)) {
      WARN("buildIpcPeerPointers: IPC handle size %zu exceeds storage %zu",
           ipcSize, sizeof(myIpcDesc.handleData));
      if (handlePtr != NULL)
        deviceAdaptor->ipcMemHandleFree(handlePtr);
      res = flagcxNotSupported;
    } else {
      res = deviceAdaptor->ipcMemHandleGet(handlePtr, exportBase);
      if (res != flagcxSuccess) {
        WARN("buildIpcPeerPointers: ipcMemHandleGet failed for allocation %p",
             exportBase);
      } else {
        res = flagcxStoreIpcHandle(&myIpcDesc.handleData, handlePtr, ipcSize);
        if (res != flagcxSuccess) {
          WARN("buildIpcPeerPointers: cannot store IPC handle of size %zu",
               ipcSize);
        } else {
          myIpcDesc.handleSize = ipcSize;
          myIpcDesc.userSize = size;
          myIpcDesc.valid = true;
        }
      }
      deviceAdaptor->ipcMemHandleFree(handlePtr);
    }
  }

  // Step 2: All-gather IPC descriptors across local ranks
  allDescs = (struct flagcxIpcPeerDesc *)malloc(
      nRanks * sizeof(struct flagcxIpcPeerDesc));
  if (!allDescs)
    return -1;
  memset(allDescs, 0, nRanks * sizeof(struct flagcxIpcPeerDesc));
  allDescs[myRank] = myIpcDesc;

  FLAGCXCHECKGOTO(bootstrapCollAllGather(comm->bootstrap, allDescs,
                                         sizeof(struct flagcxIpcPeerDesc)),
                  res, fail);

  // IPC export is optional, but every local rank must make the same decision
  // before any peer starts opening handles. In particular, do not return
  // before the all-gather when one rank cannot export its allocation; that
  // would leave its local peers blocked in this collective.
  for (int lr = 0; lr < localRanks; lr++) {
    int globalR = localRankToRank[lr];
    struct flagcxIpcPeerDesc *pd = &allDescs[globalR];
    if (!pd->valid) {
      INFO(FLAGCX_INIT,
           "buildIpcPeerPointers: rank %d has no exportable IPC handle; "
           "disabling IPC for this local group",
           globalR);
      res = flagcxNotSupported;
      goto fail;
    }
    if (pd->handleSize == 0 || pd->handleSize > sizeof(pd->handleData) ||
        pd->userOffset > pd->allocationSize ||
        pd->userSize > pd->allocationSize - pd->userOffset) {
      WARN("buildIpcPeerPointers: rank %d reported an invalid IPC range "
           "(handleSize=%zu allocationSize=%zu userOffset=%zu userSize=%zu)",
           globalR, pd->handleSize, pd->allocationSize, pd->userOffset,
           pd->userSize);
      res = flagcxInvalidUsage;
      goto fail;
    }
  }

  // Step 3: Open peer IPC handles
  hostPeerPtrs = (void **)malloc(localRanks * sizeof(void *));
  if (!hostPeerPtrs) {
    res = flagcxSystemError;
    goto fail;
  }
  memset(hostPeerPtrs, 0, localRanks * sizeof(void *));
  hostPeerBasePtrs = (void **)malloc(localRanks * sizeof(void *));
  if (!hostPeerBasePtrs) {
    res = flagcxSystemError;
    goto fail;
  }
  memset(hostPeerBasePtrs, 0, localRanks * sizeof(void *));

  for (int lr = 0; lr < localRanks; lr++) {
    int globalR = localRankToRank[lr];
    if (globalR == myRank) {
      hostPeerPtrs[lr] = buff;
    } else {
      struct flagcxIpcPeerDesc *pd = &allDescs[globalR];
      if (pd->valid && deviceAdaptor->ipcMemHandleOpen) {
        void *peerPtr = nullptr;
        flagcxIpcMemHandle_t handlePtr = (flagcxIpcMemHandle_t)&pd->handleData;
        res = deviceAdaptor->ipcMemHandleOpen(handlePtr, &peerPtr);
        if (res != flagcxSuccess) {
          WARN("buildIpcPeerPointers: ipcMemHandleOpen failed for rank %d",
               globalR);
          goto fail;
        }
        hostPeerBasePtrs[lr] = peerPtr;
        FLAGCXCHECKGOTO(flagcxResolveIpcPeerAddress(
                            peerPtr, pd->allocationSize, pd->userOffset,
                            pd->userSize, &hostPeerPtrs[lr]),
                        res, fail);
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
  comm->ipcTable[slot].hostPeerBasePtrs = hostPeerBasePtrs;
  comm->ipcTable[slot].devPeerPtrs = devPeerPtrs;
  comm->ipcTable[slot].nPeers = localRanks;
  comm->ipcTable[slot].basePtr = buff;
  comm->ipcTable[slot].inUse = true;

  INFO(FLAGCX_INIT,
       "buildIpcPeerPointers: rank %d slot %d buff=%p devPeerPtrs=%p", myRank,
       slot, buff, (void *)devPeerPtrs);
  for (int lr = 0; lr < localRanks; lr++) {
    INFO(FLAGCX_INIT,
         "buildIpcPeerPointers:   hostPeerPtrs[%d]=%p mappingBase=%p", lr,
         hostPeerPtrs[lr], hostPeerBasePtrs[lr]);
  }
  return slot;

fail:
  free(allDescs);
  if (hostPeerBasePtrs) {
    for (int i = 0; i < localRanks; i++) {
      if (hostPeerBasePtrs[i])
        deviceAdaptor->ipcMemHandleClose(hostPeerBasePtrs[i]);
    }
    free(hostPeerBasePtrs);
  }
  if (hostPeerPtrs) {
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
    if (d->hostPeerBasePtrs) {
      for (int j = 0; j < d->nPeers; j++) {
        if (d->hostPeerBasePtrs[j])
          deviceAdaptor->ipcMemHandleClose(d->hostPeerBasePtrs[j]);
      }
      free(d->hostPeerBasePtrs);
    }
    if (d->hostPeerPtrs) {
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
