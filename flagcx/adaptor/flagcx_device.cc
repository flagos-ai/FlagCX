/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-side lifecycle management for flagcxDevComm_t and flagcxDevMem_t.
 * Thin dispatcher: delegates all backend-specific logic via devApiBackend.
 ************************************************************************/

#define FLAGCX_DISABLE_DEV_COMM_CREATE_SIZE_DISPATCH
#include "device_api/flagcx_device.h"
#include "alloc.h"
#include "comm.h"
#include "flagcx_hetero.h"
#include "flagcx_kernel_internal.h"
#include "mem_alloc_registry.h"
#include "p2p.h"
#include "proxy.h"
#include "sym_heap.h"
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>

#include "dev_api_backend.h"
#include <climits>
#include <cstdint>

// ==========================================================================
// DevComm lifecycle
// ==========================================================================

flagcxResult_t flagcxDevCommRetryPendingCleanup(flagcxComm_t comm) {
  if (comm == nullptr || comm->pendingDevCommCleanup == nullptr)
    return flagcxSuccess;
  flagcxDevComm_t pending = comm->pendingDevCommCleanup;
  flagcxResult_t result = flagcxDevCommDestroy(comm, pending);
  if (result == flagcxSuccess)
    comm->pendingDevCommCleanup = nullptr;
  return result;
}

static flagcxResult_t
flagcxDevCommCreateInternal(flagcxComm_t comm,
                            const struct flagcxDevCommRequirements *reqs,
                            size_t reqsSize, flagcxDevComm_t *devComm) {
  if (comm == nullptr || reqs == nullptr || devComm == nullptr ||
      reqsSize < FLAGCX_DEV_COMM_REQUIREMENTS_LEGACY_SIZE)
    return flagcxInvalidArgument;
  *devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommRetryPendingCleanup(comm));
  if (comm->devCommCount == INT_MAX)
    return flagcxInvalidUsage;

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
        // The output remains NULL because creation failed, but the outer
        // communicator retains the partially-created owner for a later retry.
        comm->pendingDevCommCleanup = handle;
        return ret;
      }
      pthread_mutex_destroy(&handle->cachedPtrMutex);
      free(handle);
      return ret;
    }
  }

  *devComm = handle;
  handle->devCommId = ++comm->devCommCount;
  handle->previousActive = comm->activeDevComms;
  comm->activeDevComms = handle;

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
  if (comm == nullptr &&
      (devComm->cachedNetContextsPtr || devComm->cachedDevicePtr ||
       devComm->cachedGridBarrierPtr))
    return flagcxInvalidArgument;
  if (devComm->devCommId != 0 &&
      (comm == nullptr || devComm->devCommId != comm->devCommCount ||
       comm->activeDevComms != devComm))
    return flagcxInvalidUsage;
  if (deviceAdaptor != nullptr && deviceAdaptor->deviceSynchronize != nullptr)
    FLAGCXCHECK(deviceAdaptor->deviceSynchronize());

  // Stop the proxy from acquiring a new reference while the backend tears
  // down fields reachable through DevComm. Restore publication on failure so
  // the still-owned object remains retryable.
  const bool wasPublished = comm != nullptr && comm->heteroComm != nullptr &&
                            comm->heteroComm->devCommHandle == devComm;
  if (wasPublished)
    comm->heteroComm->devCommHandle = devComm->previousActive;

  if (!devComm->backendDestroyed) {
    flagcxResult_t result = devApiBackend->devCommDestroy(comm, devComm);
    if (result != flagcxSuccess) {
      if (wasPublished)
        comm->heteroComm->devCommHandle = devComm;
      return result;
    }
    devComm->backendDestroyed = true;
  }

  if (devComm->cachedNetContextsPtr != nullptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(devComm->cachedNetContextsPtr,
                                          flagcxMemDevice, nullptr));
    devComm->cachedNetContextsPtr = nullptr;
  }
  if (devComm->cachedDevicePtr != nullptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(devComm->cachedDevicePtr,
                                          flagcxMemDevice, nullptr));
    devComm->cachedDevicePtr = nullptr;
  }
  if (devComm->cachedGridBarrierPtr != nullptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(devComm->cachedGridBarrierPtr,
                                          flagcxMemDevice, nullptr));
    devComm->cachedGridBarrierPtr = nullptr;
  }
  pthread_mutex_destroy(&devComm->cachedPtrMutex);
  if (devComm->devCommId != 0)
    --comm->devCommCount;
  if (comm != nullptr && comm->activeDevComms == devComm)
    comm->activeDevComms = devComm->previousActive;
  free(devComm);
  return flagcxSuccess;
}

// ==========================================================================
// DevMem lifecycle
// ==========================================================================

static void freeUnboundDevMem(flagcxDevMem_t handle) {
  pthread_mutex_destroy(&handle->cachedPtrMutex);
  free(handle);
}

static flagcxResult_t flagcxDevMemCreateImpl(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             int preparedIpcSlot,
                                             flagcxDevMem_t *devMem) {
  if (devMem == nullptr)
    return flagcxInvalidArgument;
  *devMem = nullptr;
  if (comm == nullptr)
    return flagcxInvalidArgument;
  flagcxResult_t localPrepare =
      buff == nullptr || size == 0 ? flagcxInvalidArgument : flagcxSuccess;
  flagcxDevMem_t handle = nullptr;
  if (localPrepare == flagcxSuccess) {
    handle = (flagcxDevMem_t)malloc(sizeof(struct flagcxDevMemInternal));
    if (handle == nullptr) {
      localPrepare = flagcxSystemError;
    } else {
      memset(handle, 0, sizeof(struct flagcxDevMemInternal));
      if (pthread_mutex_init(&handle->cachedPtrMutex, NULL) != 0) {
        free(handle);
        handle = nullptr;
        localPrepare = flagcxSystemError;
      }
    }
  }
  if (localPrepare == flagcxSuccess) {
    handle->rawPtr = buff;
    handle->ipcIndex = -1;
    handle->runnerOwned = preparedIpcSlot >= 0;
  }

  // A subrange is valid only when its complete byte range lies inside the
  // exact flagcxMemAlloc allocation that contains its first byte. Unknown
  // pointers remain supported for the default/CCL backend.
  if (localPrepare == flagcxSuccess) {
    flagcxMemAllocationInfo allocation;
    flagcxResult_t allocationRes =
        globalMemAllocRegistry.findRange(buff, 1, &allocation);
    if (allocationRes == flagcxSuccess) {
      if (globalMemAllocRegistry.findRange(buff, size, &allocation) !=
          flagcxSuccess) {
        localPrepare = flagcxInvalidUsage;
      } else {
        handle->allocationTracked = true;
        handle->allocationBase = allocation.base;
        handle->allocationSize = allocation.size;
        handle->allocator = allocation.allocator;
        handle->allocBackend = allocation.backend;
      }
    } else if (allocationRes != flagcxInvalidUsage) {
      localPrepare = allocationRes;
    }
  }
  if (localPrepare != flagcxSuccess) {
    if (handle != nullptr)
      freeUnboundDevMem(handle);
    return localPrepare;
  }
  if (preparedIpcSlot >= 0 && win != nullptr) {
    freeUnboundDevMem(handle);
    return flagcxInvalidUsage;
  }
  if (win != nullptr) {
    if ((win->ownerComm != nullptr && win->ownerComm != comm) ||
        (win->vendorBase == nullptr && win->defaultBase == nullptr) ||
        win->devMemRefs == UINT32_MAX) {
      freeUnboundDevMem(handle);
      return flagcxInvalidUsage;
    }
    ++win->devMemRefs;
    handle->boundWindow = win;
  }
  if (preparedIpcSlot >= 0) {
    if (preparedIpcSlot >= comm->ipcTableCapacity) {
      freeUnboundDevMem(handle);
      return flagcxInvalidUsage;
    }
    auto &entry = comm->ipcTable[preparedIpcSlot];
    if (!entry.inUse || entry.ownerHeld || entry.basePtr != buff ||
        entry.size < size || entry.hostPeerPtrs == nullptr ||
        entry.devPeerPtrs == nullptr || entry.refCount == SIZE_MAX) {
      freeUnboundDevMem(handle);
      return flagcxInvalidUsage;
    }
    ++entry.refCount;
    handle->ipcIndex = preparedIpcSlot;
  }
  handle->activeNext = comm->activeDevMems;
  comm->activeDevMems = handle;
  // Backend-specific creation
  {
    flagcxResult_t ret =
        devApiBackend->devMemCreate(comm, buff, size, win, handle);
    if (ret != flagcxSuccess) {
      WARN("flagcxDevMemCreate: %s backend failed (%d)", devApiBackend->name,
           ret);
      flagcxResult_t cleanup = flagcxDevMemDestroy(comm, handle);
      if (cleanup != flagcxSuccess) {
        *devMem = handle;
        return cleanup;
      }
      return ret;
    }
  }

  *devMem = handle;
  return flagcxSuccess;
}

extern "C" flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t *devMem) {
  return flagcxDevMemCreateImpl(comm, buff, size, win, -1, devMem);
}

flagcxResult_t flagcxUniRunnerIpcWindowRegisterAndDevMemCreate(
    flagcxComm_t comm, void *buff, size_t size, flagcxDevMem_t *devMem) {
  if (comm == nullptr || devMem == nullptr)
    return flagcxInvalidArgument;
  *devMem = nullptr;
  int slot = -1;
  flagcxResult_t local =
      flagcxCommWindowRegisterIpcInternal(comm, buff, size, true, &slot);
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, local, &common));
  if (common == flagcxSuccess)
    local = flagcxDevMemCreateImpl(comm, buff, size, nullptr, slot, devMem);
  else
    local = common;
  if (local == flagcxSuccess && (*devMem == nullptr || !(*devMem)->hasWindow))
    local = *devMem == nullptr ? flagcxSystemError : flagcxNotSupported;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, local, &common));
  if (common == flagcxSuccess)
    return flagcxSuccess;

  // Every rank closes its imported mapping before any caller may free its
  // exporter allocation during the collective creation rollback.
  flagcxResult_t cleanup = flagcxSuccess;
  if (*devMem != nullptr) {
    cleanup = flagcxDevMemDestroy(comm, *devMem);
    if (cleanup == flagcxSuccess)
      *devMem = nullptr;
  } else if (slot >= 0 && comm->ipcTable[slot].inUse) {
    cleanup = releaseIpcTableSlot(comm, slot);
  }
  flagcxResult_t commonCleanup = flagcxSuccess;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, cleanup, &commonCleanup));
  return commonCleanup != flagcxSuccess ? commonCleanup : common;
}

extern "C" flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  if (devMem == nullptr)
    return flagcxSuccess;
  if (comm == nullptr && (devMem->ipcIndex >= 0 || devMem->cachedDevicePtr ||
                          devMem->segmentedPeerLayouts))
    return flagcxInvalidArgument;

  if (deviceAdaptor != nullptr && deviceAdaptor->deviceSynchronize != nullptr)
    FLAGCXCHECK(deviceAdaptor->deviceSynchronize());
  if (!devMem->backendDestroyed) {
    FLAGCXCHECK(devApiBackend->devMemDestroy(comm, devMem));
    devMem->backendDestroyed = true;
  }

  if (devMem->ipcIndex >= 0) {
    if (comm == nullptr || devMem->ipcIndex >= comm->ipcTableCapacity)
      return flagcxInvalidUsage;
    struct flagcxIpcTableEntry *ipcEntry = &comm->ipcTable[devMem->ipcIndex];
    if (ipcEntry->refCount == 0)
      return flagcxInvalidUsage;
    if (ipcEntry->refCount == 1 && !ipcEntry->ownerHeld) {
      ipcEntry->refCount = 0;
      flagcxResult_t release = releaseIpcTableSlot(comm, devMem->ipcIndex);
      if (release != flagcxSuccess) {
        ipcEntry->refCount = 1;
        return release;
      }
    } else {
      --ipcEntry->refCount;
    }
  }
  devMem->ipcIndex = -1;
  if (devMem->cachedDevicePtr != nullptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(devMem->cachedDevicePtr,
                                          flagcxMemDevice, nullptr));
    devMem->cachedDevicePtr = nullptr;
  }
  if (devMem->segmentedPeerLayouts != nullptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(devMem->segmentedPeerLayouts,
                                          flagcxMemDevice, nullptr));
    devMem->segmentedPeerLayouts = nullptr;
  }
  if (devMem->boundWindow != nullptr) {
    --devMem->boundWindow->devMemRefs;
    devMem->boundWindow = nullptr;
  }
  if (comm != nullptr) {
    flagcxDevMem_t *link = &comm->activeDevMems;
    while (*link != nullptr && *link != devMem)
      link = &(*link)->activeNext;
    if (*link == devMem)
      *link = devMem->activeNext;
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

flagcxResult_t flagcxCommQuiesce(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  // Backend producers (notably device FIFOs) feed the common RMA proxy. Drain
  // them first so already-accepted device work can still be enqueued.
  if (devApiBackend->commQuiesce != nullptr)
    FLAGCXCHECK(devApiBackend->commQuiesce(comm));
  if (comm->heteroComm != nullptr) {
    FLAGCXCHECK(flagcxHeteroRmaProxyQuiesce(comm->heteroComm));
    comm->heteroComm->oneSideQuiesced = true;
  }
  return flagcxSuccess;
}

// ==========================================================================
// IPC table cleanup — called after hetero resources stop using the mappings,
// while the owning communicator and device adaptor are still alive.
// ==========================================================================

flagcxResult_t flagcxCommCleanupIpcTable(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  for (int k = 0; k < comm->ipcTableCapacity; k++) {
    struct flagcxIpcTableEntry *e = &comm->ipcTable[k];
    if (e->refCount != 0)
      return flagcxInvalidUsage;
    FLAGCXCHECK(releaseIpcTableSlot(comm, k));
  }
  return flagcxSuccess;
}

flagcxResult_t releaseIpcTableSlot(flagcxComm_t comm, int slot) {
  if (comm == nullptr || slot < 0 || slot >= comm->ipcTableCapacity)
    return flagcxInvalidArgument;
  struct flagcxIpcTableEntry *e = &comm->ipcTable[slot];
  if (e->refCount != 0)
    return flagcxInvalidUsage;
  if (e->hostPeerBasePtrs != nullptr) {
    for (int i = 0; i < e->nPeers; i++) {
      if (e->hostPeerBasePtrs[i] != nullptr) {
        FLAGCXCHECK(deviceAdaptor->ipcMemHandleClose(e->hostPeerBasePtrs[i]));
        e->hostPeerBasePtrs[i] = nullptr;
      }
    }
    free(e->hostPeerBasePtrs);
    e->hostPeerBasePtrs = nullptr;
  }
  if (e->devPeerPtrs != nullptr) {
    FLAGCXCHECK(
        deviceAdaptor->deviceFree(e->devPeerPtrs, flagcxMemDevice, nullptr));
    e->devPeerPtrs = nullptr;
  }
  free(e->hostPeerPtrs);
  e->hostPeerPtrs = nullptr;
  e->basePtr = nullptr;
  e->size = 0;
  e->nPeers = 0;
  e->refCount = 0;
  e->inUse = false;
  e->ownerHeld = false;
  return flagcxSuccess;
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

flagcxResult_t flagcxConvergeBootstrapStatus(flagcxComm_t comm,
                                             flagcxResult_t local,
                                             flagcxResult_t *common) {
  if (comm == nullptr || comm->bootstrap == nullptr || common == nullptr ||
      comm->rank < 0 || comm->rank >= comm->nranks || comm->nranks <= 0)
    return flagcxInvalidArgument;

  constexpr int root = 0;
  constexpr int gatherTag = 0x5960;
  constexpr int broadcastTag = 0x5961;
  int value = static_cast<int>(local);
  int result = value;
  if (comm->rank == root) {
    for (int peer = 1; peer < comm->nranks; peer++) {
      int peerValue = static_cast<int>(flagcxSuccess);
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, gatherTag, &peerValue,
                                sizeof(peerValue)));
      if (result == flagcxSuccess && peerValue != flagcxSuccess)
        result = peerValue;
    }
    for (int peer = 1; peer < comm->nranks; peer++)
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, peer, broadcastTag, &result,
                                sizeof(result)));
  } else {
    FLAGCXCHECK(
        bootstrapSend(comm->bootstrap, root, gatherTag, &value, sizeof(value)));
    FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, broadcastTag, &result,
                              sizeof(result)));
  }
  *common = static_cast<flagcxResult_t>(result);
  return flagcxSuccess;
}

static flagcxResult_t flagcxGrowIpcTable(flagcxComm_t comm) {
  if (comm == nullptr || comm->ipcTableCapacity <= 0 ||
      comm->ipcTableCapacity > INT_MAX / 2)
    return flagcxNotSupported;
  const int newCapacity = comm->ipcTableCapacity * 2;
  auto *entries = static_cast<flagcxIpcTableEntry *>(
      calloc(static_cast<size_t>(newCapacity), sizeof(flagcxIpcTableEntry)));
  if (entries == nullptr)
    return flagcxSystemError;
  memcpy(entries, comm->ipcTable,
         static_cast<size_t>(comm->ipcTableCapacity) * sizeof(*entries));
  flagcxIpcTableEntry *old = comm->ipcTable;
  comm->ipcTable = entries;
  comm->ipcTableCapacity = newCapacity;
  if (comm->heteroComm != nullptr) {
    comm->heteroComm->ipcTable = entries;
    comm->heteroComm->ipcTableSize = newCapacity;
  }
  if (old != comm->ipcTableInline)
    free(old);
  return flagcxSuccess;
}

int buildIpcPeerPointers(flagcxComm_t comm, void *buff, size_t size) {
  flagcxResult_t res = flagcxSuccess;
  int slot = -1;
  for (int k = 0; k < comm->ipcTableCapacity; k++) {
    if (comm->ipcTable[k].hostPeerPtrs == nullptr &&
        comm->ipcTable[k].hostPeerBasePtrs == nullptr &&
        comm->ipcTable[k].devPeerPtrs == nullptr) {
      slot = k;
      break;
    }
  }
  if (slot < 0) {
    res = flagcxGrowIpcTable(comm);
    if (res == flagcxSuccess)
      slot = comm->ipcTableCapacity / 2;
  }

  int myRank = comm->rank;
  int nRanks = comm->nranks;
  int localRanks = comm->localRanks;
  int *localRankToRank = comm->localRankToRank;

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
  // Converge every rank-local preparation result before any rank enters the
  // descriptor metadata exchange. In particular, a failed export must not be
  // represented as an invalid descriptor while peers proceed to all-gather.
  flagcxResult_t localPrepare = res;
  if (localPrepare == flagcxSuccess && slot < 0)
    localPrepare = flagcxNotSupported;
  if (localPrepare == flagcxSuccess && allDescs == nullptr)
    localPrepare = flagcxSystemError;
  flagcxResult_t commonPrepare = flagcxSuccess;
  FLAGCXCHECKGOTO(
      flagcxConvergeBootstrapStatus(comm, localPrepare, &commonPrepare), res,
      fail);
  if (commonPrepare != flagcxSuccess) {
    res = commonPrepare;
    goto fail;
  }
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
  comm->ipcTable[slot].size = size;
  comm->ipcTable[slot].refCount = 0;
  comm->ipcTable[slot].inUse = true;
  comm->ipcTable[slot].ownerHeld = false;

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

flagcxResult_t flagcxCommWindowRegisterIpcInternal(flagcxComm_t comm,
                                                   void *buff, size_t size,
                                                   bool uniRunnerIpc,
                                                   int *slot) {
  if (comm == nullptr || slot == nullptr)
    return flagcxInvalidArgument;
  *slot = -1;
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(
      comm,
      buff == nullptr || size == 0 ? flagcxInvalidArgument : flagcxSuccess,
      &common));
  if (common != flagcxSuccess)
    return common;
  *slot = buildIpcPeerPointers(comm, buff, size);
  if (*slot < 0)
    return flagcxNotSupported;
  comm->ipcTable[*slot].ownerHeld = !uniRunnerIpc;
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
