/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * ACLSHMEM Device API backend for flagcxDevComm lifecycle (Ascend 910B/910C).
 * Linked instead of default_dev_api_backend.cc when USE_SHMEM=1.
 ************************************************************************/

#include "adaptor.h"
#include "aclshmem_adaptor.h"
#include "dev_api_backend.h"
#include "device_api/aclshmem_comm_traits.h"
#include "device_api/flagcx_device.h"
#include "shmem_adaptor.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <pthread.h>

// flagcx_device_core.h casts the adaptor handle straight onto the trait Comm;
// the two layouts must agree byte for byte.
static_assert(sizeof(flagcxShmemCommInternal) ==
                  sizeof(CommTraits<AclshmemBackend>::Comm),
              "ShmemCommInternal and CommTraits<AclshmemBackend>::Comm "
              "size mismatch");
static_assert(offsetof(flagcxShmemCommInternal, gridSyncState) ==
                  offsetof(CommTraits<AclshmemBackend>::Comm, gridSyncState),
              "gridSyncState offset mismatch");
static_assert(offsetof(flagcxShmemCommInternal, scratchBuffer) ==
                  offsetof(CommTraits<AclshmemBackend>::Comm, scratchBuffer),
              "scratchBuffer offset mismatch");

static flagcxResult_t
aclshmemDevApiCommCreate(flagcxComm_t comm,
                         const struct flagcxDevCommRequirements *reqs,
                         size_t reqsSize, flagcxDevComm_t devComm) {
  if (shmemAdaptor == nullptr)
    return flagcxInternalError;

  // Reference-counted; aclshmem needs the whole communicator, since PEs are
  // global ranks and the symmetric heap is shared by all of them.
  flagcxResult_t ret = shmemAdaptor->init(comm->rank, comm->nranks, nullptr);
  if (ret != flagcxSuccess)
    return ret;

  flagcxDevCommRequirements shmemReqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  size_t copySize = reqsSize < sizeof(shmemReqs) ? reqsSize : sizeof(shmemReqs);
  memcpy(&shmemReqs, reqs, copySize);
  // The common layer owns the context count; metadata must be partitioned
  // the same way the per-context Net objects are.
  shmemReqs.interContextCount = devComm->contextCount;

  flagcxShmemComm_t shmemComm = nullptr;
  ret = shmemAdaptor->devCommCreate(comm, &shmemReqs, copySize, &shmemComm);
  if (ret != flagcxSuccess) {
    shmemAdaptor->finalize();
    return ret;
  }

  devComm->devComm = (flagcxInnerDevComm_t)shmemComm;
  devComm->signalBuffer = shmemComm->signalBuffer;
  devComm->shadowBuffer = shmemComm->shadowBuffer;
  devComm->counterBuffer = shmemComm->counterBuffer;
  devComm->signalCount = shmemComm->signalCount;
  devComm->counterCount = shmemComm->counterCount;
  int intraSize = shmemComm->intraSize;
  int interSize = (intraSize > 0 && shmemComm->nRanks % intraSize == 0)
                      ? (shmemComm->nRanks / intraSize)
                      : 1;
  devComm->nInterPeers = (interSize > 1) ? (interSize - 1) : 0;
  return flagcxSuccess;
}

static flagcxResult_t aclshmemDevApiCommDestroy(flagcxComm_t comm,
                                                flagcxDevComm_t devComm) {
  (void)comm;
  if (shmemAdaptor != nullptr && devComm->devComm != nullptr) {
    shmemAdaptor->devCommDestroy((flagcxShmemComm_t)devComm->devComm);
    devComm->devComm = nullptr;
    shmemAdaptor->finalize();
  }
  return flagcxSuccess;
}

static flagcxResult_t aclshmemDevApiMemCreate(flagcxComm_t comm, void *buff,
                                              size_t size, flagcxWindow_t win,
                                              flagcxDevMem_t devMem) {
  (void)comm;
  (void)win;
  if (!devMem->allocationTracked || devMem->allocator != flagcxMemSHMEM ||
      devMem->allocBackend != flagcxMemAllocBackendShmem) {
    WARN("aclshmem Device API memory must be allocated with "
         "flagcxMemAlloc(..., flagcxMemSHMEM)");
    return flagcxInvalidUsage;
  }
  using Window = CommTraits<AclshmemBackend>::Window;
  auto *w = new (std::nothrow) Window();
  if (w == nullptr)
    return flagcxSystemError;
  // symBase is the device-visible address of the symmetric allocation; on
  // this stack host and device map it at the same VA.
  w->symBase = buff;
  w->allocSize = size;
  w->rawPtr = buff;
  w->localPtr = buff;
  devMem->window = (void *)w;
  devMem->hasWindow = true;
  devMem->isSymmetric = true;
  return flagcxSuccess;
}

static flagcxResult_t aclshmemDevApiMemDestroy(flagcxComm_t comm,
                                               flagcxDevMem_t devMem) {
  (void)comm;
  if (devMem->window) {
    delete (CommTraits<AclshmemBackend>::Window *)devMem->window;
    devMem->window = nullptr;
  }
  return flagcxSuccess;
}

static flagcxResult_t aclshmemDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                     void **devPtr) {
  if (!devComm || !devPtr)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devComm->cachedPtrMutex);
  if (devComm->cachedDevicePtr) {
    *devPtr = devComm->cachedDevicePtr;
    pthread_mutex_unlock(&devComm->cachedPtrMutex);
    return flagcxSuccess;
  }

  flagcxDevComm hostCopy(*devComm);
  hostCopy._netContexts = nullptr;

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

  // The IR builds one flagcxDevNet per launch from this bytes-identical
  // host image: { Comm _dc; int _contextId; int _nInterPeers;
  //   unsigned int *_gridBarrierState; }
  if (hostCopy._contextCount > 0) {
    using Comm = CommTraits<AclshmemBackend>::Comm;
    struct HostNet {
      Comm _dc;
      int _contextId;
      int _nInterPeers;
      unsigned int *_gridBarrierState;
    };
    size_t netSize = sizeof(HostNet);
    size_t netArraySize = (size_t)hostCopy._contextCount * netSize;
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&netDevPtr, netArraySize,
                                                flagcxMemDevice, NULL),
                    res, fail);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMemset(netDevPtr, 0, netArraySize,
                                                flagcxMemDevice, NULL),
                    res, fail);
    for (int i = 0; i < hostCopy._contextCount; i++) {
      HostNet hn;
      memset(&hn, 0, sizeof(hn));
      hn._dc = hostCopy._commBase;
      hn._contextId = i;
      hn._nInterPeers = hostCopy._nInterPeers;
      // Barriers use gridSyncState inside the trait Comm; the per-block
      // sense-reversing array is only used by the default backend.
      hn._gridBarrierState = nullptr;
      FLAGCXCHECKGOTO(deviceAdaptor->deviceMemcpy(
                          (char *)netDevPtr + i * netSize, &hn, sizeof(hn),
                          flagcxMemcpyHostToDevice, NULL, NULL),
                      res, fail);
    }
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
  if (netDevPtr)
    deviceAdaptor->deviceFree(netDevPtr, flagcxMemDevice, NULL);
  if (dPtr)
    deviceAdaptor->deviceFree(dPtr, flagcxMemDevice, NULL);
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  return res;
}

static flagcxResult_t
aclshmemDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  if (!devComm)
    return flagcxSuccess;
  pthread_mutex_lock(&devComm->cachedPtrMutex);
  if (devComm->cachedNetContextsPtr) {
    deviceAdaptor->deviceFree(devComm->cachedNetContextsPtr, flagcxMemDevice,
                              NULL);
    devComm->cachedNetContextsPtr = nullptr;
  }
  if (devComm->cachedDevicePtr) {
    deviceAdaptor->deviceFree(devComm->cachedDevicePtr, flagcxMemDevice, NULL);
    devComm->cachedDevicePtr = nullptr;
  }
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  return flagcxSuccess;
}

static flagcxResult_t aclshmemDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
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
  if (dPtr)
    deviceAdaptor->deviceFree(dPtr, flagcxMemDevice, NULL);
  pthread_mutex_unlock(&devMem->cachedPtrMutex);
  return res;
}

static flagcxResult_t aclshmemDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
  if (!devMem)
    return flagcxSuccess;
  pthread_mutex_lock(&devMem->cachedPtrMutex);
  if (devMem->cachedDevicePtr) {
    deviceAdaptor->deviceFree(devMem->cachedDevicePtr, flagcxMemDevice, NULL);
    devMem->cachedDevicePtr = nullptr;
  }
  pthread_mutex_unlock(&devMem->cachedPtrMutex);
  return flagcxSuccess;
}

// No relay, proxy or per-comm registration: the symmetric heap is the
// transport, so there is nothing to quiesce or tear down per comm.
static flagcxResult_t aclshmemDevApiCommNoop(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend aclshmemBackend = {
    .name = "aclshmem",
    .devCommCreate = aclshmemDevApiCommCreate,
    .devCommDestroy = aclshmemDevApiCommDestroy,
    .devMemCreate = aclshmemDevApiMemCreate,
    .devMemDestroy = aclshmemDevApiMemDestroy,
    .devCommGetDevicePtr = aclshmemDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = aclshmemDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = aclshmemDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = aclshmemDevApiMemFreeDevicePtr,
    .commQuiesce = aclshmemDevApiCommNoop,
    .commCleanup = aclshmemDevApiCommNoop,
    .kind = flagcxDevApiShmem,
};

struct flagcxDevApiBackend *devApiBackend = &aclshmemBackend;
