/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * XSHMEM Device API backend for flagcxDevComm / flagcxDevMem lifecycle
 * on Kunlunxin (P800). Linked when USE_KUNLUNXIN_ADAPTOR + USE_SHMEM=1.
 *
 * XSHMEM traits provide host parsing shims, matching the NVSHMEM backend, so
 * this file constructs the real public Device API value types on the host.
 ************************************************************************/

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "device_api/xshmem_comm_traits.h"
#include "global_comm.h"
#include "kunlunxin_adaptor.h"
#include "shmem_adaptor.h"
#include "xshmem_adaptor.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <xshmem/xshmem.h>

#ifdef COMPILE_KERNEL_HOST
extern "C" size_t flagcxXshmemDevCommSizeOf();
extern "C" size_t flagcxXshmemDevMemSizeOf();
extern "C" size_t flagcxXshmemDevNetSizeOf();
#else
namespace {
static size_t flagcxXshmemDevCommSizeOf() { return sizeof(flagcxDevComm); }
static size_t flagcxXshmemDevMemSizeOf() { return sizeof(flagcxDevMem); }
static size_t flagcxXshmemDevNetSizeOf() { return 0; }
} // namespace
#endif

using XshmemComm = CommTraits<XshmemBackend>::Comm;
using XshmemWindow = CommTraits<XshmemBackend>::Window;

static_assert(sizeof(flagcxShmemCommInternal) == sizeof(XshmemComm),
              "xSHMEM communicator host/trait size mismatch");
static_assert(alignof(flagcxShmemCommInternal) == alignof(XshmemComm),
              "xSHMEM communicator host/trait alignment mismatch");
static_assert(offsetof(flagcxShmemCommInternal, gridSyncState) ==
                  offsetof(XshmemComm, gridSyncState),
              "xSHMEM communicator grid-state offset mismatch");
static_assert(offsetof(flagcxShmemCommInternal, devStateHandle) ==
                  offsetof(XshmemComm, devStateHandle),
              "xSHMEM communicator final-field offset mismatch");

// ==========================================================================
// DevComm lifecycle
// ==========================================================================
static flagcxResult_t
xshmemDevApiCommCreate(flagcxComm_t comm,
                       const struct flagcxDevCommRequirements *reqs,
                       flagcxDevComm_t devComm) {
  if (shmemAdaptor == nullptr) {
    return flagcxInternalError;
  }

  if (comm->homoComm == nullptr) {
    return flagcxInternalError;
  }
  // XSHMEM is initialized from the device-specific BKCL communicator. Until
  // the public Device API can express a homogeneous sub-team independently of
  // the FlagCX world, exposing that communicator as a heterogeneous world
  // would give device kernels incorrect ranks and peer mappings.
  if (comm->homoRanks != comm->nranks) {
    WARN("xshmem Device API currently requires a homogeneous communicator "
         "(homoRanks=%d, nranks=%d)",
         comm->homoRanks, comm->nranks);
    return flagcxNotSupported;
  }
  flagcxResult_t ret = shmemAdaptor->init(comm->homoRank, comm->homoRanks,
                                          (void *)comm->homoComm->base);
  if (ret != flagcxSuccess) {
    return ret;
  }

  flagcxDevCommRequirements shmemReqs = *reqs;
  shmemReqs.interContextCount = devComm->contextCount;

  flagcxShmemComm_t shmemComm = nullptr;
  ret = shmemAdaptor->devCommCreate(comm, &shmemReqs, &shmemComm);
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
  int interSize = (shmemComm->intraSize > 0 &&
                   shmemComm->nRanks % shmemComm->intraSize == 0)
                      ? shmemComm->nRanks / shmemComm->intraSize
                      : 1;
  devComm->nInterPeers = interSize > 1 ? interSize - 1 : 0;

  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiCommDestroy(flagcxComm_t comm,
                                              flagcxDevComm_t devComm) {
  (void)comm;
  if (shmemAdaptor != nullptr && devComm->devComm != nullptr) {
    shmemAdaptor->devCommDestroy((flagcxShmemComm_t)devComm->devComm);
    devComm->devComm = nullptr;
    shmemAdaptor->finalize();
  }
  return flagcxSuccess;
}

// ==========================================================================
// DevMem lifecycle
// ==========================================================================
static flagcxResult_t xshmemDevApiMemCreate(flagcxComm_t comm, void *buff,
                                            size_t size, flagcxWindow_t win,
                                            flagcxDevMem_t devMem) {
  (void)win;
  if (!devMem->allocationTracked || devMem->allocator != flagcxMemSHMEM ||
      devMem->allocBackend != flagcxMemAllocBackendSHMEM) {
    WARN("xshmem Device API memory must be allocated with "
         "flagcxMemAlloc(..., flagcxMemSHMEM)");
    return flagcxInvalidUsage;
  }
  auto *window = new (std::nothrow) XshmemWindow();
  if (window == nullptr)
    return flagcxSystemError;
  window->symBase = buff;
  window->allocSize = size;
  window->rawPtr = buff;
  window->intraPeMap = nullptr;
  window->intraRank = comm->localRank;
  window->intraSize = comm->localRanks;
  if (comm->localRanks > 0) {
    if (comm->localRankToRank == nullptr) {
      delete window;
      return flagcxInternalError;
    }
    size_t mapBytes = (size_t)comm->localRanks * sizeof(int);
    flagcxResult_t result = deviceAdaptor->deviceMalloc(
        (void **)&window->intraPeMap, mapBytes, flagcxMemDevice, nullptr);
    if (result != flagcxSuccess) {
      delete window;
      return result;
    }
    result = deviceAdaptor->deviceMemcpy(
        window->intraPeMap, comm->localRankToRank, mapBytes,
        flagcxMemcpyHostToDevice, nullptr, nullptr);
    if (result != flagcxSuccess) {
      deviceAdaptor->deviceFree(window->intraPeMap, flagcxMemDevice, nullptr);
      delete window;
      return result;
    }
  }
  devMem->window = window;
  devMem->hasWindow = true;
  devMem->isSymmetric = true;
  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiMemDestroy(flagcxComm_t comm,
                                             flagcxDevMem_t devMem) {
  (void)comm;
  auto *window = (XshmemWindow *)devMem->window;
  if (window != nullptr && window->intraPeMap != nullptr)
    deviceAdaptor->deviceFree(window->intraPeMap, flagcxMemDevice, nullptr);
  delete window;
  devMem->window = nullptr;
  return flagcxSuccess;
}

// ==========================================================================
// Device pointer materialization
// ==========================================================================
static flagcxResult_t xshmemDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                   void **devPtr) {
  if (devComm == nullptr || devPtr == nullptr || devComm->devComm == nullptr)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devComm->cachedPtrMutex);
  if (devComm->cachedDevicePtr != nullptr) {
    *devPtr = devComm->cachedDevicePtr;
    pthread_mutex_unlock(&devComm->cachedPtrMutex);
    return flagcxSuccess;
  }

  flagcxDevComm hostCopy(*devComm);
  hostCopy._netContexts = nullptr;
  if (flagcxXshmemDevCommSizeOf() != sizeof(hostCopy)) {
    pthread_mutex_unlock(&devComm->cachedPtrMutex);
    return flagcxInternalError;
  }

  void *deviceCopy = nullptr;
  void *netContexts = nullptr;
  flagcxResult_t result = flagcxSuccess;
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&deviceCopy,
                                              sizeof(flagcxDevComm),
                                              flagcxMemDevice, nullptr),
                  result, fail);

  if (hostCopy._contextCount > 0) {
    struct HostNet {
      XshmemComm _dc;
      int _contextId;
      int _nInterPeers;
      unsigned int *_gridBarrierState;
    };
    size_t netSize = flagcxXshmemDevNetSizeOf();
    if (netSize == 0)
      netSize = sizeof(HostNet);
    if (netSize != sizeof(HostNet)) {
      result = flagcxInternalError;
      goto fail;
    }
    size_t bytes = (size_t)hostCopy._contextCount * netSize;
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&netContexts, bytes,
                                                flagcxMemDevice, nullptr),
                    result, fail);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMemset(netContexts, 0, bytes,
                                                flagcxMemDevice, nullptr),
                    result, fail);
    for (int i = 0; i < hostCopy._contextCount; ++i) {
      HostNet net;
      memset(&net, 0, sizeof(net));
      net._dc = hostCopy._commBase;
      net._contextId = i;
      net._nInterPeers = hostCopy._nInterPeers;
      FLAGCXCHECKGOTO(deviceAdaptor->deviceMemcpy(
                          (char *)netContexts + (size_t)i * netSize, &net,
                          sizeof(net), flagcxMemcpyHostToDevice, nullptr,
                          nullptr),
                      result, fail);
    }
    hostCopy._netContexts = netContexts;
  }

  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMemcpy(deviceCopy, &hostCopy, sizeof(flagcxDevComm),
                                  flagcxMemcpyHostToDevice, nullptr, nullptr),
      result, fail);
  FLAGCXCHECKGOTO(deviceAdaptor->deviceSynchronize(), result, fail);

  devComm->cachedDevicePtr = deviceCopy;
  devComm->cachedNetContextsPtr = netContexts;
  *devPtr = deviceCopy;
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  return flagcxSuccess;

fail:
  if (netContexts)
    deviceAdaptor->deviceFree(netContexts, flagcxMemDevice, nullptr);
  if (deviceCopy)
    deviceAdaptor->deviceFree(deviceCopy, flagcxMemDevice, nullptr);
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  return result;
}

static flagcxResult_t xshmemDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  if (devComm == nullptr)
    return flagcxSuccess;
  pthread_mutex_lock(&devComm->cachedPtrMutex);
  if (devComm->cachedNetContextsPtr) {
    deviceAdaptor->deviceFree(devComm->cachedNetContextsPtr, flagcxMemDevice,
                              nullptr);
    devComm->cachedNetContextsPtr = nullptr;
  }
  if (devComm->cachedDevicePtr) {
    deviceAdaptor->deviceFree(devComm->cachedDevicePtr, flagcxMemDevice,
                              nullptr);
    devComm->cachedDevicePtr = nullptr;
  }
  pthread_mutex_unlock(&devComm->cachedPtrMutex);
  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
                                                  void **devPtr) {
  if (devMem == nullptr || devPtr == nullptr || devMem->window == nullptr)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devMem->cachedPtrMutex);
  if (devMem->cachedDevicePtr != nullptr) {
    *devPtr = devMem->cachedDevicePtr;
    pthread_mutex_unlock(&devMem->cachedPtrMutex);
    return flagcxSuccess;
  }

  flagcxDevMem hostCopy(*devMem);
  if (flagcxXshmemDevMemSizeOf() != sizeof(hostCopy)) {
    pthread_mutex_unlock(&devMem->cachedPtrMutex);
    return flagcxInternalError;
  }

  void *deviceCopy = nullptr;
  flagcxResult_t result = deviceAdaptor->deviceMalloc(
      &deviceCopy, sizeof(flagcxDevMem), flagcxMemDevice, nullptr);
  if (result == flagcxSuccess)
    result =
        deviceAdaptor->deviceMemcpy(deviceCopy, &hostCopy, sizeof(flagcxDevMem),
                                    flagcxMemcpyHostToDevice, nullptr, nullptr);
  if (result == flagcxSuccess)
    result = deviceAdaptor->deviceSynchronize();
  if (result != flagcxSuccess) {
    if (deviceCopy)
      deviceAdaptor->deviceFree(deviceCopy, flagcxMemDevice, nullptr);
    pthread_mutex_unlock(&devMem->cachedPtrMutex);
    return result;
  }

  devMem->cachedDevicePtr = deviceCopy;
  *devPtr = deviceCopy;
  pthread_mutex_unlock(&devMem->cachedPtrMutex);
  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
  if (devMem == nullptr)
    return flagcxSuccess;
  pthread_mutex_lock(&devMem->cachedPtrMutex);
  if (devMem->cachedDevicePtr) {
    deviceAdaptor->deviceFree(devMem->cachedDevicePtr, flagcxMemDevice,
                              nullptr);
    devMem->cachedDevicePtr = nullptr;
  }
  pthread_mutex_unlock(&devMem->cachedPtrMutex);
  return flagcxSuccess;
}

static flagcxResult_t xshmemDevApiCommCleanup(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend xshmemBackend = {
    .name = "xshmem",
    .devCommCreate = xshmemDevApiCommCreate,
    .devCommDestroy = xshmemDevApiCommDestroy,
    .devMemCreate = xshmemDevApiMemCreate,
    .devMemDestroy = xshmemDevApiMemDestroy,
    .devCommGetDevicePtr = xshmemDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = xshmemDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = xshmemDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = xshmemDevApiMemFreeDevicePtr,
    .commCleanup = xshmemDevApiCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &xshmemBackend;
