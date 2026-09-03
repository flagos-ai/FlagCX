/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM Device API backend for flagcxDevComm lifecycle.
 * Linked when USE_SHMEM=1.
 ************************************************************************/

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "device_api/nvshmem_comm_traits.h"
#include "nvshmem_adaptor.h"
#include "shmem_adaptor.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>

// Host-visible helper from device_api_host_helpers.cu (size query only)
#ifdef COMPILE_KERNEL_HOST
extern "C" size_t flagcxDevNetSizeOf();
#else
static size_t flagcxDevNetSizeOf() { return 0; }
#endif

// Verify that flagcxShmemCommInternal and CommTraits<NvshmemBackend>::Comm have
// compatible layout — the constructor in flagcx_device_core.h does a raw cast.
static_assert(
    sizeof(flagcxShmemCommInternal) == sizeof(CommTraits<NvshmemBackend>::Comm),
    "ShmemCommInternal and CommTraits<NvshmemBackend>::Comm size mismatch");
static_assert(
    offsetof(flagcxShmemCommInternal, gridSyncState) ==
        offsetof(CommTraits<NvshmemBackend>::Comm, gridSyncState),
    "ShmemCommInternal and CommTraits::Comm last-field offset mismatch");

static flagcxResult_t
nvshmemDevApiCommCreate(flagcxComm_t comm,
                        const struct flagcxDevCommRequirements *reqs,
                        flagcxDevComm_t devComm) {
  if (shmemAdaptor == nullptr) {
    return flagcxInternalError;
  }

  // Initialize NVSHMEM (reference-counted, safe to call multiple times).
  // NVSHMEM does not require a vendor-specific initialization handle.
  flagcxResult_t ret = shmemAdaptor->init(comm->rank, comm->nranks, nullptr);
  if (ret != flagcxSuccess) {
    return ret;
  }

  // NVSHMEM has no user-visible transport-context API, but FlagCX logical
  // contexts still require disjoint signal/counter/shadow namespaces.  Pass
  // the context count selected by the common Device API layer to the SHMEM
  // adaptor so its metadata allocation matches the device-visible Net array.
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
  // All logical contexts share NVSHMEM's transport, while their metadata is
  // partitioned by context in CommTraits<NvshmemBackend>::Net.
  // NVSHMEM doesn't need a host-side relay, but the World barrier uses
  // nInterPeers to decide whether to compose the inter-node barrier phase.
  int intraSize = shmemComm->intraSize;
  int interSize = (intraSize > 0 && shmemComm->nRanks % intraSize == 0)
                      ? (shmemComm->nRanks / intraSize)
                      : 1;
  devComm->nInterPeers = (interSize > 1) ? (interSize - 1) : 0;

  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiCommDestroy(flagcxComm_t comm,
                                               flagcxDevComm_t devComm) {
  (void)comm;
  if (shmemAdaptor != nullptr && devComm->devComm != nullptr) {
    shmemAdaptor->devCommDestroy((flagcxShmemComm_t)devComm->devComm);
    devComm->devComm = nullptr;
    shmemAdaptor->finalize();
  }
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t devMem) {
  (void)comm;
  (void)win;
  if (!devMem->allocationTracked || devMem->allocator != flagcxMemSHMEM ||
      devMem->allocBackend != flagcxMemAllocBackendSHMEM) {
    WARN("nvshmem Device API memory must be allocated with "
         "flagcxMemAlloc(..., flagcxMemSHMEM)");
    return flagcxInvalidUsage;
  }
  using Window = CommTraits<NvshmemBackend>::Window;
  auto *w = new (std::nothrow) Window();
  if (w == nullptr)
    return flagcxSystemError;
  w->symBase = buff;
  w->allocSize = size;
  w->rawPtr = buff;
  devMem->window = (void *)w;
  devMem->hasWindow = true;
  devMem->isSymmetric = true;
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  (void)comm;
  if (devMem->window) {
    delete (CommTraits<NvshmemBackend>::Window *)devMem->window;
    devMem->window = nullptr;
  }
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                    void **devPtr) {
  if (!devComm || !devPtr)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&devComm->cachedPtrMutex);

  if (devComm->cachedDevicePtr) {
    *devPtr = devComm->cachedDevicePtr;
    pthread_mutex_unlock(&devComm->cachedPtrMutex);
    return flagcxSuccess;
  }

  // Construct value struct on host stack, then copy to device.
  // Note: _gridBarrierState is intentionally left nullptr for NVSHMEM.
  // NVSHMEM barriers use nvshmemx_barrier_block() + per-block arrive/release
  // flags internally, so the IR-level flagcxGridSync (sense-reversing) is not
  // needed and the null check in the IR functions will skip it.
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

  // Allocate + construct net context array on device.
  // flagcxDevNet is device-only (#ifdef FLAGCX_DEVICE_COMPILE), so we build the
  // equivalent bytes on the host.  NVSHMEM flagcxDevNet layout:
  //   struct { Comm _dc; int _contextId; } (base: DeviceAPI::Net)
  //   int _nInterPeers;
  //   unsigned int* _gridBarrierState;
  // The kernel-launch approach (flagcxDevNetLaunchConstruct) fails with
  // "invalid resource handle" when the library's device fatbinary isn't
  // registered in the calling process.  Use host memcpy instead.
  if (hostCopy._contextCount > 0) {
    using Comm = CommTraits<NvshmemBackend>::Comm;
    struct HostNet {
      Comm _dc;
      int _contextId;
      int _nInterPeers;
      unsigned int *_gridBarrierState;
    };
    size_t netSize = flagcxDevNetSizeOf();
    if (netSize == 0)
      netSize = sizeof(HostNet);
    size_t netArraySize = hostCopy._contextCount * netSize;
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&netDevPtr, netArraySize,
                                                flagcxMemDevice, NULL),
                    res, fail);
    // Zero-init to avoid uninitialized padding bytes
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMemset(netDevPtr, 0, netArraySize,
                                                flagcxMemDevice, NULL),
                    res, fail);
    for (int i = 0; i < hostCopy._contextCount; i++) {
      HostNet hn;
      memset(&hn, 0, sizeof(hn));
      hn._dc = hostCopy._commBase;
      hn._contextId = i;
      hn._nInterPeers = hostCopy._nInterPeers;
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

static flagcxResult_t nvshmemDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
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

static flagcxResult_t nvshmemDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
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

static flagcxResult_t nvshmemDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
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

static flagcxResult_t nvshmemDevApiCommCleanup(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend nvshmemBackend = {
    .name = "nvshmem",
    .devCommCreate = nvshmemDevApiCommCreate,
    .devCommDestroy = nvshmemDevApiCommDestroy,
    .devMemCreate = nvshmemDevApiMemCreate,
    .devMemDestroy = nvshmemDevApiMemDestroy,
    .devCommGetDevicePtr = nvshmemDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = nvshmemDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = nvshmemDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = nvshmemDevApiMemFreeDevicePtr,
    .commCleanup = nvshmemDevApiCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &nvshmemBackend;
