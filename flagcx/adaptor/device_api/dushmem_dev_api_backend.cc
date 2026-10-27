/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * DUSHMEM Device API backend for flagcxDevComm / flagcxDevMem lifecycle
 * on the DU (HIP/DTK) platform. Linked when USE_DU_ADAPTOR + USE_SHMEM=1.
 *
 * DUSHMEM traits provide host parsing shims, matching the NVSHMEM backend,
 * so this file constructs the real public Device API value types on the host.
 ************************************************************************/

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "device_api/dushmem_comm_traits.h"
#include "dushmem_adaptor.h"
#include "global_comm.h"
#include "shmem_adaptor.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>

// Host-visible helper from flagcx/adaptor/kernel/du/device_api_host_helpers.hip
// (size query only)
#ifdef COMPILE_KERNEL_HOST
extern "C" size_t flagcxDevNetSizeOf();
#else
static size_t flagcxDevNetSizeOf() { return 0; }
#endif

using DushmemComm = CommTraits<DushmemBackend>::Comm;
using DushmemWindow = CommTraits<DushmemBackend>::Window;

// Verify that flagcxShmemCommInternal and CommTraits<DushmemBackend>::Comm
// have compatible layout — the constructor in flagcx_device_core.h does a
// raw cast.
static_assert(sizeof(flagcxShmemCommInternal) == sizeof(DushmemComm),
              "DUSHMEM communicator host/trait size mismatch");
static_assert(alignof(flagcxShmemCommInternal) == alignof(DushmemComm),
              "DUSHMEM communicator host/trait alignment mismatch");
static_assert(offsetof(flagcxShmemCommInternal, gridSyncState) ==
                  offsetof(DushmemComm, gridSyncState),
              "DUSHMEM communicator grid-state offset mismatch");

// ==========================================================================
// DevComm lifecycle
// ==========================================================================
static flagcxResult_t
dushmemDevApiCommCreate(flagcxComm_t comm,
                        const struct flagcxDevCommRequirements *reqs,
                        size_t reqsSize, flagcxDevComm_t devComm) {
  if (shmemAdaptor == nullptr) {
    return flagcxInternalError;
  }

  // DUSHMEM is initialized with the environment-selected bootstrap; no
  // vendor-specific handle is required (pass nullptr). A borrowed
  // dushmemx_uniqueid_t can be passed instead once the public Device API
  // can exchange it through the FlagCX bootstrap.
  if (comm->homoRanks != comm->nranks) {
    WARN("dushmem Device API currently requires a homogeneous communicator "
         "(homoRanks=%d, nranks=%d)",
         comm->homoRanks, comm->nranks);
    return flagcxNotSupported;
  }
  flagcxResult_t ret = shmemAdaptor->init(comm->homoRank, comm->homoRanks,
                                          nullptr);
  if (ret != flagcxSuccess) {
    return ret;
  }

  // DUSHMEM has no user-visible transport-context API, but FlagCX logical
  // contexts still require disjoint signal/counter/shadow namespaces. Pass
  // the context count selected by the common Device API layer to the SHMEM
  // adaptor so its metadata allocation matches the device-visible Net array.
  flagcxDevCommRequirements shmemReqs =
      FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  size_t copySize = reqsSize < sizeof(shmemReqs) ? reqsSize : sizeof(shmemReqs);
  memcpy(&shmemReqs, reqs, copySize);
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
  // All logical contexts share DUSHMEM's transport, while their metadata is
  // partitioned by context in CommTraits<DushmemBackend>::Net.
  // DUSHMEM doesn't need a host-side relay, but the World barrier uses
  // nInterPeers to decide whether to compose the inter-node barrier phase.
  int intraSize = shmemComm->intraSize;
  int interSize = (intraSize > 0 && shmemComm->nRanks % intraSize == 0)
                      ? (shmemComm->nRanks / intraSize)
                      : 1;
  devComm->nInterPeers = (interSize > 1) ? (interSize - 1) : 0;

  return flagcxSuccess;
}

static flagcxResult_t dushmemDevApiCommDestroy(flagcxComm_t comm,
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
static flagcxResult_t dushmemDevApiMemCreate(flagcxComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t win,
                                             flagcxDevMem_t devMem) {
  (void)comm;
  (void)win;
  if (!devMem->allocationTracked || devMem->allocator != flagcxMemSHMEM ||
      devMem->allocBackend != flagcxMemAllocBackendShmem) {
    WARN("dushmem Device API memory must be allocated with "
         "flagcxMemAlloc(..., flagcxMemSHMEM)");
    return flagcxInvalidUsage;
  }
  auto *w = new (std::nothrow) DushmemWindow();
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

static flagcxResult_t dushmemDevApiMemDestroy(flagcxComm_t comm,
                                              flagcxDevMem_t devMem) {
  (void)comm;
  if (devMem->window) {
    delete (DushmemWindow *)devMem->window;
    devMem->window = nullptr;
  }
  return flagcxSuccess;
}

// ==========================================================================
// Device pointer materialization
// ==========================================================================
static flagcxResult_t dushmemDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
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
  // Note: _gridBarrierState is intentionally left nullptr for DUSHMEM.
  // DUSHMEM barriers use dushmemx_barrier_block() + per-block arrive/release
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
  // flagcxDevNet is device-only (#ifdef FLAGCX_DEVICE_COMPILE), so we build
  // the equivalent bytes on the host. DUSHMEM flagcxDevNet layout:
  //   struct { Comm _dc; int _contextId; } (base: DeviceAPI::Net)
  //   int _nInterPeers;
  //   unsigned int* _gridBarrierState;
  if (hostCopy._contextCount > 0) {
    struct HostNet {
      DushmemComm _dc;
      int _contextId;
      int _nInterPeers;
      unsigned int *_gridBarrierState;
    };
    size_t netSize = flagcxDevNetSizeOf();
    if (netSize == 0)
      netSize = sizeof(HostNet);
    if (netSize != sizeof(HostNet)) {
      res = flagcxInternalError;
      goto fail;
    }
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

static flagcxResult_t dushmemDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
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

static flagcxResult_t dushmemDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
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
  FLAGCXCHECKGOTO(deviceAdaptor->deviceSynchronize(), res, fail);

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

static flagcxResult_t dushmemDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
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

static flagcxResult_t dushmemDevApiCommCleanup(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend dushmemBackend = {
    .name = "dushmem",
    .devCommCreate = dushmemDevApiCommCreate,
    .devCommDestroy = dushmemDevApiCommDestroy,
    .devMemCreate = dushmemDevApiMemCreate,
    .devMemDestroy = dushmemDevApiMemDestroy,
    .devCommGetDevicePtr = dushmemDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = dushmemDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = dushmemDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = dushmemDevApiMemFreeDevicePtr,
    .commQuiesce = dushmemDevApiCommCleanup,
    .commCleanup = dushmemDevApiCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &dushmemBackend;
