/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * ECCL (vendor) Device API backend for flagcxDevComm / flagcxDevMem lifecycle.
 * Linked when FLAGCX_COMM_TRAITS_CCL is defined on Enflame.
 *
 * Structure mirrors nccl_dev_api_backend.cc. The one thing worth spelling out
 * is how the opaque vendor handles reach the device:
 *   - Comm: flagcxInnerDevComm holds ecclDevComm_t by value and is
 *     layout-compatible with CommTraits<EcclBackend>::Comm, so the by-value
 *     copy performed here is exactly what the device pass reinterprets.
 *   - Window: flagcxInnerWindow holds the registration's ecclWindow_t, and
 *     CommTraits<EcclBackend>::Window::populateFromHost() pulls it back out.
 ************************************************************************/

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"

#include <cstddef>

#ifdef COMPILE_KERNEL_HOST
extern "C" size_t flagcxDevNetSizeOf();
extern "C" void flagcxDevNetLaunchConstruct(void *devNets, void *devComm,
                                            int count, void *stream);
#else
static size_t flagcxDevNetSizeOf() { return 0; }
static void flagcxDevNetLaunchConstruct(void *, void *, int, void *) {}
#endif

static flagcxResult_t
ecclDevApiCommCreate(flagcxComm_t comm,
                     const struct flagcxDevCommRequirements *reqs,
                     size_t reqsSize, flagcxDevComm_t devComm) {
  // The CCL adaptor consumes only the frozen requirements prefix.
  (void)reqsSize;
  flagcxInnerComm_t innerComm = comm->homoComm;
  if (innerComm == nullptr ||
      cclAdaptors[flagcxCCLAdaptorDevice]->devCommCreate == NULL) {
    return flagcxInternalError;
  }

  flagcxInnerDevComm_t innerDevComm = nullptr;
  flagcxResult_t ret = cclAdaptors[flagcxCCLAdaptorDevice]->devCommCreate(
      innerComm, reqs, &innerDevComm);
  if (ret != flagcxSuccess) {
    return ret;
  }

  devComm->devComm = innerDevComm;

  // Inter-node peer info, derived from the uniform node layout
  // (nNodes = nRanks / localRanks). Kernels use it to decide whether a peer is
  // reachable through LSA or has to go over GIN.
  int nRanks = comm->nranks;
  int localRanks = comm->localRanks;

  if (localRanks > 0 && nRanks > localRanks) {
    int nNodes = nRanks / localRanks;
    devComm->nInterPeers = nNodes - 1;
    devComm->teamRank = comm->rank / localRanks;
    devComm->nTeamRanks = nNodes;
  }

  return flagcxSuccess;
}

static flagcxResult_t ecclDevApiCommDestroy(flagcxComm_t comm,
                                            flagcxDevComm_t devComm) {
  if (comm != nullptr && devComm->devComm != nullptr) {
    flagcxInnerComm_t innerComm = comm->homoComm;
    if (innerComm != nullptr &&
        cclAdaptors[flagcxCCLAdaptorDevice]->devCommDestroy != NULL) {
      cclAdaptors[flagcxCCLAdaptorDevice]->devCommDestroy(innerComm,
                                                          devComm->devComm);
      devComm->devComm = nullptr;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t ecclDevApiMemCreate(flagcxComm_t comm, void *buff,
                                          size_t size, flagcxWindow_t win,
                                          flagcxDevMem_t devMem) {
  (void)buff;
  (void)size;

  if (comm != nullptr) {
    devMem->intraRank = comm->localRank;
  }

  if (win != nullptr && win->vendorBase) {
    devMem->hasWindow = true;
    devMem->isSymmetric = (win->winFlags & FLAGCX_WIN_COLL_SYMMETRIC) != 0;
    devMem->winHandle = (void *)win;
  }

  auto *kWin = new (std::nothrow) typename DeviceAPI::Window{};
  if (kWin == nullptr) {
    return flagcxSystemError;
  }
  kWin->populateFromHost(win, devMem->rawPtr, devMem->intraRank,
                         devMem->mrIndex, devMem->mrBase, devMem->ipcIndex,
                         nullptr);
  devMem->window = kWin;
  devMem->hasWindow = kWin->hasAccess();

  return flagcxSuccess;
}

static flagcxResult_t ecclDevApiMemDestroy(flagcxComm_t comm,
                                           flagcxDevMem_t devMem) {
  (void)comm;
  if (devMem != nullptr && devMem->window != nullptr) {
    delete static_cast<typename DeviceAPI::Window *>(devMem->window);
    devMem->window = nullptr;
  }
  return flagcxSuccess;
}

static flagcxResult_t ecclDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
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

  FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(&dPtr, sizeof(flagcxDevComm),
                                              flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(
      deviceAdaptor->deviceMemcpy(dPtr, &hostCopy, sizeof(flagcxDevComm),
                                  flagcxMemcpyHostToDevice, NULL, NULL),
      res, fail);

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

static flagcxResult_t ecclDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  if (!devComm)
    return flagcxSuccess;

  pthread_mutex_lock(&devComm->cachedPtrMutex);
  if (devComm->cachedGridBarrierPtr) {
    deviceAdaptor->deviceFree(devComm->cachedGridBarrierPtr, flagcxMemDevice,
                              NULL);
    devComm->cachedGridBarrierPtr = nullptr;
  }
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

static flagcxResult_t ecclDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
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

static flagcxResult_t ecclDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
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

static flagcxResult_t ecclCommCleanup(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend ecclBackend = {
    .name = "eccl",
    .devCommCreate = ecclDevApiCommCreate,
    .devCommDestroy = ecclDevApiCommDestroy,
    .devMemCreate = ecclDevApiMemCreate,
    .devMemDestroy = ecclDevApiMemDestroy,
    .devCommGetDevicePtr = ecclDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = ecclDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = ecclDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = ecclDevApiMemFreeDevicePtr,
    .commCleanup = ecclCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &ecclBackend;
