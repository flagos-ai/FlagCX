/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NCCL (vendor) Device API backend for flagcxDevComm lifecycle.
 * Linked when FLAGCX_COMM_TRAITS_CCL is defined (NCCL >= 2.29).
 ************************************************************************/

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"

static flagcxResult_t
ncclDevApiCommCreate(flagcxComm_t comm,
                     const struct flagcxDevCommRequirements *reqs,
                     flagcxDevComm_t devComm) {
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
  return flagcxSuccess;
}

static flagcxResult_t ncclDevApiCommDestroy(flagcxComm_t comm,
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

static flagcxResult_t ncclDevApiMemCreate(flagcxComm_t comm, void *buff,
                                          size_t size, flagcxWindow_t win,
                                          flagcxDevMem_t devMem) {
  (void)buff;
  (void)size;

  // On Vendor path, we only need the ncclWindow_t from the vendor window.
  // No IPC peer pointers needed — NCCL GIN handles all transport.
  if (comm != nullptr) {
    devMem->intraRank = comm->localRank;
  }

  if (win != nullptr && !win->isSymmetricDefault && win->vendorBase) {
    devMem->hasWindow = true;
    devMem->isSymmetric = (win->winFlags & FLAGCX_WIN_COLL_SYMMETRIC) != 0;
    devMem->winHandle = (void *)win;
  }

  // Allocate and populate kernel Window (wraps ncclWindow_t)
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

static flagcxResult_t ncclDevApiMemDestroy(flagcxComm_t comm,
                                           flagcxDevMem_t devMem) {
  (void)comm;
  if (devMem != nullptr && devMem->window != nullptr) {
    delete static_cast<typename DeviceAPI::Window *>(devMem->window);
    devMem->window = nullptr;
  }
  return flagcxSuccess;
}

static flagcxResult_t ncclDevApiCommGetDevicePtr(flagcxDevComm_t devComm,
                                                 void **devPtr) {
  (void)devComm;
  (void)devPtr;
  return flagcxNotSupported;
}

static flagcxResult_t ncclDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  (void)devComm;
  return flagcxNotSupported;
}

static flagcxResult_t ncclDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
                                                void **devPtr) {
  (void)devMem;
  (void)devPtr;
  return flagcxNotSupported;
}

static flagcxResult_t ncclDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
  (void)devMem;
  return flagcxNotSupported;
}

static flagcxResult_t ncclCommCleanup(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

static struct flagcxDevApiBackend ncclBackend = {
    .name = "nccl",
    .devCommCreate = ncclDevApiCommCreate,
    .devCommDestroy = ncclDevApiCommDestroy,
    .devMemCreate = ncclDevApiMemCreate,
    .devMemDestroy = ncclDevApiMemDestroy,
    .devCommGetDevicePtr = ncclDevApiCommGetDevicePtr,
    .devCommFreeDevicePtr = ncclDevApiCommFreeDevicePtr,
    .devMemGetDevicePtr = ncclDevApiMemGetDevicePtr,
    .devMemFreeDevicePtr = ncclDevApiMemFreeDevicePtr,
    .commCleanup = ncclCommCleanup,
};

struct flagcxDevApiBackend *devApiBackend = &ncclBackend;
