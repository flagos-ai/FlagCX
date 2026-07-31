/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM Device API backend for flagcxDevComm lifecycle.
 * Linked when USE_SHMEM=1.
 ************************************************************************/

#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "device_api/nvshmem_comm_traits.h"
#include "nvshmem_adaptor.h"
#include "shmem_adaptor.h"

#include <cstddef>
#include <cstdio>
#include <new>

// Verify that flagcxShmemCommInternal and CommTraits<NvshmemBackend>::Comm have
// compatible layout — the constructor in flagcx_device_core.h does a raw cast.
static_assert(
    sizeof(flagcxShmemCommInternal) == sizeof(CommTraits<NvshmemBackend>::Comm),
    "ShmemCommInternal and CommTraits<NvshmemBackend>::Comm size mismatch");
static_assert(
    offsetof(flagcxShmemCommInternal, worldBarrierCount) ==
        offsetof(CommTraits<NvshmemBackend>::Comm, worldBarrierCount),
    "ShmemCommInternal and CommTraits::Comm last-field offset mismatch");

static flagcxResult_t
nvshmemDevApiCommCreate(flagcxComm_t comm,
                        const struct flagcxDevCommRequirements *reqs,
                        flagcxDevComm_t devComm) {
  if (shmemAdaptor == nullptr) {
    return flagcxInternalError;
  }

  // Initialize NVSHMEM (reference-counted, safe to call multiple times)
  flagcxResult_t ret = shmemAdaptor->init(comm->rank, comm->nranks);
  if (ret != flagcxSuccess) {
    return ret;
  }

  flagcxShmemComm_t shmemComm = nullptr;
  ret = shmemAdaptor->devCommCreate(comm, reqs, &shmemComm);
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
  // NVSHMEM does not use FIFO contexts — leave contextCount at 0.
  devComm->contextCount = 0;
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
  // NVSHMEM kernels use __constant__ symbol path, not DevicePtr.
  // Return host handle for API compatibility (lifecycle tests, Triton stubs).
  WARN("nvshmem: DevCommGetDevicePtr returns host handle; "
       "NVSHMEM kernels use __constant__ symbol path");
  *devPtr = (void *)devComm;
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiCommFreeDevicePtr(flagcxDevComm_t devComm) {
  (void)devComm;
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemGetDevicePtr(flagcxDevMem_t devMem,
                                                   void **devPtr) {
  if (!devMem || !devPtr)
    return flagcxInvalidArgument;
  WARN("nvshmem: DevMemGetDevicePtr returns host handle; "
       "NVSHMEM kernels use __constant__ symbol path");
  *devPtr = (void *)devMem;
  return flagcxSuccess;
}

static flagcxResult_t nvshmemDevApiMemFreeDevicePtr(flagcxDevMem_t devMem) {
  (void)devMem;
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
