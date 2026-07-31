/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Unified host-side backend interface for DeviceAPI comm/mem lifecycle.
 * Link-time selection: only one backend .cc is linked per build.
 ************************************************************************/

#ifndef FLAGCX_DEV_API_BACKEND_H_
#define FLAGCX_DEV_API_BACKEND_H_

#include "device_api/flagcx_device.h"

struct flagcxDevApiBackend {
  const char *name;

  // DevComm lifecycle
  flagcxResult_t (*devCommCreate)(flagcxComm_t comm,
                                  const struct flagcxDevCommRequirements *reqs,
                                  flagcxDevComm_t devComm);
  flagcxResult_t (*devCommDestroy)(flagcxComm_t comm, flagcxDevComm_t devComm);

  // DevMem lifecycle
  flagcxResult_t (*devMemCreate)(flagcxComm_t comm, void *buff, size_t size,
                                 flagcxWindow_t win, flagcxDevMem_t devMem);
  flagcxResult_t (*devMemDestroy)(flagcxComm_t comm, flagcxDevMem_t devMem);

  // Device pointer materialization (Triton)
  flagcxResult_t (*devCommGetDevicePtr)(flagcxDevComm_t devComm, void **devPtr);
  flagcxResult_t (*devCommFreeDevicePtr)(flagcxDevComm_t devComm);
  flagcxResult_t (*devMemGetDevicePtr)(flagcxDevMem_t devMem, void **devPtr);
  flagcxResult_t (*devMemFreeDevicePtr)(flagcxDevMem_t devMem);

  // Comm-level cleanup (called once from flagcxCommDestroy in flagcx.cc)
  flagcxResult_t (*commCleanup)(flagcxComm_t comm);
};

extern struct flagcxDevApiBackend *devApiBackend;

#endif // FLAGCX_DEV_API_BACKEND_H_
