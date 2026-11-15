/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host lifecycle hooks for the NVIDIA default GDA adapter.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_XPU_NETIO_HOST_H_
#define FLAGCX_NVIDIA_XPU_NETIO_HOST_H_

#include "flagcx_device_api.h"
#include "flagcx_device_internal.h"

namespace DefaultNvidiaGda {

flagcxResult_t createComm(flagcxComm_t comm, flagcxDevComm_t devComm,
                          flagcxDevNetGdaPolicy_t policy,
                          unsigned requiredOperations);
flagcxResult_t destroyComm(flagcxDevComm_t devComm);
flagcxResult_t createWindow(flagcxComm_t comm, flagcxDevComm_t devComm,
                            flagcxDevMem_t devMem, size_t bytes,
                            void *deviceWindow);
flagcxResult_t destroyWindow(flagcxDevMem_t devMem);

} // namespace DefaultNvidiaGda

#endif // FLAGCX_NVIDIA_XPU_NETIO_HOST_H_
