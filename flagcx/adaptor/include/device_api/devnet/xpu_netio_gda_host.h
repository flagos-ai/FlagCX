/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host lifecycle and diagnostics for the XPU NetIO GDA adapter.
 ************************************************************************/

#ifndef FLAGCX_XPU_NETIO_GDA_HOST_H_
#define FLAGCX_XPU_NETIO_GDA_HOST_H_

#include "device_api/flagcx_device_internal.h"
#include "flagcx_device_api.h"

namespace XpuNetioGda {

flagcxResult_t createComm(flagcxComm_t comm, flagcxDevComm_t devComm,
                          const flagcxDevCommRequirements *reqs, int netType);
flagcxResult_t destroyComm(flagcxDevComm_t devComm);
flagcxResult_t registerWindow(flagcxComm_t comm, flagcxDevComm_t devComm,
                              flagcxWindow_t window);
flagcxResult_t prepareWindowDeregister(flagcxWindow_t window);
flagcxResult_t deregisterWindow(flagcxWindow_t window);
flagcxResult_t bindDevMem(flagcxWindow_t window, flagcxDevMem_t devMem,
                          size_t bytes, void *deviceWindow);
flagcxResult_t unbindDevMem(flagcxDevMem_t devMem);
flagcxResult_t queryComm(flagcxDevComm_t devComm, unsigned *capabilities,
                         uint32_t **deviceErrorFlag, uint64_t *maxChunkBytes,
                         unsigned *requiredAlignment);
flagcxResult_t clearError(flagcxDevComm_t devComm);
flagcxResult_t pollError(flagcxDevComm_t devComm);

} // namespace XpuNetioGda

#endif // FLAGCX_XPU_NETIO_GDA_HOST_H_
