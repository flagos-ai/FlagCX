/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Build-selected GPU runtime binding for the XPU NetIO host implementation.
 ************************************************************************/

#ifndef FLAGCX_XPU_NETIO_GDA_RUNTIME_H_
#define FLAGCX_XPU_NETIO_GDA_RUNTIME_H_

#include <xpu_netio/provider.h>
#include <xpu_netio/runtime.h>

struct XpuNetioGdaRuntime {
  const xpu_netio_runtime_ops_v1 *ops;
  xpu_netio_memory_type_t memoryType;
  uint64_t memoryCapability;
  uint32_t platformId;
};

// Exactly one platform binding is linked with the XPU NetIO GDA host code.
XpuNetioGdaRuntime flagcxXpuNetioGdaRuntime();

#endif // FLAGCX_XPU_NETIO_GDA_RUNTIME_H_
