/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVIDIA GPU runtime binding for the platform-neutral XPU NetIO host owner.
 ************************************************************************/

#include "device_api/devnet/xpu_netio_gda_runtime.h"

XpuNetioGdaRuntime flagcxXpuNetioGdaRuntime() {
  return {xpu_netio_cuda_runtime(), XPU_NETIO_MEMORY_TYPE_CUDA,
          XPU_NETIO_MEMORY_CUDA, XPU_NETIO_PLATFORM_NVIDIA};
}
