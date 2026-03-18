/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * NVIDIA platform adaptor header for FlagCX CCL plugins.
 * Copied from flagcx/adaptor/include/nvidia_adaptor.h with the
 * USE_NVIDIA_ADAPTOR guard removed — plugin authors targeting NVIDIA
 * include this header directly.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_ADAPTOR_H_
#define FLAGCX_NVIDIA_ADAPTOR_H_

#include "flagcx.h"
#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"

struct stagedBuffer {
  void *buff;
  ncclWindow_t win;
};
typedef struct stagedBuffer *stagedBuffer_t;

#else

typedef void *stagedBuffer_t;
typedef void ncclDevComm;

#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)

struct flagcxInnerComm {
  ncclComm_t base;
  ncclDevComm *devBase;
  stagedBuffer_t sendStagedBuff;
  stagedBuffer_t recvStagedBuff;
};

struct flagcxStream {
  cudaStream_t base;
};

struct flagcxEvent {
  cudaEvent_t base;
};

struct flagcxIpcMemHandle {
  cudaIpcMemHandle_t base;
};

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
struct flagcxWindow {
  ncclWindow_t base;
  int winFlags;
};
#else
struct flagcxWindow {
  int winFlags;
};
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // FLAGCX_NVIDIA_ADAPTOR_H_
