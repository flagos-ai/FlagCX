#ifdef USE_NVIDIA_ADAPTOR

#include "flagcx.h"
#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"

#define NCCL_ADAPTOR_DEVICE_CTA_COUNT 36
#define NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA 512

struct flagcxInnerDevComm {
  ncclDevComm base;
};

#else

typedef void ncclDevComm;
struct flagcxInnerDevComm {};

#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)

struct flagcxInnerComm {
  ncclComm_t base;
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

#endif // USE_NVIDIA_ADAPTOR