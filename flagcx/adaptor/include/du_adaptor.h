#ifdef USE_DU_ADAPTOR

#include "flagcx.h"
#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
struct flagcxInnerDevComm {};

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

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_DU_ADAPTOR