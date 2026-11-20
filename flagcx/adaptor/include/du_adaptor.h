#ifdef USE_DU_ADAPTOR

#include "flagcx.h"
#include <hip/hip_runtime.h>
#include <map>
#include <rccl/rccl.h>
struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  ncclComm_t base;
};

struct flagcxStream {
  hipStream_t base;
};

struct flagcxEvent {
  hipEvent_t base;
};

struct flagcxIpcMemHandle {
  hipIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  do {                                                                         \
    hipError_t ret = (func);                                                   \
    if (ret != hipSuccess)                                                     \
      return flagcxUnhandledDeviceError;                                       \
  } while (0)

#endif // USE_DU_ADAPTOR
