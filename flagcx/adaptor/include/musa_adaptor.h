#ifdef USE_MUSA_ADAPTOR

#include "flagcx.h"
#include "mccl.h"
#include <map>
#include <musa.h>
#include <musa_runtime.h>
struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  mcclComm_t base;
};

struct flagcxStream {
  musaStream_t base;
};

struct flagcxEvent {
  musaEvent_t base;
};

struct flagcxIpcMemHandle {
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != musaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_MUSA_ADAPTOR