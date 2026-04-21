#ifdef USE_SUNRISE_ADAPTOR
#include "tang.h"
#include "tang_runtime.h"
#include "pccl.h"
#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "flagcx.h"
#include <map>
struct flagcxInnerComm {
  pcclComm_t base;
};
struct flagcxStream {
  tangStream_t base;
};

struct flagcxEvent {
  tangEvent_t base;
};

struct flagcxIpcMemHandle {
  tangIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    tangError_t ret = func;                                                    \
    if (ret != tangSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_SUNRISE_ADAPTOR
