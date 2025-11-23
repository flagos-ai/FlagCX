#ifdef USE_TSM_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "flagcx.h"
#include "tx_runtime.h"
#include "tccl.h"

#include <map>
struct flagcxInnerComm {
  tcclComm_t base;
};

struct flagcxStream {
  txStream_t base;
};

struct flagcxEvent {
  txEvent_t base;
};

struct flagcxIpcMemHandle {
  txIpcMemHandle_t base;
};
                                   
#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != TX_SUCCESS)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_TSM_ADAPTOR