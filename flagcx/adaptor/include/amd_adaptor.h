#ifdef USE_AMD_ADAPTOR

#include "flagcx.h"
#include "rccl.h"
#include <hip/hip_runtime.h>
#include <map>
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
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  do {                                                                         \
    int ret = func;                                                            \
    if (ret != hipSuccess)                                                     \
      return flagcxUnhandledDeviceError;                                       \
  } while (0);

#define CCLCHECKGOTO(call, RES, label)                                         \
  do {                                                                         \
    RES = call;                                                                \
    if (RES != ncclSuccess && RES != ncclInProgress) {                         \
      /* Print the back trace*/                                                \
      if (flagcxDebugNoWarn == 0)                                              \
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#endif // USE_AMD_ADAPTOR
