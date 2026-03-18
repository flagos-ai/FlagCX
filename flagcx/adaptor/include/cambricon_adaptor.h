#ifdef USE_CAMBRICON_ADAPTOR
#ifndef SRC_ADAPTOR_API_MLU_ADAPTOR_H
#define SRC_ADAPTOR_API_MLU_ADAPTOR_H

#include "cncl.h"
#include "cnrt.h"
#include "flagcx.h"
#include <map>
struct flagcxInnerComm {
  cnclComm_t base;
};

struct flagcxStream {
  cnrtQueue_t base;
};

struct flagcxEvent {
  cnrtNotifier_t base;
};

struct flagcxIpcMemHandle {
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cnrtSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // SRC_ADAPTOR_API_MLU_ADAPTOR_H
#endif // USE_CAMBRICON_ADAPTOR
