/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#ifdef USE_ENFLAME_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "eccl.h"
#include "flagcx.h"
#include <map>
#include <tops/tops_runtime_api.h>

struct flagcxInnerComm {
  ecclComm_t base;
};

struct flagcxStream {
  topsStream_t base;
};

struct flagcxEvent {
  topsEvent_t base;
};

struct flagcxIpcMemHandle {
  topsIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != topsSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_ENFLAME_ADAPTOR
