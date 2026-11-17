/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#ifdef USE_ENFLAME_ADAPTOR

#include "eccl.h"
#ifdef FLAGCX_ENFLAME_DEVICE_API
#include "eccl_device.h"
#endif
#include "flagcx.h"
#include <map>
#include <tops/tops_runtime.h>

struct flagcxInnerDevComm {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  ecclDevComm_t base;
#endif
};

struct flagcxInnerWindow {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  ecclWindow_t base;
#else
  void *base;
#endif
  int winFlags;
};

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
