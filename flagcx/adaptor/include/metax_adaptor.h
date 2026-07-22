/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#ifdef USE_METAX_ADAPTOR

#include "flagcx.h"
#include "mccl.h"
#include <map>
#include <mcr/mc_runtime.h>
#if MCCL_VERSION_CODE >= MCCL_VERSION(2, 30, 4)
#include "mccl/mccl_device.h"

struct flagcxInnerDevComm {
  mcclDevComm base;
};
struct flagcxInnerWindow {
  mcclWindow_t base;
  int winFlags;
};
#else
struct flagcxInnerDevComm {};
#endif

struct flagcxInnerComm {
  mcclComm_t base;
};

struct flagcxStream {
  mcStream_t base;
};

struct flagcxEvent {
  mcEvent_t base;
};

struct flagcxIpcMemHandle {
  mcIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != mcSuccess)                                                      \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_METAX_ADAPTOR
