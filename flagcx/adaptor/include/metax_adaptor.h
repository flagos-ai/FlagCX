/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#ifdef USE_METAX_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "flagcx.h"
#include "mccl.h"
#include <map>
#include <mcr/mc_runtime.h>
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
