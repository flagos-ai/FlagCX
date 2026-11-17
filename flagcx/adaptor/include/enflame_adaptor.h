/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#ifdef USE_ENFLAME_ADAPTOR

#include "eccl.h"
#include "flagcx.h"
#include <map>
#include <tops/tops_runtime.h>

// The device-properties struct is `topsDeviceProp_t` on the 1.9.x SDK and
// `topsDeviceProp` from 4.x on, with no compat alias on either side — so a
// build has to pick one. TOPSRT_VERSION exists only on 4.x (and is 4010 there,
// 1.9.29 does not define it at all), which makes it the discriminator; the
// spelling itself is not a version number, so deriving it from the value would
// age badly.
#if defined(TOPSRT_VERSION)
typedef struct topsDeviceProp topsDevicePropCompat_t;
#else
typedef topsDeviceProp_t topsDevicePropCompat_t;
#endif

struct flagcxInnerDevComm {};

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
