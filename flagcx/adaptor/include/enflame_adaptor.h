/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#ifdef USE_ENFLAME_ADAPTOR

#include "eccl.h"
#ifdef FLAGCX_ENFLAME_DEVICE_API
#include "eccl_device.h"
#endif
#include "flagcx.h"
#include "flagcx_device_api.h"
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

// Keep optional ECCL SDK declarations behind one compatibility boundary. The
// adaptor always exposes the same callbacks, including with runtime-only SDKs.
static inline bool flagcxEcclDeviceApiSupported() {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  return true;
#else
  return false;
#endif
}

static inline flagcxResult_t flagcxEcclMemAlloc(void **ptr, size_t size) {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  return (flagcxResult_t)ecclMemAlloc(ptr, size);
#else
  return (flagcxResult_t)topsMalloc(ptr, size);
#endif
}

static inline flagcxResult_t flagcxEcclMemFree(void *ptr) {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  return (flagcxResult_t)ecclMemFree(ptr);
#else
  return (flagcxResult_t)topsFree(ptr);
#endif
}

static inline flagcxResult_t
flagcxEcclWindowRegister(struct flagcxInnerComm *comm, void *buff, size_t size,
                         struct flagcxInnerWindow *win, int winFlags) {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  return (flagcxResult_t)ecclCommWindowRegister(comm->base, buff, size,
                                                &win->base, winFlags);
#else
  (void)comm;
  (void)buff;
  (void)size;
  (void)win;
  (void)winFlags;
  return flagcxNotSupported;
#endif
}

static inline flagcxResult_t
flagcxEcclWindowDeregister(struct flagcxInnerComm *comm,
                           struct flagcxInnerWindow *win) {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  return (flagcxResult_t)ecclCommWindowDeregister(comm->base, win->base);
#else
  (void)comm;
  (void)win;
  return flagcxNotSupported;
#endif
}

static inline flagcxResult_t
flagcxEcclDevCommReqsInit(struct flagcxInnerComm *comm,
                          flagcxDevCommRequirements *reqs,
                          int intraBarrierCount) {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  ecclCommProperties_t props = ECCL_COMM_PROPERTIES_INITIALIZER;
  flagcxResult_t result =
      (flagcxResult_t)ecclCommQueryProperties(comm->base, &props);
  if (result != flagcxSuccess)
    return result;
  if (!props.deviceApiSupport)
    return flagcxNotSupported;
  *reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs->intraBarrierCount = intraBarrierCount;
  reqs->intraMulticast = props.multimemSupport;
  return flagcxSuccess;
#else
  (void)comm;
  (void)reqs;
  (void)intraBarrierCount;
  return flagcxNotSupported;
#endif
}

static inline flagcxResult_t
flagcxEcclDevCommCreate(struct flagcxInnerComm *comm,
                        const flagcxDevCommRequirements *reqs,
                        struct flagcxInnerDevComm *devComm) {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  ecclDevCommRequirements_t nativeReqs = ECCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  nativeReqs.barrierCount = reqs->barrierCount;
  nativeReqs.lsaBarrierCount = reqs->intraBarrierCount;
  nativeReqs.lsaLLA2ABlockCount = reqs->intraLLA2ABlockCount;
  nativeReqs.lsaLLA2ASlotCount = reqs->intraLLA2ASlotCount;
  // Preserve the GIN resource counts consumed by the device backend. The
  // Enflame inter-node path still requires hardware validation.
  nativeReqs.ginContextCount = reqs->interContextCount;
  nativeReqs.railGinBarrierCount = reqs->interBarrierCount;
  nativeReqs.ginCounterCount = reqs->interCounterCount;
  nativeReqs.ginConnectionType = ECCL_GIN_CONNECTION_FULL;
  return (flagcxResult_t)ecclDevCommCreate(comm->base, &nativeReqs,
                                           &devComm->base);
#else
  (void)comm;
  (void)reqs;
  (void)devComm;
  return flagcxNotSupported;
#endif
}

static inline flagcxResult_t
flagcxEcclDevCommDestroy(struct flagcxInnerComm *comm,
                         struct flagcxInnerDevComm *devComm) {
#ifdef FLAGCX_ENFLAME_DEVICE_API
  return (flagcxResult_t)ecclDevCommDestroy(comm->base, &devComm->base);
#else
  (void)comm;
  (void)devComm;
  return flagcxNotSupported;
#endif
}

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != topsSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_ENFLAME_ADAPTOR
