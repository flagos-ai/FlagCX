#ifndef FLAGCX_PPU_ADAPTOR_H_
#define FLAGCX_PPU_ADAPTOR_H_

#ifdef USE_PPU_ADAPTOR

#include "flagcx.h"
#include <hggc.h>
#include <hggc_runtime.h>
#include <pccl.h>

struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  pcclComm_t base;
};

struct flagcxStream {
  hggcStream_t base;
};

static_assert(offsetof(flagcxStream, base) == 0 &&
                  sizeof(flagcxStream) == sizeof(hggcStream_t),
              "FlagCX stream wrappers must expose the native handle first");

struct flagcxEvent {
  hggcEvent_t base;
};

struct flagcxIpcMemHandle {
  hggcIpcMemHandle_t base;
};

inline flagcxResult_t hggcResultToFlagcx(hggcError_t result) {
  if (result == hggcSuccess)
    return flagcxSuccess;
  if (result == hggcErrorNotReady)
    return flagcxInProgress;
  return flagcxUnhandledDeviceError;
}

inline flagcxResult_t hggcDriverResultToFlagcx(HGresult result) {
  if (result == HGGC_SUCCESS)
    return flagcxSuccess;
  if (result == HGGC_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}

#define HGGCCHECK(func)                                                        \
  do {                                                                         \
    const flagcxResult_t hggcResult_ = hggcResultToFlagcx(func);               \
    if (hggcResult_ != flagcxSuccess)                                          \
      return hggcResult_;                                                      \
  } while (0)

#define HGDRVCHECK(func)                                                       \
  do {                                                                         \
    const flagcxResult_t hggcResult_ = hggcDriverResultToFlagcx(func);         \
    if (hggcResult_ != flagcxSuccess)                                          \
      return hggcResult_;                                                      \
  } while (0)

#define DEVCHECK(func) HGGCCHECK(func)

#endif // USE_PPU_ADAPTOR
#endif // FLAGCX_PPU_ADAPTOR_H_
