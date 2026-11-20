#ifndef FLAGCX_RUNTIME_CAPTURE_H_
#define FLAGCX_RUNTIME_CAPTURE_H_

#include "flagcx_device_adaptor.h"
#include <dlfcn.h>

// Runtime APIs are resolved separately from the SDK headers so old runtimes
// without graph support can still execute ordinary stream operations. Each
// backend supplies its own capture entry-point names and unwraps its stream.
typedef int (*flagcxRuntimeCaptureQuery_t)(void *, int *);

static inline flagcxResult_t
flagcxRuntimeDecodeCaptureStatus(int nativeResult, int nativeStatus,
                                 flagcxStreamCaptureStatus_t *status) {
  if (status == NULL)
    return flagcxInvalidArgument;
  *status = flagcxStreamCaptureInvalidated;
  if (nativeResult != 0)
    return flagcxUnhandledDeviceError;
  if (nativeStatus < 0 || nativeStatus > 2)
    return flagcxInternalError;
  *status = (flagcxStreamCaptureStatus_t)nativeStatus;
  return flagcxSuccess;
}

static inline flagcxResult_t
flagcxRuntimeQueryCapture(flagcxRuntimeCaptureQuery_t query,
                          const void *beginCapture, void *stream,
                          flagcxStreamCaptureStatus_t *status) {
  if (status == NULL)
    return flagcxInvalidArgument;
  *status = flagcxStreamCaptureInvalidated;
  if (query == NULL) {
    // A runtime with no capture entry points cannot have an active capture.
    // If it exposes capture but not introspection, fail before submitting.
    if (beginCapture != NULL)
      return flagcxNotSupported;
    *status = flagcxStreamCaptureNone;
    return flagcxSuccess;
  }
  int nativeStatus = -1;
  int result = query(stream, &nativeStatus);
  return flagcxRuntimeDecodeCaptureStatus(result, nativeStatus, status);
}

#endif
