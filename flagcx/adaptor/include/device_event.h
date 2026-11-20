#ifndef FLAGCX_DEVICE_EVENT_H_
#define FLAGCX_DEVICE_EVENT_H_

#include "flagcx.h"
#include <stdlib.h>

// A failed native creation owns no valid native event. Keep its unpublished
// host wrapper separate from the normal eventDestroy path.
struct flagcxEventCreateOps {
  void *(*allocate)(size_t bytes);
  void (*release)(void *wrapper);
  flagcxResult_t (*nativeCreate)(void *wrapper, unsigned int flags);
};

static inline void *flagcxEventAllocateWrapper(size_t bytes) {
  return calloc(1, bytes);
}

static inline flagcxResult_t
flagcxEventCreateTransactional(void **output, size_t bytes, unsigned int flags,
                               const struct flagcxEventCreateOps *ops) {
  if (output == NULL)
    return flagcxInvalidArgument;
  *output = NULL;
  if (ops == NULL || ops->allocate == NULL || ops->release == NULL ||
      ops->nativeCreate == NULL || bytes == 0)
    return flagcxInvalidArgument;
  void *wrapper = ops->allocate(bytes);
  if (wrapper == NULL)
    return flagcxSystemError;
  flagcxResult_t result = ops->nativeCreate(wrapper, flags);
  if (result != flagcxSuccess) {
    ops->release(wrapper);
    return result;
  }
  *output = wrapper;
  return flagcxSuccess;
}

#endif
