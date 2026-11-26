#ifndef FLAGCX_UNI_RUNNER_STREAM_FLAGS_H_
#define FLAGCX_UNI_RUNNER_STREAM_FLAGS_H_

#include "flagcx_device_adaptor.h"
#include <cstdlib>
#include <limits>

// The caller drains all users before resizing, resetting, or releasing this
// pool. A failed rollback keeps its allocation for a later cleanup attempt.
struct flagcxUniRunnerStreamFlagPool {
  void *buffer;
  void *pendingFree;
  void **addresses;
  size_t capacity;
};

static inline flagcxResult_t
flagcxUniRunnerReleasePendingFlags(const flagcxDeviceAdaptor *adaptor,
                                   flagcxUniRunnerStreamFlagPool *pool) {
  if (pool->pendingFree == nullptr)
    return flagcxSuccess;
  flagcxResult_t result =
      adaptor->deviceFree(pool->pendingFree, flagcxMemDevice, nullptr);
  if (result == flagcxSuccess)
    pool->pendingFree = nullptr;
  return result;
}

static inline flagcxResult_t
flagcxUniRunnerReserveStreamFlags(const flagcxDeviceAdaptor *adaptor,
                                  flagcxUniRunnerStreamFlagPool *pool,
                                  size_t required) {
  flagcxResult_t result = flagcxUniRunnerReleasePendingFlags(adaptor, pool);
  if (result != flagcxSuccess || required <= pool->capacity)
    return result;
  const size_t maxSize = std::numeric_limits<size_t>::max();
  if (required > maxSize / sizeof(uint64_t) ||
      required > maxSize / sizeof(void *))
    return flagcxInvalidArgument;
  size_t capacity = pool->capacity == 0 ? 1 : pool->capacity;
  while (capacity < required) {
    if (capacity > maxSize / 2 || capacity * 2 > maxSize / sizeof(uint64_t) ||
        capacity * 2 > maxSize / sizeof(void *)) {
      capacity = required;
      break;
    }
    capacity *= 2;
  }
  void **addresses = static_cast<void **>(calloc(capacity, sizeof(void *)));
  if (addresses == nullptr)
    return flagcxSystemError;
  void *buffer = nullptr;
  result = flagcxDeviceAllocWithFlags(
      adaptor, &buffer, capacity * sizeof(uint64_t), flagcxMemDevice, nullptr,
      FLAGCX_DEVICE_ALLOC_UNIRUNNER);
  if (result == flagcxSuccess && buffer == nullptr)
    result = flagcxSystemError;
  if (result == flagcxSuccess)
    result = adaptor->deviceMemset(buffer, 0, capacity * sizeof(uint64_t),
                                   flagcxMemDevice, nullptr);
  if (result == flagcxSuccess && pool->buffer != nullptr)
    result = adaptor->deviceFree(pool->buffer, flagcxMemDevice, nullptr);
  if (result != flagcxSuccess) {
    free(addresses);
    pool->pendingFree = buffer;
    // Preserve the original error; a failed cleanup retains ownership.
    (void)flagcxUniRunnerReleasePendingFlags(adaptor, pool);
    return result;
  }
  for (size_t i = 0; i < capacity; ++i)
    addresses[i] = static_cast<char *>(buffer) + i * sizeof(uint64_t);
  free(pool->addresses);
  pool->addresses = addresses;
  pool->buffer = buffer;
  pool->capacity = capacity;
  return flagcxSuccess;
}

static inline flagcxResult_t
flagcxUniRunnerReleaseStreamFlags(const flagcxDeviceAdaptor *adaptor,
                                  flagcxUniRunnerStreamFlagPool *pool) {
  flagcxResult_t result = flagcxUniRunnerReleasePendingFlags(adaptor, pool);
  if (result != flagcxSuccess)
    return result;
  if (pool->buffer != nullptr) {
    result = adaptor->deviceFree(pool->buffer, flagcxMemDevice, nullptr);
    if (result != flagcxSuccess)
      return result;
    pool->buffer = nullptr;
  }
  free(pool->addresses);
  pool->addresses = nullptr;
  pool->capacity = 0;
  return flagcxSuccess;
}

#endif
