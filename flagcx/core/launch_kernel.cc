#include "launch_kernel.h"
#include "group.h"
#include <climits>
#include <map>
#include <stdio.h>

flagcxLaunchFunc_t deviceAsyncKernel = NULL;
flagcxDeviceSemaphoreBufferPool deviceSemaphoreBufferPool;

FLAGCX_PARAM(SemaphoreBufferPoolCapacity, "SEMAPHORE_BUFFER_POOL_CAPACITY", 32);

flagcxDeviceSemaphoreBufferPool::flagcxDeviceSemaphoreBufferPool()
    : capacity(0), chunks(NULL), pending(NULL) {}

flagcxDeviceSemaphoreBufferPool::~flagcxDeviceSemaphoreBufferPool() {
  (void)clear();
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::getSlotId(int *id) {
  if (id == NULL)
    return flagcxInvalidArgument;
  flagcxResult_t result = flagcxSuccess;
  *id = -1;
  PTHREADCHECKGOTO(pthread_mutex_lock(&mutex), "pthread_mutex_lock", result,
                   exit);
  for (;;) {
    for (Chunk *chunk = chunks; chunk != NULL; chunk = chunk->next) {
      for (int i = 0; i < chunk->capacity; ++i) {
        if (chunk->inUse[i])
          continue;
        chunk->inUse[i] = 1;
        memset(chunk->signals + (size_t)FLAGCX_SIGNALS_PER_SEMAPHORE * i, 0,
               FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int));
        *id = chunk->first + i;
        goto unlock;
      }
    }
    FLAGCXCHECKGOTO(grow(), result, unlock);
    if (result != flagcxSuccess)
      goto unlock;
  }
unlock:
  pthread_mutex_unlock(&mutex);
exit:
  return result;
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::grow() {
  if (pending == NULL) {
    int count = flagcxParamSemaphoreBufferPoolCapacity();
    if (count <= 0 || count > INT_MAX - capacity ||
        (size_t)count > SIZE_MAX / FLAGCX_SIGNALS_PER_SEMAPHORE / sizeof(int))
      return flagcxInvalidArgument;
    pending = (Chunk *)calloc(1, sizeof(Chunk));
    if (pending == NULL)
      return flagcxSystemError;
    pending->capacity = count;
    pending->first = capacity;
  }
  if (pending->inUse == NULL) {
    pending->inUse = (int *)calloc(pending->capacity, sizeof(int));
    if (pending->inUse == NULL)
      return flagcxSystemError;
  }
  if (!pending->allocated) {
    // Some adaptors return an owner even when allocation fails. Retain it
    // until cleanup succeeds; never publish it as a usable signal buffer.
    if (pending->signals != NULL) {
      flagcxResult_t result =
          deviceAdaptor->deviceFree(pending->signals, flagcxMemHost, NULL);
      if (result != flagcxSuccess)
        return result;
      pending->signals = NULL;
    }
    flagcxResult_t result = deviceAdaptor->deviceMalloc(
        (void **)&pending->signals,
        (size_t)pending->capacity * FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int),
        flagcxMemHost, NULL);
    if (result != flagcxSuccess)
      return result;
    if (pending->signals == NULL)
      return flagcxSystemError;
    pending->allocated = 1;
  }
  flagcxResult_t result =
      deviceAdaptor->hostGetDevicePointer(&pending->dSignals, pending->signals);
  if (result != flagcxSuccess)
    return result;
  if (pending->dSignals == NULL)
    return flagcxSystemError;
  capacity += pending->capacity;
  pending->next = chunks;
  chunks = pending;
  pending = NULL;
  return flagcxSuccess;
}

flagcxDeviceSemaphoreBufferPool::Chunk *
flagcxDeviceSemaphoreBufferPool::findChunk(int id) {
  for (Chunk *chunk = chunks; chunk != NULL; chunk = chunk->next)
    if (id >= chunk->first && id - chunk->first < chunk->capacity)
      return chunk;
  assert(false);
  return NULL;
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::releaseSlot(int id) {
  flagcxResult_t result = flagcxSuccess;
  PTHREADCHECKGOTO(pthread_mutex_lock(&mutex), "pthread_mutex_lock", result,
                   exit);
  {
    Chunk *chunk = findChunk(id);
    if (chunk == NULL || !chunk->inUse[id - chunk->first])
      result = flagcxInvalidUsage;
    else
      chunk->inUse[id - chunk->first] = 0;
  }
  pthread_mutex_unlock(&mutex);
exit:
  return result;
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::clear() {
  flagcxResult_t result = flagcxSuccess;
  PTHREADCHECKGOTO(pthread_mutex_lock(&mutex), "pthread_mutex_lock", result,
                   exit);
  for (Chunk *chunk = chunks; chunk != NULL; chunk = chunk->next)
    for (int i = 0; i < chunk->capacity; ++i)
      if (chunk->inUse[i]) {
        result = flagcxInProgress;
        goto unlock;
      }
  {
    Chunk **lists[] = {&pending, &chunks};
    for (unsigned int i = 0; i < 2; ++i) {
      Chunk **list = lists[i];
      while (*list != NULL) {
        Chunk *chunk = *list;
        if (chunk->signals != NULL) {
          FLAGCXCHECKGOTO(
              deviceAdaptor->deviceFree(chunk->signals, flagcxMemHost, NULL),
              result, unlock);
          if (result != flagcxSuccess)
            goto unlock;
        }
        *list = chunk->next;
        free(chunk->inUse);
        free(chunk);
      }
    }
  }
  capacity = 0;
unlock:
  pthread_mutex_unlock(&mutex);
exit:
  return result;
}

int *flagcxDeviceSemaphoreBufferPool::getHostPtr(int id) {
  pthread_mutex_lock(&mutex);
  Chunk *chunk = findChunk(id);
  int *pointer = chunk->signals +
                 (size_t)FLAGCX_SIGNALS_PER_SEMAPHORE * (id - chunk->first);
  pthread_mutex_unlock(&mutex);
  return pointer;
}
void *flagcxDeviceSemaphoreBufferPool::getDevicePtr(int id) {
  pthread_mutex_lock(&mutex);
  Chunk *chunk = findChunk(id);
  void *pointer =
      (char *)chunk->dSignals +
      (size_t)FLAGCX_SIGNALS_PER_SEMAPHORE * (id - chunk->first) * sizeof(int);
  pthread_mutex_unlock(&mutex);
  return pointer;
}

void cpuAsyncKernel(void *args) {
  struct flagcxGroupCallback *context = (struct flagcxGroupCallback *)args;
  struct flagcxHostSemaphore *semaphore = context->semaphore;
  semaphore->runCallback();
  __atomic_store_n(&context->finished, 1, __ATOMIC_RELEASE);
  // Last object access: proxy owners cannot retire before this reference drops.
  __atomic_fetch_sub(&semaphore->callbacks, 1, __ATOMIC_RELEASE);
}

static pthread_mutex_t semaphoreOwnerMutex = PTHREAD_MUTEX_INITIALIZER;
flagcxResult_t flagcxReleaseSemaphore(std::shared_ptr<flagcxSemaphore> *owner) {
  flagcxResult_t result = flagcxSuccess;
  PTHREADCHECKGOTO(pthread_mutex_lock(&semaphoreOwnerMutex),
                   "pthread_mutex_lock", result, exit);
  if (*owner && owner->use_count() == 1) {
    result = (*owner)->release();
    if (result != flagcxSuccess)
      goto unlock;
  }
  owner->reset();
unlock:
  pthread_mutex_unlock(&semaphoreOwnerMutex);
exit:
  return result;
}

flagcxResult_t flagcxSemaphore::releaseDefaultStream() {
  if (ownedDefaultStream != NULL) {
    FLAGCXCHECKSTRICT(deviceAdaptor->streamFree(ownedDefaultStream));
    ownedDefaultStream = NULL;
  }
  return flagcxSuccess;
}

void flagcxSemaphore::cancelNotifications(flagcxResult_t error) {
  pthread_mutex_lock(&notificationMutex);
  if (!isCancelled()) {
    recordNotificationError(error == flagcxSuccess || error == flagcxInProgress
                                ? flagcxUnhandledDeviceError
                                : error);
    __atomic_store_n(&notificationNeedsDrain, true, __ATOMIC_RELEASE);
    __atomic_store_n(&notificationCancelled, true, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&notificationMutex);
}

flagcxResult_t flagcxSemaphore::drainNotifications() {
  if (!isSealed())
    return flagcxInProgress;
  flagcxResult_t result = flagcxSuccess;
  PTHREADCHECKGOTO(pthread_mutex_lock(&notificationMutex), "pthread_mutex_lock",
                   result, exit);
  if (__atomic_load_n(&notificationNeedsDrain, __ATOMIC_ACQUIRE)) {
    for (size_t i = 0; i < notificationStreams.size(); ++i) {
      // Query is a nonblocking SDK completion proof. No tail event or device
      // synchronization is introduced; an uncertain owner stays retryable.
      FLAGCXCHECKGOTO(deviceAdaptor->streamQuery(notificationStreams[i]),
                      result, unlock);
      if (result != flagcxSuccess)
        goto unlock;
    }
    __atomic_store_n(&notificationNeedsDrain, false, __ATOMIC_RELEASE);
  }
unlock:
  pthread_mutex_unlock(&notificationMutex);
exit:
  return result;
}

void flagcxHostSemaphore::runCallback() {
  int expected = 0;
  if (__atomic_compare_exchange_n(&callbackStarted, &expected, 1, false,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    if (!isCancelled()) {
      signalStart();
      wait();
    }
  } else {
    while (!isCancelled() && __atomic_load_n(&counter, __ATOMIC_ACQUIRE) != 0)
      sched_yield();
  }
}

flagcxResult_t flagcxHostSemaphore::drainNotifications() {
  FLAGCXCHECKSTRICT(flagcxSemaphore::drainNotifications());
  if (isCancelled()) {
    for (struct flagcxGroupCallback *context = callbackContexts;
         context != NULL; context = context->next) {
      if (__atomic_exchange_n(&context->finished, 1, __ATOMIC_ACQ_REL) == 0)
        __atomic_fetch_sub(&callbacks, 1, __ATOMIC_RELEASE);
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHostSemaphore::release() {
  FLAGCXCHECKSTRICT(drainNotifications());
  if (__atomic_load_n(&callbacks, __ATOMIC_ACQUIRE) != 0)
    return flagcxInProgress;
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i] == NULL)
      continue;
    FLAGCXCHECKSTRICT(deviceAdaptor->eventDestroy(events[i]));
    events[i] = NULL;
  }
  FLAGCXCHECKSTRICT(releaseDefaultStream());
  while (callbackContexts != NULL) {
    struct flagcxGroupCallback *context = callbackContexts;
    callbackContexts = context->next;
    free(context);
  }
  notificationStreams.clear();
  return flagcxSuccess;
}

void flagcxDeviceSemaphore::cancelNotifications(flagcxResult_t error) {
  pthread_mutex_lock(&signalMutex);
  flagcxSemaphore::cancelNotifications(error);
  if (signals != NULL)
    __atomic_store_n(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, 0,
                     __ATOMIC_RELEASE);
  pthread_mutex_unlock(&signalMutex);
}
flagcxResult_t flagcxDeviceSemaphore::drainNotifications() {
  FLAGCXCHECKSTRICT(flagcxSemaphore::drainNotifications());
  if (isCancelled())
    __atomic_store_n(&completionDrained, true, __ATOMIC_RELEASE);
  return flagcxSuccess;
}
flagcxResult_t flagcxDeviceSemaphore::release() {
  FLAGCXCHECKSTRICT(drainNotifications());
  if (!notificationStreams.empty() && !pollEnd())
    return flagcxInProgress;
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i] == NULL)
      continue;
    FLAGCXCHECKSTRICT(deviceAdaptor->eventDestroy(events[i]));
    events[i] = NULL;
  }
  FLAGCXCHECKSTRICT(releaseDefaultStream());
  if (slotId >= 0) {
    FLAGCXCHECKSTRICT(deviceSemaphoreBufferPool.releaseSlot(slotId));
    slotId = -1;
  }
  notificationStreams.clear();
  headEvent = NULL;
  return flagcxSuccess;
}
flagcxResult_t flagcxStreamValueSemaphore::drainNotifications() {
  FLAGCXCHECKSTRICT(flagcxSemaphore::drainNotifications());
  if (isCancelled()) {
    markCompletionDrained();
    completionUnsafe = false;
  }
  return flagcxSuccess;
}
flagcxResult_t flagcxStreamValueSemaphore::release() {
  FLAGCXCHECKSTRICT(drainNotifications());
  if (completionUnsafe)
    return flagcxInProgress;
  if (completionEvent != NULL) {
    FLAGCXCHECKSTRICT(deviceAdaptor->eventDestroy(completionEvent));
    completionEvent = NULL;
  }
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i] == NULL)
      continue;
    FLAGCXCHECKSTRICT(deviceAdaptor->eventDestroy(events[i]));
    events[i] = NULL;
  }
  FLAGCXCHECKSTRICT(releaseDefaultStream());
  streamValueBufferPool.release(signals);
  signals = NULL;
  notificationStreams.clear();
  return flagcxSuccess;
}

flagcxStreamValueBufferPool streamValueBufferPool;

flagcxStreamValueBufferPool::~flagcxStreamValueBufferPool() {
  flagcxResult_t result = clear();
  if (result != flagcxSuccess)
    WARN("Failed to clear stream-value semaphore pool: %d", result);
}

flagcxResult_t flagcxStreamValueBufferPool::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &chunk : chunks) {
    for (int slot = 0; slot < chunk.capacity; ++slot) {
      if (chunk.inUse[slot] != 0)
        return flagcxInvalidUsage;
    }
  }
  while (!chunks.empty()) {
    Chunk &chunk = chunks.back();
    flagcxResult_t result =
        deviceAdaptor->deviceFree(chunk.signals, flagcxMemHost, nullptr);
    if (result != flagcxSuccess)
      return result;
    free(chunk.inUse);
    chunks.pop_back();
  }
  return flagcxSuccess;
}

namespace {

std::mutex streamValueProbeMutex;
std::map<const flagcxDeviceAdaptor *, std::map<int, bool>>
    streamValueProbeCache;

struct flagcxStreamValueProbeResources {
  flagcxStreamValueProbeResources *next;
  flagcxDeviceAdaptor *adaptor;
  int device;
  flagcxStream_t stream;
  uint64_t *hostValue;
  bool writeQueued;
};
flagcxStreamValueProbeResources *pendingProbes;

static flagcxResult_t releaseProbe(flagcxStreamValueProbeResources *probe) {
  if (probe->writeQueued) {
    flagcxResult_t result = probe->adaptor->streamSynchronize(probe->stream);
    if (result != flagcxSuccess)
      return result;
    probe->writeQueued = false;
  }
  if (probe->hostValue != NULL) {
    flagcxResult_t result =
        probe->adaptor->deviceFree(probe->hostValue, flagcxMemHost, NULL);
    if (result != flagcxSuccess)
      return result;
    probe->hostValue = NULL;
  }
  if (probe->stream != NULL) {
    flagcxResult_t result = probe->adaptor->streamDestroy(probe->stream);
    if (result != flagcxSuccess)
      return result;
    probe->stream = NULL;
  }
  return flagcxSuccess;
}
static flagcxResult_t retryProbes(flagcxDeviceAdaptor *adaptor, int device) {
  auto **link = &pendingProbes;
  while (*link != NULL) {
    auto *probe = *link;
    if (probe->adaptor != adaptor || probe->device != device) {
      link = &probe->next;
      continue;
    }
    flagcxResult_t result = releaseProbe(probe);
    if (result != flagcxSuccess)
      return result;
    *link = probe->next;
    free(probe);
  }
  return flagcxSuccess;
}

// The stream is private to this probe. A failed drain retains its owner and
// prevents plugin unload; no device-wide fallback can wait for unrelated work.
flagcxResult_t probeStreamValueOps(flagcxDeviceAdaptor *adaptor, int device) {
  flagcxResult_t result = retryProbes(adaptor, device);
  if (result != flagcxSuccess)
    return result;
  auto *probe = (flagcxStreamValueProbeResources *)calloc(
      1, sizeof(flagcxStreamValueProbeResources));
  if (probe == NULL)
    return flagcxSystemError;
  probe->adaptor = adaptor;
  probe->device = device;
  probe->next = pendingProbes;
  pendingProbes = probe;
  result = adaptor->streamCreate(&probe->stream);
  if (result == flagcxSuccess && probe->stream == NULL)
    result = flagcxSystemError;
  if (result == flagcxSuccess)
    result = adaptor->deviceMalloc((void **)&probe->hostValue, sizeof(uint64_t),
                                   flagcxMemHost, NULL);
  if (result == flagcxSuccess && probe->hostValue == NULL)
    result = flagcxSystemError;
  void *deviceValue = NULL;
  if (result == flagcxSuccess) {
    *probe->hostValue = 0;
    result = adaptor->hostGetDevicePointer(&deviceValue, probe->hostValue);
    if (result == flagcxSuccess && deviceValue == NULL)
      result = flagcxSystemError;
  }
  if (result == flagcxSuccess) {
    result = adaptor->streamWriteValue64(probe->stream, deviceValue, 1, 0);
    probe->writeQueued = result == flagcxSuccess;
  }
  if (result == flagcxSuccess)
    result = adaptor->streamWaitValue64(probe->stream, deviceValue, 1, 0);
  flagcxResult_t cleanup = retryProbes(adaptor, device);
  return cleanup == flagcxSuccess ? result : cleanup;
}

} // namespace

flagcxResult_t flagcxStreamValueSupported(bool *supported) {
  if (supported == nullptr)
    return flagcxInvalidArgument;
  *supported = false;
  flagcxDeviceAdaptor *adaptor = deviceAdaptor;
  if (adaptor == nullptr || adaptor->streamWaitValue64 == nullptr ||
      adaptor->streamWriteValue64 == nullptr ||
      adaptor->hostGetDevicePointer == nullptr)
    return flagcxSuccess;

  int device = -1;
  FLAGCXCHECK(adaptor->getDevice(&device));
  std::lock_guard<std::mutex> lock(streamValueProbeMutex);
  auto &devices = streamValueProbeCache[adaptor];
  auto found = devices.find(device);
  if (found != devices.end()) {
    *supported = found->second;
    return flagcxSuccess;
  }

  flagcxResult_t result = probeStreamValueOps(adaptor, device);
  if (result == flagcxNotSupported) {
    devices[device] = false;
    return flagcxSuccess;
  }
  if (result != flagcxSuccess)
    return result;
  devices[device] = true;
  *supported = true;
  return flagcxSuccess;
}

flagcxResult_t flagcxStreamValueProbeCleanup() {
  std::lock_guard<std::mutex> lock(streamValueProbeMutex);
  while (pendingProbes != NULL) {
    auto *probe = pendingProbes;
    flagcxResult_t result = probe->adaptor->setDevice(probe->device);
    if (result != flagcxSuccess)
      return result;
    result = retryProbes(probe->adaptor, probe->device);
    if (result != flagcxSuccess)
      return result;
  }
  return flagcxSuccess;
}

void flagcxClearStreamValueProbeCache() {
  std::lock_guard<std::mutex> lock(streamValueProbeMutex);
  streamValueProbeCache.clear();
}

flagcxResult_t flagcxStreamValueBufferPool::grow() {
  Chunk chunk{};
  chunk.capacity = flagcxParamSemaphoreBufferPoolCapacity();
  if (chunk.capacity < 1) {
    chunk.capacity = 1;
  }
  flagcxResult_t result = deviceAdaptor->deviceMalloc(
      reinterpret_cast<void **>(&chunk.signals),
      static_cast<size_t>(chunk.capacity) * 2 * sizeof(uint64_t), flagcxMemHost,
      nullptr);
  if (result != flagcxSuccess)
    return result;
  if (chunk.signals == nullptr)
    return flagcxSystemError;
  result = deviceAdaptor->hostGetDevicePointer(&chunk.dSignals, chunk.signals);
  if (result != flagcxSuccess || chunk.dSignals == nullptr) {
    deviceAdaptor->deviceFree(chunk.signals, flagcxMemHost, nullptr);
    return result == flagcxSuccess ? flagcxSystemError : result;
  }
  result = flagcxCalloc(&chunk.inUse, chunk.capacity);
  if (result != flagcxSuccess) {
    deviceAdaptor->deviceFree(chunk.signals, flagcxMemHost, nullptr);
    return result;
  }
  chunks.push_back(chunk);
  return flagcxSuccess;
}

flagcxResult_t flagcxStreamValueBufferPool::reserve(size_t requiredFreeSlots) {
  if (requiredFreeSlots == 0)
    return flagcxSuccess;
  std::lock_guard<std::mutex> lock(mutex);
  size_t missing = requiredFreeSlots;
  for (const auto &chunk : chunks) {
    for (int slot = 0; slot < chunk.capacity; ++slot) {
      if (chunk.inUse[slot] == 0 && --missing == 0)
        return flagcxSuccess;
    }
  }
  while (missing != 0) {
    FLAGCXCHECK(grow());
    const size_t added = static_cast<size_t>(chunks.back().capacity);
    if (added >= missing)
      return flagcxSuccess;
    missing -= added;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxStreamValueBufferPool::acquire(uint64_t **signals,
                                                    void **dSignals) {
  std::lock_guard<std::mutex> lock(mutex);
  for (auto &chunk : chunks) {
    for (int slot = 0; slot < chunk.capacity; ++slot) {
      if (chunk.inUse[slot] == 0) {
        chunk.inUse[slot] = 1;
        *signals = chunk.signals + 2 * slot;
        *dSignals = static_cast<char *>(chunk.dSignals) +
                    static_cast<size_t>(2 * slot) * sizeof(uint64_t);
        memset(*signals, 0, 2 * sizeof(uint64_t));
        return flagcxSuccess;
      }
    }
  }

  FLAGCXCHECK(grow());
  Chunk &chunk = chunks.back();
  chunk.inUse[0] = 1;
  *signals = chunk.signals;
  *dSignals = chunk.dSignals;
  memset(*signals, 0, 2 * sizeof(uint64_t));
  return flagcxSuccess;
}

void flagcxStreamValueBufferPool::release(uint64_t *signals) {
  if (signals == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex);
  const uintptr_t address = reinterpret_cast<uintptr_t>(signals);
  for (auto &chunk : chunks) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(chunk.signals);
    const uintptr_t end =
        begin + static_cast<size_t>(chunk.capacity) * 2 * sizeof(uint64_t);
    if (address >= begin && address < end) {
      const size_t offset = address - begin;
      assert(offset % (2 * sizeof(uint64_t)) == 0);
      chunk.inUse[offset / (2 * sizeof(uint64_t))] = 0;
      return;
    }
  }
  assert(false);
}

flagcxResult_t
flagcxStreamValueSemaphore::enqueueCompletion(flagcxStream_t launchStream) {
  // Log both sides of each existing call so a blocked enqueue is visible.
  auto traceCall = [&](const char *stage, int result) {
    TRACE(FLAGCX_PROXY,
          "stream-value semaphore sem=%p stage=%s stream=%p signals=%p "
          "device_signals=%p event=%p result=%d",
          static_cast<void *>(this), stage, static_cast<void *>(launchStream),
          static_cast<void *>(signals), dSignals,
          static_cast<void *>(completionEvent), result);
  };
  traceCall("event-create-enter", -1);
  flagcxResult_t result =
      deviceAdaptor->eventCreate(&completionEvent, flagcxEventDisableTiming);
  traceCall("event-create-return", result);
  FLAGCXCHECK(result);
  traceCall("start-write-enter", -1);
  result = deviceAdaptor->streamWriteValue64(launchStream, dSignals, 1, 0);
  traceCall("start-write-return", result);
  FLAGCXCHECK(result);
  traceCall("end-wait-enter", -1);
  result = deviceAdaptor->streamWaitValue64(
      launchStream, static_cast<char *>(dSignals) + sizeof(uint64_t), 1, 0);
  traceCall("end-wait-return", result);
  FLAGCXCHECK(result);
  traceCall("event-record-enter", -1);
  result = deviceAdaptor->eventRecord(completionEvent, launchStream);
  traceCall("event-record-return", result);
  FLAGCXCHECK(result);
  // Publish the event handle only after it has captured the stream work.
  __atomic_store_n(&completionState, 1, __ATOMIC_RELEASE);
  traceCall("event-published", flagcxSuccess);
  return flagcxSuccess;
}

void flagcxStreamValueSemaphore::subCounter(int opId) {
  auto it = stepInfo.find(opId);
  assert(it != stepInfo.end());
  const int idx = it->second;
  const int completed =
      __atomic_fetch_add(&steps[idx].first, 1, __ATOMIC_ACQ_REL) + 1;
  const int required = __atomic_load_n(&steps[idx].second, __ATOMIC_ACQUIRE);
  if (completed == required) {
    if (__atomic_sub_fetch(&counter, 1, __ATOMIC_ACQ_REL) == 0) {
      __atomic_store_n(signals + 1, uint64_t(1), __ATOMIC_RELEASE);
      TRACE(FLAGCX_PROXY,
            "stream-value semaphore sem=%p stage=end_signal_published "
            "op_id=%d signals=%p end_signal=1",
            static_cast<void *>(this), opId, static_cast<void *>(signals));
    }
  }
  if (__atomic_load_n(&flagcxDebugLevel, __ATOMIC_ACQUIRE) >=
          FLAGCX_LOG_TRACE &&
      (flagcxDebugMask & FLAGCX_PROXY) != 0) {
    TRACE(FLAGCX_PROXY,
          "stream-value semaphore sem=%p stage=sub-counter op_id=%d "
          "completed_steps=%d required_steps=%d remaining_counter=%d",
          static_cast<void *>(this), opId, completed, required, getCounter());
  }
}

void flagcxStreamValueSemaphore::traceCompletion(const char *reason,
                                                 int remaining, int state,
                                                 bool queried,
                                                 flagcxResult_t queryResult) {
  if (__atomic_load_n(&flagcxDebugLevel, __ATOMIC_ACQUIRE) < FLAGCX_LOG_TRACE ||
      (flagcxDebugMask & FLAGCX_PROXY) == 0)
    return;
  std::unique_lock<std::mutex> lock(traceMutex, std::try_to_lock);
  if (!lock.owns_lock())
    return;
  const uint64_t start =
      signals == nullptr ? 0 : __atomic_load_n(signals, __ATOMIC_ACQUIRE);
  const uint64_t end =
      signals == nullptr ? 0 : __atomic_load_n(signals + 1, __ATOMIC_ACQUIRE);
  const std::array<uint64_t, 6> snapshot = {static_cast<uint64_t>(remaining),
                                            static_cast<uint64_t>(state),
                                            start,
                                            end,
                                            static_cast<uint64_t>(queried),
                                            static_cast<uint64_t>(queryResult)};
  const auto now = std::chrono::steady_clock::now();
  if (traceSeen && snapshot == traceSnapshot &&
      now - traceTime < std::chrono::seconds(5))
    return;
  traceSeen = true;
  traceSnapshot = snapshot;
  traceTime = now;
  // completionEvent is written by the submitting thread. Read it only after
  // the acquire load of completionState has observed its publication.
  TRACE(FLAGCX_PROXY,
        "stream-value semaphore sem=%p reason=%s counter=%d "
        "signals=%p device_signals=%p start_signal=%llu end_signal=%llu "
        "completion_state=%d event=%p event_query=%s event_result=%d",
        static_cast<void *>(this), reason, remaining,
        static_cast<void *>(signals), dSignals,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end), state,
        state == 0 ? nullptr : static_cast<void *>(completionEvent),
        queried ? "queried" : "not_queried", queried ? int(queryResult) : -1);
}

int flagcxStreamValueSemaphore::pollEnd() {
  if (!isSealed())
    return 0;
  const int remaining = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
  const int state = __atomic_load_n(&completionState, __ATOMIC_ACQUIRE);
  // Cancellation can include operations that were never published to a proxy.
  // Their counters cannot retire, but a successful stream drain proves that
  // no notification still uses the semaphore.
  if (isCancelled() && state == 2) {
    traceCompletion("cancelled_drained", remaining, state, false,
                    flagcxSuccess);
    return 1;
  }
  if (remaining != 0) {
    traceCompletion("counter_pending", remaining, state, false, flagcxSuccess);
    return 0;
  }
  if (state == 0) {
    traceCompletion("event_unpublished", remaining, state, false,
                    flagcxSuccess);
    return 0;
  }
  if (state == 2) {
    traceCompletion("complete_drained", remaining, state, false, flagcxSuccess);
    return 1;
  }
  const flagcxResult_t result = deviceAdaptor->eventQuery(completionEvent);
  recordNotificationError(result);
  traceCompletion(result == flagcxSuccess      ? "complete"
                  : result == flagcxInProgress ? "event_pending"
                                               : "event_error",
                  remaining, state, true, result);
  return result == flagcxSuccess;
}
