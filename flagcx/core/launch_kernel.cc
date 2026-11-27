#include "launch_kernel.h"
#include "group.h"
#include <climits>
#include <map>
#include <new>
#include <stdio.h>
#include <utility>

flagcxLaunchFunc_t deviceAsyncKernel = NULL;

FLAGCX_PARAM(SemaphoreBufferPoolCapacity, "SEMAPHORE_BUFFER_POOL_CAPACITY", 32);

flagcxDeviceSemaphoreBufferPool deviceSemaphoreBufferPool;

flagcxDeviceSemaphoreBufferPool::flagcxDeviceSemaphoreBufferPool()
    : capacity(0), chunks(NULL), pending(NULL) {}

flagcxDeviceSemaphoreBufferPool::~flagcxDeviceSemaphoreBufferPool() {
  (void)clear();
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::getSlotId(int *id) {
  *id = -1;
  std::lock_guard<std::mutex> lock(mutex);
  for (;;) {
    for (auto *chunk = chunks; chunk != NULL; chunk = chunk->next) {
      for (int i = 0; i < chunk->capacity; ++i) {
        if (chunk->inUse[i])
          continue;
        chunk->inUse[i] = 1;
        memset(chunk->signals + (size_t)FLAGCX_SIGNALS_PER_SEMAPHORE * i, 0,
               FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int));
        *id = chunk->first + i;
        return flagcxSuccess;
      }
    }
    // Completed groups may still have live Comm owners. Grow without moving
    // their signals or reclaiming any slot before its owner releases it.
    flagcxResult_t result = grow();
    if (result != flagcxSuccess)
      return result;
  }
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
  for (auto *chunk = chunks; chunk != NULL; chunk = chunk->next)
    if (id >= chunk->first && id - chunk->first < chunk->capacity)
      return chunk;
  assert(false);
  return NULL;
}

void flagcxDeviceSemaphoreBufferPool::releaseSlot(int id) {
  std::lock_guard<std::mutex> lock(mutex);
  auto *chunk = findChunk(id);
  assert(chunk->inUse[id - chunk->first]);
  chunk->inUse[id - chunk->first] = 0;
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  for (auto *chunk = chunks; chunk != NULL; chunk = chunk->next)
    for (int i = 0; i < chunk->capacity; ++i)
      if (chunk->inUse[i])
        return flagcxInProgress;
  // Each successful free removes its owner. A later failure retains only the
  // remaining chunks, including any partially initialized growth allocation.
  Chunk **lists[] = {&pending, &chunks};
  for (auto **list : lists) {
    while (*list != NULL) {
      auto *chunk = *list;
      if (chunk->signals != NULL) {
        flagcxResult_t result =
            deviceAdaptor->deviceFree(chunk->signals, flagcxMemHost, NULL);
        if (result != flagcxSuccess)
          return result;
      }
      *list = chunk->next;
      free(chunk->inUse);
      free(chunk);
    }
  }
  capacity = 0;
  return flagcxSuccess;
}

int *flagcxDeviceSemaphoreBufferPool::getHostPtr(int id) {
  std::lock_guard<std::mutex> lock(mutex);
  auto *chunk = findChunk(id);
  return chunk->signals +
         (size_t)FLAGCX_SIGNALS_PER_SEMAPHORE * (id - chunk->first);
}
void *flagcxDeviceSemaphoreBufferPool::getDevicePtr(int id) {
  std::lock_guard<std::mutex> lock(mutex);
  auto *chunk = findChunk(id);
  return (char *)chunk->dSignals + (size_t)FLAGCX_SIGNALS_PER_SEMAPHORE *
                                       (id - chunk->first) * sizeof(int);
}

void cpuAsyncKernel(void *args) {
  flagcxHostSemaphore *semaphore = (flagcxHostSemaphore *)args;
  semaphore->signalStart();
  semaphore->wait();
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
std::map<const flagcxDeviceAdaptor *, std::map<std::pair<int, bool>, bool>>
    streamValueProbeCache;

struct flagcxStreamValueProbeResources {
  flagcxStreamValueProbeResources *next;
  flagcxDeviceAdaptor *adaptor;
  int device;
  flagcxStream_t stream;
  uint64_t *value;
  flagcxMemType_t memoryType;
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
  if (probe->value != NULL) {
    flagcxResult_t result =
        probe->adaptor->deviceFree(probe->value, probe->memoryType, NULL);
    if (result != flagcxSuccess)
      return result;
    probe->value = NULL;
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
flagcxResult_t probeStreamValueOps(flagcxDeviceAdaptor *adaptor, int device,
                                   bool deviceControl, bool *unavailable) {
  *unavailable = false;
  flagcxResult_t result = retryProbes(adaptor, device);
  if (result != flagcxSuccess)
    return result;
  auto *probe = (flagcxStreamValueProbeResources *)calloc(
      1, sizeof(flagcxStreamValueProbeResources));
  if (probe == NULL)
    return flagcxSystemError;
  probe->adaptor = adaptor;
  probe->device = device;
  probe->memoryType = deviceControl ? flagcxMemDevice : flagcxMemHost;
  probe->next = pendingProbes;
  pendingProbes = probe;
  result = adaptor->streamCreate(&probe->stream);
  if (result == flagcxSuccess && probe->stream == NULL)
    result = flagcxSystemError;
  if (result == flagcxSuccess)
    result =
        flagcxDeviceAllocWithFlags(adaptor, (void **)&probe->value,
                                   sizeof(uint64_t), probe->memoryType, NULL,
                                   deviceControl ? FLAGCX_DEVICE_ALLOC_UNIRUNNER
                                                 : FLAGCX_DEVICE_ALLOC_DEFAULT);
  if (result == flagcxSuccess && probe->value == NULL)
    result = flagcxSystemError;
  void *deviceValue = probe->value;
  if (result == flagcxSuccess && !deviceControl) {
    *probe->value = 0;
    result = adaptor->hostGetDevicePointer(&deviceValue, probe->value);
    if (result == flagcxSuccess && deviceValue == NULL)
      result = flagcxSystemError;
  }
  const uint64_t expected = deviceControl ? UINT64_C(0xfedcba9876543210) : 1;
  if (result == flagcxSuccess) {
    result =
        adaptor->streamWriteValue64(probe->stream, deviceValue, expected, 0);
    probe->writeQueued = result == flagcxSuccess;
  }
  if (result == flagcxSuccess)
    result =
        adaptor->streamWaitValue64(probe->stream, deviceValue, expected, 0);
  if (result == flagcxSuccess && deviceControl) {
    result = adaptor->streamSynchronize(probe->stream);
    if (result == flagcxSuccess) {
      probe->writeQueued = false;
      uint64_t observed = 0;
      result = adaptor->deviceMemcpy(&observed, deviceValue, sizeof(observed),
                                     flagcxMemcpyDeviceToHost, NULL, NULL);
      if (result == flagcxSuccess && observed != expected)
        result = flagcxInternalError;
    }
  }
  flagcxResult_t cleanup = retryProbes(adaptor, device);
  *unavailable = cleanup == flagcxSuccess && result == flagcxNotSupported;
  return cleanup == flagcxSuccess ? result : cleanup;
}

} // namespace

static flagcxResult_t streamValueSupported(bool *supported,
                                           bool deviceControl) {
  if (supported == nullptr)
    return flagcxInvalidArgument;
  *supported = false;
  flagcxDeviceAdaptor *adaptor = deviceAdaptor;
  if (adaptor == nullptr || adaptor->streamWaitValue64 == nullptr ||
      adaptor->streamWriteValue64 == nullptr || adaptor->getDevice == nullptr ||
      adaptor->streamCreate == nullptr ||
      adaptor->streamSynchronize == nullptr ||
      adaptor->streamDestroy == nullptr || adaptor->deviceFree == nullptr ||
      (!deviceControl && adaptor->hostGetDevicePointer == nullptr) ||
      (deviceControl && adaptor->deviceMemcpy == nullptr))
    return flagcxSuccess;

  int device = -1;
  FLAGCXCHECK(adaptor->getDevice(&device));
  std::lock_guard<std::mutex> lock(streamValueProbeMutex);
  auto &devices = streamValueProbeCache[adaptor];
  const auto key = std::make_pair(device, deviceControl);
  auto found = devices.find(key);
  if (found != devices.end()) {
    *supported = found->second;
    return flagcxSuccess;
  }

  bool unavailable = false;
  flagcxResult_t result =
      probeStreamValueOps(adaptor, device, deviceControl, &unavailable);
  if (unavailable) {
    devices[key] = false;
    return flagcxSuccess;
  }
  if (result != flagcxSuccess)
    return result;
  devices[key] = true;
  *supported = true;
  return flagcxSuccess;
}

flagcxResult_t flagcxStreamValueSupported(bool *supported) {
  return streamValueSupported(supported, false);
}

flagcxResult_t flagcxUniRunnerDeviceControlSupported(bool *supported) {
  return streamValueSupported(supported, true);
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
  if (isCompletionCancelled())
    return 1;
  const int remaining = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
  const int state = __atomic_load_n(&completionState, __ATOMIC_ACQUIRE);
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
  traceCompletion(result == flagcxSuccess      ? "complete"
                  : result == flagcxInProgress ? "event_pending"
                                               : "event_error",
                  remaining, state, true, result);
  return result == flagcxSuccess;
}
