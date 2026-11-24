#include "launch_kernel.h"
#include "group.h"
#include <climits>
#include <map>
#include <new>
#include <stdio.h>

flagcxLaunchFunc_t deviceAsyncKernel = NULL;

FLAGCX_PARAM(SemaphoreBufferPoolCapacity, "SEMAPHORE_BUFFER_POOL_CAPACITY", 32);

flagcxDeviceSemaphoreBufferPool deviceSemaphoreBufferPool;

flagcxDeviceSemaphoreBufferPool::flagcxDeviceSemaphoreBufferPool()
    : capacity(-1), slotId(-1), signalsPool(nullptr), dSignalsPool(nullptr),
      events(nullptr), initError(flagcxSuccess) {}

flagcxDeviceSemaphoreBufferPool::~flagcxDeviceSemaphoreBufferPool() {
  (void)clear();
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::getSlotId(int *id) {
  *id = -1;
  std::unique_lock<std::mutex> lock(mutex);
  if (initError != flagcxSuccess) {
    lock.unlock();
    flagcxResult_t result = clear();
    if (result != flagcxSuccess)
      return result;
    lock.lock();
  }
  if (capacity == -1) {
    initError = initialize();
    if (initError != flagcxSuccess)
      return initError;
  }
  for (int i = 0; i < capacity; ++i) {
    int candidate = (slotId + i) % capacity;
    if (inUse[candidate])
      continue;
    inUse[candidate] = 1;
    memset(signalsPool + FLAGCX_SIGNALS_PER_SEMAPHORE * candidate, 0,
           FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int));
    slotId = (candidate + 1) % capacity;
    *id = candidate;
    return flagcxSuccess;
  }
  // A busy slot belongs to its previous Comm until that owner releases it.
  return flagcxInProgress;
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::initialize() {
  capacity = flagcxParamSemaphoreBufferPoolCapacity();
  if (capacity <= 0 || capacity > INT_MAX / FLAGCX_SIGNALS_PER_SEMAPHORE)
    return flagcxInvalidArgument;
  slotId = 0;
  // Allocate host-pinned memory for all semaphores (3 ints each)
  flagcxResult_t result = deviceAdaptor->deviceMalloc(
      (void **)&signalsPool,
      capacity * FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int), flagcxMemHost,
      nullptr);
  if (result != flagcxSuccess)
    return result;
  // Get device pointer alias
  result =
      deviceAdaptor->hostGetDevicePointer(&dSignalsPool, (void *)signalsPool);
  if (result != flagcxSuccess)
    return result;
  // Init events to nullptr
  result = flagcxCalloc(&events, capacity);
  if (result != flagcxSuccess)
    return result;
  try {
    inUse.assign(capacity, 0);
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

void flagcxDeviceSemaphoreBufferPool::releaseSlot(int id) {
  std::lock_guard<std::mutex> lock(mutex);
  assert(id >= 0 && id < capacity && inUse[id]);
  events[id] = nullptr;
  inUse[id] = 0;
}

flagcxResult_t flagcxDeviceSemaphoreBufferPool::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  for (int used : inUse)
    if (used)
      return flagcxInProgress;
  if (signalsPool != nullptr) {
    flagcxResult_t result =
        deviceAdaptor->deviceFree(signalsPool, flagcxMemHost, nullptr);
    if (result != flagcxSuccess)
      return result;
    signalsPool = nullptr;
  }
  free(events);
  events = nullptr;
  dSignalsPool = nullptr;
  inUse.clear();
  capacity = slotId = -1;
  initError = flagcxSuccess;
  return flagcxSuccess;
}

// Set event for a semaphore
void flagcxDeviceSemaphoreBufferPool::setEvent(int id, flagcxEvent_t event) {
  assert(id >= 0 && id < capacity);
  // events[id] should be set to nullptr before
  events[id] = event;
}

// Return pointer to the start of a semaphore’s signals (host/device)
int *flagcxDeviceSemaphoreBufferPool::getHostPtr(int id) {
  assert(id >= 0 && id < capacity);
  return signalsPool + FLAGCX_SIGNALS_PER_SEMAPHORE * id;
}
void *flagcxDeviceSemaphoreBufferPool::getDevicePtr(int id) {
  assert(id >= 0 && id < capacity);
  return static_cast<void *>((static_cast<char *>(dSignalsPool) +
                              FLAGCX_SIGNALS_PER_SEMAPHORE * id * sizeof(int)));
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
  FLAGCXCHECK(
      deviceAdaptor->eventCreate(&completionEvent, flagcxEventDisableTiming));
  FLAGCXCHECK(deviceAdaptor->streamWriteValue64(launchStream, dSignals, 1, 0));
  FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
      launchStream, static_cast<char *>(dSignals) + sizeof(uint64_t), 1, 0));
  FLAGCXCHECK(deviceAdaptor->eventRecord(completionEvent, launchStream));
  // Publish the event handle only after it has captured the stream work.
  __atomic_store_n(&completionState, 1, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

int flagcxStreamValueSemaphore::pollEnd() {
  if (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) != 0) {
    return 0;
  }
  const int state = __atomic_load_n(&completionState, __ATOMIC_ACQUIRE);
  if (state == 0)
    return 0;
  if (state == 2)
    return 1;
  return deviceAdaptor->eventQuery(completionEvent) == flagcxSuccess;
}
