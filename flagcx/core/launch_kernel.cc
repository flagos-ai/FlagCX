#include "launch_kernel.h"
#include "group.h"
#include <map>
#include <stdio.h>

flagcxLaunchFunc_t deviceAsyncKernel = NULL;

FLAGCX_PARAM(SemaphoreBufferPoolCapacity, "SEMAPHORE_BUFFER_POOL_CAPACITY", 32);

flagcxDeviceSemaphoreBufferPool::flagcxDeviceSemaphoreBufferPool()
    : capacity(-1), slotId(-1), signalsPool(nullptr), dSignalsPool(nullptr),
      events(nullptr) {}

flagcxDeviceSemaphoreBufferPool::~flagcxDeviceSemaphoreBufferPool() {
  free(events);
  dSignalsPool = nullptr;
  if (signalsPool != nullptr) {
    deviceAdaptor->deviceFree((void *)signalsPool, flagcxMemHost, nullptr);
  }
}

int flagcxDeviceSemaphoreBufferPool::getSlotId() {
  assert(capacity != -1);
  if (events[slotId] != nullptr) {
    // wait for the previous event to complete
    while (deviceAdaptor->eventQuery(events[slotId]) != flagcxSuccess) {
      sched_yield();
    }
    events[slotId] = nullptr;
  }
  // set this slot signals to zero
  int offset = FLAGCX_SIGNALS_PER_SEMAPHORE * slotId;
  memset(signalsPool + offset, 0, FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int));
  int ret = slotId;
  // Move to next slot
  slotId = (slotId + 1) % capacity;
  return ret;
}

void flagcxDeviceSemaphoreBufferPool::initialize() {
  capacity = flagcxParamSemaphoreBufferPoolCapacity();
  slotId = 0;
  // Allocate host-pinned memory for all semaphores (3 ints each)
  deviceAdaptor->deviceMalloc((void **)&signalsPool,
                              capacity * FLAGCX_SIGNALS_PER_SEMAPHORE *
                                  sizeof(int),
                              flagcxMemHost, nullptr);
  // Get device pointer alias
  deviceAdaptor->hostGetDevicePointer(&dSignalsPool, (void *)signalsPool);
  // Init events to nullptr
  flagcxCalloc(&events, capacity);
  for (int i = 0; i < capacity; i++) {
    events[i] = nullptr;
  }
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

// The write precedes the wait on the same private stream, so a supported wait
// cannot block indefinitely. A queued write must finish before its memory is
// released, including when the wait operation returns NotSupported.
flagcxResult_t probeStreamValueOps(flagcxDeviceAdaptor *adaptor) {
  flagcxStream_t stream = nullptr;
  uint64_t *hostValue = nullptr;
  void *deviceValue = nullptr;
  bool writeQueued = false;

  flagcxResult_t result = adaptor->streamCreate(&stream);
  if (result != flagcxSuccess)
    return result;
  if (stream == nullptr)
    return flagcxSystemError;

  result = adaptor->deviceMalloc(reinterpret_cast<void **>(&hostValue),
                                 sizeof(*hostValue), flagcxMemHost, nullptr);
  if (result == flagcxSuccess && hostValue == nullptr)
    result = flagcxSystemError;
  if (result == flagcxSuccess) {
    *hostValue = 0;
    result = adaptor->hostGetDevicePointer(&deviceValue, hostValue);
    if (result == flagcxSuccess && deviceValue == nullptr)
      result = flagcxSystemError;
  }
  if (result == flagcxSuccess) {
    result = adaptor->streamWriteValue64(stream, deviceValue, 1, 0);
    writeQueued = result == flagcxSuccess;
  }
  if (writeQueued && result == flagcxSuccess)
    result = adaptor->streamWaitValue64(stream, deviceValue, 1, 0);

  if (writeQueued) {
    flagcxResult_t syncResult = adaptor->streamSynchronize(stream);
    if (syncResult != flagcxSuccess) {
      // An asynchronous write may still reference hostValue. Keep both probe
      // resources alive if even a device-wide synchronization cannot drain it.
      if (adaptor->deviceSynchronize() != flagcxSuccess) {
        WARN("Stream-value probe could not drain its private stream");
        return syncResult;
      }
      result = syncResult;
    }
  }

  flagcxResult_t cleanupResult = flagcxSuccess;
  if (hostValue != nullptr)
    cleanupResult = adaptor->deviceFree(hostValue, flagcxMemHost, nullptr);
  flagcxResult_t destroyResult = adaptor->streamDestroy(stream);
  if (cleanupResult != flagcxSuccess)
    return cleanupResult;
  if (destroyResult != flagcxSuccess)
    return destroyResult;
  return result;
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
  flagcxResult_t deviceResult = adaptor->getDevice(&device);
  if (deviceResult != flagcxSuccess)
    return deviceResult;
  std::lock_guard<std::mutex> lock(streamValueProbeMutex);
  auto &devices = streamValueProbeCache[adaptor];
  auto found = devices.find(device);
  if (found != devices.end()) {
    *supported = found->second;
    return flagcxSuccess;
  }

  flagcxResult_t result = probeStreamValueOps(adaptor);
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
  return flagcxSuccess;
}

int flagcxStreamValueSemaphore::pollEnd() {
  if (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) != 0 ||
      completionEvent == nullptr) {
    return 0;
  }
  return deviceAdaptor->eventQuery(completionEvent) == flagcxSuccess;
}
