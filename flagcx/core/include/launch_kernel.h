#ifndef FLAGCX_LAUNCH_KERNEL_H_
#define FLAGCX_LAUNCH_KERNEL_H_
#pragma once
#include "adaptor.h"
#include "check.h"
#include "debug.h"
#include "flagcx.h"
#include "param.h"
#include "topo.h"
#include "utils.h"
#include <array>
#include <dlfcn.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <mutex>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct flagcxHostSemaphore;
struct flagcxGroupCallback {
  struct flagcxGroupCallback *next;
  struct flagcxHostSemaphore *semaphore;
  int submitted, finished;
};

struct flagcxSemaphore {
  flagcxResult_t initResult = flagcxSuccess;
  bool notificationSealed = true, notificationCancelled = false;
  flagcxResult_t notificationError = flagcxSuccess;
  pthread_mutex_t notificationMutex = PTHREAD_MUTEX_INITIALIZER;
  flagcxStream_t ownedDefaultStream = NULL;
  // Borrowed only by the builder; retirement never dereferences these wrappers.
  std::vector<flagcxStream_t> notificationStreams;
  void prepareNotifications() { notificationSealed = false; }
  void sealNotifications() {
    __atomic_store_n(&notificationSealed, true, __ATOMIC_RELEASE);
  }
  bool isCancelled() {
    return __atomic_load_n(&notificationCancelled, __ATOMIC_ACQUIRE);
  }
  bool isSealed() {
    return __atomic_load_n(&notificationSealed, __ATOMIC_ACQUIRE);
  }
  virtual void cancelNotifications(flagcxResult_t error = flagcxRemoteError);
  virtual flagcxResult_t drainNotifications();
  virtual flagcxResult_t captureCancellationCompletion(flagcxStream_t) {
    return flagcxSuccess;
  }
  virtual flagcxResult_t release() {
    {
      flagcxResult_t cleanupStatus = (drainNotifications());
      FLAGCXCHECK(cleanupStatus);
      if (cleanupStatus == flagcxInProgress)
        return cleanupStatus;
    }
    return releaseDefaultStream();
  }
  flagcxResult_t releaseDefaultStream();
  void recordNotificationError(flagcxResult_t error) {
    if (error == flagcxSuccess || error == flagcxInProgress)
      return;
    flagcxResult_t expected = flagcxSuccess;
    __atomic_compare_exchange_n(&notificationError, &expected, error, false,
                                __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  }
  flagcxResult_t getNotificationError() {
    return __atomic_load_n(&notificationError, __ATOMIC_ACQUIRE);
  }
  flagcxSemaphore() = default;
  virtual ~flagcxSemaphore() { pthread_mutex_destroy(&notificationMutex); }

  virtual flagcxEvent_t getEvent() = 0;
  virtual void signalStart() = 0;
  virtual void *getSignals() = 0;
  virtual void subCounter(int opId = 0) = 0;
  virtual void addCounter(int opId = 0) = 0;
  virtual int getCounter() = 0;
  virtual int pollStart(int opId = 0, int step = 0) = 0;
  virtual int pollEnd() = 0;
  virtual void wait() = 0;
};

#define FLAGCX_OPS_PER_SEMAPHORE 64
#define FLAGCX_SIGNALS_PER_SEMAPHORE (2 * FLAGCX_OPS_PER_SEMAPHORE + 1)
#define FLAGCX_SIGNAL_CURSTEP_OFFSET 0
#define FLAGCX_SIGNAL_NSTEPS_OFFSET FLAGCX_OPS_PER_SEMAPHORE
#define FLAGCX_SIGNAL_COUNTER_OFFSET (2 * FLAGCX_OPS_PER_SEMAPHORE)

// Host semaphore derived class
struct flagcxHostSemaphore : public flagcxSemaphore {
  int callbacks = 0, callbackStarted = 0;
  struct flagcxGroupCallback *callbackContexts = NULL;
  int counter;                              // total ops
  std::unordered_map<int, int> stepInfo;    // opId -> sigalId
  std::vector<std::pair<int, int>> signals; // [curStep, nSteps]
  std::vector<flagcxEvent_t> events;
  bool frozen; // true during execution phase

  flagcxHostSemaphore() {
    counter = 0;
    frozen = false;
    stepInfo.reserve(FLAGCX_OPS_PER_SEMAPHORE);
    signals.reserve(FLAGCX_SIGNALS_PER_SEMAPHORE);
    events.reserve(FLAGCX_SIGNALS_PER_SEMAPHORE);
  }
  ~flagcxHostSemaphore() override { (void)release(); }
  flagcxResult_t release() override;
  flagcxResult_t drainNotifications() override;
  void runCallback();
  flagcxEvent_t getEvent() override {
    events.push_back(nullptr);
    auto &event = events.back();
    initResult = deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    if (initResult != flagcxSuccess)
      return NULL;
    return event;
  }
  void signalStart() override {
    frozen =
        true; // freeze: no more structural mutations until wait() completes
    for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
      __atomic_store_n(&signals[it->second].first, 0, __ATOMIC_RELEASE);
    }
  }
  void *getSignals() override { return nullptr; }
  void subCounter(int opId = 0) override {
    auto it = stepInfo.find(opId);
    assert(it != stepInfo.end());
    int idx = it->second;
    __atomic_fetch_add(&signals[idx].first, 1, __ATOMIC_RELEASE);
    TRACE(FLAGCX_PROXY,
          "SubCounter curStep[%d] = %d, nSteps[%d] = %d, counter %d", opId,
          signals[idx].first, opId, signals[idx].second, counter);
  }
  void addCounter(int opId = 0) override {
    assert(!frozen); // must not mutate during execution phase
    auto it = stepInfo.find(opId);
    if (it != stepInfo.end()) {
      __atomic_fetch_add(&signals[it->second].second, 1, __ATOMIC_RELEASE);
    } else {
      signals.emplace_back(-1, 1);
      stepInfo[opId] = (int)signals.size() - 1;
      __atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE);
    }
  }
  int getCounter() override { return counter; }
  int pollStart(int opId = 0, int step = 0) override {
    auto it = stepInfo.find(opId);
    assert(it != stepInfo.end());
    return (signals[it->second].first >= step);
  }
  int pollEnd() override {
    return isSealed() &&
           (isCancelled() ||
            __atomic_load_n(&counter, __ATOMIC_ACQUIRE) == 0) &&
           __atomic_load_n(&callbacks, __ATOMIC_ACQUIRE) == 0;
  }
  void wait() override {
    int nDone = 0;
    int nOps = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
    while (nDone < nOps && !isCancelled()) {
      for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
        if (__atomic_load_n(&signals[it->second].first, __ATOMIC_ACQUIRE) ==
            __atomic_load_n(&signals[it->second].second, __ATOMIC_ACQUIRE)) {
          __atomic_fetch_add(&signals[it->second].first, 1, __ATOMIC_RELEASE);
          nDone++;
        }
      }
      sched_yield();
    }
    frozen = false; // Last publication follows every access to semaphore state.
    __atomic_store_n(&counter, 0, __ATOMIC_RELEASE);
  }
};

// Used for flagcxDeviceSemaphore to manage a buffer pool
struct flagcxDeviceSemaphoreBufferPool {
  struct Chunk {
    Chunk *next;
    int first, capacity, allocated;
    int *signals, *inUse;
    void *dSignals;
  };
  int capacity;
  Chunk *chunks, *pending;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

  flagcxDeviceSemaphoreBufferPool();
  ~flagcxDeviceSemaphoreBufferPool();
  flagcxResult_t getSlotId(int *id);
  flagcxResult_t releaseSlot(int id);
  flagcxResult_t clear();
  int *getHostPtr(int id);
  void *getDevicePtr(int id);

private:
  flagcxResult_t grow();
  Chunk *findChunk(int id);
};
extern flagcxDeviceSemaphoreBufferPool deviceSemaphoreBufferPool;

struct flagcxStreamValueBufferPool {
  struct Chunk {
    int capacity;
    uint64_t *signals;
    void *dSignals;
    int *inUse;
  };

  std::mutex mutex;
  std::vector<Chunk> chunks;

  ~flagcxStreamValueBufferPool();
  flagcxResult_t reserve(size_t requiredFreeSlots);
  flagcxResult_t acquire(uint64_t **signals, void **dSignals);
  void release(uint64_t *signals);
  flagcxResult_t clear();

private:
  flagcxResult_t grow();
};
extern flagcxStreamValueBufferPool streamValueBufferPool;

// Probe on a private stream before group operations are submitted. Results are
// cached for the current adaptor and device until the adaptor is unloaded.
flagcxResult_t flagcxStreamValueSupported(bool *supported);
flagcxResult_t flagcxStreamValueProbeCleanup();
void flagcxClearStreamValueProbeCache();

// Device semaphore derived class
struct flagcxDeviceSemaphore : public flagcxSemaphore {
  int slotId;
  int opOffset;
  int *signals; // [curStep,...,nSteps,..., counter]
  void *dSignals;
  flagcxEvent_t headEvent;
  std::map<int, int> curStep; // current step of each op
  std::map<int, int> nSteps;  // total steps of each op
  std::vector<flagcxEvent_t> events;

  bool completionArmed = false, completionDrained = false;
  bool notificationAttempted = false;
  pthread_mutex_t signalMutex = PTHREAD_MUTEX_INITIALIZER;
  flagcxDeviceSemaphore() {
    slotId = -1;
    opOffset = 0;
    initResult = deviceSemaphoreBufferPool.getSlotId(&slotId);
    signals = initResult == flagcxSuccess
                  ? deviceSemaphoreBufferPool.getHostPtr(slotId)
                  : NULL;
    dSignals = initResult == flagcxSuccess
                   ? deviceSemaphoreBufferPool.getDevicePtr(slotId)
                   : NULL;
    headEvent = NULL;
    if (initResult == flagcxSuccess && (signals == NULL || dSignals == NULL))
      initResult = flagcxSystemError;
  }
  ~flagcxDeviceSemaphore() override {
    (void)release();
    pthread_mutex_destroy(&signalMutex);
  }
  flagcxResult_t release() override;
  void cancelNotifications(flagcxResult_t error = flagcxRemoteError) override;
  flagcxResult_t drainNotifications() override;
  flagcxResult_t captureCancellationCompletion(flagcxStream_t stream) override;
  flagcxEvent_t getEvent() override {
    events.push_back(NULL);
    flagcxEvent_t &event = events.back();
    initResult = deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    if (initResult != flagcxSuccess)
      return NULL;
    if (events.size() == 1)
      headEvent = event;
    return event;
  }
  // Since the device kernel handles the signaling,
  // host-side signalStart/End are intentionally no-op and not needed
  void signalStart() override {}
  void *getSignals() override { return dSignals; }
  void subCounter(int opId = 0) override {
    pthread_mutex_lock(&signalMutex);
    if (isCancelled()) {
      pthread_mutex_unlock(&signalMutex);
      return;
    }
    assert(curStep.find(opId) != curStep.end());
    assert(nSteps.find(opId) != nSteps.end());
    if (signals[curStep[opId]] + 1 == signals[nSteps[opId]]) {
      __atomic_fetch_sub(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, 1,
                         __ATOMIC_RELEASE);
    } else {
      __atomic_fetch_add(signals + curStep[opId], 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&signalMutex);
  }
  void addCounter(int opId = 0) override {
    if (nSteps.find(opId) != nSteps.end()) {
      __atomic_fetch_add(signals + nSteps[opId], 1, __ATOMIC_RELEASE);
    } else {
      // Make sure that opOffset is not used up
      assert(opOffset < FLAGCX_OPS_PER_SEMAPHORE);
      curStep[opId] = FLAGCX_SIGNAL_CURSTEP_OFFSET + opOffset;
      nSteps[opId] = FLAGCX_SIGNAL_NSTEPS_OFFSET + opOffset;
      opOffset++;
      __atomic_store_n(signals + curStep[opId], -1, __ATOMIC_RELEASE);
      __atomic_store_n(signals + nSteps[opId], 1, __ATOMIC_RELEASE);
      __atomic_fetch_add(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, 1,
                         __ATOMIC_RELEASE);
    }
  }
  int getCounter() override {
    return __atomic_load_n(signals + FLAGCX_SIGNAL_COUNTER_OFFSET,
                           __ATOMIC_ACQUIRE);
  }
  int pollStart(int opId = 0, int step = 0) override {
    assert(curStep.find(opId) != curStep.end());
    return (__atomic_load_n(signals + curStep[opId], __ATOMIC_ACQUIRE) >= step);
  }
  int pollEnd() override {
    if (!isSealed())
      return 0;
    if (__atomic_load_n(&completionDrained, __ATOMIC_ACQUIRE))
      return 1;
    if (isCancelled())
      return 0;
    if (notificationStreams.empty())
      return 1;
    if (!__atomic_load_n(&completionArmed, __ATOMIC_ACQUIRE))
      return 0;
    if (getCounter() != 0)
      return 0;
    flagcxResult_t result = deviceAdaptor->eventQuery(headEvent);
    recordNotificationError(result);
    return result == flagcxSuccess;
  }
  // Since the device kernel handles the signaling,
  // host-side wait is intentionally no-op and not needed
  void wait() override {}
};

struct flagcxStreamValueSemaphore : public flagcxSemaphore {
  int counter;
  uint64_t *signals;
  void *dSignals;
  flagcxEvent_t completionEvent;
  // 0: not recorded, 1: recorded, 2: failed submission safely drained.
  int completionState;
  bool completionUnsafe;
  std::map<int, int> stepInfo;
  std::vector<std::pair<int, int>> steps;
  std::vector<flagcxEvent_t> events;
  // A group can be polled by several proxy threads. Contention only drops a
  // diagnostic snapshot; it must never delay progress or semaphore retirement.
  std::mutex traceMutex;
  bool traceSeen = false;
  std::array<uint64_t, 6> traceSnapshot{};
  std::chrono::steady_clock::time_point traceTime{};

  flagcxStreamValueSemaphore()
      : counter(0), signals(nullptr), dSignals(nullptr),
        completionEvent(nullptr), completionState(0), completionUnsafe(false) {
    initResult = streamValueBufferPool.acquire(&signals, &dSignals);
  }
  ~flagcxStreamValueSemaphore() override { (void)release(); }
  flagcxResult_t release() override;
  flagcxResult_t drainNotifications() override;
  void cancelNotifications(flagcxResult_t error = flagcxRemoteError) override;
  flagcxResult_t captureCancellationCompletion(flagcxStream_t stream) override;

  flagcxResult_t enqueueCompletion(flagcxStream_t launchStream);
  void traceCompletion(const char *reason, int remaining, int state,
                       bool queried, flagcxResult_t queryResult);
  void markCompletionDrained() {
    __atomic_store_n(&completionUnsafe, false, __ATOMIC_RELEASE);
    __atomic_store_n(&completionState, 2, __ATOMIC_RELEASE);
  }
  void markCompletionUnsafe() { completionUnsafe = true; }

  flagcxEvent_t getEvent() override {
    events.push_back(nullptr);
    auto &event = events.back();
    initResult = deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    if (initResult != flagcxSuccess)
      return NULL;
    return event;
  }
  void signalStart() override {}
  void *getSignals() override { return dSignals; }
  void subCounter(int opId = 0) override;
  void addCounter(int opId = 0) override {
    auto it = stepInfo.find(opId);
    if (it != stepInfo.end()) {
      __atomic_fetch_add(&steps[it->second].second, 1, __ATOMIC_RELEASE);
    } else {
      stepInfo[opId] = static_cast<int>(steps.size());
      steps.emplace_back(0, 1);
      __atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE);
    }
  }
  int getCounter() override {
    return __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
  }
  int pollStart(int opId = 0, int step = 0) override {
    auto it = stepInfo.find(opId);
    assert(it != stepInfo.end());
    return __atomic_load_n(signals, __ATOMIC_ACQUIRE) >= 1 &&
           __atomic_load_n(&steps[it->second].first, __ATOMIC_ACQUIRE) >= step;
  }
  int pollEnd() override;
  void wait() override {}
};

flagcxResult_t flagcxReleaseSemaphore(std::shared_ptr<flagcxSemaphore> *owner);
void cpuAsyncKernel(void *args);
extern flagcxLaunchFunc_t deviceAsyncKernel;

#endif
