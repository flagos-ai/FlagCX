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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct flagcxSemaphore {
  flagcxResult_t initResult = flagcxSuccess;
  flagcxSemaphore() = default;
  virtual ~flagcxSemaphore() = default;

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
  ~flagcxHostSemaphore() override {
    for (auto event : events) {
      deviceAdaptor->eventDestroy(event);
    }
  }
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
    return (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) == 0);
  }
  void wait() override {
    int nDone = 0;
    int nOps = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
    while (nDone < nOps) {
      for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
        if (__atomic_load_n(&signals[it->second].first, __ATOMIC_ACQUIRE) ==
            __atomic_load_n(&signals[it->second].second, __ATOMIC_ACQUIRE)) {
          __atomic_fetch_add(&signals[it->second].first, 1, __ATOMIC_RELEASE);
          nDone++;
        }
      }
      sched_yield();
    }
    __atomic_store_n(&counter, 0, __ATOMIC_RELEASE);
    frozen = false; // unfreeze: allow addCounter for next round
  }
};

// Used for flagcxDeviceSemaphore to manage a buffer pool
struct flagcxDeviceSemaphoreBufferPool {
  int capacity;          // total slots
  int slotId;            // slot index in the pool
  int *signalsPool;      // Host-mapped memory region
  void *dSignalsPool;    // Device alias
  flagcxEvent_t *events; // store first event of each semaphore

  flagcxDeviceSemaphoreBufferPool();
  ~flagcxDeviceSemaphoreBufferPool();
  flagcxResult_t getSlotId(int *id);
  flagcxResult_t initialize();
  flagcxResult_t clear();
  void setEvent(int id, flagcxEvent_t event);
  int *getHostPtr(int id);
  void *getDevicePtr(int id);
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

  flagcxDeviceSemaphore() {
    slotId = -1;
    signals = NULL;
    dSignals = NULL;
    headEvent = NULL;
    opOffset = 0;
    initResult = deviceSemaphoreBufferPool.initialize();
    if (initResult != flagcxSuccess)
      return;
    initResult = deviceSemaphoreBufferPool.getSlotId(&slotId);
    if (initResult != flagcxSuccess)
      return;
    signals = deviceSemaphoreBufferPool.getHostPtr(slotId);
    dSignals = deviceSemaphoreBufferPool.getDevicePtr(slotId);
    headEvent = nullptr;
  }
  ~flagcxDeviceSemaphore() override {
    // Clear event in the pool
    if (slotId >= 0)
      deviceSemaphoreBufferPool.setEvent(slotId, nullptr);
    for (auto event : events) {
      deviceAdaptor->eventDestroy(event);
    }
  }
  flagcxEvent_t getEvent() override {
    events.push_back(nullptr);
    auto &event = events.back();
    initResult = deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    if (initResult != flagcxSuccess)
      return NULL;
    // Set the first event to the pool
    if (events.size() == 1) {
      headEvent = event;
      deviceSemaphoreBufferPool.setEvent(slotId, event);
    }
    return event;
  }
  // Since the device kernel handles the signaling,
  // host-side signalStart/End are intentionally no-op and not needed
  void signalStart() override {}
  void *getSignals() override { return dSignals; }
  void subCounter(int opId = 0) override {
    assert(curStep.find(opId) != curStep.end());
    assert(nSteps.find(opId) != nSteps.end());
    if (signals[curStep[opId]] + 1 == signals[nSteps[opId]]) {
      __atomic_fetch_sub(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, 1,
                         __ATOMIC_RELEASE);
    } else {
      __atomic_fetch_add(signals + curStep[opId], 1, __ATOMIC_RELEASE);
    }
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
    return (__atomic_load_n(signals + FLAGCX_SIGNAL_COUNTER_OFFSET,
                            __ATOMIC_ACQUIRE) == 0);
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
  ~flagcxStreamValueSemaphore() override {
    // A failed synchronization may leave GPU work using the event or slot.
    // Keep both allocated so the pool cannot hand the slot out again.
    if (completionUnsafe)
      return;
    if (completionEvent != nullptr) {
      deviceAdaptor->eventDestroy(completionEvent);
    }
    for (auto event : events) {
      deviceAdaptor->eventDestroy(event);
    }
    streamValueBufferPool.release(signals);
  }

  flagcxResult_t enqueueCompletion(flagcxStream_t launchStream);
  void traceCompletion(const char *reason, int remaining, int state,
                       bool queried, flagcxResult_t queryResult);
  void markCompletionDrained() {
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

void cpuAsyncKernel(void *args);
extern flagcxLaunchFunc_t deviceAsyncKernel;

#endif
