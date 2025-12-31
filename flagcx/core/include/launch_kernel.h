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

#ifdef __cplusplus
extern "C" {
#endif

flagcxResult_t loadKernelSymbol(const char *path, const char *name,
                                flagcxLaunchFunc_t *fn);

#ifdef __cplusplus
}
#endif

struct flagcxSemaphore {
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

  flagcxHostSemaphore() {
    counter = 0;
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
    deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    return event;
  }
  void signalStart() override {
    for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
      __atomic_store_n(&signals[it->second].first, 0, __ATOMIC_RELEASE);
    }
  }
  void *getSignals() override { return nullptr; }
  void subCounter(int opId = 0) override {
    assert(stepInfo.find(opId) != stepInfo.end());
    __atomic_fetch_add(&signals[stepInfo[opId]].first, 1, __ATOMIC_RELEASE);
    INFO(FLAGCX_PROXY,
         "SubCounter curStep[%d] = %d, nSteps[%d] = %d, counter %d", opId,
         signals[stepInfo[opId]].first, opId, signals[stepInfo[opId]].second,
         counter);
  }
  void addCounter(int opId = 0) override {
    if (stepInfo.find(opId) != stepInfo.end()) {
      __atomic_fetch_add(&signals[stepInfo[opId]].second, 1, __ATOMIC_RELEASE);
    } else {
      signals.emplace_back(-1, 1);
      stepInfo[opId] = (int)signals.size() - 1;
      __atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE);
    }
  }
  int getCounter() override { return counter; }
  int pollStart(int opId = 0, int step = 0) override {
    assert(stepInfo.find(opId) != stepInfo.end());
    return (signals[stepInfo[opId]].first >= step);
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
  int getSlotId();
  void initialize();
  void setEvent(int id, flagcxEvent_t event);
  int *getHostPtr(int id);
  void *getDevicePtr(int id);
};
static flagcxDeviceSemaphoreBufferPool deviceSemaphoreBufferPool;

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
    if (deviceSemaphoreBufferPool.capacity == -1) {
      deviceSemaphoreBufferPool.initialize();
    }
    opOffset = 0;
    slotId = deviceSemaphoreBufferPool.getSlotId();
    signals = deviceSemaphoreBufferPool.getHostPtr(slotId);
    dSignals = deviceSemaphoreBufferPool.getDevicePtr(slotId);
    headEvent = nullptr;
  }
  ~flagcxDeviceSemaphore() override {
    // Clear event in the pool
    deviceSemaphoreBufferPool.setEvent(slotId, nullptr);
    for (auto event : events) {
      deviceAdaptor->eventDestroy(event);
    }
  }
  flagcxEvent_t getEvent() override {
    events.push_back(nullptr);
    auto &event = events.back();
    deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
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

void cpuAsyncKernel(void *args);
extern flagcxLaunchFunc_t deviceAsyncKernel;

#endif