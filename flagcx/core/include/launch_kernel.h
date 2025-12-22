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

// Host semaphore derived class
struct flagcxHostSemaphore : public flagcxSemaphore {
  int counter;                // total ops
  std::map<int, int> curStep; // current step of each op
  std::map<int, int> nSteps;  // total steps of each op
  std::mutex mapMutex;
  std::vector<flagcxEvent_t> events;

  flagcxHostSemaphore() : counter(0) {}
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
    std::lock_guard<std::mutex> lock(mapMutex);
    for (auto it = curStep.begin(); it != curStep.end(); ++it) {
      __atomic_store_n(&it->second, 0, __ATOMIC_RELEASE);
    }
    // printf("counter = %d\n", counter);
    // for (auto it = curStep.begin(); it != curStep.end(); ++it) {
    //   printf("curStep[%d] = %d, nSteps[%d] = %d\n", it->first, it->second,
    //          it->first, nSteps[it->first]);
    // }
  }
  void *getSignals() override { return nullptr; }
  void subCounter(int opId = 0) override {
    std::lock_guard<std::mutex> lock(mapMutex);
    // printf("Enter SubCounter curStep[%d] = %d, nSteps[%d] = %d, counter =
    // %d\n",
    //        opId, curStep[opId], opId, nSteps[opId], counter);
    assert(curStep.find(opId) != curStep.end());
    assert(nSteps.find(opId) != nSteps.end());
    if (curStep[opId] + 1 == nSteps[opId]) {
      __atomic_fetch_sub(&counter, 1, __ATOMIC_RELEASE);
      // printf(
      //     "Next SubCounter curStep[%d] = %d, nSteps[%d] = %d, counter =
      //     %d\n", opId, curStep[opId], opId, nSteps[opId], counter);
    } else {
      // printf(
      //     "Before SubCounter curStep[%d] = %d, nSteps[%d] = %d, counter =
      //     %d\n", opId, curStep[opId], opId, nSteps[opId], counter);
      __atomic_fetch_add(&curStep[opId], 1, __ATOMIC_RELEASE);
      // printf(
      //     "After SubCounter curStep[%d] = %d, nSteps[%d] = %d, counter =
      //     %d\n", opId, curStep[opId], opId, nSteps[opId], counter);
    }
    // for (auto it = curStep.begin(); it != curStep.end(); ++it) {
    //   printf("SubCounter curStep[%d] = %d, nSteps[%d] = %d\n", it->first,
    //   it->second, it->first, nSteps[it->first]);
    // }
  }
  void addCounter(int opId = 0) override {
    std::lock_guard<std::mutex> lock(mapMutex);
    if (nSteps.find(opId) != nSteps.end()) {
      __atomic_fetch_add(&nSteps[opId], 1, __ATOMIC_RELEASE);
    } else {
      curStep[opId] = 0;
      nSteps[opId] = 0;
      __atomic_store_n(&curStep[opId], -1, __ATOMIC_RELEASE);
      __atomic_store_n(&nSteps[opId], 1, __ATOMIC_RELEASE);
      __atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE);
    }
  }
  int getCounter() override { return counter; }
  int pollStart(int opId = 0, int step = 0) override {
    // printf("PollStart curStep[%d] = %d, nSteps[%d] = %d, counter = %d, step =
    // %d\n", opId, curStep[opId], opId, nSteps[opId], counter, step);
    return (__atomic_load_n(&curStep[opId], __ATOMIC_ACQUIRE) == step);
  }
  int pollEnd() override {
    return (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) == 0);
  }
  void wait() override {
    while (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) > 0) {
      // printf("Waiting, counter = %d\n", counter);
      sched_yield();
    }
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

#define FLAGCX_OPS_PER_SEMAPHORE 16
#define FLAGCX_SIGNALS_PER_SEMAPHORE (2 * FLAGCX_OPS_PER_SEMAPHORE + 1)
#define FLAGCX_SIGNAL_CURSTEP_OFFSET 0
#define FLAGCX_SIGNAL_NSTRPS_OFFSET FLAGCX_OPS_PER_SEMAPHORE
#define FLAGCX_SIGNAL_COUNTER_OFFSET (2 * FLAGCX_OPS_PER_SEMAPHORE)
// Device semaphore derived class
struct flagcxDeviceSemaphore : public flagcxSemaphore {
  int slotId;
  int opOffset;
  int *signals; // [curStep,...,nSteps,..., counter]
  void *dSignals;
  flagcxEvent_t headEvent;
  std::map<int, int> curStep; // current step of each op
  std::map<int, int> nSteps;  // total steps of each op
  std::mutex mapMutex;
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
    std::lock_guard<std::mutex> lock(mapMutex);
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
    std::lock_guard<std::mutex> lock(mapMutex);
    if (nSteps.find(opId) != nSteps.end()) {
      __atomic_fetch_add(signals + nSteps[opId], 1, __ATOMIC_RELEASE);
    } else {
      assert(opOffset < FLAGCX_OPS_PER_SEMAPHORE);
      curStep[opId] = FLAGCX_SIGNAL_CURSTEP_OFFSET + opOffset;
      nSteps[opId] = FLAGCX_SIGNAL_NSTRPS_OFFSET + opOffset;
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
    return (__atomic_load_n(signals + curStep[opId], __ATOMIC_ACQUIRE) == step);
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