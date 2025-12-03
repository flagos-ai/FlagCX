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
  virtual void signalEnd() = 0;
  virtual void *getSignals() = 0;
  virtual void subCounter(int value) = 0;
  virtual void addCounter(int value) = 0;
  virtual int getCounter() = 0;
  virtual int pollStart() = 0;
  virtual int pollEnd() = 0;
  virtual void wait() = 0;
};

// Host semaphore derived class
struct flagcxHostSemaphore : public flagcxSemaphore {
  int start;   // started or not
  int end;     // ended or not
  int counter; // total operations to wait for inside the group;
  std::vector<flagcxEvent_t> events;

  flagcxHostSemaphore() : start(0), end(0), counter(0) {}
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
  void signalStart() override { __atomic_store_n(&start, 1, __ATOMIC_RELEASE); }
  void signalEnd() override { __atomic_store_n(&end, 1, __ATOMIC_RELEASE); }
  void *getSignals() override { return nullptr; }
  void subCounter(int value) override {
    __atomic_fetch_sub(&counter, value, __ATOMIC_RELEASE);
  }
  void addCounter(int value) override {
    __atomic_fetch_add(&counter, value, __ATOMIC_RELEASE);
  }
  int getCounter() override { return counter; }
  int pollStart() override { return __atomic_load_n(&start, __ATOMIC_ACQUIRE); }
  int pollEnd() override { return __atomic_load_n(&end, __ATOMIC_ACQUIRE); }
  void wait() override {
    while (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) > 0) {
      sched_yield();
    }
  }
};

// Used for flagcxDeviceSemaphore to manage a buffer pool
struct flagcxDeviceSemaphoreBufferPool {
  int capacity;          // total slots
  int slotId;            // slot index in the pool
  int *signalsPool;      // Host-mapped memory region, [start, end, counter] *
                         // capacity
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

#define FLAGCX_SIGNALS_PER_SEMAPHORE 3
#define FLAGCX_SIGNAL_START_OFFSET 0
#define FLAGCX_SIGNAL_END_OFFSET 1
#define FLAGCX_SIGNAL_COUNTER_OFFSET 2
// Device semaphore derived class
struct flagcxDeviceSemaphore : public flagcxSemaphore {
  int slotId;
  int *signals; // [start, end, counter]
  void *dSignals;
  flagcxEvent_t headEvent;
  std::vector<flagcxEvent_t> events;

  flagcxDeviceSemaphore() {
    if (deviceSemaphoreBufferPool.capacity == -1) {
      deviceSemaphoreBufferPool.initialize();
    }
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
  void signalEnd() override {}
  void *getSignals() override { return dSignals; }
  void subCounter(int value) override {
    __atomic_fetch_sub(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, value,
                       __ATOMIC_RELEASE);
  }
  void addCounter(int value) override {
    __atomic_fetch_add(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, value,
                       __ATOMIC_RELEASE);
  }
  int getCounter() override {
    return __atomic_load_n(signals + FLAGCX_SIGNAL_COUNTER_OFFSET,
                           __ATOMIC_ACQUIRE);
  }
  int pollStart() override {
    return __atomic_load_n(signals + FLAGCX_SIGNAL_START_OFFSET,
                           __ATOMIC_ACQUIRE);
  }
  int pollEnd() override {
    return __atomic_load_n(signals + FLAGCX_SIGNAL_END_OFFSET,
                           __ATOMIC_ACQUIRE);
  }
  // Since the device kernel handles the signaling,
  // host-side wait is intentionally no-op and not needed
  void wait() override {}
};

void cpuAsyncKernel(void *args);
extern flagcxLaunchFunc_t deviceAsyncKernel;

#endif