#ifndef FLAGCX_LAUNCH_KERNEL_H_
#define FLAGCX_LAUNCH_KERNEL_H_
#pragma once
#include "adaptor.h"
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

extern flagcxLaunchFunc_t deviceAsyncStore;
extern flagcxLaunchFunc_t deviceAsyncLoad;

void cpuAsyncStore(void *args);
void cpuAsyncLoad(void *args);
void cpuAsyncLoadWithMaxSpinCount(void *args);

struct flagcxHostSemaphore {
  int start = 0;   // started or not
  int end = 0;     // ended or not
  int counter = 0; // total operations to wait for inside the group
  std::vector<flagcxEvent_t> events;

  ~flagcxHostSemaphore() {
    for (auto event : events) {
      deviceAdaptor->eventDestroy(event);
    }
  }
  flagcxEvent_t getEvent() {
    events.push_back(nullptr);
    auto &event = events.back();
    deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    return event;
  }
  void signalStart() { __atomic_store_n(&start, 1, __ATOMIC_RELEASE); }
  void signalEnd() { __atomic_store_n(&end, 1, __ATOMIC_RELEASE); }
  void signalCounter(int value) {
    __atomic_fetch_sub(&counter, value, __ATOMIC_RELEASE);
  }
  int pollStart() { return __atomic_load_n(&start, __ATOMIC_ACQUIRE); }
  int pollEnd() { return __atomic_load_n(&end, __ATOMIC_ACQUIRE); }
  void wait() {
    while (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) > 0) {
      sched_yield();
    }
  }
};

void cpuAsyncKernel(void *args);

#endif