/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_KERNEL_PROXY_COMPLETION_H_
#define FLAGCX_KERNEL_PROXY_COMPLETION_H_

#include "flagcx_kernel_core.h"

// Host-only checked arithmetic shared by allocation and FIFO consumption.
static inline bool flagcxKernelCompletionBytes(size_t contexts, size_t count,
                                               size_t *bytes) {
  if (contexts == 0 || count > flagcxDeviceTriggerCompletionCapacity ||
      contexts > SIZE_MAX / sizeof(uint64_t) ||
      count > SIZE_MAX / sizeof(uint64_t) / contexts)
    return false;
  *bytes = contexts * count * sizeof(uint64_t);
  return true;
}

static inline bool flagcxKernelCompletionLayout(
    int userSignals, int barrierChannels, int teamRanks, int counters,
    int contexts, int *signalCount, size_t *signalBytes, size_t *counterBytes) {
  if (userSignals < 0 || barrierChannels < 0 || teamRanks < 0 || counters < 0 ||
      contexts <= 0 || (barrierChannels > 0 && teamRanks == 0))
    return false;
  uint64_t signals =
      uint64_t(userSignals) + uint64_t(barrierChannels) * uint64_t(teamRanks);
  if (signals > flagcxDeviceTriggerCompletionCapacity ||
      !flagcxKernelCompletionBytes(size_t(contexts), size_t(signals),
                                   signalBytes) ||
      !flagcxKernelCompletionBytes(size_t(contexts), size_t(counters),
                                   counterBytes))
    return false;
  *signalCount = int(signals);
  return true;
}

static inline flagcxResult_t
flagcxKernelProxyValidateFormat(flagcxDeviceTrigger &trigger) {
  return !flagcxDeviceTriggerUsesVersionedLayout(trigger.getPrim()) ||
                 trigger.getFormatVersion() == flagcxDeviceTriggerFormatVersion
             ? flagcxSuccess
             : flagcxInvalidArgument;
}

// The FIFO identifies the context. Its completion ID never contains a context
// stride, including for fused PutSignal and barrier-generated Signal entries.
static inline flagcxResult_t
flagcxKernelProxyCompletionOffset(flagcxDeviceTrigger &trigger, int contextId,
                                  int contextCount, int signalCount,
                                  int counterCount, size_t *byteOffset) {
  uint64_t prim = trigger.getPrim();
  if ((prim != flagcxDevicePrimPutSignal && prim != flagcxDevicePrimSignal &&
       prim != flagcxDevicePrimSignalValue) ||
      flagcxKernelProxyValidateFormat(trigger) != flagcxSuccess ||
      contextId < 0 || contextId >= contextCount)
    return flagcxInvalidArgument;
  uint64_t bufferType = trigger.getBufferType();
  if (bufferType > 1 || (prim == flagcxDevicePrimPutSignal && bufferType != 0))
    return flagcxInvalidArgument;
  int count = bufferType == 0 ? signalCount : counterCount;
  uint64_t slot = trigger.getSignalIdx();
  size_t contextBytes = 0;
  if (count <= 0 || slot >= uint64_t(count) ||
      !flagcxKernelCompletionBytes(size_t(contextCount), size_t(count),
                                   &contextBytes))
    return flagcxInvalidArgument;
  *byteOffset =
      (size_t(contextId) * size_t(count) + size_t(slot)) * sizeof(uint64_t);
  return flagcxSuccess;
}

#endif // FLAGCX_KERNEL_PROXY_COMPLETION_H_
