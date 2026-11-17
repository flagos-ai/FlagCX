/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * MUSA collective-kernel launcher for the unified runner.
 * The implementation follows the FlagCX default path and dispatches platform
 * operations through DeviceAPI::Atomic and DeviceAPI::Intrin.
 ************************************************************************/

#include "device_api/comm_traits.h"
#include "flagcx.h"
#include "flagcx_kernel_internal.h"

#define SLOT_IDX 4
#define FST_IDX 5
#define SND_IDX 6
#define OUT_IDX 7
#define COUNT_IDX 8
#define NTHREADS_IDX 9
#define DATATYPE_IDX 10
#define REDOP_IDX 11

FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getInput1() {
  return value[0];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getInput2() {
  return value[1];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getOutput() {
  return value[2];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getCount() {
  return value[3] >> flagcxReduceTriggerOffCount &
         flagcxTriggerMask(flagcxReduceTriggerBitsCount);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getNThreads() {
  return value[3] >> flagcxReduceTriggerOffNThreads &
         flagcxTriggerMask(flagcxReduceTriggerBitsNThreads);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getDatatype() {
  return value[3] >> flagcxReduceTriggerOffDatatype &
         flagcxTriggerMask(flagcxReduceTriggerBitsDatatype);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getRedop() {
  return value[3] >> flagcxReduceTriggerOffRedop &
         flagcxTriggerMask(flagcxReduceTriggerBitsRedop);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getState() {
  return value[3] >> flagcxReduceTriggerOffState &
         flagcxTriggerMask(flagcxReduceTriggerBitsState);
}
FLAGCX_DEVICE_INLINE_DECORATOR void flagcxReduceTrigger::setComplete() {
  DeviceAPI::Atomic::fetchOr(
      reinterpret_cast<uint64_t *>(value) + 3,
      (uint64_t)((flagcxReduceTriggerComplete &
                  flagcxTriggerMask(flagcxReduceTriggerBitsState))
                 << flagcxReduceTriggerOffState),
      flagcxDeviceMemoryOrderRelease);
}

// Match the host's MUSA DefaultBackend control-word width. Do not inherit
// the CoreX platform's low-32-bit FIFO control representation.
static_assert(sizeof(DeviceAPI::CompletionWord) == sizeof(uint64_t),
              "MUSA collective FIFO requires matching 64-bit host controls");

FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *fifoControlWord(uint64_t *buffer,
                                                         int index) {
  return buffer + index;
}

FLAGCX_DEVICE_INLINE_DECORATOR bool musaDequeue(uint64_t *buffer, uint64_t *idx) {
  for (;;) {
    uint64_t oldConsumed = DeviceAPI::Atomic::load(
        fifoControlWord(buffer, flagcxFifoIdxConsumed),
        flagcxDeviceMemoryOrderAcquire);
    const uint64_t curProduced = DeviceAPI::Atomic::load(
        fifoControlWord(buffer, flagcxFifoIdxProduced),
        flagcxDeviceMemoryOrderAcquire);
    if (oldConsumed == curProduced)
      return false;
    uint64_t expected = oldConsumed;
    if (DeviceAPI::Atomic::compareExchange(
            fifoControlWord(buffer, flagcxFifoIdxConsumed), expected,
            oldConsumed + uint64_t{1}, flagcxDeviceMemoryOrderAcqRel)) {
      *idx = oldConsumed;
      return true;
    }
  }
}

FLAGCX_DEVICE_DECORATOR void
flagcxReduceKernel(uint64_t fst, uint64_t snd, uint64_t out, uint64_t count,
                   uint64_t nthreads, uint64_t datatype, uint64_t redOp) {
  // This baseline supports float/Sum, matching the reference implementation.
  // Host enqueue validates that contract before publishing a FIFO descriptor.
  // Other datatypes/ops return flagcxNotSupported on the host, not false success.
  int tid = FLAGCX_THREAD_IDX_X;
  float *fstPtr = reinterpret_cast<float *>(fst);
  float *sndPtr = reinterpret_cast<float *>(snd);
  float *outPtr = reinterpret_cast<float *>(out);
  for (uint64_t i = static_cast<uint64_t>(tid); i < count; i += nthreads)
    outPtr[i] = fstPtr[i] + sndPtr[i];
}

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(1024) flagcxCollectiveKernel(void *fifoBuffer) {
  FLAGCX_SHARED uint64_t shm[16];
  uint64_t *vBuf = static_cast<uint64_t *>(fifoBuffer);
  int emptyIter = 0;
  int cap = -1;
  uint64_t c = 0;
  uint64_t p = 0;
  int term = -1;
  int slot = -1;
  int tid = FLAGCX_THREAD_IDX_X;

  if (tid == 0)
    shm[flagcxFifoIdxCapacity] = vBuf[flagcxFifoIdxCapacity];
  FLAGCX_DEVICE_SYNC_THREADS();
  cap = shm[flagcxFifoIdxCapacity];

  while (true) {
    if (tid == 0) {
      shm[flagcxFifoIdxConsumed] = DeviceAPI::Atomic::load(
          fifoControlWord(vBuf, flagcxFifoIdxConsumed),
          flagcxDeviceMemoryOrderAcquire);
      shm[flagcxFifoIdxProduced] = DeviceAPI::Atomic::load(
          fifoControlWord(vBuf, flagcxFifoIdxProduced),
          flagcxDeviceMemoryOrderAcquire);
      shm[flagcxFifoIdxTerminate] = DeviceAPI::Atomic::load(
          fifoControlWord(vBuf, flagcxFifoIdxTerminate),
          flagcxDeviceMemoryOrderAcquire);
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    c = static_cast<uint64_t>(shm[flagcxFifoIdxConsumed]);
    p = static_cast<uint64_t>(shm[flagcxFifoIdxProduced]);
    term = shm[flagcxFifoIdxTerminate];

    // consumed/produced are modulo-2^64 sequence numbers. The producer keeps
    // the outstanding distance below the FIFO capacity, so equality is the
    // only valid empty test; relational comparisons fail across wraparound.
    if (c == p) {
      if (term == 1)
        break;
      DeviceAPI::Intrin::spinBackoff(++emptyIter);
      continue;
    }

    if (tid == 0) {
      uint64_t myIdx = 0;
      bool hasWork = musaDequeue(vBuf, &myIdx);
      slot = (int)(myIdx % static_cast<uint64_t>(cap));
      shm[SLOT_IDX] = hasWork ? slot : cap;
      if (hasWork) {
        flagcxReduceTrigger *trigger =
            reinterpret_cast<flagcxReduceTrigger *>(vBuf + flagcxFifoIdxData) +
            slot;
        shm[FST_IDX] = trigger->getInput1();
        shm[SND_IDX] = trigger->getInput2();
        shm[OUT_IDX] = trigger->getOutput();
        shm[COUNT_IDX] = trigger->getCount();
        shm[NTHREADS_IDX] = trigger->getNThreads();
        shm[DATATYPE_IDX] = trigger->getDatatype();
        shm[REDOP_IDX] = trigger->getRedop();
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();

    slot = shm[SLOT_IDX];
    if (slot == cap) {
      if (term == 1)
        break;
      DeviceAPI::Intrin::spinBackoff(++emptyIter);
      continue;
    }

    emptyIter = 0;
    flagcxReduceKernel(shm[FST_IDX], shm[SND_IDX], shm[OUT_IDX], shm[COUNT_IDX],
                       shm[NTHREADS_IDX], shm[DATATYPE_IDX], shm[REDOP_IDX]);
    FLAGCX_DEVICE_SYNC_THREADS();
    FLAGCX_DEVICE_THREAD_FENCE();

    if (tid == 0) {
      flagcxReduceTrigger *trigger =
          reinterpret_cast<flagcxReduceTrigger *>(vBuf + flagcxFifoIdxData) +
          slot;
      trigger->setComplete();
    }
  }
}

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream) {
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(fifoBuffer);
}
