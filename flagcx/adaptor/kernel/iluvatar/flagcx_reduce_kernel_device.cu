/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Iluvatar/CoreX collective-kernel launcher for the unified runner.
 * The implementation follows the CUDA default path and dispatches platform
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
  // CoreX ivcore11 has no 64-bit RMW.  State lives in bits 57:56 of value[3],
  // so update only the containing high uint32_t word.
  uint32_t *stateWord = reinterpret_cast<uint32_t *>(value + 3) + 1;
  constexpr unsigned stateWordOffset = flagcxReduceTriggerOffState - 32;
  DeviceAPI::Atomic::fetchOr(
      stateWord,
      (uint32_t)((flagcxReduceTriggerComplete &
                  flagcxTriggerMask(flagcxReduceTriggerBitsState))
                 << stateWordOffset),
      flagcxDeviceMemoryOrderRelease);
}

FLAGCX_DEVICE_INLINE_DECORATOR uint32_t *fifoControlWord(uint64_t *buffer,
                                                         int index) {
  // FIFO control slots remain physically uint64_t for ABI/layout stability;
  // CoreX owns and atomically updates their low 32-bit word.
  return reinterpret_cast<uint32_t *>(buffer + index);
}

FLAGCX_DEVICE_INLINE_DECORATOR bool dequeue(uint64_t *buffer, uint32_t *idx) {
  while (true) {
    uint32_t oldConsumed = *fifoControlWord(buffer, flagcxFifoIdxConsumed);
    uint32_t curProduced = *fifoControlWord(buffer, flagcxFifoIdxProduced);
    if (oldConsumed == curProduced)
      return false;
    uint32_t expected = oldConsumed;
    if (DeviceAPI::Atomic::compareExchange(
            fifoControlWord(buffer, flagcxFifoIdxConsumed), expected,
            oldConsumed + 1, flagcxDeviceMemoryOrderAcqRel)) {
      *idx = oldConsumed;
      return true;
    }
  }
}

FLAGCX_DEVICE_DECORATOR void
flagcxReduceKernel(uint64_t fst, uint64_t snd, uint64_t out, uint64_t count,
                   uint64_t nthreads, uint64_t datatype, uint64_t redOp) {
  // Keep parity with the current CUDA default-path kernel. Vendor-specialized
  // datatype/reduction dispatch can replace this baseline independently.
  int tid = FLAGCX_THREAD_IDX_X;
  float *fstPtr = reinterpret_cast<float *>(fst);
  float *sndPtr = reinterpret_cast<float *>(snd);
  float *outPtr = reinterpret_cast<float *>(out);
  for (uint64_t i = static_cast<uint64_t>(tid); i < count; i += nthreads)
    outPtr[i] = fstPtr[i] + sndPtr[i];
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer) {
  FLAGCX_SHARED uint64_t shm[16];
  uint64_t *vBuf = static_cast<uint64_t *>(fifoBuffer);
  int emptyIter = 0;
  int cap = -1;
  uint32_t c = 0;
  uint32_t p = 0;
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
    c = static_cast<uint32_t>(shm[flagcxFifoIdxConsumed]);
    p = static_cast<uint32_t>(shm[flagcxFifoIdxProduced]);
    term = shm[flagcxFifoIdxTerminate];

    // consumed/produced are modulo-2^32 sequence numbers. The producer keeps
    // the outstanding distance below the FIFO capacity, so equality is the
    // only valid empty test; relational comparisons fail across wraparound.
    if (c == p) {
      if (term == 1)
        break;
      DeviceAPI::Intrin::spinBackoff(++emptyIter);
      continue;
    }

    if (tid == 0) {
      uint32_t myIdx = 0;
      bool hasWork = dequeue(vBuf, &myIdx);
      slot = (int)(myIdx % (uint32_t)cap);
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
