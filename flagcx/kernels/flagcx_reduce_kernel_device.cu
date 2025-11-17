#include "flagcx.h"
#include "flagcx_kernel.h"

#define WARP_SIZE 32

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
FLAGCX_DEVICE_INLINE_DECORATOR void setComplete() {
  value[3] |= (((uint64_t)flagcxReduceTriggerComplete &
                flagcxTriggerMask(flagcxReduceTriggerBitsState))
               << flagcxReduceTriggerOffState);
}

FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                                      int *idx) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  while (true) {
    int old_c = *((uint64_t *)fifoBuffer + 1);
    int cur_p = *((uint64_t *)fifoBuffer + 2);
    if (old_c >= cur_p) {
      // no-op, task dequeued by other consumers
      *idx = -1;
      break;
    }
    // set consumed from `old_c` to `old_c+1`
    int prev = atomicCAS((uint64_t *)fifoBuffer + 1, old_c, old_c + 1);
    if (prev == old_c) {
      *idx = old_c;
      break;
    }
  }
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR void flagcxReduceKernel(void *fst, void *snd, void *out,
                                                size_t count, size_t nthreads,
                                                flagcxDataType_t datatype,
                                                flagcxRedOp_t redOp) {
  // to be implemented by vendors
  int tid = threadIdx.x;
  float *fst_ptr = (float *)fst;
  float *snd_ptr = (float *)snd;
  float *out_ptr = (float *)out;
  for (int i = tid; i < count; i += nthreads) {
    out_ptr[i] = fst_ptr[i] + snd_ptr[i];
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer) {
  const unsigned FULL_MASK = 0xffffffff;
  int empty_iter = 0; // backoff counter

  while (true) {
    // (1) terminate condition
    if (__ldg(*((uint64_t *)fifoBuffer + 3)) == 1)
      break;

    // (2) dequeue
    int p = __ldg(*((uint64_t *)fifoBuffer + 2)); // produced
    int c = __ldg(*((uint64_t *)fifoBuffer + 1)); // consumed

    // (3) backoff if queue empty
    if (c >= p) {
      empty_iter++;
      spinBackoff(empty_iter);
      // check terminate again
      if (__ldg(*((uint64_t *)fifoBuffer + 3)) == 1)
        break;
      continue;
    }

    // (4) dequeue task (thread 0 in a block)
    int myIdx = -1;
    int tid = threadIdx.x;
    if (tid == 0) {
      dequeue(fifoBuffer, &myIdx);
    }
    // sync myIdx to warp
    myIdx = __shfl_sync(FULL_MASK, t, 0);
    if (myIdx < 0) {
      // backoff if no task is performed
      empty_iter++;
      spinBackoff(empty_iter);
      continue;
    }

    // (5) perform reduce task
    empty_iter = 0;
    int slot = myIdx & (*(uint64_t *)fifoBuffer - 1);
    flagcxReduceTrigger *t =
        (flagcxReduceTrigger *)((uint64_t *)fifoBuffer + 4) + myIdx;
    flagcxReduceKernel(t->getInput1(), t->getInput2(), t->getOutput(),
                       t->getCount(), t->getNthreads(), t->getDatatype(),
                       t->getRedop());

    // (6) set completion flag
    FLAGCX_DEVICE_THREAD_FENCE();
    t->setComplete();
  }
}