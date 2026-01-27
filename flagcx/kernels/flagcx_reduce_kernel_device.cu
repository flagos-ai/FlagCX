#include "device_utils.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

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
  atomicOr(reinterpret_cast<unsigned long long *>(value) + 3,
           (flagcxReduceTriggerComplete &
            flagcxTriggerMask(flagcxReduceTriggerBitsState))
               << flagcxReduceTriggerOffState);
}

FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(volatile uint64_t *buffer,
                                                      int *idx) {
  while (true) {
    unsigned long long int oldConsumed = *(buffer + 1);
    unsigned long long int curProduced = *(buffer + 2);
    if (oldConsumed >= curProduced) {
      // no-op, task dequeued by other consumers
      *idx = -1;
      break;
    }
    // set consumed from `oldConsumed` to `oldConsumed+1`
    unsigned long long int prev = atomicCAS(
        (unsigned long long int *)(buffer + 1), oldConsumed, oldConsumed + 1);
    if (prev == oldConsumed) {
      *idx = oldConsumed;
      break;
    }
  }
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR void
flagcxReduceKernel(uint64_t fst, uint64_t snd, uint64_t out, uint64_t count,
                   uint64_t nthreads, uint64_t datatype, uint64_t redOp) {
  // to be implemented by vendors
  int tid = threadIdx.x;
  float *fstPtr = (float *)fst;
  float *sndPtr = (float *)snd;
  float *outPtr = (float *)out;
  for (int i = tid; i < count; i += nthreads) {
    outPtr[i] = fstPtr[i] + sndPtr[i];
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer) {
  volatile uint64_t *vBuf = (volatile uint64_t *)fifoBuffer;
  int emptyIter = 0; // backoff counter

  while (true) {
    // (1) dequeue
    int tid = threadIdx.x;
    int myIdx = -1;
    int c = -1;
    int p = -1;
    int term = -1;
    if (tid == 0) {
      c = vBuf[1]; // consumed
      p = vBuf[2]; // produced
      term = vBuf[3];
    }
    c = __shfl_sync(FULL_MASK, c, 0);
    p = __shfl_sync(FULL_MASK, p, 0);
    term = __shfl_sync(FULL_MASK, term, 0);

    // (2) backoff if queue empty
    if (c >= p) {
      // check terminate
      if (term == 1)
        break;
      emptyIter++;
      spinBackoff(emptyIter);
      continue;
    }

    // (3) dequeue task (lane 0 in a warp)
    if (tid == 0) {
      dequeue(vBuf, &myIdx);
    }
    // sync myIdx to warp
    myIdx = __shfl_sync(FULL_MASK, myIdx, 0);
    if (myIdx < 0) {
      if (term == 1)
        break;
      // backoff if no task is performed
      emptyIter++;
      spinBackoff(emptyIter);
      continue;
    }

    // (4) perform reduce task
    emptyIter = 0;
    uint64_t fst;
    uint64_t snd;
    uint64_t out;
    uint64_t count;
    uint64_t nthreads;
    uint64_t datatype;
    uint64_t redop;
    int slot = myIdx & (*vBuf - 1);
    if (tid == 0) {
      flagcxReduceTrigger *t = (flagcxReduceTrigger *)(vBuf + 4) + slot;
      fst = t->getInput1();
      snd = t->getInput2();
      out = t->getOutput();
      count = t->getCount();
      nthreads = t->getNThreads();
      datatype = t->getDatatype();
      redop = t->getRedop();
    }
    fst = __shfl_sync(FULL_MASK, fst, 0);
    snd = __shfl_sync(FULL_MASK, snd, 0);
    out = __shfl_sync(FULL_MASK, out, 0);
    count = __shfl_sync(FULL_MASK, count, 0);
    nthreads = __shfl_sync(FULL_MASK, nthreads, 0);
    datatype = __shfl_sync(FULL_MASK, datatype, 0);
    redop = __shfl_sync(FULL_MASK, redop, 0);
    flagcxReduceKernel(fst, snd, out, count, nthreads, datatype, redop);
    __syncthreads();
    FLAGCX_DEVICE_THREAD_FENCE();

    // (5) set completion flag
    if (tid == 0) {
      flagcxReduceTrigger *t = (flagcxReduceTrigger *)(vBuf + 4) + slot;
      t->setComplete();
    }
  }
}

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream) {
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(fifoBuffer);
}
