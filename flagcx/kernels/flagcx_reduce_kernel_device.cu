#include "device_utils.h"
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
FLAGCX_DEVICE_INLINE_DECORATOR void flagcxReduceTrigger::setComplete() {
  // atomicOr(reinterpret_cast<unsigned long long *>(value) + 3,
  //          (flagcxReduceTriggerComplete &
  //           flagcxTriggerMask(flagcxReduceTriggerBitsState))
  //              << flagcxReduceTriggerOffState);
  uint64_t* ptr =
    reinterpret_cast<uint64_t*>(value) + 3;
  uint64_t mask =
    (flagcxReduceTriggerComplete &
     flagcxTriggerMask(flagcxReduceTriggerBitsState))
        << flagcxReduceTriggerOffState;
  *ptr |= mask;
  //FLAGCX_DEVICE_THREAD_FENCE(); 
}

FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                                      int *idx) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  while (true) {
    unsigned long long int old_c = *(buffer + 1);
    unsigned long long int cur_p = *(buffer + 2);
    if (old_c >= cur_p) {
      // no-op, task dequeued by other consumers
      *idx = -1;
      break;
    }
    // set consumed from `old_c` to `old_c+1`
    unsigned long long int prev = atomicCAS(reinterpret_cast<unsigned long long int *>(buffer + 1),
                              old_c, old_c + 1);
    if (prev == old_c) {
      *idx = old_c;
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
    // debug: timeout
    // if (empty_iter > 1000) {
    //   printf("reduce kernel timeout\n");
    //   break;
    // }

    // (1) terminate condition
    // if (__ldg(static_cast<const uint64_t *>(fifoBuffer) + 3) == 1)
    //   break;

    // (2) dequeue
    int tid = threadIdx.x;
    int myIdx = -1;
    int c = -1;
    int p = -1;
    int term = -1;
    volatile uint64_t *vBuf =  (volatile uint64_t*)fifoBuffer;
    if (tid == 0) {
      c = vBuf[1]; // consumed
      p = vBuf[2]; // produced
      term = vBuf[3];
    }
    c = __shfl_sync(FULL_MASK, c, 0);
    p = __shfl_sync(FULL_MASK, p, 0);
    term = __shfl_sync(FULL_MASK, term, 0);

    // (3) backoff if queue empty
    if (c >= p) {
      // check terminate
      if (term == 1)
        break;
      empty_iter++;
      spinBackoff(empty_iter);
      continue;
    }

    // (4) dequeue task (thread 0 in a block)
    if (tid == 0) {
      dequeue(fifoBuffer, &myIdx);
    }
    // sync myIdx to warp
    myIdx = __shfl_sync(FULL_MASK, myIdx, 0);
    if (myIdx < 0) {
      if (term == 1)
        break;
      // backoff if no task is performed
      empty_iter++;
      spinBackoff(empty_iter);
      continue;
    }

    // (5) perform reduce task
    empty_iter = 0;
    int slot = myIdx & (*(uint64_t *)fifoBuffer - 1);
    flagcxReduceTrigger *t =
        ((flagcxReduceTrigger *)((uint64_t *)fifoBuffer + 4)) + slot;
    flagcxReduceKernel(t->getInput1(), t->getInput2(), t->getOutput(),
                       t->getCount(), t->getNThreads(), t->getDatatype(),
                       t->getRedop());
    FLAGCX_DEVICE_THREAD_FENCE();

    // (6) set completion flag
    if (tid == 0) {
      reinterpret_cast<flagcxReduceTrigger *>((uint64_t *)fifoBuffer + 4)->setComplete();
      t->setComplete();
    }
  }
  //FLAGCX_DEVICE_THREAD_FENCE();
}

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream) {
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(fifoBuffer);
}
