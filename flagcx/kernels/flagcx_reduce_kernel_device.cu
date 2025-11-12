#include "flagcx.h"
#include "flagcx_kernel.h"

#define WARP_SIZE 32

FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getInput1() {
  // wip
}
FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getInput2() {
  // wip
}
FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getOutput() {
  // wip
}
FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getCount() {
  // wip
}
FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getNThreads() {
  // wip
}
FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getDatatype() {
  // wip
}
FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getRedop() {
  // wip
}
FLAGCX_DEVICE_DECORATOR uint64_t flagcxReduceTrigger::getState() {
  // wip
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                               flagcxReduceTrigger_t trigger) {
  // wip
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
      while (true) {
        int old_c = *((uint64_t *)fifoBuffer + 1);
        int cur_p = *((uint64_t *)fifoBuffer + 2);
        if (old_c >= cur_p) {
          myIdx = -1; // no-op, task dequeued by other consumers
          break;
        }
        // set consumed from `old_c` to `old_c+1`
        int prev = atomicCAS((uint64_t *)fifoBuffer + 1, old_c, old_c + 1);
        if (prev == old_c) {
          myIdx = old_c;
          break;
        }
      }
    }
    // sync myIdx to warp
    myIdx = __shfl_sync(FULL_MASK, myIdx, 0);
    if (myIdx < 0) {
      // backoff if no task is performed
      empty_iter++;
      spinBackoff(empty_iter);
      continue;
    }

    // (5) perform reduce task
    empty_iter = 0;
    int slot = myIdx & (*(uint64_t *)fifoBuffer - 1); // myIdx % capacity
    flagcxReduceTrigger *t =
        (flagcxReduceTrigger *)((uint64_t *)fifoBuffer + 4) + myIdx;
    flagcxReduceKernel(t->getInput1(), t->getInput2(), t->getOutput(),
                       t->getCount(), t->getNthreads(), t->getDatatype(),
                       t->getRedop());

    // (6) set completion flag
    FLAGCX_DEVICE_THREAD_FENCE();
    *((uint64_t *)fifoBuffer + 4 +
      myIdx * sizeof(flagcxReduceTrigger) / sizeof(uint64_t)) |= 1;
  }
}