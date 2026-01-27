#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_PARAM(ReduceFifoCapacity, "REDUCE_FIFO_CAPACITY", FLAGCX_FIFO_CAPACITY);

FLAGCX_HOST_DECORATOR void
flagcxReduceTrigger::setValue(uint64_t fst, uint64_t snd, uint64_t out,
                              size_t count, size_t nthreads,
                              flagcxDataType_t datatype, flagcxRedOp_t redOp,
                              flagcxReduceTriggerState state) {
  uint64_t tmp[4];
  tmp[0] = fst;
  tmp[1] = snd;
  tmp[2] = out;
  tmp[3] = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
               << flagcxReduceTriggerOffCount |
           (nthreads & flagcxTriggerMask(flagcxReduceTriggerBitsNThreads))
               << flagcxReduceTriggerOffNThreads |
           (datatype & flagcxTriggerMask(flagcxReduceTriggerBitsDatatype))
               << flagcxReduceTriggerOffDatatype |
           (redOp & flagcxTriggerMask(flagcxReduceTriggerBitsRedop))
               << flagcxReduceTriggerOffRedop |
           (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
               << flagcxReduceTriggerOffState;
  memcpy(this->value, tmp, 4 * sizeof(uint64_t));
}

FLAGCX_HOST_DECORATOR uint64_t flagcxReduceTrigger::pollState() {
  uint64_t currVal = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  return currVal >> flagcxReduceTriggerOffState &
         flagcxTriggerMask(flagcxReduceTriggerBitsState);
}

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setState(int state) {
  uint64_t currVal = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  currVal &= ~(flagcxTriggerMask(flagcxReduceTriggerBitsState)
               << flagcxReduceTriggerOffState);
  currVal |= (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
             << flagcxReduceTriggerOffState;
  __atomic_store_n(&this->value[3], currVal, __ATOMIC_RELEASE);
  TRACE(FLAGCX_KERNEL, "setState called, new state=%llu",
        currVal >> flagcxReduceTriggerOffState &
            flagcxTriggerMask(flagcxReduceTriggerBitsState));
}

FLAGCX_HOST_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr1,
                                             uint64_t addr2, uint64_t addr3,
                                             size_t count, size_t nthreads,
                                             flagcxDataType_t datatype,
                                             flagcxRedOp_t redop, int *ret) {
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[0];
  int distance = buffer[2] - buffer[1];
  // red buffer full, wait for kernel to consume
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  idx = buffer[2] % capacity;
  flagcxReduceTrigger *trigger = ((flagcxReduceTrigger *)(buffer + 4)) + idx;

  // kernel reduce work in progress
  if (trigger->pollState() != flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  trigger->setValue(addr1, addr2, addr3, count, nthreads, datatype, redop,
                    flagcxReduceTriggerEnqueued);
  __atomic_fetch_add(buffer + 2, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enqueue red: count=%lu, nthreads=%lu, datatype=%d, redop=%d, idx=%d",
        count, nthreads, datatype, redop, idx);

  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoInit() {
  TRACE(FLAGCX_INIT, "flagcxRedFifoInit called");
  uint64_t flagcxReduceFifoCapacity = flagcxParamReduceFifoCapacity();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          4 * sizeof(uint64_t) +
                                              flagcxReduceFifoCapacity *
                                                  sizeof(flagcxReduceTrigger),
                                          flagcxMemHost, NULL));
  buffer[0] = flagcxReduceFifoCapacity;
  buffer[1] = 0;
  buffer[2] = 0;
  buffer[3] = 0;
  memset((void *)(buffer + 4), 0,
         flagcxReduceFifoCapacity * sizeof(flagcxReduceTrigger));
  __sync_synchronize();
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxRedFifoDestroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}
