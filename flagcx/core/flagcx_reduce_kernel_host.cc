#include "flagcx.h"
#include "flagcx_kernel.h"

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
  TRACE(
      FLAGCX_KERNEL,
      "setValue called: fst=0x%016lx, snd=0x%016lx, out=0x%016lx, count=%lu, "
      "nthreads=%lu, "
      "datatype=%d, redop=%d, state=%d, value[0]=0x%016lx, value[1]=0x%016lx, "
      "value[2]=0x%016lx, value[3]=0x%016lx",
      fst, snd, out, count, nthreads, datatype, redOp, state, value[0],
      value[1], value[2], value[3]);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxReduceTrigger::pollState() {
  uint64_t curr_val = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  return curr_val >> flagcxReduceTriggerOffState &
         flagcxTriggerMask(flagcxReduceTriggerBitsState);
}

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setState(int state) {
  uint64_t curr_val = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  curr_val &= ~(flagcxTriggerMask(flagcxReduceTriggerBitsState)
                << flagcxReduceTriggerOffState);
  curr_val |= (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
              << flagcxReduceTriggerOffState;
  __atomic_store_n(&this->value[3], curr_val, __ATOMIC_RELEASE);
  TRACE(FLAGCX_KERNEL, "setState called, new state=%llu",
        curr_val >> flagcxReduceTriggerOffState &
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
  while (distance >= capacity) {
    sched_yield();
    distance = buffer[2] - buffer[1];
  }
  idx = buffer[2] % capacity;
  flagcxReduceTrigger *trigger = ((flagcxReduceTrigger *)(buffer + 4)) + idx;

  while (trigger->pollState() != flagcxReduceTriggerAvailable) {
    sched_yield();
  }
  trigger->setValue(addr1, addr2, addr3, count, nthreads, datatype, redop,
                    flagcxReduceTriggerEnqueued);
  __atomic_fetch_add(buffer + 2, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enq red called: addr1=0x%016lx, addr2=0x%016lx, addr3=0x%016lx, "
        "count=%lu, nthreads=%lu, datatype=%d, redop=%d, idx=%d",
        addr1, addr2, addr3, count, nthreads, datatype, redop, idx);
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoInit() {
  // TODO: use a better way to initialize FIFO
  TRACE(FLAGCX_INIT, "flagcxRedFifoInit called");
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          4 * sizeof(uint64_t) +
                                              FLAGCX_KERNEL_FIFO_CAPACITY *
                                                  sizeof(flagcxReduceTrigger),
                                          flagcxMemHost, NULL));
  buffer[0] = FLAGCX_KERNEL_FIFO_CAPACITY;
  buffer[1] = 0;
  buffer[2] = 0;
  buffer[3] = 0;
  memset((void *)(buffer + 4), 0,
         FLAGCX_KERNEL_FIFO_CAPACITY * sizeof(flagcxReduceTrigger));
  __sync_synchronize();
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxRedFifoDestroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}
