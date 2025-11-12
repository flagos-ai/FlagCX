#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_HOST_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr1,
                                             uint64_t addr2, uint64_t addr3,
                                             int count, int nthreads,
                                             flagcxDataType_t datatype,
                                             flagcxRedOp_t redop) {
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
  // check the state of reduction workload
  while (trigger->getState() != flagcxReduceTriggerAvailable) {
    sched_yield();
  }
  trigger->setValue(addr1, addr2, addr3, count, nthreads, datatype, redop);
  __sync_synchronize();
  __atomic_fetch_add(buffer + 2, 1ul, __ATOMIC_RELAXED);
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoInit() {
  // TODO: use a better way to initialize FIFO
  INFO(FLAGCX_KERNEL, "flagcxRedFifoInit called");
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
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxRedFifoDestroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}
