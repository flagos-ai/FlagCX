#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_PARAM(KernelFifoCapacity, "KERNEL_FIFO_CAPACITY", FLAGCX_FIFO_CAPACITY);

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getAddr() { return fst; }

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getCount() {
  return snd >> flagcxDeviceTriggerOffCount &
         flagcxTriggerMask(flagcxDeviceTriggerBitsCount);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getPeerRank() {
  return snd >> flagcxDeviceTriggerOffPeerRank &
         flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getDatatype() {
  return snd >> flagcxDeviceTriggerOffDatatype &
         flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getType() {
  return snd >> flagcxDeviceTriggerOffPrim &
         flagcxTriggerMask(flagcxDeviceTriggerBitsPrim);
}

flagcxResult_t flagcxFifo::flagcxFifoInit() {
  INFO(FLAGCX_KERNEL, "flagcxFifoInit called");
  uint64_t flagcxKernelFifoCapacity = flagcxParamKernelFifoCapacity();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          3 * sizeof(uint64_t) +
                                              flagcxKernelFifoCapacity *
                                                  sizeof(flagcxDeviceTrigger),
                                          flagcxMemHost, NULL));
  buffer[0] = flagcxKernelFifoCapacity;
  buffer[1] = 0;
  buffer[2] = 0;
  memset((void *)(buffer + 3), 0,
         flagcxKernelFifoCapacity * sizeof(flagcxDeviceTrigger));
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxFifoDestroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger) {
  volatile uint64_t *buffer = (volatile uint64_t *)fifoBuffer;
  uint64_t capacity = buffer[0];
  uint64_t cons = buffer[1];
  uint64_t prod = buffer[2];

  if (prod > cons) {
    // Get pointer to slot's raw uint64_t fields
    uint64_t idx = cons % capacity;
    volatile uint64_t *slotFst =
        buffer + 3 + idx * (sizeof(flagcxDeviceTrigger) / sizeof(uint64_t));
    volatile uint64_t *slotSnd = slotFst + 1;

    // Wait for valid bit to be set (data is committed by producer)
    // Use hybrid approach: spin vigorously first, then yield to reduce CPU
    // usage
    int spins = 0;
    while (!(*slotSnd & flagcxDeviceTriggerValidMask)) {
      __sync_synchronize();
      if (++spins > 1000) {
        sched_yield();
        spins = 0;
      }
    }

    // Memory fence before reading
    __sync_synchronize();

    // Copy data (clear valid bit in the copy)
    trigger->fst = *slotFst;
    trigger->snd = *slotSnd & ~flagcxDeviceTriggerValidMask;

    // Clear valid bit in slot for reuse
    *slotSnd = 0;

    // Memory fence before updating consumed
    __sync_synchronize();

    // Update consumed counter
    buffer[1] = cons + 1;
  } else {
    memset((void *)trigger, 0, sizeof(flagcxDeviceTrigger));
  }
  return flagcxSuccess;
}
