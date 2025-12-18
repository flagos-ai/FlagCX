#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

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
  // TODO: use a better way to initialize FIFO
  INFO(FLAGCX_KERNEL, "flagcxFifoInit called");
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          3 * sizeof(uint64_t) +
                                              FLAGCX_KERNEL_FIFO_CAPACITY *
                                                  sizeof(flagcxDeviceTrigger),
                                          flagcxMemHost, NULL));
  buffer[0] = FLAGCX_KERNEL_FIFO_CAPACITY;
  buffer[1] = 0;
  buffer[2] = 0;
  memset((void *)(buffer + 3), 0,
         FLAGCX_KERNEL_FIFO_CAPACITY * sizeof(flagcxDeviceTrigger));
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxFifoDestroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger) {
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  uint64_t capacity = buffer[0];
  uint64_t cons = buffer[1];
  uint64_t prod = buffer[2];
  int distance = prod - cons;
  if (distance > 0) {
    idx = cons % capacity;
    memcpy((void *)trigger,
           (void *)(buffer + 3 +
                    sizeof(flagcxDeviceTrigger) / sizeof(uint64_t) * idx),
           sizeof(flagcxDeviceTrigger));
    __sync_synchronize();
    buffer[1] = buffer[1] + 1;
  } else {
    memset((void *)trigger, 0, sizeof(flagcxDeviceTrigger));
  }
  return flagcxSuccess;
}
