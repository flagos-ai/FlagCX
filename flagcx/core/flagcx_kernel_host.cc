#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_PARAM(KernelFifoCapacity, "KERNEL_FIFO_CAPACITY", FLAGCX_FIFO_CAPACITY);

// ==========================================================================
// flagcxDeviceTrigger accessors — read from trd (common) / fst,snd (payload)
// ==========================================================================

// Common accessors (trd)
FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getPrim() {
  return (trd >> flagcxDeviceTriggerOffPrim) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsPrim);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getPeerRank() {
  return (trd >> flagcxDeviceTriggerOffPeerRank) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getSlotIdx() {
  return (trd >> flagcxDeviceTriggerOffSlotIdx) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsSlotIdx);
}

// Backward compat alias
FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getType() {
  return getPrim();
}

// Two-sided accessors (Send/Recv)
FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getAddr() { return fst; }

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getDatatype() {
  return (trd >> flagcxDeviceTriggerOffDatatype) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getCount() {
  return (trd >> flagcxDeviceTriggerOffCount) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsCount);
}

// One-sided accessors (Put/PutSignal/PutValue)
FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getSrcMrIdx() {
  return (trd >> flagcxDeviceTriggerOffSrcMrIdx) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsSrcMrIdx);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getDstMrIdx() {
  return (trd >> flagcxDeviceTriggerOffDstMrIdx) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsDstMrIdx);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getSize() {
  return (snd >> flagcxDeviceTriggerOffSize) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsSize);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getSrcOffset() {
  return (fst >> flagcxDeviceTriggerOffSrcOffset) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsSrcOffset);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getDstOffset() {
  return (fst >> flagcxDeviceTriggerOffDstOffset) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsDstOffset);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getValue() { return snd; }

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getSignalIdx() {
  // PutSignal uses trd[21:14], Signal/WaitSignal uses trd[33:26]
  uint64_t prim = getPrim();
  if (prim == flagcxDevicePrimPutSignal) {
    return (trd >> flagcxDeviceTriggerOffSignalIdx) &
           flagcxTriggerMask(flagcxDeviceTriggerBitsSignalIdx);
  }
  // Signal, WaitSignal
  return (trd >> flagcxDeviceTriggerOffSignalIdxSig) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsSignalIdxSig);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getSignalValue() {
  // PutSignal stores signalValue in snd[15:0], Signal stores in trd[25:10]
  uint64_t prim = getPrim();
  if (prim == flagcxDevicePrimPutSignal) {
    return (snd >> flagcxDeviceTriggerOffSignalValuePut) &
           flagcxTriggerMask(flagcxDeviceTriggerBitsSignalValuePut);
  }
  return (trd >> flagcxDeviceTriggerOffSignalValue) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsSignalValue);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getExpectedValue() {
  return (trd >> flagcxDeviceTriggerOffSignalValue) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsSignalValue);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getBufferType() {
  return (trd >> flagcxDeviceTriggerOffBufferType) &
         flagcxTriggerMask(flagcxDeviceTriggerBitsBufferType);
}

// Term accessor
FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getTotalCoops() {
  return fst;
}

// ==========================================================================
// FIFO init / destroy
// ==========================================================================

flagcxResult_t flagcxFifo::flagcxFifoInit() {
  INFO(FLAGCX_KERNEL, "flagcxFifoInit called");
  uint64_t flagcxKernelFifoCapacity = flagcxParamKernelFifoCapacity();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          flagcxFifoIdxData * sizeof(uint64_t) +
                                              flagcxKernelFifoCapacity *
                                                  sizeof(flagcxDeviceTrigger),
                                          flagcxMemHost, NULL));
  buffer[flagcxFifoIdxCapacity] = flagcxKernelFifoCapacity;
  buffer[flagcxFifoIdxConsumed] = 0;
  buffer[flagcxFifoIdxProduced] = 0;
  buffer[flagcxFifoIdxCompleted] = 0; // IB completion count (GPU polls this)
  memset((void *)(buffer + flagcxFifoIdxData), 0,
         flagcxKernelFifoCapacity * sizeof(flagcxDeviceTrigger));
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxFifoDestroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}

// ==========================================================================
// Host-side dequeue — polls trd (word2) for valid bit
// ==========================================================================

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  uint64_t capacity = buffer[flagcxFifoIdxCapacity];
  // Use atomic loads for GPU-written fields: the GPU writes produced via
  // system-scoped atomics (atom.acq_rel.sys).  A plain volatile read may
  // return stale data on ARM hosts and can be speculatively reordered on x86.
  // __atomic_load_n with ACQUIRE ensures we observe the latest GPU write.
  uint64_t cons =
      __atomic_load_n(&buffer[flagcxFifoIdxConsumed], __ATOMIC_RELAXED);
  uint64_t prod =
      __atomic_load_n(&buffer[flagcxFifoIdxProduced], __ATOMIC_ACQUIRE);

  if (prod > cons) {
    // Get pointer to slot's raw uint64_t fields (3 words per entry)
    uint64_t idx = cons % capacity;
    uint64_t *slotFst = buffer + flagcxFifoIdxData +
                        idx * (sizeof(flagcxDeviceTrigger) / sizeof(uint64_t));
    uint64_t *slotSnd = slotFst + 1;
    uint64_t *slotTrd = slotFst + 2;

    // Wait for valid bit on trd (word2, written last by producer with release)
    int spins = 0;
    while (!(__atomic_load_n(slotTrd, __ATOMIC_ACQUIRE) &
             flagcxDeviceTriggerValidMask)) {
      if (++spins > 1000) {
        if (spins == 1001) {
          INFO(FLAGCX_P2P,
               "dequeue: spinning on valid bit prod=%lu cons=%lu idx=%lu "
               "slotTrd=0x%lx validMask=0x%lx",
               prod, cons, idx, __atomic_load_n(slotTrd, __ATOMIC_RELAXED),
               flagcxDeviceTriggerValidMask);
        }
        sched_yield();
        spins = 0;
      }
    }

    // Acquire on slotTrd above ensures payload (fst/snd) is visible
    trigger->fst = __atomic_load_n(slotFst, __ATOMIC_RELAXED);
    trigger->snd = __atomic_load_n(slotSnd, __ATOMIC_RELAXED);
    trigger->trd = __atomic_load_n(slotTrd, __ATOMIC_RELAXED) &
                   ~flagcxDeviceTriggerValidMask;

    TRACE(FLAGCX_P2P,
          "dequeue: got entry prod=%lu cons=%lu prim=%lu peer=%lu "
          "fst=0x%lx snd=0x%lx trd=0x%lx",
          prod, cons, trigger->getPrim(), trigger->getPeerRank(), trigger->fst,
          trigger->snd, trigger->trd);

    // Clear trd valid bit in slot for reuse
    __atomic_store_n(slotTrd, (uint64_t)0, __ATOMIC_RELAXED);
  } else {
    memset((void *)trigger, 0, sizeof(flagcxDeviceTrigger));
  }
  return flagcxSuccess;
}
