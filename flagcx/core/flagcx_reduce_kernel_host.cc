#include "device_api/completion_word.h"
#include "flagcx.h"
#include "flagcx_kernel_internal.h"

#include <limits>

FLAGCX_PARAM(ReduceFifoCapacity, "REDUCE_FIFO_CAPACITY", FLAGCX_FIFO_CAPACITY);
FLAGCX_PARAM(IpcFifoCapacity, "IPC_FIFO_CAPACITY", FLAGCX_FIFO_CAPACITY);

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setValue(
    uint64_t fst, uint64_t snd, uint64_t out, size_t count, size_t nthreads,
    flagcxDataType_t datatype, flagcxRedOp_t redOp,
    flagcxReduceTriggerState state, uint64_t flagIn, uint64_t flagOut) {
  uint64_t tmp[12] = {};
  tmp[0] = fst;
  tmp[1] = snd;
  tmp[2] = out;
  tmp[4] = flagIn;
  tmp[5] = flagOut;
  tmp[3] = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
               << flagcxReduceTriggerOffCount |
           (nthreads & flagcxTriggerMask(flagcxReduceTriggerBitsNThreads))
               << flagcxReduceTriggerOffNThreads |
           (datatype & flagcxTriggerMask(flagcxReduceTriggerBitsDatatype))
               << flagcxReduceTriggerOffDatatype |
           (redOp & flagcxTriggerMask(flagcxReduceTriggerBitsRedop))
               << flagcxReduceTriggerOffRedop |
           (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
               << flagcxReduceTriggerOffState |
           (flagcxComputeTriggerReduce &
            flagcxTriggerMask(flagcxReduceTriggerBitsComputeType))
               << flagcxReduceTriggerOffComputeType;
  memcpy(this->value, tmp, sizeof(tmp));
}

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setGemmValue(
    uint64_t a, uint64_t b, uint64_t c, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, size_t nthreads,
    flagcxDataType_t datatype, int accumulate, uint32_t workerId,
    uint32_t workerCount, uint64_t completionCounter,
    flagcxReduceTriggerState state, uint64_t flagIn, uint64_t flagOut) {
  uint64_t tmp[12] = {};
  tmp[0] = a;
  tmp[1] = b;
  tmp[2] = c;
  tmp[3] = (nthreads & flagcxTriggerMask(flagcxReduceTriggerBitsNThreads))
               << flagcxReduceTriggerOffNThreads |
           (datatype & flagcxTriggerMask(flagcxReduceTriggerBitsDatatype))
               << flagcxReduceTriggerOffDatatype |
           (flagcxRedNoOp & flagcxTriggerMask(flagcxReduceTriggerBitsRedop))
               << flagcxReduceTriggerOffRedop |
           (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
               << flagcxReduceTriggerOffState |
           (flagcxComputeTriggerGemm &
            flagcxTriggerMask(flagcxReduceTriggerBitsComputeType))
               << flagcxReduceTriggerOffComputeType;
  tmp[4] = flagIn;
  tmp[5] = flagOut;
  tmp[6] = static_cast<uint64_t>(m) << 32 | n;
  tmp[7] = static_cast<uint64_t>(k) << 32 | lda;
  tmp[8] = static_cast<uint64_t>(ldb) << 32 | ldc;
  tmp[9] = (static_cast<uint64_t>(workerId) &
            flagcxTriggerMask(flagcxGemmTriggerBitsWorkerId))
               << flagcxGemmTriggerOffWorkerId |
           (static_cast<uint64_t>(workerCount) &
            flagcxTriggerMask(flagcxGemmTriggerBitsWorkerCount))
               << flagcxGemmTriggerOffWorkerCount |
           (static_cast<uint64_t>(accumulate != 0) &
            flagcxTriggerMask(flagcxGemmTriggerBitsAccumulate))
               << flagcxGemmTriggerOffAccumulate;
  tmp[10] = completionCounter;
  memcpy(this->value, tmp, sizeof(tmp));
}

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setReduceWorkerValue(
    uint64_t fst, uint64_t snd, uint64_t out, size_t count, size_t nthreads,
    flagcxDataType_t datatype, flagcxRedOp_t redOp, uint32_t workerId,
    uint32_t workerCount, uint64_t completionCounter,
    flagcxReduceTriggerState state, uint64_t flagIn, uint64_t flagOut) {
  uint64_t tmp[12] = {};
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
               << flagcxReduceTriggerOffState |
           (flagcxComputeTriggerReduce &
            flagcxTriggerMask(flagcxReduceTriggerBitsComputeType))
               << flagcxReduceTriggerOffComputeType;
  tmp[4] = flagIn;
  tmp[5] = flagOut;
  tmp[6] = static_cast<uint64_t>(workerCount) << 32 | workerId;
  tmp[7] = completionCounter;
  memcpy(this->value, tmp, sizeof(tmp));
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

FLAGCX_HOST_DECORATOR flagcxResult_t
enqueue(void *fifoBuffer, uint64_t addr1, uint64_t addr2, uint64_t addr3,
        size_t count, size_t nthreads, flagcxDataType_t datatype,
        flagcxRedOp_t redop, uint64_t flagIn, uint64_t flagOut, int *ret) {
  if (fifoBuffer == NULL || ret == NULL || static_cast<int>(datatype) < 0 ||
      static_cast<int>(datatype) >= flagcxNumTypes ||
      getFlagcxDataTypeSize(datatype) == 0 ||
      static_cast<int>(redop) < static_cast<int>(flagcxSum) ||
      static_cast<int>(redop) >= static_cast<int>(flagcxNumRedOps) ||
      count > flagcxTriggerMask(flagcxReduceTriggerBitsCount) ||
      nthreads == 0 ||
      nthreads > flagcxTriggerMask(flagcxReduceTriggerBitsNThreads)) {
    return flagcxInvalidArgument;
  }
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[flagcxFifoIdxCapacity];
  flagcxCompletionWord_t *produced =
      flagcxFifoControlPtr(buffer, flagcxFifoIdxProduced);
  flagcxCompletionWord_t *consumed =
      flagcxFifoControlPtr(buffer, flagcxFifoIdxConsumed);
  flagcxCompletionWord_t prod = __atomic_load_n(produced, __ATOMIC_ACQUIRE);
  flagcxCompletionWord_t cons = __atomic_load_n(consumed, __ATOMIC_RELAXED);
  flagcxCompletionWord_t distance = prod - cons;
  // red buffer full, wait for kernel to consume
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  idx = prod % capacity;
  flagcxReduceTrigger *trigger =
      ((flagcxReduceTrigger *)(buffer + flagcxFifoIdxData)) + idx;

  // kernel reduce work in progress
  if (trigger->pollState() != flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  trigger->setValue(addr1, addr2, addr3, count, nthreads, datatype, redop,
                    flagcxReduceTriggerEnqueued, flagIn, flagOut);
  __atomic_fetch_add(produced, flagcxCompletionWord_t{1}, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enqueue red: count=%lu, nthreads=%lu, datatype=%d, redop=%d, idx=%d",
        count, nthreads, datatype, redop, idx);

  return flagcxSuccess;
}

FLAGCX_HOST_DECORATOR flagcxResult_t enqueueReduceWorker(
    void *fifoBuffer, uint64_t addr1, uint64_t addr2, uint64_t addr3,
    size_t count, size_t nthreads, flagcxDataType_t datatype,
    flagcxRedOp_t redop, uint32_t workerId, uint32_t workerCount,
    uint64_t completionCounter, uint64_t flagIn, uint64_t flagOut, int *ret) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[flagcxFifoIdxCapacity];
  int distance = buffer[flagcxFifoIdxProduced] - buffer[flagcxFifoIdxConsumed];
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  int idx = buffer[flagcxFifoIdxProduced] % capacity;
  flagcxReduceTrigger *trigger =
      ((flagcxReduceTrigger *)(buffer + flagcxFifoIdxData)) + idx;
  if (trigger->pollState() != flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  trigger->setReduceWorkerValue(addr1, addr2, addr3, count, nthreads, datatype,
                                redop, workerId, workerCount, completionCounter,
                                flagcxReduceTriggerEnqueued, flagIn, flagOut);
  __atomic_fetch_add(buffer + flagcxFifoIdxProduced, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL, "enqueue red worker: count=%lu, worker=%u/%u, idx=%d",
        count, workerId, workerCount, idx);
  return flagcxSuccess;
}

FLAGCX_HOST_DECORATOR flagcxResult_t
enqueueGemm(void *fifoBuffer, uint64_t a, uint64_t b, uint64_t c, uint32_t m,
            uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc,
            size_t nthreads, flagcxDataType_t datatype, int accumulate,
            uint32_t workerId, uint32_t workerCount, uint64_t completionCounter,
            uint64_t flagIn, uint64_t flagOut, int *ret) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[flagcxFifoIdxCapacity];
  int distance = buffer[flagcxFifoIdxProduced] - buffer[flagcxFifoIdxConsumed];
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  int idx = buffer[flagcxFifoIdxProduced] % capacity;
  flagcxReduceTrigger *trigger =
      ((flagcxReduceTrigger *)(buffer + flagcxFifoIdxData)) + idx;
  if (trigger->pollState() != flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  trigger->setGemmValue(a, b, c, m, n, k, lda, ldb, ldc, nthreads, datatype,
                        accumulate, workerId, workerCount, completionCounter,
                        flagcxReduceTriggerEnqueued, flagIn, flagOut);
  __atomic_fetch_add(buffer + flagcxFifoIdxProduced, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enqueue gemm worker: m=%u, n=%u, k=%u, worker=%u/%u, idx=%d", m, n, k,
        workerId, workerCount, idx);
  return flagcxSuccess;
}

flagcxResult_t flagcxCheckedFifoAllocationSize(size_t capacity,
                                               size_t elementSize,
                                               size_t *bytes) {
  if (bytes == NULL || elementSize == 0 ||
      capacity > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }
  const size_t headerBytes = flagcxFifoIdxData * sizeof(uint64_t);
  if (capacity >
      (std::numeric_limits<size_t>::max() - headerBytes) / elementSize) {
    return flagcxInvalidArgument;
  }
  *bytes = headerBytes + capacity * elementSize;
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoInit() {
  const int64_t configuredCapacity = flagcxParamReduceFifoCapacity();
  if (configuredCapacity <= 0) {
    return flagcxInvalidArgument;
  }
  return flagcxRedFifoInit(static_cast<size_t>(configuredCapacity));
}

flagcxResult_t flagcxFifo::flagcxRedFifoInit(size_t numTriggers) {
  TRACE(FLAGCX_INIT, "flagcxRedFifoInit called");
  if (buffer != NULL || numTriggers == 0) {
    return flagcxInvalidArgument;
  }
  size_t bytes = 0;
  FLAGCXCHECK(flagcxCheckedFifoAllocationSize(
      numTriggers, sizeof(flagcxReduceTrigger), &bytes));
  uint64_t *newBuffer = NULL;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(reinterpret_cast<void **>(&newBuffer),
                                          bytes, flagcxMemHost, NULL));
  if (newBuffer == NULL) {
    return flagcxSystemError;
  }
  buffer = newBuffer;
  buffer[flagcxFifoIdxCapacity] = numTriggers;
  buffer[flagcxFifoIdxConsumed] = 0;
  buffer[flagcxFifoIdxProduced] = 0;
  buffer[flagcxFifoIdxTerminate] = 0;
  buffer[flagcxFifoIdxTerminalStatus] = 0;
  buffer[flagcxFifoIdxProducerState] = 0;
  memset((void *)(buffer + flagcxFifoIdxData), 0,
         numTriggers * sizeof(flagcxReduceTrigger));
  __sync_synchronize();
  return flagcxSuccess;
}

FLAGCX_HOST_DECORATOR flagcxResult_t
enqueueIpc(void *fifoBuffer, uint64_t srcOffsetBytes, uint64_t dstOffsetBytes,
           uint64_t bytes, uint64_t chunkSize,
           flagcxIpcBufferType srcBufferType, int peerLocalRank,
           uint32_t readySlot, uint64_t epoch, uint32_t parentFlagsOffset,
           uint32_t numParentFlags, uint64_t flagOut, int *ret) {
  if (chunkSize == 0) {
    return flagcxInvalidArgument;
  }
  uint64_t numChunks = bytes / chunkSize + (bytes % chunkSize != 0);
  if (numChunks == 0)
    numChunks = 1;
  if (numChunks > ~uint32_t(0)) {
    return flagcxInvalidArgument;
  }

  uint64_t *buffer = static_cast<uint64_t *>(fifoBuffer);
  int capacity = static_cast<int>(buffer[flagcxFifoIdxCapacity]);
  uint64_t produced =
      __atomic_load_n(buffer + flagcxFifoIdxProduced, __ATOMIC_ACQUIRE);
  uint64_t consumed =
      __atomic_load_n(buffer + flagcxFifoIdxConsumed, __ATOMIC_ACQUIRE);
  if (produced - consumed >= static_cast<uint64_t>(capacity)) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }

  int idx = static_cast<int>(produced % capacity);
  flagcxIpcTrigger *trigger =
      reinterpret_cast<flagcxIpcTrigger *>(buffer + flagcxFifoIdxData) + idx;
  if (__atomic_load_n(&trigger->state, __ATOMIC_ACQUIRE) !=
      flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }

  trigger->srcOffsetBytes = srcOffsetBytes;
  trigger->dstOffsetBytes = dstOffsetBytes;
  trigger->bytes = bytes;
  trigger->chunkSize = chunkSize;
  trigger->flagOut = flagOut;
  trigger->epoch = epoch;
  trigger->srcBufferType = static_cast<uint32_t>(srcBufferType);
  trigger->peerLocalRank = static_cast<uint32_t>(peerLocalRank);
  trigger->readySlot = readySlot;
  trigger->parentFlagsOffset = parentFlagsOffset;
  trigger->numParentFlags = numParentFlags;
  trigger->numChunks = static_cast<uint32_t>(numChunks);
  trigger->nextChunk = 0;
  trigger->completedChunks = 0;
  __atomic_store_n(&trigger->state, flagcxReduceTriggerEnqueued,
                   __ATOMIC_RELEASE);
  __atomic_fetch_add(buffer + flagcxFifoIdxProduced, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enqueue ipc: bytes=%lu chunkSize=%lu chunks=%lu peerLocalRank=%d "
        "slot=%u epoch=%lu idx=%d",
        bytes, chunkSize, numChunks, peerLocalRank, readySlot, epoch, idx);
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxIpcFifoInit() {
  const int64_t configuredCapacity = flagcxParamIpcFifoCapacity();
  if (configuredCapacity <= 0) {
    return flagcxInvalidArgument;
  }
  return flagcxIpcFifoInit(static_cast<size_t>(configuredCapacity));
}

flagcxResult_t flagcxFifo::flagcxIpcFifoInit(size_t numTriggers) {
  TRACE(FLAGCX_INIT, "flagcxIpcFifoInit called");
  if (buffer != NULL || numTriggers == 0) {
    return flagcxInvalidArgument;
  }
  size_t bytes = 0;
  FLAGCXCHECK(flagcxCheckedFifoAllocationSize(
      numTriggers, sizeof(flagcxIpcTrigger), &bytes));
  uint64_t *newBuffer = NULL;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(reinterpret_cast<void **>(&newBuffer),
                                          bytes, flagcxMemHost, NULL));
  if (newBuffer == NULL) {
    return flagcxSystemError;
  }
  buffer = newBuffer;
  buffer[flagcxFifoIdxCapacity] = numTriggers;
  buffer[flagcxFifoIdxConsumed] = 0;
  buffer[flagcxFifoIdxProduced] = 0;
  buffer[flagcxFifoIdxTerminate] = 0;
  memset(buffer + flagcxFifoIdxData, 0, numTriggers * sizeof(flagcxIpcTrigger));
  __sync_synchronize();
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxRedFifoDestroy called");
  if (buffer != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
    buffer = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxIpcFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxIpcFifoDestroy called");
  if (buffer != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(static_cast<void *>(buffer),
                                          flagcxMemHost, NULL));
    buffer = NULL;
  }
  return flagcxSuccess;
}
