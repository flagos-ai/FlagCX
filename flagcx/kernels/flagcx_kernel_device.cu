#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

#ifndef flagcxTriggerMask
#define flagcxTriggerMask(w) ((w == 64) ? ~0ull : ((1ull << w) - 1))
#endif

FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iter) {
  int delay = 1 << (iter < 15 ? iter : 15);
#if __CUDA_ARCH__ >= 700
  __nanosleep(delay);
#else
  uint64_t start = clock64();
  while (clock64() - start < (uint64_t)delay) { /* spin */
  }
#endif
}

FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype) {
  switch (dtype) {
    // case flagcxInt8:
    case flagcxChar:
      return sizeof(char); // 1 byte
    case flagcxUint8:
      return sizeof(unsigned char); // 1 byte
    // case flagcxInt32:
    case flagcxInt:
      return sizeof(int); // 4 bytes
    case flagcxUint32:
      return sizeof(unsigned int); // 4 bytes
    case flagcxInt64:
      return sizeof(long long); // 8 bytes
    case flagcxUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case flagcxFloat16:
    case flagcxHalf:
      return 2; // Half precision float is 2 bytes
    // case flagcxFloat32:
    case flagcxFloat:
      return sizeof(float); // 4 bytes
    // case flagcxFloat64:
    case flagcxDouble:
      return sizeof(double); // 8 bytes
    case flagcxBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      return 0;
  }
}

FLAGCX_DEVICE_DECORATOR FLAGCX_HOST_DECORATOR uint64_t
flagcxTriggerMask(size_t w) {
  return (w == 64) ? ~0ull : ((1ull << w) - 1);
}

FLAGCX_HOST_DECORATOR FLAGCX_DEVICE_DECORATOR
flagcxDeviceTrigger::flagcxDeviceTrigger(uint64_t addr, uint64_t count, uint64_t peerRank,
                                         uint64_t datatype, uint64_t type) {
  printf("init trigger\n");
  fst = addr;
  snd = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
            << flagcxDeviceTriggerOffCount |
        (peerRank & flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank))
            << flagcxDeviceTriggerOffPeerRank |
        (datatype & flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype))
            << flagcxDeviceTriggerOffDatatype |
        (type & flagcxTriggerMask(flagcxDeviceTriggerBitsPrim))
            << flagcxDeviceTriggerOffPrim;
  printf("addr 0x%016lx, count %lu, peer %lu, datatype %lu, type %lu, snd 0x%016lx\n", addr, count, peerRank, datatype, type, snd);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getAddr() {
  return fst;
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getCount() {
  printf("getCount %lu, snd=%016lx\n", snd >> flagcxDeviceTriggerOffCount &
         flagcxTriggerMask(flagcxDeviceTriggerBitsCount), snd);
  return snd >> flagcxDeviceTriggerOffCount &
         flagcxTriggerMask(flagcxDeviceTriggerBitsCount);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getPeerRank() {
  printf("getPeer %lu, snd=0x%016lx\n", snd >> flagcxDeviceTriggerOffPeerRank &
         flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank), snd);
  return snd >> flagcxDeviceTriggerOffPeerRank &
         flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getDatatype() {
  printf("getDatatype %lu, snd=0x%016lx\n", snd >> flagcxDeviceTriggerOffDatatype &
         flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype), snd);
  return snd >> flagcxDeviceTriggerOffDatatype &
         flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype);
}

FLAGCX_HOST_DECORATOR uint64_t flagcxDeviceTrigger::getType() {
  // printf("getType %lu, snd=0x%016lx\n", snd >> flagcxDeviceTriggerOffPrim &
  //        flagcxTriggerMask(flagcxDeviceTriggerBitsPrim), snd);
  return snd >> flagcxDeviceTriggerOffPrim &
         flagcxTriggerMask(flagcxDeviceTriggerBitsPrim);
}

FLAGCX_DEVICE_DECORATOR void
flagcxDeviceTrigger::setValue(uint64_t addr, uint64_t count, uint64_t peerRank,
                              uint64_t datatype, uint64_t type) {
  printf("setValue\n");
  fst = addr;
  snd = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
            << flagcxDeviceTriggerOffCount |
        (peerRank & flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank))
            << flagcxDeviceTriggerOffPeerRank |
        (datatype & flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype))
            << flagcxDeviceTriggerOffDatatype |
        (type & flagcxTriggerMask(flagcxDeviceTriggerBitsPrim))
            << flagcxDeviceTriggerOffPrim;
  printf("addr 0x%016lx, count %lu, peer %lu, datatype %lu, type %lu, snd 0x%016lx\n", addr, count, peerRank, datatype, type, snd);
}

flagcxResult_t flagcxFifo::flagcxFifoInit() {
  // TODO: use a better way to initialize FIFO
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
  INFO(FLAGCX_INIT, "fifo destroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceSend(const void *sendbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer) {
  enqueue(fifoBuffer, (uint64_t)((uintptr_t)sendbuff), count, peer, datatype,
          flagcxDevicePrimSend);
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceRecv(void *recvbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer) {
  enqueue(fifoBuffer, (uint64_t)((uintptr_t)recvbuff), count, peer, datatype,
          flagcxDevicePrimRecv);
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceTerm(void *fifoBuffer) {
  enqueue(fifoBuffer, 0, 0, 0, 0, flagcxDevicePrimTerm);
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceWait(void *fifoBuffer) {
  enqueue(fifoBuffer, 0, 0, 0, 0, flagcxDevicePrimWait);
  unsigned long long int *buffer = (unsigned long long int *)fifoBuffer;
  int distance = buffer[2] - buffer[1];
  int iter = 0;
  while (distance > 0) {
    spinBackoff(iter);
    iter++;
    uint64_t cons = buffer[1];
    uint64_t prod = buffer[2];
    distance = prod - cons;
  }
  return flagcxSuccess;
}

FLAGCX_DEVICE_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr,
                                               uint64_t count,
                                               uint64_t peerRank,
                                               uint64_t datatype,
                                               uint64_t type) {
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[0];
  int distance = buffer[2] - buffer[1];
  int iter = 0;
  while (distance >= capacity) {
    spinBackoff(iter);
    iter++;
    distance = buffer[2] - buffer[1];
  }
  idx = buffer[2] % capacity;
  flagcxDeviceTrigger *trigger = ((flagcxDeviceTrigger *)(buffer + 3)) + idx;
  // trigger->addr = addr;
  // trigger->count = count;
  // trigger->peerRank = peerRank;
  // trigger->datatype = datatype;
  // trigger->type = type;
  printf("device enqueue buffer=0x%016lx, trigger=0x%016lx, idx=%d\n", (uintptr_t)buffer, (uintptr_t)trigger, idx);
  trigger->setValue(addr, count, peerRank, datatype, type);
  printf("enqueued addr=0x%016lx, snd=0x%016lx\n", trigger->fst, trigger->snd);
  FLAGCX_DEVICE_THREAD_FENCE();
  buffer[2] = buffer[2] + 1;
  return flagcxSuccess;
}

// __device__ flagcxResult_t flagcxFifo::dequeue(flagcxReduceTrigger_t trigger)
// {
//   // to be implemented
//   return flagcxNotSupported;
// }
