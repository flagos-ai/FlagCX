#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/flagcx_device.h"
#include "global_comm.h"

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

FLAGCX_GLOBAL_DECORATOR void flagcxOnesidedSendKernel(const void *srcbuff,
                                                       size_t srcOffset,
                                                       size_t dstOffset,
                                                       size_t signalOffset,
                                                       size_t count,
                                                       flagcxDataType_t datatype,
                                                       int peer, flagcxDevComm devComm) {
  int tid = threadIdx.x;
  if (tid == 0) {
    void *fifoBuffer = devComm.getFifoBuffer();
    flagcxDevicePut(srcbuff, srcOffset, dstOffset, count, datatype, peer,
                    fifoBuffer);
    flagcxDeviceSignal(signalOffset, peer, fifoBuffer);
    flagcxDevNet net(devComm);
    net.term();
    net.wait();
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxOnesidedRecvKernel(volatile uint64_t *waitAddr,
                                                       uint64_t expectedValue,
                                                       flagcxDevComm devComm) {
  int tid = threadIdx.x;
  if (tid == 0) {
    int iter = 0;
    while (*waitAddr != expectedValue) {
      spinBackoff(iter);
      iter++;
    }
    flagcxDevNet net(devComm);
    net.term();
    net.wait();
  }
}

void flagcxOnesidedSendDemo(const void *srcbuff, size_t srcOffset,
                             size_t dstOffset, size_t signalOffset, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxDevComm_t devComm, flagcxStream_t stream) {
  flagcxDevComm dc(*devComm);
  flagcxOnesidedSendKernel<<<1, 1, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      srcbuff, srcOffset, dstOffset, signalOffset, count, datatype, peer, dc);
}

void flagcxOnesidedRecvDemo(volatile uint64_t *waitAddr, uint64_t expectedValue,
                             flagcxDevComm_t devComm, flagcxStream_t stream) {
  flagcxDevComm dc(*devComm);
  flagcxOnesidedRecvKernel<<<1, 1, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      waitAddr, expectedValue, dc);
}

