#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/flagcx_device.h"
#include "global_comm.h"


FLAGCX_GLOBAL_DECORATOR void flagcxOnesidedSendKernel(size_t srcOffset,
                                                      size_t dstOffset,
                                                      size_t signalOffset,
                                                      size_t count,
                                                      flagcxDataType_t datatype,
                                                      int peer,
                                                      flagcxDevComm devComm) {
  int tid = threadIdx.x;
  if (tid == 0) {
    flagcxDevNet net(devComm);
    net.put(srcOffset, dstOffset, count, datatype, peer);
    net.signal(signalOffset, peer);
    net.term();
    net.wait();
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxOnesidedRecvKernel(
    volatile uint64_t *waitAddr, uint64_t expectedValue, volatile int *errorFlag,
    flagcxDevComm devComm) {
  int tid = threadIdx.x;
  if (tid == 0) {
    int iter = 0;
    constexpr int kMaxIters = 1 << 24;
    while (*waitAddr != expectedValue) {
      spinBackoff(iter);
      iter++;
      if (iter >= kMaxIters) {
        if (errorFlag)
          *errorFlag = 1;
        break;
      }
    }
    flagcxDevNet net(devComm);
    net.term();
    net.wait();
  }
}

void flagcxOnesidedSendDemo(size_t srcOffset, size_t dstOffset,
                            size_t signalOffset, size_t count,
                            flagcxDataType_t datatype, int peer,
                            flagcxDevComm_t devComm, flagcxStream_t stream) {
  flagcxDevComm dc(*devComm);
  flagcxOnesidedSendKernel<<<1, 1, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      srcOffset, dstOffset, signalOffset, count, datatype, peer, dc);
}

void flagcxOnesidedRecvDemo(volatile uint64_t *waitAddr, uint64_t expectedValue,
                            volatile int *errorFlag, flagcxDevComm_t devComm,
                            flagcxStream_t stream) {
  flagcxDevComm dc(*devComm);
  flagcxOnesidedRecvKernel<<<1, 1, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      waitAddr, expectedValue, errorFlag, dc);
}

