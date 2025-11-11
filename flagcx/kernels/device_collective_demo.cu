#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "global_comm.h"
FLAGCX_GLOBAL_DECORATOR void flagcxP2pKernel(const void *sendbuff,
                                             void *recvbuff, size_t count,
                                             flagcxDataType_t datatype,
                                             int sendPeer, int recvPeer,
                                             void *fifoBuffer) {
  // multiple threads will be supported in future
  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 1; i++) {
      const void *sendaddr = static_cast<const void *>(
          static_cast<char *>(const_cast<void *>(sendbuff)));
      flagcxDeviceSend(sendaddr, count, datatype, sendPeer, fifoBuffer);
    }
    for (int i = 0; i < 1; i++) {
      void *recvaddr = static_cast<void *>(static_cast<char *>(recvbuff));
      flagcxDeviceRecv(recvaddr, count, datatype, recvPeer, fifoBuffer);
    }
    flagcxDeviceTerm(fifoBuffer);
    flagcxDeviceWait(fifoBuffer);
  }
}

void flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                   flagcxDataType_t datatype, int sendPeer, int recvPeer,
                   flagcxComm_t comm, flagcxStream_t stream) {
  void *fifo = NULL;
  flagcxCommFifoBuffer(comm, &fifo);
  flagcxP2pKernel<<<1, 1, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      sendbuff, recvbuff, count, datatype, sendPeer, recvPeer, fifo);
}
