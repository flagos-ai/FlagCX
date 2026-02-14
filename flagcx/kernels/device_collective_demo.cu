#include "comm.h"
#include "flagcx_kernel.h"
#include "global_comm.h"

#define NBLOCKS 1
#define NTHREADS_PER_BLOCK 32

// P2P kernel implementing alltoall pattern (one thread per peer)
// Each thread handles all communication with its assigned peer
// This preserves send/recv ordering per-peer for correct P2P matching
// Note: Uses single block so __syncthreads() can synchronize all threads
// Buffer layout: [rank0_data][rank1_data]...[rankN_data], each of size count
// sendbuff: data at offset peerRank * count is sent to peerRank
// recvbuff: data from peerRank is stored at offset peerRank * count
FLAGCX_GLOBAL_DECORATOR void flagcxP2pKernel(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, int myRank, int nRanks, void *fifoBuffer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Each thread handles one peer (tid = peer index)
  // Skip if tid >= nRanks or tid == myRank (no self-communication)
  if (tid < nRanks && tid != myRank) {
    int peerRank = tid;

    // Calculate offsets for this peer's send and receive buffers
    size_t elementSize = getFlagcxDataTypeSizeDevice(datatype);
    size_t offset = peerRank * count * elementSize;
    const void *peerSendBuff = (const char *)sendbuff + offset;
    void *peerRecvBuff = (char *)recvbuff + offset;

    // Send to peer and receive from peer
    // Each thread's operations are ordered: send then recv
    flagcxDeviceSend(peerSendBuff, count, datatype, peerRank, fifoBuffer);
    flagcxDeviceRecv(peerRecvBuff, count, datatype, peerRank, fifoBuffer);
  }

  // Ensure all threads finish enqueuing before termination
  FLAGCX_DEVICE_SYNC_THREADS();

  // Only thread 0 sends termination and waits
  if (tid == 0) {
    flagcxDeviceTerm(fifoBuffer);
    flagcxDeviceWait(fifoBuffer);
  }
}

// Alltoall demo: each rank sends different data to each peer and receives from all
// sendbuff: size = nRanks * count elements (data for peer i at offset i * count)
// recvbuff: size = nRanks * count elements (data from peer i at offset i * count)
flagcxResult_t flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, flagcxComm_t comm,
                             flagcxStream_t stream) {
  void *fifo = NULL;
  FLAGCXCHECK(flagcxCommFifoBuffer(comm, &fifo));

  int myRank, nRanks;
  FLAGCXCHECK(flagcxCommUserRank(comm, &myRank));
  FLAGCXCHECK(flagcxCommCount(comm, &nRanks));

  // Launch kernel with (NBLOCKS, NTHREADS_PER_BLOCK) (one thread per potential peer)
  // Single block ensures __syncthreads() synchronizes all threads before Term/Wait
  // Each thread handles communication with one peer, preserving ordering
  flagcxP2pKernel<<<NBLOCKS, NTHREADS_PER_BLOCK, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      sendbuff, recvbuff, count, datatype, myRank, nRanks, fifo);
  return flagcxSuccess;
}
