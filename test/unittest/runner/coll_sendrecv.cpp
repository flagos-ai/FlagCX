// Point-to-point Send/Recv correctness test.
// Ring pattern: each rank sends to (rank+1)%nranks, receives from
// (rank-1+nranks)%nranks. Each rank fills its sendbuff with its own rank ID.
// After the exchange, each rank verifies it received the sender's rank ID.

#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <iostream>

TEST_F(FlagCXCollTest, SendRecv) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int sendPeer = (rank + 1) % nranks;
  int recvPeer = (rank - 1 + nranks) % nranks;

  // Fill sendbuff with my rank
  float *hsend = static_cast<float *>(hostsendbuff);
  for (size_t i = 0; i < count; i++) {
    hsend[i] = static_cast<float>(rank);
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Use group API for concurrent send/recv
  flagcxGroupStart(comm);
  flagcxSend(sendbuff, count, flagcxFloat, sendPeer, comm, stream);
  flagcxRecv(recvbuff, count, flagcxFloat, recvPeer, comm, stream);
  flagcxGroupEnd(comm);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: all received elements should equal recvPeer's rank
  float *hrecv = static_cast<float *>(hostrecvbuff);
  float expected = static_cast<float>(recvPeer);
  bool success = true;
  for (size_t i = 0; i < count; i++) {
    if (hrecv[i] != expected) {
      if (rank == 0) {
        std::cout << "Mismatch at index " << i << ": expected " << expected
                  << ", got " << hrecv[i] << std::endl;
      }
      success = false;
      break;
    }
  }
  EXPECT_TRUE(success);
}
