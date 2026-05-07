// AlltoAll correctness test.
// Each rank fills sendbuff so that chunk i contains data identifying the
// sender. After AlltoAll, each rank verifies it received the correct data from
// each peer.

#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <iostream>

TEST_F(FlagCXCollTest, AlltoAll) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  size_t countPerRank = count / nranks;

  // Fill sendbuff: chunk[i] = rank * nranks + i (so receiver can verify sender)
  float *hsend = static_cast<float *>(hostsendbuff);
  for (int i = 0; i < nranks; i++) {
    for (size_t j = 0; j < countPerRank; j++) {
      hsend[i * countPerRank + j] = static_cast<float>(rank * nranks + i);
    }
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxAlltoAll(sendbuff, recvbuff, countPerRank, flagcxFloat, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: chunk[i] in recvbuff should be data from rank i,
  // which sent rank i's chunk destined for us = i * nranks + rank
  float *hrecv = static_cast<float *>(hostrecvbuff);
  bool success = true;
  for (int i = 0; i < nranks; i++) {
    float expected = static_cast<float>(i * nranks + rank);
    for (size_t j = 0; j < countPerRank; j++) {
      size_t idx = i * countPerRank + j;
      if (hrecv[idx] != expected) {
        if (rank == 0) {
          std::cout << "Mismatch at chunk " << i << " index " << j
                    << ": expected " << expected << ", got " << hrecv[idx]
                    << std::endl;
        }
        success = false;
        break;
      }
    }
    if (!success)
      break;
  }
  EXPECT_TRUE(success);
}
