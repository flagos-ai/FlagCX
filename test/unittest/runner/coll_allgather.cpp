// AllGather correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <iostream>

TEST_F(FlagCXCollTest, AllGather) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxAllGather(sendbuff, recvbuff, count / nranks, flagcxFloat, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // AllGather: each rank contributes count/nranks elements.
  // All ranks sent the same data, so each chunk in recvbuff should match sendbuff.
  size_t chunkCount = count / nranks;
  EXPECT_TRUE(verifyBuffer(static_cast<float *>(hostrecvbuff),
                            static_cast<float *>(hostsendbuff), chunkCount));
}
