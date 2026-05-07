// Scatter correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <iostream>
#include <vector>

TEST_F(FlagCXCollTest, Scatter) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  if (rank == 0) {
    for (size_t i = 0; i < count; i++) {
      ((float *)hostsendbuff)[i] = static_cast<float>(i);
    }

    devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                            flagcxMemcpyHostToDevice, stream);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxScatter(sendbuff, recvbuff, count / nranks, flagcxFloat, 0, comm,
                stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size / nranks,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Scatter from root=0: each rank receives chunk[rank] of root's sendbuff.
  size_t chunkCount = count / nranks;
  size_t chunkStart = rank * chunkCount;
  std::vector<float> expected(chunkCount);
  for (size_t i = 0; i < chunkCount; i++) {
    expected[i] = static_cast<float>(chunkStart + i);
  }

  EXPECT_TRUE(verifyBuffer(static_cast<float *>(hostrecvbuff), expected.data(),
                           chunkCount));
}
