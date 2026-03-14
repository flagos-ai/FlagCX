// Reduce correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <iostream>
#include <vector>

TEST_F(FlagCXCollTest, Reduce) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxReduce(sendbuff, recvbuff, count, flagcxFloat, flagcxSum, 0, comm,
               stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    // Reduce with Sum to root=0: result[i] = sendbuff[i] * nranks
    std::vector<float> expected(count);
    for (size_t i = 0; i < count; i++) {
      expected[i] = static_cast<float>(i % 10) * nranks;
    }
    EXPECT_TRUE(verifyBuffer(static_cast<float *>(hostrecvbuff),
                              expected.data(), count));
  }
}
