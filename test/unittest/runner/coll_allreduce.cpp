// AllReduce correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <vector>

TEST_F(FlagCXCollTest, AllReduce) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = rank * 1000.0f + (i % 10);
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxAllReduce(sendbuff, recvbuff, count, flagcxFloat, flagcxSum, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // AllReduce with Sum: result[i] = sum over all ranks of (rank * 1000.0f + (i
  // % 10))
  //   = nranks*(nranks-1)/2 * 1000.0f + nranks * (i % 10)
  std::vector<float> expected(count);
  for (size_t i = 0; i < count; i++) {
    expected[i] =
        nranks * (nranks - 1) / 2.0f * 1000.0f + nranks * (float)(i % 10);
  }
  EXPECT_TRUE(
      verifyBuffer(static_cast<float *>(hostrecvbuff), expected.data(), count));
}
