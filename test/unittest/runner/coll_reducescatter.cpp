// ReduceScatter correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <vector>

TEST_F(FlagCXCollTest, ReduceScatter) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = rank * 1000.0f + (i % 10);
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxReduceScatter(sendbuff, recvbuff, count / nranks, flagcxFloat,
                      flagcxSum, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size / nranks,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // ReduceScatter with Sum: each rank gets chunk[rank] of the sum.
  // result[i] = sum over all ranks of (rank * 1000.0f + ((chunkStart + i) %
  // 10))
  size_t chunkCount = count / nranks;
  size_t chunkStart = rank * chunkCount;
  std::vector<float> expected(chunkCount);
  for (size_t i = 0; i < chunkCount; i++) {
    expected[i] = nranks * (nranks - 1) / 2.0f * 1000.0f +
                  nranks * (float)((chunkStart + i) % 10);
  }

  EXPECT_TRUE(verifyBuffer(static_cast<float *>(hostrecvbuff), expected.data(),
                           chunkCount));
}
