// Gather correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <vector>

TEST_F(FlagCXCollTest, Gather) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = rank * 1000.0f + (i % 10);
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxGather(sendbuff, recvbuff, count / nranks, flagcxFloat, 0, comm,
               stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    // Gather to root=0: chunk r should contain rank r's data.
    size_t chunkCount = count / nranks;
    std::vector<float> expected(chunkCount);
    for (int r = 0; r < nranks; r++) {
      for (size_t i = 0; i < chunkCount; i++) {
        expected[i] = r * 1000.0f + (i % 10);
      }
      EXPECT_TRUE(
          verifyBuffer(static_cast<float *>(hostrecvbuff) + r * chunkCount,
                       expected.data(), chunkCount))
          << "Mismatch in chunk from rank " << r;
    }
  }
}
