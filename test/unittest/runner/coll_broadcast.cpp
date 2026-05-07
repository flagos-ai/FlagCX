// Broadcast correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <vector>

TEST_F(FlagCXCollTest, Broadcast) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  // Only root (rank 0) initializes sendbuff; other ranks leave it zeroed
  // so stale data would be detected if broadcast fails.
  if (rank == 0) {
    for (size_t i = 0; i < count; i++) {
      ((float *)hostsendbuff)[i] = 42.0f + (i % 10);
    }
  }
  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxBroadcast(sendbuff, recvbuff, count, flagcxFloat, 0, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Broadcast from root=0: all ranks should receive root's data.
  std::vector<float> expected(count);
  for (size_t i = 0; i < count; i++) {
    expected[i] = 42.0f + (i % 10);
  }
  EXPECT_TRUE(
      verifyBuffer(static_cast<float *>(hostrecvbuff), expected.data(), count));
}
