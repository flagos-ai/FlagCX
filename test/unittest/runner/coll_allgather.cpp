// AllGather correctness test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <vector>

TEST_F(FlagCXCollTest, AllGather) {

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = rank * 1000.0f + (i % 10);
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxAllGather(sendbuff, recvbuff, count / nranks, flagcxFloat, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  // AllGather: each rank contributes count/nranks elements.
  // Chunk r should contain rank r's data: r * 1000.0f + (i % 10).
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

TEST_F(FlagCXCollTest, AllGatherInsideGroup) {
  for (size_t i = 0; i < count / nranks; ++i)
    static_cast<float *>(hostsendbuff)[i] = rank * 1000.0f + (i % 10);
  ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);

  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  const flagcxResult_t submitResult = flagcxAllGather(
      sendbuff, recvbuff, count / nranks, flagcxFloat, comm, stream);
  const flagcxResult_t groupResult = flagcxGroupEnd(comm);
  ASSERT_EQ(submitResult, flagcxSuccess);
  ASSERT_EQ(groupResult, flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

  const size_t chunkCount = count / nranks;
  std::vector<float> expected(chunkCount);
  for (int source = 0; source < nranks; ++source) {
    for (size_t i = 0; i < chunkCount; ++i)
      expected[i] = source * 1000.0f + (i % 10);
    EXPECT_TRUE(
        verifyBuffer(static_cast<float *>(hostrecvbuff) + source * chunkCount,
                     expected.data(), chunkCount));
  }
}
