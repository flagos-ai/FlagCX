#include "comm_lifecycle_check.hpp"
#include "runner_fixtures.hpp"

TEST_F(FlagCXCollTest, FinalizeThenRankLocalDestroy) {
  ASSERT_EQ(devHandle->deviceMemset(sendbuff, 0, size, flagcxMemDevice, stream),
            flagcxSuccess);
  ASSERT_EQ(flagcxAllGather(sendbuff, recvbuff, count / nranks, flagcxFloat,
                            comm, stream),
            flagcxSuccess);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
  ASSERT_EQ(flagcxTestFinalizeAndDestroyRankLocally(&comm, rank, nranks),
            flagcxSuccess);
}
