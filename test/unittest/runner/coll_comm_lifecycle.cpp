#include "comm_lifecycle_check.hpp"
#include "runner_fixtures.hpp"

TEST_F(FlagCXCollTest, FinalizeThenDestroy) {
  ASSERT_EQ(devHandle->deviceMemset(sendbuff, 0, size, flagcxMemDevice, stream),
            flagcxSuccess);
  ASSERT_EQ(flagcxAllGather(sendbuff, recvbuff, count / nranks, flagcxFloat,
                            comm, stream),
            flagcxSuccess);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
  ASSERT_EQ(flagcxTestFinalizeAndDestroyAcrossRanks(&comm), flagcxSuccess);
}
