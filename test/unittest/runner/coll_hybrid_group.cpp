#include "global_comm.h"
#include "runner_fixtures.hpp"

#include <cstdlib>
#include <cstring>

TEST_F(FlagCXCollTest, HybridPlannerRejectsOuterGroup) {
  const char *useHetero = std::getenv("FLAGCX_USE_HETERO_COMM");
  const char *useHost = std::getenv("FLAGCX_USE_HOST_COMM");
  int localHybrid =
      comm->commType == flagcxCommunicatorHybrid &&
              comm->hasSingleRankHomoComm == 0 &&
              (useHetero == nullptr || std::strcmp(useHetero, "1") != 0) &&
              (useHost == nullptr || std::strcmp(useHost, "1") != 0)
          ? 1
          : 0;
  int allHybrid = 0;
  MPI_Allreduce(&localHybrid, &allHybrid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allHybrid)
    GTEST_SKIP() << "requires HybridRunner on every rank";

  // The first call creates and caches a planner. The second uses that cache.
  for (int attempt = 0; attempt < 2; ++attempt) {
    ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
    EXPECT_EQ(flagcxReduce(sendbuff, recvbuff, 1, flagcxFloat, flagcxSum, 0,
                           comm, stream),
              flagcxInvalidUsage);
    EXPECT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  }
  EXPECT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
}
