#include "bootstrap.h"
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <thread>

namespace {

class IntraNodeBootstrapStatus : public ::testing::Test {
protected:
  bootstrapState *states[4] = {};
  volatile uint32_t abortFlags[4] = {};

  void SetUp() override {
    ASSERT_EQ(bootstrapNetInit(), flagcxSuccess);
    flagcxBootstrapHandle handle;
    ASSERT_EQ(bootstrapGetUniqueId(&handle), flagcxSuccess);
    flagcxResult_t results[4];
    std::thread threads[4];
    for (int rank = 0; rank < 4; rank++)
      threads[rank] = std::thread([&, rank] {
        results[rank] = bootstrapCollInit(&handle, rank, 4, handle.magic,
                                          &abortFlags[rank], &states[rank]);
      });
    for (int rank = 0; rank < 4; rank++) {
      threads[rank].join();
      ASSERT_EQ(results[rank], flagcxSuccess);
    }
  }

  void TearDown() override {
    for (int rank = 0; rank < 4; rank++)
      if (states[rank] != nullptr)
        EXPECT_EQ(bootstrapClose(states[rank]), flagcxSuccess);
  }
};

TEST(BootstrapIntraStatusArguments, SingleRankAndInvalidGroups) {
  flagcxResult_t common = flagcxSuccess;
  EXPECT_EQ(bootstrapCollIntraNodeStatus(nullptr, nullptr, 0, 1, 100,
                                         flagcxRemoteError, &common),
            flagcxSuccess);
  EXPECT_EQ(common, flagcxRemoteError);
  EXPECT_EQ(bootstrapCollIntraNodeStatus(nullptr, nullptr, 0, 2, 100,
                                         flagcxSuccess, &common),
            flagcxInvalidArgument);
  EXPECT_EQ(bootstrapCollIntraNodeStatus(nullptr, nullptr, 1, 1, 100,
                                         flagcxSuccess, &common),
            flagcxInvalidArgument);
  bootstrapCollState coll = {};
  coll.nranks = 4;
  bootstrapState state = {};
  state.coll = &coll;
  int duplicate[] = {0, 0};
  EXPECT_EQ(bootstrapCollIntraNodeStatus(&state, duplicate, 0, 2, 100,
                                         flagcxSuccess, &common),
            flagcxInvalidArgument);
}

TEST_F(IntraNodeBootstrapStatus, SparseGroupsConvergeAndRetryIndependently) {
  int groups[2][2] = {{0, 2}, {1, 3}};
  for (int round = 0; round < 2; round++) {
    flagcxResult_t common[4] = {};
    flagcxResult_t results[4] = {};
    std::thread threads[4];
    for (int rank = 0; rank < 4; rank++)
      threads[rank] = std::thread([&, rank] {
        flagcxResult_t local =
            round == 0 && rank >= 2
                ? (rank == 2 ? flagcxSystemError : flagcxRemoteError)
                : flagcxSuccess;
        results[rank] = bootstrapCollIntraNodeStatus(
            states[rank], groups[rank % 2], rank / 2, 2, 100 + 2 * round, local,
            &common[rank]);
      });
    for (int rank = 0; rank < 4; rank++) {
      threads[rank].join();
      EXPECT_EQ(results[rank], flagcxSuccess);
      EXPECT_EQ(common[rank], round == 0 ? (rank % 2 == 0 ? flagcxSystemError
                                                          : flagcxRemoteError)
                                         : flagcxSuccess);
    }
  }
}

TEST_F(IntraNodeBootstrapStatus, DoesNotWaitForOtherNodeGroup) {
  int ranks[] = {0, 2};
  std::future<flagcxResult_t> results[2];
  for (int localRank = 0; localRank < 2; localRank++)
    results[localRank] = std::async(std::launch::async, [&, localRank] {
      flagcxResult_t common;
      flagcxResult_t result = bootstrapCollIntraNodeStatus(
          states[ranks[localRank]], ranks, localRank, 2, 200, flagcxSuccess,
          &common);
      return result == flagcxSuccess ? common : result;
    });
  for (int rank = 0; rank < 2; rank++) {
    bool ready = results[rank].wait_for(std::chrono::seconds(5)) ==
                 std::future_status::ready;
    EXPECT_TRUE(ready);
    if (!ready)
      for (int peer = 0; peer < 4; peer++)
        __atomic_store_n(&abortFlags[peer], 1, __ATOMIC_RELEASE);
    EXPECT_EQ(results[rank].get(), flagcxSuccess);
  }
}

} // namespace
