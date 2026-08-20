#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "uni_runner_impl.h"

namespace {

uniRunnerDagExecutionPlan makePlan(size_t numHostNodes, size_t numRedNodes,
                                   size_t numIpcNodes) {
  static int dummyOrder = 0;
  uniRunnerDagExecutionPlan plan = {};
  plan.numNodes = numHostNodes + numRedNodes + numIpcNodes;
  plan.numHostNodes = numHostNodes;
  plan.numRedNodes = numRedNodes;
  plan.numIpcNodes = numIpcNodes;
  plan.topoOrder = plan.numNodes == 0 ? nullptr : &dummyOrder;
  return plan;
}

} // namespace

TEST(UniRunnerDagShape, LeavesHostOnlyShapeEmpty) {
  uniRunnerDagExecutionPlan plan = makePlan(4, 0, 0);
  size_t nthreads = 99;
  size_t redBlocks = 99;
  size_t ipcBlocks = 99;
  ASSERT_EQ(flagcxSuccess, resolveUniRunnerDagLaunchShape(
                               &plan, 262144, 512, 32, &nthreads,
                               &redBlocks, &ipcBlocks));
  EXPECT_EQ(0u, nthreads);
  EXPECT_EQ(0u, redBlocks);
  EXPECT_EQ(0u, ipcBlocks);
}

TEST(UniRunnerDagShape, DerivesUnifiedThreadsAndExecutorDemand) {
  uniRunnerDagExecutionPlan plan = makePlan(0, 4, 3);
  plan.maxParallelRedNodes = 2;
  plan.maxRedNodeCount = 300;
  plan.maxIpcNodeBytes = 1048577;
  size_t nthreads = 0;
  size_t redBlocks = 0;
  size_t ipcBlocks = 0;

  ASSERT_EQ(flagcxSuccess, resolveUniRunnerDagLaunchShape(
                               &plan, 262144, 512, 32, &nthreads,
                               &redBlocks, &ipcBlocks));
  EXPECT_EQ(512u, nthreads);
  EXPECT_EQ(2u, redBlocks);
  EXPECT_EQ(5u, ipcBlocks);
}

TEST(UniRunnerDagShape, UsesWarpMinimumAndPowerOfTwoCaps) {
  uniRunnerDagExecutionPlan plan = makePlan(0, 0, 1);
  plan.maxIpcNodeBytes = 17;
  size_t nthreads = 0;
  size_t redBlocks = 0;
  size_t ipcBlocks = 0;
  ASSERT_EQ(flagcxSuccess, resolveUniRunnerDagLaunchShape(
                               &plan, 262144, 512, 32, &nthreads,
                               &redBlocks, &ipcBlocks));
  EXPECT_EQ(32u, nthreads);
  EXPECT_EQ(0u, redBlocks);
  EXPECT_EQ(1u, ipcBlocks);

  plan.maxIpcNodeBytes = 1048576;
  ASSERT_EQ(flagcxSuccess, resolveUniRunnerDagLaunchShape(
                               &plan, 262144, 300, 2, &nthreads,
                               &redBlocks, &ipcBlocks));
  EXPECT_EQ(256u, nthreads);
  EXPECT_EQ(2u, ipcBlocks);
}

TEST(UniRunnerStaticSchedule, LeavesHostOnlyDagWithoutExecutorBlocks) {
  uniRunnerDagExecutionPlan plan = makePlan(7, 0, 0);
  uniRunnerStaticExecutorSchedule schedule;

  EXPECT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 0, 0, 0, 0, &schedule));
  EXPECT_EQ(0u, schedule.numRedTasks);
  EXPECT_EQ(0u, schedule.numIpcTasks);
  EXPECT_EQ(0u, schedule.numRedBlocks);
  EXPECT_EQ(0u, schedule.numIpcBlocks);
}

TEST(UniRunnerStaticSchedule, CapsBlocksByTasksWithoutEmptyExecutors) {
  const std::array<size_t, 6> taskCounts = {{1, 2, 127, 128, 129, 513}};
  const std::array<size_t, 4> requestedBlocks = {{1, 2, 3, 1024}};

  for (size_t numTasks : taskCounts) {
    for (size_t requested : requestedBlocks) {
      uniRunnerDagExecutionPlan plan = makePlan(3, numTasks, 0);
      uniRunnerStaticExecutorSchedule schedule;
      ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                                   &plan, requested, 0, 0, 2048,
                                   &schedule));
      EXPECT_EQ(numTasks, schedule.numRedTasks);
      EXPECT_EQ(std::min(numTasks, requested), schedule.numRedBlocks);
      EXPECT_EQ(0u, schedule.numIpcBlocks);
    }
  }
}

TEST(UniRunnerStaticSchedule, AssignsEveryTaskOnceInPerBlockTopoOrder) {
  const size_t numTasks = 131;
  const size_t numBlocks = 3;
  std::vector<size_t> lastOrdinal(numBlocks, 0);
  std::vector<size_t> blockCounts(numBlocks, 0);
  std::vector<bool> seen(numTasks, false);

  for (size_t expectedBlock = 0; expectedBlock < numBlocks;
       ++expectedBlock) {
    for (size_t ordinal = expectedBlock; ordinal < numTasks;
         ordinal += numBlocks) {
      size_t blockIdx = 0;
      size_t blockTaskOrdinal = 0;
      ASSERT_EQ(flagcxSuccess, getUniRunnerStaticTaskAssignment(
                                   ordinal, numTasks, numBlocks, &blockIdx,
                                   &blockTaskOrdinal));
      ASSERT_LT(blockIdx, numBlocks);
      EXPECT_EQ(expectedBlock, blockIdx);
      EXPECT_EQ(ordinal / numBlocks, blockTaskOrdinal);
      EXPECT_FALSE(seen[ordinal]);
      seen[ordinal] = true;
      if (blockCounts[blockIdx] != 0) {
        EXPECT_LT(lastOrdinal[blockIdx], ordinal);
      }
      lastOrdinal[blockIdx] = ordinal;
      EXPECT_EQ(blockCounts[blockIdx]++, blockTaskOrdinal);
    }
  }

  for (bool wasSeen : seen) {
    EXPECT_TRUE(wasSeen);
  }
  EXPECT_EQ(numTasks, blockCounts[0] + blockCounts[1] + blockCounts[2]);
}

TEST(UniRunnerStaticSchedule, SupportsMoreBlocksThanTasks) {
  const size_t numTasks = 3;
  const size_t numBlocks = 8;
  for (size_t ordinal = 0; ordinal < numTasks; ++ordinal) {
    size_t blockIdx = 0;
    size_t blockTaskOrdinal = 99;
    ASSERT_EQ(flagcxSuccess, getUniRunnerStaticTaskAssignment(
                                 ordinal, numTasks, numBlocks, &blockIdx,
                                 &blockTaskOrdinal));
    EXPECT_EQ(ordinal, blockIdx);
    EXPECT_EQ(0u, blockTaskOrdinal);
  }
}

TEST(UniRunnerStaticSchedule,
     CapsIpcBlocksByChunkParallelismRatherThanLogicalTriggers) {
  uniRunnerDagExecutionPlan plan = makePlan(0, 0, 1);
  uniRunnerStaticExecutorSchedule schedule;

  ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 0, 8, 8, 8, &schedule));
  EXPECT_EQ(1u, schedule.numIpcTasks);
  EXPECT_EQ(8u, schedule.numIpcBlocks);

  plan = makePlan(0, 0, 100);
  ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 0, 16, 2, 16, &schedule));
  EXPECT_EQ(100u, schedule.numIpcTasks);
  EXPECT_EQ(2u, schedule.numIpcBlocks);

  const size_t numChunks = 17;
  const size_t numBlocks = 4;
  std::array<size_t, numBlocks> blockCounts = {};
  for (size_t chunk = 0; chunk < numChunks; ++chunk) {
    size_t blockIdx = 0;
    size_t blockChunkOrdinal = 0;
    ASSERT_EQ(flagcxSuccess, getUniRunnerStaticTaskAssignment(
                                 chunk, numChunks, numBlocks, &blockIdx,
                                 &blockChunkOrdinal));
    EXPECT_EQ(chunk % numBlocks, blockIdx);
    EXPECT_EQ(blockCounts[blockIdx]++, blockChunkOrdinal);
  }
  EXPECT_EQ((std::array<size_t, numBlocks>{5, 4, 4, 4}), blockCounts);
}

TEST(UniRunnerStaticSchedule, ClampsMixedExecutorsWithinResidencyBudget) {
  uniRunnerDagExecutionPlan plan = makePlan(4, 10, 10);
  uniRunnerStaticExecutorSchedule schedule;

  ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 10, 10, 10, 5, &schedule));
  EXPECT_EQ(10u, schedule.numRedTasks);
  EXPECT_EQ(10u, schedule.numIpcTasks);
  EXPECT_EQ(3u, schedule.numRedBlocks);
  EXPECT_EQ(2u, schedule.numIpcBlocks);

  ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 2, 3, 10, 5, &schedule));
  EXPECT_EQ(2u, schedule.numRedBlocks);
  EXPECT_EQ(3u, schedule.numIpcBlocks);

  plan = makePlan(0, 2, 100);
  ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 2, 100, 100, 4, &schedule));
  EXPECT_EQ(1u, schedule.numRedBlocks);
  EXPECT_EQ(3u, schedule.numIpcBlocks);

  plan = makePlan(0, 100, 2);
  ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 100, 2, 2, 4, &schedule));
  EXPECT_EQ(3u, schedule.numRedBlocks);
  EXPECT_EQ(1u, schedule.numIpcBlocks);

  ASSERT_EQ(flagcxSuccess, resolveUniRunnerStaticExecutorSchedule(
                               &plan, 100, 2, 2, 2, &schedule));
  EXPECT_EQ(1u, schedule.numRedBlocks);
  EXPECT_EQ(1u, schedule.numIpcBlocks);
}

TEST(UniRunnerStaticSchedule, RejectsZeroBlocksAndInsufficientBudget) {
  uniRunnerStaticExecutorSchedule schedule;
  uniRunnerDagExecutionPlan redPlan = makePlan(1, 2, 0);
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &redPlan, 0, 0, 0, 4,
                                          &schedule));
  EXPECT_EQ(0u, schedule.numRedTasks);

  uniRunnerDagExecutionPlan mixedPlan = makePlan(1, 2, 2);
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &mixedPlan, 1, 1, 2, 1,
                                          &schedule));
  EXPECT_EQ(0u, schedule.numRedBlocks);
  EXPECT_EQ(0u, schedule.numIpcBlocks);

  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &mixedPlan, 1, 1, 0, 4,
                                          &schedule));
  uniRunnerDagExecutionPlan redOnlyPlan = makePlan(0, 2, 0);
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &redOnlyPlan, 1, 0, 1, 4,
                                          &schedule));
}

TEST(UniRunnerStaticSchedule, RejectsMalformedPlansAndNullOutputs) {
  uniRunnerStaticExecutorSchedule schedule;
  uniRunnerDagExecutionPlan plan = makePlan(1, 1, 0);

  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          nullptr, 1, 0, 0, 1, &schedule));
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &plan, 1, 0, 0, 1, nullptr));

  plan.topoOrder = nullptr;
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &plan, 1, 0, 0, 1, &schedule));

  plan = makePlan(1, 1, 0);
  plan.numHostNodes = std::numeric_limits<size_t>::max();
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &plan, 1, 0, 0, 1, &schedule));

  plan = {};
  int unexpectedOrder = 0;
  plan.topoOrder = &unexpectedOrder;
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &plan, 0, 0, 0, 0, &schedule));

  plan.numNodes =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  plan.numRedNodes = plan.numNodes;
  EXPECT_EQ(flagcxInvalidArgument, resolveUniRunnerStaticExecutorSchedule(
                                          &plan, 1, 0, 0, 1, &schedule));
}

TEST(UniRunnerStaticSchedule, RejectsInvalidTaskAssignments) {
  size_t blockIdx = 0;
  size_t blockTaskOrdinal = 0;
  EXPECT_EQ(flagcxInvalidArgument, getUniRunnerStaticTaskAssignment(
                                          0, 0, 1, &blockIdx,
                                          &blockTaskOrdinal));
  EXPECT_EQ(flagcxInvalidArgument, getUniRunnerStaticTaskAssignment(
                                          1, 1, 1, &blockIdx,
                                          &blockTaskOrdinal));
  EXPECT_EQ(flagcxInvalidArgument, getUniRunnerStaticTaskAssignment(
                                          0, 1, 0, &blockIdx,
                                          &blockTaskOrdinal));
  EXPECT_EQ(flagcxInvalidArgument, getUniRunnerStaticTaskAssignment(
                                          0, 1, 1, nullptr,
                                          &blockTaskOrdinal));
  EXPECT_EQ(flagcxInvalidArgument, getUniRunnerStaticTaskAssignment(
                                          0, 1, 1, &blockIdx, nullptr));
}

TEST(UniRunnerStaticResidency, ReservesOneWholeSmForProducerStreams) {
  size_t budget = 0;
  EXPECT_EQ(flagcxSuccess,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 132, 3, 1024, 256, &budget));
  EXPECT_EQ(131u, budget);

  EXPECT_EQ(flagcxSuccess,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 2, 1, 1024, 1024, &budget));
  EXPECT_EQ(1u, budget);
}

TEST(UniRunnerStaticResidency, RejectsUnsupportedProgressCapabilities) {
  size_t budget = 99;
  EXPECT_EQ(flagcxNotSupported,
            resolveUniRunnerStaticExecutorResidencyBudget(
                false, true, 132, 1, 1024, 32, &budget));
  EXPECT_EQ(0u, budget);
  EXPECT_EQ(flagcxNotSupported,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, false, 132, 1, 1024, 32, &budget));
  EXPECT_EQ(flagcxNotSupported,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 1, 1, 1024, 32, &budget));
  EXPECT_EQ(flagcxNotSupported,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 132, 0, 1024, 32, &budget));
}

TEST(UniRunnerStaticResidency, RejectsInvalidThreadsAndArithmeticOverflow) {
  size_t budget = 99;
  EXPECT_EQ(flagcxInvalidArgument,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 132, 1, 1024, 0, &budget));
  EXPECT_EQ(0u, budget);
  EXPECT_EQ(flagcxInvalidArgument,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 132, 1, 1024, 1025, &budget));
  EXPECT_EQ(flagcxInvalidArgument,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 2, std::numeric_limits<size_t>::max(), 1024, 32,
                &budget));
  EXPECT_EQ(flagcxInvalidArgument,
            resolveUniRunnerStaticExecutorResidencyBudget(
                true, true, 132, 1, 1024, 32, nullptr));
}
