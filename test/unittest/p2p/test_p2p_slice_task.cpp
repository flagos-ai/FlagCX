// Unit tests for FlagcxSlice and FlagcxTransferTask logic.
// These are header-only tests (no IB hardware required) that verify
// the correctness of markSuccess/markFailed, isAllDone, and hasErrors.

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

#include "flagcx_p2p.h"

// ---------------------------------------------------------------------------
// Test FlagcxTransferTask basic lifecycle
// ---------------------------------------------------------------------------
TEST(SliceTaskTest, TaskStartsEmpty) {
  FlagcxTransferTask task;
  EXPECT_EQ(task.sliceCount.load(), 0);
  EXPECT_EQ(task.doneSliceCount.load(), 0);
  EXPECT_EQ(task.failedCount.load(), 0);
  EXPECT_FALSE(task.isAllDone());
  EXPECT_FALSE(task.hasErrors());
}

TEST(SliceTaskTest, TaskWithZeroSlicesIsNotDone) {
  FlagcxTransferTask task;
  // isAllDone() requires total > 0
  EXPECT_FALSE(task.isAllDone());
}

TEST(SliceTaskTest, TaskIsAllDoneWhenCountsMatch) {
  FlagcxTransferTask task;
  task.sliceCount.store(3);
  EXPECT_FALSE(task.isAllDone());

  task.doneSliceCount.store(1);
  EXPECT_FALSE(task.isAllDone());

  task.doneSliceCount.store(2);
  EXPECT_FALSE(task.isAllDone());

  task.doneSliceCount.store(3);
  EXPECT_TRUE(task.isAllDone());
}

TEST(SliceTaskTest, HasErrorsReturnsFalseInitially) {
  FlagcxTransferTask task;
  EXPECT_FALSE(task.hasErrors());
}

TEST(SliceTaskTest, HasErrorsReturnsTrueAfterFailed) {
  FlagcxTransferTask task;
  task.failedCount.store(1);
  EXPECT_TRUE(task.hasErrors());
}

// ---------------------------------------------------------------------------
// Test FlagcxSlice default initialization
// ---------------------------------------------------------------------------
TEST(SliceTest, SliceDefaultInitialization) {
  FlagcxSlice slice;
  EXPECT_EQ(slice.srcVa, 0);
  EXPECT_EQ(slice.dstVa, 0);
  EXPECT_EQ(slice.length, 0);
  EXPECT_EQ(slice.lkey, 0);
  EXPECT_EQ(slice.rkey, 0);
  EXPECT_EQ(slice.opcode, FLAGCX_SLICE_OP_WRITE);
  EXPECT_EQ(slice.task, nullptr);
  EXPECT_EQ(slice.qpDepth, nullptr);
}

// ---------------------------------------------------------------------------
// Test markSuccess increments doneSliceCount
// ---------------------------------------------------------------------------
TEST(SliceTest, MarkSuccessIncrementsDoneCount) {
  FlagcxTransferTask task;
  FlagcxSlice slice;
  slice.task = &task;

  slice.markSuccess();
  EXPECT_EQ(task.doneSliceCount.load(), 1);
  EXPECT_EQ(task.failedCount.load(), 0);
  EXPECT_FALSE(task.hasErrors());
}

TEST(SliceTest, MarkSuccessWithNullTaskIsSafe) {
  FlagcxSlice slice;
  slice.task = nullptr;
  // Should not crash
  slice.markSuccess();
}

// ---------------------------------------------------------------------------
// Test markFailed increments both doneSliceCount and failedCount
// ---------------------------------------------------------------------------
TEST(SliceTest, MarkFailedIncrementsBothCounters) {
  FlagcxTransferTask task;
  FlagcxSlice slice;
  slice.task = &task;

  slice.markFailed();
  EXPECT_EQ(task.doneSliceCount.load(), 1);
  EXPECT_EQ(task.failedCount.load(), 1);
  EXPECT_TRUE(task.hasErrors());
}

TEST(SliceTest, MarkFailedWithNullTaskIsSafe) {
  FlagcxSlice slice;
  slice.task = nullptr;
  // Should not crash
  slice.markFailed();
}

// ---------------------------------------------------------------------------
// Test mixed success/failure scenario
// ---------------------------------------------------------------------------
TEST(SliceTest, MixedSuccessAndFailure) {
  FlagcxTransferTask task;
  task.sliceCount.store(5);

  FlagcxSlice slices[5];
  for (int i = 0; i < 5; i++)
    slices[i].task = &task;

  // Mark 3 successful, 2 failed
  slices[0].markSuccess();
  slices[1].markSuccess();
  slices[2].markFailed();
  slices[3].markSuccess();
  slices[4].markFailed();

  EXPECT_EQ(task.sliceCount.load(), 5);
  EXPECT_EQ(task.doneSliceCount.load(), 5);
  EXPECT_EQ(task.failedCount.load(), 2);
  EXPECT_TRUE(task.isAllDone());
  EXPECT_TRUE(task.hasErrors());
}

// ---------------------------------------------------------------------------
// Test concurrent marking (atomicity check)
// ---------------------------------------------------------------------------
TEST(SliceTest, ConcurrentMarkingIsAtomic) {
  FlagcxTransferTask task;
  const int numSlices = 1000;
  task.sliceCount.store(numSlices);

  std::vector<FlagcxSlice> slices(numSlices);
  for (auto &s : slices)
    s.task = &task;

  // Mark all slices from multiple threads
  std::vector<std::thread> threads;
  const int numThreads = 4;
  const int slicesPerThread = numSlices / numThreads;

  for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([&, t]() {
      int start = t * slicesPerThread;
      int end = (t == numThreads - 1) ? numSlices : (start + slicesPerThread);
      for (int i = start; i < end; i++) {
        if (i % 10 == 0)
          slices[i].markFailed();
        else
          slices[i].markSuccess();
      }
    });
  }

  for (auto &th : threads)
    th.join();

  EXPECT_EQ(task.doneSliceCount.load(), numSlices);
  EXPECT_EQ(task.failedCount.load(), numSlices / 10);
  EXPECT_TRUE(task.isAllDone());
  EXPECT_TRUE(task.hasErrors());
}

// ---------------------------------------------------------------------------
// Test aggregate initialization still works with default initializers
// ---------------------------------------------------------------------------
TEST(SliceTest, AggregateInitializationWorks) {
  FlagcxTransferTask task;
  // Positional aggregate-init (as used in flagcx_p2p.cc)
  FlagcxSlice *slice = new FlagcxSlice{
      0x1000,               // srcVa
      0x2000,               // dstVa
      1024,                 // length
      0xAABB,               // lkey
      0xCCDD,               // rkey
      FLAGCX_SLICE_OP_READ, // opcode
      &task,                // task
      nullptr               // qpDepth
  };

  EXPECT_EQ(slice->srcVa, 0x1000);
  EXPECT_EQ(slice->dstVa, 0x2000);
  EXPECT_EQ(slice->length, 1024);
  EXPECT_EQ(slice->lkey, 0xAABB);
  EXPECT_EQ(slice->rkey, 0xCCDD);
  EXPECT_EQ(slice->opcode, FLAGCX_SLICE_OP_READ);
  EXPECT_EQ(slice->task, &task);
  EXPECT_EQ(slice->qpDepth, nullptr);

  delete slice;
}

// ---------------------------------------------------------------------------
// Test partial aggregate initialization (new capability)
// ---------------------------------------------------------------------------
TEST(SliceTest, PartialAggregateInitializationUsesDefaults) {
  FlagcxTransferTask task;
  // Only initialize first 3 fields; rest should use defaults
  FlagcxSlice slice = {0x1000, 0x2000, 1024};

  EXPECT_EQ(slice.srcVa, 0x1000);
  EXPECT_EQ(slice.dstVa, 0x2000);
  EXPECT_EQ(slice.length, 1024);
  EXPECT_EQ(slice.lkey, 0);                       // default
  EXPECT_EQ(slice.rkey, 0);                       // default
  EXPECT_EQ(slice.opcode, FLAGCX_SLICE_OP_WRITE); // default
  EXPECT_EQ(slice.task, nullptr);                 // default
  EXPECT_EQ(slice.qpDepth, nullptr);              // default
}
