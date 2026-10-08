// Unit tests for FlagcxSlice and FlagcxTransferTask logic.
// These are header-only tests (no IB hardware required) that verify
// the correctness of markSuccess/markFailed, isAllDone, and hasErrors.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "flagcx_p2p.h"
#include "p2p_scheduler.h"

namespace {

using EngineCreateSignature = FlagcxP2pEngine *(*)();
using EngineReadSignature = int (*)(FlagcxP2pConn *, FlagcxP2pMr, const void *,
                                    size_t, FlagcxP2pRdmaDesc, uint64_t *);
using EngineWriteSignature = int (*)(FlagcxP2pConn *, FlagcxP2pMr, const void *,
                                     size_t, FlagcxP2pRdmaDesc, uint64_t *);
using EngineStatusSignature = bool (*)(FlagcxP2pConn *, uint64_t);

static_assert(std::is_same<decltype(&flagcxP2pEngineCreate),
                           EngineCreateSignature>::value,
              "flagcxP2pEngineCreate public signature changed");
static_assert(
    std::is_same<decltype(&flagcxP2pEngineRead), EngineReadSignature>::value,
    "flagcxP2pEngineRead public signature changed");
static_assert(
    std::is_same<decltype(&flagcxP2pEngineWrite), EngineWriteSignature>::value,
    "flagcxP2pEngineWrite public signature changed");
static_assert(std::is_same<decltype(&flagcxP2pEngineXferStatus),
                           EngineStatusSignature>::value,
              "flagcxP2pEngineXferStatus public signature changed");

static_assert(sizeof(FlagcxP2pRdmaDesc) == 64,
              "FlagcxP2pRdmaDesc wire size changed");
static_assert(offsetof(FlagcxP2pRdmaDesc, addr) == 0,
              "FlagcxP2pRdmaDesc.addr offset changed");
static_assert(offsetof(FlagcxP2pRdmaDesc, size) == 8,
              "FlagcxP2pRdmaDesc.size offset changed");
static_assert(offsetof(FlagcxP2pRdmaDesc, rkey) == 12,
              "FlagcxP2pRdmaDesc.rkey offset changed");
static_assert(offsetof(FlagcxP2pRdmaDesc, nmsgs) == 16,
              "FlagcxP2pRdmaDesc.nmsgs offset changed");
static_assert(offsetof(FlagcxP2pRdmaDesc, rid) == 20,
              "FlagcxP2pRdmaDesc.rid offset changed");
static_assert(offsetof(FlagcxP2pRdmaDesc, idx) == 24,
              "FlagcxP2pRdmaDesc.idx offset changed");
static_assert(offsetof(FlagcxP2pRdmaDesc, padding) == 32,
              "FlagcxP2pRdmaDesc.padding offset changed");

} // namespace

TEST(P2pPublicContractTest, RdmaDescriptorSerializationPreservesEveryByte) {
  FlagcxP2pRdmaDesc input{};
  input.addr = UINT64_C(0x1020304050607080);
  input.size = UINT32_C(0x11223344);
  input.rkey = UINT32_C(0x55667788);
  input.nmsgs = UINT32_C(0x99aabbcc);
  input.rid = UINT32_C(0xddeeff00);
  input.idx = UINT64_C(0x8877665544332211);
  for (size_t i = 0; i < sizeof(input.padding); ++i)
    input.padding[i] = static_cast<char>(i * 7 + 3);

  char wire[sizeof(FlagcxP2pRdmaDesc)]{};
  flagcxP2pSerializeRdmaDesc(input, wire);

  FlagcxP2pRdmaDesc output{};
  flagcxP2pDeserializeRdmaDesc(wire, &output);
  EXPECT_EQ(memcmp(&input, &output, sizeof(input)), 0);
}

TEST(P2pSchedulingTest, AlignedAddressesUseAllDefaultWorkers) {
  int counts[4] = {};
  for (uintptr_t address = 0x100000; address < 0x104000; address += 16) {
    const int owner = flagcxP2pScheduling::workerForAddress(address, 4);
    ASSERT_GE(owner, 0);
    ASSERT_LT(owner, 4);
    counts[owner]++;
    EXPECT_EQ(owner, flagcxP2pScheduling::workerForAddress(address, 4));
  }
  for (int count : counts)
    EXPECT_GT(count, 0);
}

TEST(P2pSchedulingTest, SingleSliceSubmissionsRotateAcrossChannels) {
  const size_t expected[] = {0, 1, 2, 3, 0, 1, 2, 3};
  for (uint64_t ticket = 0; ticket < 8; ticket++)
    EXPECT_EQ(flagcxP2pScheduling::channelForTicket(ticket, 0, 4),
              expected[ticket]);
}

TEST(P2pSchedulingTest, CompletionTrackerWaitsForAcceptedWork) {
  flagcxP2pScheduling::CompletionTracker tracker;
  tracker.pending.store(3, std::memory_order_release);
  tracker.complete(2, 2); // failed work plus one work not submitted

  auto waiter = std::async(std::launch::async, [&tracker] {
    tracker.wait();
    return true;
  });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);

  tracker.complete(); // callback for the one accepted work
  EXPECT_EQ(waiter.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  EXPECT_TRUE(waiter.get());
  EXPECT_EQ(tracker.failed.load(std::memory_order_acquire), 2);
}

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
