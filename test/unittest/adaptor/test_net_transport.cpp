/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "ib_transport.h"
#include "net.h"
#include "net_transport.h"
#include "onesided_types.h"

#include <cerrno>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

TEST(NetGetVisibilityTest, TracksIssuedCompletedTargetAndVisibleSequences) {
  flagcxNetGetVisibilityDomain domain = {};
  ASSERT_EQ(flagcxNetGetVisibilityDomainInit(&domain, 3, 17), flagcxSuccess);
  uint64_t first = 0;
  uint64_t second = 0;
  ASSERT_EQ(flagcxNetGetVisibilityIssue(&domain, &first), flagcxSuccess);
  ASSERT_EQ(flagcxNetGetVisibilityIssue(&domain, &second), flagcxSuccess);
  EXPECT_EQ(first, 1u);
  EXPECT_EQ(second, 2u);
  EXPECT_EQ(domain.issuedGetSequence, 2u);

  ASSERT_EQ(flagcxNetGetVisibilityAdvanceData(&domain, 2), flagcxSuccess);
  uint64_t target = 0;
  ASSERT_EQ(flagcxNetGetVisibilityBeginFlush(&domain, &target), flagcxSuccess);
  EXPECT_EQ(target, 2u);
  EXPECT_EQ(domain.flushTargetGetSequence, 2u);
  ASSERT_EQ(flagcxNetGetVisibilityCompleteFlush(&domain, flagcxSuccess),
            flagcxSuccess);
  EXPECT_EQ(domain.visibleGetSequence, 2u);
}

TEST(NetGetVisibilityTest, CancelOnlyRollsBackUncompletedTail) {
  flagcxNetGetVisibilityDomain domain = {};
  ASSERT_EQ(flagcxNetGetVisibilityDomainInit(&domain, 0, 0), flagcxSuccess);
  uint64_t first = 0;
  uint64_t second = 0;
  ASSERT_EQ(flagcxNetGetVisibilityIssue(&domain, &first), flagcxSuccess);
  ASSERT_EQ(flagcxNetGetVisibilityIssue(&domain, &second), flagcxSuccess);
  EXPECT_EQ(flagcxNetGetVisibilityCancelIssue(&domain, first),
            flagcxInvalidArgument);
  ASSERT_EQ(flagcxNetGetVisibilityCancelIssue(&domain, second), flagcxSuccess);
  ASSERT_EQ(flagcxNetGetVisibilityAdvanceData(&domain, first), flagcxSuccess);
  EXPECT_EQ(flagcxNetGetVisibilityCancelIssue(&domain, first),
            flagcxInvalidArgument);
  ASSERT_EQ(flagcxNetGetVisibilityAdvanceVisible(&domain, first),
            flagcxSuccess);
  EXPECT_EQ(domain.visibleGetSequence, first);
}

TEST(NetGetVisibilityTest, RemoveIssueCompactsEveryWatermark) {
  flagcxNetGetVisibilityDomain domain = {};
  ASSERT_EQ(flagcxNetGetVisibilityDomainInit(&domain, 0, 0), flagcxSuccess);
  uint64_t sequence = 0;
  for (int i = 0; i < 3; ++i)
    ASSERT_EQ(flagcxNetGetVisibilityIssue(&domain, &sequence), flagcxSuccess);
  ASSERT_EQ(flagcxNetGetVisibilityAdvanceData(&domain, 3), flagcxSuccess);
  uint64_t target = 0;
  ASSERT_EQ(flagcxNetGetVisibilityBeginFlush(&domain, &target), flagcxSuccess);
  ASSERT_EQ(flagcxNetGetVisibilityCompleteFlush(&domain, flagcxSuccess),
            flagcxSuccess);

  ASSERT_EQ(flagcxNetGetVisibilityRemoveIssue(&domain, 2), flagcxSuccess);
  EXPECT_EQ(domain.issuedGetSequence, 2u);
  EXPECT_EQ(domain.dataCompletedGetSequence, 2u);
  EXPECT_EQ(domain.flushTargetGetSequence, 2u);
  EXPECT_EQ(domain.visibleGetSequence, 2u);
}

TEST(NetGetVisibilityTest, RemovingInflightRangeStillRetiresFlushRequest) {
  flagcxNetGetVisibilityDomain domain = {};
  ASSERT_EQ(flagcxNetGetVisibilityDomainInit(&domain, 0, 0), flagcxSuccess);
  uint64_t sequence = 0;
  ASSERT_EQ(flagcxNetGetVisibilityIssue(&domain, &sequence), flagcxSuccess);
  ASSERT_EQ(flagcxNetGetVisibilityAdvanceData(&domain, sequence),
            flagcxSuccess);
  uint64_t target = 0;
  ASSERT_EQ(flagcxNetGetVisibilityBeginFlush(&domain, &target), flagcxSuccess);
  domain.flushRequest = reinterpret_cast<void *>(0x1);

  ASSERT_EQ(flagcxNetGetVisibilityRemoveIssue(&domain, sequence),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetGetVisibilityCompleteFlush(&domain, flagcxRemoteError),
            flagcxSuccess);
  EXPECT_EQ(domain.flushRequest, nullptr);
  EXPECT_EQ(domain.visibleGetSequence, 0u);
}

TEST(IbTransportEntryTest, OneSidedOperationsRejectNullCommunicator) {
  ASSERT_NE(flagcxNetIb.iput, nullptr);
  ASSERT_NE(flagcxNetIb.iget, nullptr);
  ASSERT_NE(flagcxNetIb.iputSignal, nullptr);

  void *request = reinterpret_cast<void *>(1);
  EXPECT_EQ(
      flagcxNetIb.iput(nullptr, 0, 0, 8, 0, 1, nullptr, nullptr, &request),
      flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);

  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(
      flagcxNetIb.iget(nullptr, 0, 0, 8, 0, 1, nullptr, nullptr, &request),
      flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);

  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(flagcxNetIb.iputSignal(nullptr, 0, 0, 0, 0, 1, nullptr, nullptr, 0,
                                   nullptr, 1, &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);
}

TEST(NetTransportLaneTest, OrderedKeysAreStableAndDomainZeroUsesLaneZero) {
  flagcxNetLaneSet lanes = {4, 2};
  uint32_t lane = UINT32_MAX;

  ASSERT_EQ(flagcxNetSelectLane(&lanes, FLAGCX_NET_LANE_ORDERED, 0, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane, 0u);
  ASSERT_EQ(flagcxNetSelectLane(&lanes, FLAGCX_NET_LANE_ORDERED, 5, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane, 1u);
  ASSERT_EQ(flagcxNetSelectLane(&lanes, FLAGCX_NET_LANE_ORDERED, 5, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane, 1u);
  EXPECT_EQ(lanes.unorderedCursor, 2u);
}

TEST(NetTransportLaneTest, UnorderedLaneAdvancesOnlyAfterCommit) {
  flagcxNetLaneSet lanes = {3, 1};
  uint32_t lane = UINT32_MAX;

  ASSERT_EQ(flagcxNetSelectLane(&lanes, FLAGCX_NET_LANE_UNORDERED, 0, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane, 1u);
  ASSERT_EQ(flagcxNetSelectLane(&lanes, FLAGCX_NET_LANE_UNORDERED, 0, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane, 1u);
  ASSERT_EQ(flagcxNetCommitLane(&lanes, FLAGCX_NET_LANE_UNORDERED, lane),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetSelectLane(&lanes, FLAGCX_NET_LANE_UNORDERED, 0, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane, 2u);
}

TEST(NetTransportLaneTest, RejectsInvalidLaneSetsAndCommits) {
  flagcxNetLaneSet empty = {};
  flagcxNetLaneSet lanes = {2, 0};
  uint32_t lane = 0;
  EXPECT_EQ(flagcxNetSelectLane(&empty, FLAGCX_NET_LANE_ORDERED, 0, &lane),
            flagcxInvalidArgument);
  EXPECT_EQ(
      flagcxNetSelectLane(&lanes, static_cast<flagcxNetLaneMode>(99), 0, &lane),
      flagcxInvalidArgument);
  EXPECT_EQ(flagcxNetCommitLane(&lanes, FLAGCX_NET_LANE_UNORDERED, 1),
            flagcxInvalidArgument);
}

TEST(NetTransportSubmitContextTest, ThreadLocalContextKeepsAdaptorAbiStable) {
  flagcxNetSubmitContext context = {};
  context.orderingKey = 7;
  context.groupId = 11;
  context.generation = 3;
  context.sequence = 19;
  context.flags = FLAGCX_NET_SUBMIT_DATA | FLAGCX_NET_SUBMIT_INDEPENDENT;

  ASSERT_EQ(flagcxNetSetSubmitContext(&context), flagcxSuccess);
  flagcxNetSubmitContext observed = {};
  ASSERT_EQ(flagcxNetGetSubmitContext(&observed), flagcxSuccess);
  EXPECT_EQ(observed.orderingKey, 7u);
  EXPECT_EQ(observed.groupId, 11u);
  EXPECT_EQ(observed.generation, 3u);
  EXPECT_EQ(observed.sequence, 19u);
  EXPECT_NE(observed.flags & FLAGCX_NET_SUBMIT_INDEPENDENT, 0u);

  flagcxNetClearSubmitContext();
  EXPECT_EQ(flagcxNetGetSubmitContext(&observed), flagcxNotSupported);
}

flagcxNetSubmitContext makeSubmitContext(uint64_t sequence, uint64_t generation,
                                         uint64_t groupId = 0,
                                         uint64_t orderingKey = 0) {
  flagcxNetSubmitContext context = {};
  context.orderingKey = orderingKey;
  context.groupId = groupId;
  context.generation = generation;
  context.sequence = sequence;
  return context;
}

TEST(NetTransportScoreboardTest, AdvancesOnlyContiguousCompletedSequence) {
  flagcxNetCompletionEntry entries[4] = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  flagcxNetReleaseGroup group = {};
  ASSERT_EQ(flagcxNetCompletionScoreboardInit(&scoreboard, entries, 4, 7, 10),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupInit(&group, 19, 7), flagcxSuccess);

  flagcxNetSubmitContext first = makeSubmitContext(10, 7, 19, 1);
  flagcxNetSubmitContext second = makeSubmitContext(11, 7, 19, 2);
  flagcxNetSubmitContext third = makeSubmitContext(12, 7, 19, 1);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &first, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &second, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &third, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);

  int done = -1;
  int releaseAllowed = -1;
  EXPECT_EQ(flagcxNetReleaseGroupTest(&group, &done, &releaseAllowed),
            flagcxSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(releaseAllowed, 0);

  uint32_t advanced = UINT32_MAX;
  ASSERT_EQ(
      flagcxNetTrackCompletion(&scoreboard, &third, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 0u);
  ASSERT_EQ(
      flagcxNetTrackCompletion(&scoreboard, &first, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  ASSERT_EQ(
      flagcxNetTrackCompletion(&scoreboard, &second, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 2u);

  EXPECT_EQ(flagcxNetReleaseGroupTest(&group, &done, &releaseAllowed),
            flagcxSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(releaseAllowed, 1);

  uint64_t nextSequence = 0;
  uint32_t inFlight = UINT32_MAX;
  flagcxResult_t firstError = flagcxInternalError;
  ASSERT_EQ(flagcxNetCompletionScoreboardQuery(&scoreboard, &nextSequence,
                                               &inFlight, &firstError),
            flagcxSuccess);
  EXPECT_EQ(nextSequence, 13u);
  EXPECT_EQ(inFlight, 0u);
  EXPECT_EQ(firstError, flagcxSuccess);
}

TEST(NetTransportScoreboardTest, CancelsUnpublishedGroupTail) {
  flagcxNetCompletionEntry entries[2] = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  flagcxNetReleaseGroup group = {};
  ASSERT_EQ(flagcxNetCompletionScoreboardInit(&scoreboard, entries, 2, 6, 4),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupInit(&group, 9, 6), flagcxSuccess);

  flagcxNetSubmitContext context = makeSubmitContext(4, 6, 9);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &context, &group), flagcxSuccess);
  EXPECT_EQ(group.pending, 1u);
  EXPECT_EQ(group.members, 1u);
  ASSERT_EQ(flagcxNetTrackCancel(&scoreboard, &context), flagcxSuccess);
  EXPECT_EQ(group.pending, 0u);
  EXPECT_EQ(group.members, 0u);
  EXPECT_EQ(scoreboard.inFlight, 0u);

  // The same tail sequence can be prepared again because cancellation did not
  // publish or retire it.
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &context, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);
  uint32_t advanced = 0;
  ASSERT_EQ(
      flagcxNetTrackCompletion(&scoreboard, &context, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
}

TEST(NetTransportReleaseGroupTest, SuppressesReleaseAndKeepsFirstError) {
  flagcxNetCompletionEntry entries[2] = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  flagcxNetReleaseGroup group = {};
  ASSERT_EQ(flagcxNetCompletionScoreboardInit(&scoreboard, entries, 2, 4, 0),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupInit(&group, 8, 4), flagcxSuccess);
  flagcxNetSubmitContext first = makeSubmitContext(0, 4, 8);
  flagcxNetSubmitContext second = makeSubmitContext(1, 4, 8);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &first, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &second, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);

  uint32_t advanced = 0;
  ASSERT_EQ(flagcxNetTrackCompletion(&scoreboard, &second, flagcxRemoteError,
                                     &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 0u);
  ASSERT_EQ(flagcxNetTrackCompletion(&scoreboard, &first, flagcxSystemError,
                                     &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 2u);

  int done = 0;
  int releaseAllowed = 1;
  EXPECT_EQ(flagcxNetReleaseGroupTest(&group, &done, &releaseAllowed),
            flagcxRemoteError);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(releaseAllowed, 0);

  uint64_t nextSequence = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(flagcxNetCompletionScoreboardQuery(&scoreboard, &nextSequence,
                                               &inFlight, &firstError),
            flagcxSuccess);
  EXPECT_EQ(firstError, flagcxRemoteError);
}

TEST(NetTransportReleaseGroupTest, AggregatesConcurrentCompletions) {
  constexpr uint32_t count = 32;
  flagcxNetCompletionEntry entries[count] = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  flagcxNetReleaseGroup group = {};
  ASSERT_EQ(
      flagcxNetCompletionScoreboardInit(&scoreboard, entries, count, 2, 0),
      flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupInit(&group, 6, 2), flagcxSuccess);

  flagcxNetSubmitContext contexts[count] = {};
  for (uint32_t i = 0; i < count; ++i) {
    contexts[i] = makeSubmitContext(i, 2, 6, i % 4);
    ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &contexts[i], &group),
              flagcxSuccess);
  }
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);

  flagcxResult_t completionResults[count] = {};
  std::vector<std::thread> threads;
  threads.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    threads.emplace_back([&, i] {
      uint32_t advanced = 0;
      const flagcxResult_t terminal =
          i == 7 ? flagcxRemoteError : flagcxSuccess;
      completionResults[i] = flagcxNetTrackCompletion(&scoreboard, &contexts[i],
                                                      terminal, &advanced);
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  for (flagcxResult_t result : completionResults)
    EXPECT_EQ(result, flagcxSuccess);

  int done = 0;
  int releaseAllowed = 1;
  EXPECT_EQ(flagcxNetReleaseGroupTest(&group, &done, &releaseAllowed),
            flagcxRemoteError);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(releaseAllowed, 0);
  uint64_t nextSequence = 0;
  uint32_t inFlight = count;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(flagcxNetCompletionScoreboardQuery(&scoreboard, &nextSequence,
                                               &inFlight, &firstError),
            flagcxSuccess);
  EXPECT_EQ(nextSequence, count);
  EXPECT_EQ(inFlight, 0u);
  EXPECT_EQ(firstError, flagcxRemoteError);
}

TEST(NetTransportReleaseGroupTest, EmptySealedGroupCanRelease) {
  flagcxNetReleaseGroup group = {};
  ASSERT_EQ(flagcxNetReleaseGroupInit(&group, 1, 3), flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);
  int done = 0;
  int releaseAllowed = 0;
  EXPECT_EQ(flagcxNetReleaseGroupTest(&group, &done, &releaseAllowed),
            flagcxSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(releaseAllowed, 1);
}

TEST(NetTransportScoreboardTest, InProgressDoesNotCompleteMember) {
  flagcxNetCompletionEntry entry = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  flagcxNetReleaseGroup group = {};
  ASSERT_EQ(flagcxNetCompletionScoreboardInit(&scoreboard, &entry, 1, 1, 0),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupInit(&group, 1, 1), flagcxSuccess);
  flagcxNetSubmitContext context = makeSubmitContext(0, 1, 1);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &context, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);

  uint32_t advanced = 1;
  EXPECT_EQ(flagcxNetTrackCompletion(&scoreboard, &context, flagcxInProgress,
                                     &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 0u);
  int done = 1;
  int releaseAllowed = 1;
  EXPECT_EQ(flagcxNetReleaseGroupTest(&group, &done, &releaseAllowed),
            flagcxSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(releaseAllowed, 0);
}

TEST(NetTransportScoreboardTest, RejectsStaleGenerationAndCapacityOverflow) {
  flagcxNetCompletionEntry entries[2] = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  ASSERT_EQ(flagcxNetCompletionScoreboardInit(&scoreboard, entries, 2, 5, 20),
            flagcxSuccess);
  flagcxNetSubmitContext stale = makeSubmitContext(20, 4);
  flagcxNetSubmitContext outsideWindow = makeSubmitContext(22, 5);
  EXPECT_EQ(flagcxNetTrackSubmit(&scoreboard, &stale, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxNetTrackSubmit(&scoreboard, &outsideWindow, nullptr),
            flagcxInProgress);
}

TEST(NetTransportOrderingGenerationTest, ReusesOnlyCompletedNewGeneration) {
  flagcxNetCompletionEntry entry = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  flagcxNetReleaseGroup group = {};
  ASSERT_EQ(flagcxNetCompletionScoreboardInit(&scoreboard, &entry, 1, 9, 0),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupInit(&group, 7, 9), flagcxSuccess);
  EXPECT_EQ(flagcxNetReleaseGroupReset(&group, 7, 10), flagcxInvalidArgument);
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupReset(&group, 7, 10), flagcxSuccess);
  ASSERT_EQ(flagcxNetCompletionScoreboardReset(&scoreboard, 10, 0),
            flagcxSuccess);

  flagcxNetSubmitContext stale = makeSubmitContext(0, 9, 7);
  flagcxNetSubmitContext current = makeSubmitContext(0, 10, 7);
  EXPECT_EQ(flagcxNetTrackSubmit(&scoreboard, &stale, &group),
            flagcxInvalidArgument);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &current, &group), flagcxSuccess);
  ASSERT_EQ(flagcxNetReleaseGroupSeal(&group), flagcxSuccess);
  uint32_t advanced = 0;
  EXPECT_EQ(
      flagcxNetTrackCompletion(&scoreboard, &stale, flagcxSuccess, &advanced),
      flagcxInvalidArgument);
  ASSERT_EQ(
      flagcxNetTrackCompletion(&scoreboard, &current, flagcxSuccess, &advanced),
      flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
}

TEST(NetTransportScoreboardTest, UngroupedSubmissionRequiresGroupIdZero) {
  flagcxNetCompletionEntry entry = {};
  flagcxNetCompletionScoreboard scoreboard = {};
  ASSERT_EQ(flagcxNetCompletionScoreboardInit(&scoreboard, &entry, 1, 2, 0),
            flagcxSuccess);
  flagcxNetSubmitContext grouped = makeSubmitContext(0, 2, 4);
  EXPECT_EQ(flagcxNetTrackSubmit(&scoreboard, &grouped, nullptr),
            flagcxInvalidArgument);
  flagcxNetSubmitContext ungrouped = makeSubmitContext(0, 2);
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard, &ungrouped, nullptr),
            flagcxSuccess);
  uint32_t advanced = 0;
  ASSERT_EQ(flagcxNetTrackCompletion(&scoreboard, &ungrouped, flagcxSuccess,
                                     &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
}

TEST(NetTransportCreditTest, ReportsBackpressureAndRejectsUnderflow) {
  flagcxNetCredit credit = {};
  ASSERT_EQ(flagcxNetCreditInit(&credit, 4), flagcxSuccess);
  ASSERT_EQ(flagcxNetCreditAcquire(&credit, 3), flagcxSuccess);
  EXPECT_EQ(flagcxNetCreditAcquire(&credit, 2), flagcxInProgress);
  uint32_t available = 0;
  ASSERT_EQ(flagcxNetCreditAvailable(&credit, &available), flagcxSuccess);
  EXPECT_EQ(available, 1u);
  EXPECT_EQ(flagcxNetCreditRelease(&credit, 4), flagcxInvalidArgument);
  ASSERT_EQ(flagcxNetCreditRelease(&credit, 2), flagcxSuccess);
  ASSERT_EQ(flagcxNetCreditAcquire(&credit, 2), flagcxSuccess);
  ASSERT_EQ(flagcxNetCreditAvailable(&credit, &available), flagcxSuccess);
  EXPECT_EQ(available, 1u);
}

TEST(NetTransportRequestTest, CompositeRequestCompletesAfterEveryEvent) {
  flagcxNetRequestCore core = {};
  flagcxNetRequestCoreInit(&core);
  ASSERT_EQ(flagcxNetRequestCoreAcquire(&core), flagcxSuccess);
  EXPECT_EQ(flagcxNetRequestCoreAcquire(&core), flagcxInProgress);
  ASSERT_EQ(flagcxNetRequestCoreAddPending(&core, 3), flagcxSuccess);

  int done = -1;
  ASSERT_EQ(flagcxNetRequestCoreTest(&core, &done), flagcxSuccess);
  EXPECT_EQ(done, 0);
  ASSERT_EQ(flagcxNetRequestCoreComplete(&core, 1, flagcxSuccess),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetRequestCoreComplete(&core, 1, flagcxRemoteError),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetRequestCoreTest(&core, &done), flagcxSuccess);
  EXPECT_EQ(done, 0);
  ASSERT_EQ(flagcxNetRequestCoreComplete(&core, 1, flagcxSystemError),
            flagcxSuccess);
  EXPECT_EQ(flagcxNetRequestCoreTest(&core, &done), flagcxRemoteError);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(flagcxNetRequestCoreRelease(&core), flagcxSuccess);
  EXPECT_EQ(flagcxNetRequestCoreAcquire(&core), flagcxSuccess);
}

TEST(NetTransportRequestTest, ZeroEventRequestCanFinishExplicitly) {
  flagcxNetRequestCore core = {};
  flagcxNetRequestCoreInit(&core);
  ASSERT_EQ(flagcxNetRequestCoreAcquire(&core), flagcxSuccess);
  ASSERT_EQ(flagcxNetRequestCoreFinish(&core, flagcxSuccess), flagcxSuccess);
  int done = 0;
  EXPECT_EQ(flagcxNetRequestCoreTest(&core, &done), flagcxSuccess);
  EXPECT_EQ(done, 1);
}

TEST(NetTransportRequestTest, RejectsCompletionUnderflow) {
  flagcxNetRequestCore core = {};
  flagcxNetRequestCoreInit(&core);
  ASSERT_EQ(flagcxNetRequestCoreAcquire(&core), flagcxSuccess);
  ASSERT_EQ(flagcxNetRequestCoreAddPending(&core, 1), flagcxSuccess);
  EXPECT_EQ(flagcxNetRequestCoreComplete(&core, 2, flagcxSuccess),
            flagcxInvalidArgument);
}

TEST(NetTransportPostTest, ValidatesAcceptedPrefix) {
  flagcxNetPostResult post = {};
  ASSERT_EQ(flagcxNetPostResultInit(&post, 4, 2, flagcxInProgress),
            flagcxSuccess);
  EXPECT_EQ(post.requested, 4);
  EXPECT_EQ(post.accepted, 2);
  EXPECT_EQ(post.result, flagcxInProgress);
  EXPECT_EQ(flagcxNetPostResultInit(&post, 2, 3, flagcxSuccess),
            flagcxInvalidArgument);
}

int transportPostResult = 0;
int transportPostCalls = 0;
ibv_send_wr *transportFirstRejected = nullptr;

int fakeTransportPostSend(ibv_qp *, ibv_send_wr *, ibv_send_wr **badWr) {
  ++transportPostCalls;
  if (badWr != nullptr)
    *badWr = transportFirstRejected;
  return transportPostResult;
}

TEST(IbTransportPostTest, ReportsAcceptedPrefixFromLinkedWrList) {
  ibv_context context = {};
  ibv_qp nativeQp = {};
  flagcxIbQp qp = {};
  flagcxIbLane lane = {};
  ibv_send_wr wrs[3] = {};
  wrs[0].next = &wrs[1];
  wrs[1].next = &wrs[2];
  context.ops.post_send = fakeTransportPostSend;
  nativeQp.context = &context;
  qp.qp = &nativeQp;
  lane.ibQp = &qp;
  transportPostResult = ENOMEM;
  transportPostCalls = 0;
  transportFirstRejected = &wrs[2];

  flagcxNetPostResult post = {};
  ASSERT_EQ(flagcxIbPostSendList(&lane, wrs, 3, true, &post), flagcxSuccess);
  EXPECT_EQ(transportPostCalls, 1);
  EXPECT_EQ(post.result, flagcxInProgress);
  EXPECT_EQ(post.requested, 3);
  EXPECT_EQ(post.accepted, 2);
}

TEST(IbTransportPostTest, RejectsBadWrOutsideSubmittedList) {
  ibv_context context = {};
  ibv_qp nativeQp = {};
  flagcxIbQp qp = {};
  flagcxIbLane lane = {};
  ibv_send_wr wr = {};
  ibv_send_wr unrelated = {};
  context.ops.post_send = fakeTransportPostSend;
  nativeQp.context = &context;
  qp.qp = &nativeQp;
  lane.ibQp = &qp;
  transportPostResult = EINVAL;
  transportPostCalls = 0;
  transportFirstRejected = &unrelated;

  flagcxNetPostResult post = {};
  EXPECT_EQ(flagcxIbPostSendList(&lane, &wr, 1, true, &post), flagcxSuccess);
  EXPECT_EQ(post.result, flagcxInternalError);
  EXPECT_EQ(post.accepted, 0);
}

TEST(NetTransportRangeTest, ResolvesInteriorPointerAndMrInfo) {
  uintptr_t bases[2] = {0x1000, 0x4000};
  size_t sizes[2] = {0x1000, 0x2000};
  flagcxNetMrInfo mrInfos[2] = {};
  flagcxOneSideHandleInfo info = {};
  info.baseVas = bases;
  info.regionSizes = sizes;
  info.mrInfos = mrInfos;
  info.nRanks = 2;

  flagcxNetResolvedRange range = {};
  ASSERT_EQ(flagcxNetResolveOneSideRange(&info, 1, 0x120, 0x80, &range),
            flagcxSuccess);
  EXPECT_EQ(range.address, 0x4120u);
  EXPECT_EQ(range.size, 0x80u);
  EXPECT_EQ(range.mrInfo, &mrInfos[1]);
}

TEST(NetTransportRangeTest, RejectsOutOfBoundsAndAddressOverflow) {
  uintptr_t base = UINTPTR_MAX - 15;
  size_t size = 32;
  flagcxNetMrInfo mrInfo = {};
  flagcxOneSideHandleInfo info = {};
  info.baseVas = &base;
  info.regionSizes = &size;
  info.mrInfos = &mrInfo;
  info.nRanks = 1;
  flagcxNetResolvedRange range = {};

  EXPECT_EQ(flagcxNetResolveOneSideRange(&info, 0, 16, 1, &range),
            flagcxInvalidArgument);
  base = 0x1000;
  EXPECT_EQ(flagcxNetResolveOneSideRange(&info, 0, 31, 2, &range),
            flagcxInvalidArgument);
}

flagcxResult_t fakeBatchTest(void *request, int *done, int *) {
  const uintptr_t value = reinterpret_cast<uintptr_t>(request);
  *done = value != 2;
  return value == 3 ? flagcxRemoteError : flagcxSuccess;
}

TEST(NetTransportBatchTest, CollectsCompletionAndFirstError) {
  void *requests[4] = {nullptr, reinterpret_cast<void *>(1),
                       reinterpret_cast<void *>(2),
                       reinterpret_cast<void *>(3)};
  int doneFlags[4] = {};
  int doneCount = -1;
  EXPECT_EQ(flagcxNetTestBatchCommon(requests, 4, doneFlags, &doneCount,
                                     fakeBatchTest),
            flagcxRemoteError);
  EXPECT_EQ(doneCount, 3);
  EXPECT_EQ(doneFlags[0], 1);
  EXPECT_EQ(doneFlags[1], 1);
  EXPECT_EQ(doneFlags[2], 0);
  EXPECT_EQ(doneFlags[3], 1);
  EXPECT_EQ(requests[0], nullptr);
  EXPECT_EQ(requests[1], nullptr);
  EXPECT_EQ(requests[2], reinterpret_cast<void *>(2));
  EXPECT_EQ(requests[3], nullptr);
}

TEST(IbTransportLaneTest, CompatibilityOrderKeySelectsQpZero) {
  flagcxIbNetCommBase comm = {};
  ibv_qp nativeQps[2] = {};
  comm.ready = 1;
  comm.nqps = 2;
  comm.qpIndex = 1;
  comm.qps[0].qp = &nativeQps[0];
  comm.qps[0].devIndex = 0;
  comm.qps[0].remDevIdx = 1;
  comm.qps[1].qp = &nativeQps[1];
  comm.qps[1].devIndex = 1;
  comm.qps[1].remDevIdx = 0;

  flagcxIbLane lane = {};
  ASSERT_EQ(flagcxIbSelectLane(&comm, FLAGCX_NET_LANE_ORDERED, 0, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane.base.index, 0u);
  EXPECT_EQ(lane.base.localDevIndex, 0);
  EXPECT_EQ(lane.base.remoteDevIndex, 1);
  EXPECT_EQ(lane.ibQp, &comm.qps[0]);
  EXPECT_EQ(comm.qpIndex, 1);
}

TEST(IbTransportLaneTest, UnorderedSelectionPreservesLegacyCursor) {
  flagcxIbNetCommBase comm = {};
  ibv_qp nativeQps[2] = {};
  comm.ready = 1;
  comm.nqps = 2;
  comm.qpIndex = 1;
  comm.qps[0].qp = &nativeQps[0];
  comm.qps[1].qp = &nativeQps[1];

  flagcxIbLane lane = {};
  ASSERT_EQ(flagcxIbSelectLane(&comm, FLAGCX_NET_LANE_UNORDERED, 0, &lane),
            flagcxSuccess);
  EXPECT_EQ(lane.base.index, 1u);
  EXPECT_EQ(comm.qpIndex, 1);
  ASSERT_EQ(flagcxIbCommitLane(&comm, FLAGCX_NET_LANE_UNORDERED, &lane),
            flagcxSuccess);
  EXPECT_EQ(comm.qpIndex, 0);
}

TEST(IbTransportLaneTest, IndependentDomainUsesStableStripedLaneGroup) {
  flagcxIbNetCommBase comm = {};
  ibv_qp nativeQps[4] = {};
  comm.ready = 1;
  comm.nqps = 4;
  comm.qpIndex = 3;
  for (int i = 0; i < comm.nqps; ++i)
    comm.qps[i].qp = &nativeQps[i];

  flagcxNetSubmitContext submit = {};
  uint64_t laneMask = 0;
  submit.orderingKey = 6;
  submit.flags = FLAGCX_NET_SUBMIT_DATA | FLAGCX_NET_SUBMIT_INDEPENDENT;
  submit.laneMask = &laneMask;
  ASSERT_EQ(flagcxNetSetSubmitContext(&submit), flagcxSuccess);

  flagcxIbDataLanePolicy policy = {};
  flagcxIbGetDataLanePolicy(&policy);
  EXPECT_EQ(policy.mode, FLAGCX_NET_LANE_ORDERED);
  flagcxIbLane first = {};
  flagcxIbLane second = {};
  ASSERT_EQ(flagcxIbSelectDataLane(&comm, &policy, 0, &first), flagcxSuccess);
  ASSERT_EQ(flagcxIbSelectDataLane(&comm, &policy, 1, &second), flagcxSuccess);
  EXPECT_EQ(first.base.index, 2u);
  EXPECT_EQ(second.base.index, 3u);
  EXPECT_EQ(comm.qpIndex, 3);
  EXPECT_EQ(flagcxIbCommitDataLane(&comm, &policy, &first), flagcxSuccess);
  EXPECT_EQ(comm.qpIndex, 3);
  EXPECT_EQ(laneMask, 1ULL << first.base.index);
  flagcxNetClearSubmitContext();
}

TEST(IbTransportLaneTest, CompatibilityDataPolicyRetainsRoundRobin) {
  flagcxIbNetCommBase comm = {};
  ibv_qp nativeQps[2] = {};
  comm.ready = 1;
  comm.nqps = 2;
  comm.qpIndex = 1;
  comm.qps[0].qp = &nativeQps[0];
  comm.qps[1].qp = &nativeQps[1];

  flagcxIbDataLanePolicy policy = {};
  flagcxIbGetDataLanePolicy(&policy);
  EXPECT_EQ(policy.mode, FLAGCX_NET_LANE_UNORDERED);
  flagcxIbLane lane = {};
  ASSERT_EQ(flagcxIbSelectDataLane(&comm, &policy, 0, &lane), flagcxSuccess);
  EXPECT_EQ(lane.base.index, 1u);
  ASSERT_EQ(flagcxIbCommitDataLane(&comm, &policy, &lane), flagcxSuccess);
  EXPECT_EQ(comm.qpIndex, 0);
}

TEST(IbTransportLaneTest, DataLaneGeometryRequiresExactPeerAgreement) {
  EXPECT_EQ(flagcxIbValidateDataLaneGeometry(2, 0, 2, 0), flagcxSuccess);
  EXPECT_EQ(flagcxIbValidateDataLaneGeometry(2, 1, 2, 1), flagcxSuccess);
  EXPECT_EQ(flagcxIbValidateDataLaneGeometry(1, 0, 2, 0), flagcxInvalidUsage);
  EXPECT_EQ(flagcxIbValidateDataLaneGeometry(2, 0, 2, 1), flagcxInvalidUsage);
  EXPECT_EQ(flagcxIbValidateDataLaneGeometry(0, 0, 2, 0),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxIbValidateDataLaneGeometry(2, 2, 2, 0),
            flagcxInvalidArgument);
}

TEST(IbTransportKeyTest, UsesLaneLocalAndRemoteNicKeys) {
  uintptr_t base = 0x1000;
  size_t regionSize = 0x1000;
  flagcxNetMrInfo mrInfo = {};
  mrInfo.nKeys = 2;
  mrInfo.lkeys[0] = 10;
  mrInfo.lkeys[1] = 11;
  mrInfo.rkeys[0] = 20;
  mrInfo.rkeys[1] = 21;
  flagcxOneSideHandleInfo info = {};
  info.baseVas = &base;
  info.regionSizes = &regionSize;
  info.mrInfos = &mrInfo;
  info.nRanks = 1;
  flagcxIbLane lane = {};
  lane.base.localDevIndex = 1;
  lane.base.remoteDevIndex = 0;

  flagcxNetResolvedRange range = {};
  uint32_t key = 0;
  ASSERT_EQ(
      flagcxIbResolveOneSideRange(&info, 0, 16, 32, &lane, true, &range, &key),
      flagcxSuccess);
  EXPECT_EQ(range.address, 0x1010u);
  EXPECT_EQ(key, 11u);
  ASSERT_EQ(
      flagcxIbResolveOneSideRange(&info, 0, 16, 32, &lane, false, &range, &key),
      flagcxSuccess);
  EXPECT_EQ(key, 20u);
}

} // namespace
