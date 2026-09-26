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

namespace {

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
