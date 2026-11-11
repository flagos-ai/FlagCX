/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "launch_kernel.h"
#include "net_transport.h"
#include "p2p.h"
#include "p2p_transport.h"
#include "proxy.h"
#include <gtest/gtest.h>

TEST(P2pTransportPlan, ResolvesAllRegistrationCombinations) {
  flagcxP2pTransferPlan plan = flagcxP2pPlanPending;

  ASSERT_EQ(flagcxP2pResolveTransferPlan(0, 0, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanFifo);

  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 0, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanRead);

  ASSERT_EQ(flagcxP2pResolveTransferPlan(0, 1, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanWrite);

  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 1, 0, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanWrite);
}

TEST(P2pTransportPlan, RejectsInvalidOffers) {
  flagcxP2pTransferPlan plan = flagcxP2pPlanPending;
  EXPECT_EQ(flagcxP2pResolveTransferPlan(2, 0, 0, 0, &plan),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxP2pResolveTransferPlan(0, -1, 0, 0, &plan),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxP2pResolveTransferPlan(0, 0, 0, 0, nullptr),
            flagcxInvalidArgument);
}

TEST(P2pTransportPlan, IpcFailureForcesACommonFifoFallback) {
  flagcxP2pTransferPlan plan = flagcxP2pPlanPending;
  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 1, 1, 0, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanFifo);
  ASSERT_EQ(flagcxP2pResolveTransferPlan(1, 1, 0, 1, &plan), flagcxSuccess);
  EXPECT_EQ(plan, flagcxP2pPlanFifo);
  EXPECT_EQ(flagcxP2pResolveTransferPlan(1, 1, 2, 0, &plan),
            flagcxInvalidArgument);
}

TEST(TransportCompletion, DistinguishesPendingFromPermanentFailure) {
  int completed = -1;
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxSuccess, &completed),
            flagcxSuccess);
  EXPECT_EQ(completed, 1);

  completed = -1;
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxInProgress, &completed),
            flagcxSuccess);
  EXPECT_EQ(completed, 0);

  completed = -1;
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxSystemError, &completed),
            flagcxSystemError);
  EXPECT_EQ(completed, 0);
  EXPECT_EQ(flagcxTransportClassifyCompletion(flagcxSuccess, nullptr),
            flagcxInvalidArgument);
}

TEST(TransportNeutralFoundation, RequestAndCreditShareCanonicalState) {
  flagcxTransportCredit credit = {};
  ASSERT_EQ(flagcxTransportCreditInit(&credit, 2), flagcxSuccess);
  ASSERT_EQ(flagcxTransportCreditAcquire(&credit, 2), flagcxSuccess);
  EXPECT_EQ(flagcxTransportCreditAcquire(&credit, 1), flagcxInProgress);
  ASSERT_EQ(flagcxTransportCreditRelease(&credit, 2), flagcxSuccess);

  flagcxTransportRequest request = {};
  flagcxTransportRequestInit(&request);
  ASSERT_EQ(flagcxTransportRequestAcquire(&request), flagcxSuccess);
  ASSERT_EQ(flagcxTransportRequestAddPending(&request, 2), flagcxSuccess);
  ASSERT_EQ(flagcxTransportRequestComplete(&request, 1, flagcxSuccess),
            flagcxSuccess);
  ASSERT_EQ(flagcxTransportRequestComplete(&request, 1, flagcxRemoteError),
            flagcxSuccess);
  int done = 0;
  EXPECT_EQ(flagcxTransportRequestTest(&request, &done), flagcxRemoteError);
  EXPECT_EQ(done, 1);
}

namespace {
class P2pCompletionSemaphore : public flagcxSemaphore {
public:
  int releases = 0;
  flagcxEvent_t getEvent() override { return nullptr; }
  void signalStart() override {}
  void *getSignals() override { return nullptr; }
  void subCounter(int) override { ++releases; }
  void addCounter(int) override {}
  int getCounter() override { return 1 - releases; }
  int pollStart(int, int) override { return 1; }
  int pollEnd() override { return releases == 1; }
  void wait() override {}
};
} // namespace

TEST(P2pTransportPlan, CompletedTransferRetiresAfterPeerReusesOffer) {
  for (bool sender : {false, true}) {
    for (auto plan :
         {flagcxP2pPlanFifo, flagcxP2pPlanRead, flagcxP2pPlanWrite}) {
      SCOPED_TRACE(::testing::Message()
                   << "sender=" << sender << " plan=" << plan);
      flagcxP2pShm shm = {};
      flagcxP2pResources resources = {};
      resources.proxyInfo.shm = &shm;
      auto semaphore = std::make_shared<P2pCompletionSemaphore>();
      auto peerSemaphore = std::make_shared<P2pCompletionSemaphore>();
      flagcxProxyArgs args = {}, peerArgs = {};
      args.semaphore = semaphore;
      args.p2pPlan = plan;
      args.p2pSlotIdx = 0;
      args.p2pPeerSlotIdx = 16;
      args.p2pOpHash = 1;
      args.p2pPeerOpHash = 2;
      args.chunkSteps = args.copied = args.transmitted = 1;
      peerArgs = args;
      peerArgs.semaphore = peerSemaphore;
      peerArgs.p2pSlotIdx = args.p2pPeerSlotIdx;
      peerArgs.p2pPeerSlotIdx = args.p2pSlotIdx;
      peerArgs.p2pOpHash = args.p2pPeerOpHash;
      peerArgs.p2pPeerOpHash = args.p2pOpHash;
      shm.slots[0].opHash = args.p2pOpHash;
      shm.slots[16].opHash = args.p2pPeerOpHash;
      shm.regInfos[0].offerReady = shm.regInfos[16].offerReady = 1;

      auto progress = sender ? flagcxP2pProxySend : flagcxP2pProxyRecv;
      auto peerProgress = sender ? flagcxP2pProxyRecv : flagcxP2pProxySend;
      ASSERT_EQ(progress(&resources, nullptr, 4, &args), flagcxSuccess);
      ASSERT_EQ(args.done, 0);
      ASSERT_EQ(peerProgress(&resources, nullptr, 4, &peerArgs), flagcxSuccess);
      ASSERT_EQ(peerArgs.done, 1);
      ASSERT_EQ(peerSemaphore->releases, 1);
      ASSERT_EQ(shm.slots[16].opHash, UINT64_MAX);

      // A different-size transfer can reuse the peer slot before we retire.
      flagcxProxyArgs nextArgs = {};
      nextArgs.semaphore = std::make_shared<P2pCompletionSemaphore>();
      nextArgs.p2pPlan = flagcxP2pPlanPending;
      nextArgs.p2pSlotIdx = 16;
      nextArgs.p2pPeerSlotIdx = 0;
      nextArgs.p2pOpHash = 3;
      nextArgs.p2pPeerOpHash = 4;
      nextArgs.chunkSteps = 1;
      ASSERT_EQ(peerProgress(&resources, nullptr, 8, &nextArgs), flagcxSuccess);
      ASSERT_EQ(shm.slots[16].opHash, nextArgs.p2pOpHash);
      ASSERT_EQ(shm.regInfos[16].offerReady, 0);

      ASSERT_EQ(progress(&resources, nullptr, 4, &args), flagcxSuccess);
      EXPECT_EQ(args.done, 1);
      EXPECT_EQ(shm.slots[0].opHash, UINT64_MAX);
      EXPECT_EQ(semaphore->releases, 1);
      ASSERT_EQ(progress(&resources, nullptr, 4, &args), flagcxSuccess);
      EXPECT_EQ(semaphore->releases, 1);
      EXPECT_EQ(shm.slots[16].opHash, nextArgs.p2pOpHash);
      EXPECT_EQ(shm.slots[16].peerDone, 0);
      EXPECT_EQ(shm.regInfos[16].offerReady, 0);
    }
  }
}

TEST(P2pTransportPlan, PendingTransferWaitsForPeerOffer) {
  for (bool sender : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "sender=" << sender);
    flagcxP2pShm shm = {};
    flagcxP2pResources resources = {};
    resources.proxyInfo.shm = &shm;
    auto semaphore = std::make_shared<P2pCompletionSemaphore>();
    flagcxProxyArgs args = {};
    args.semaphore = semaphore;
    args.p2pPlan = flagcxP2pPlanPending;
    args.p2pSlotIdx = 0;
    args.p2pPeerSlotIdx = 16;
    args.p2pOpHash = 1;
    args.p2pPeerOpHash = 2;
    args.chunkSteps = 1;
    shm.slots[0].opHash = args.p2pOpHash;
    shm.slots[16].opHash = args.p2pPeerOpHash;

    auto progress = sender ? flagcxP2pProxySend : flagcxP2pProxyRecv;
    ASSERT_EQ(progress(&resources, nullptr, 4, &args), flagcxSuccess);
    EXPECT_EQ(args.p2pPlan, flagcxP2pPlanPending);
    EXPECT_EQ(args.done, 0);
    EXPECT_EQ(args.copied, 0u);
    EXPECT_EQ(args.transmitted, 0u);
    EXPECT_EQ(semaphore->releases, 0);
    EXPECT_EQ(shm.regInfos[0].offerReady, 1);
    EXPECT_EQ(shm.regInfos[16].offerReady, 0);
  }
}
