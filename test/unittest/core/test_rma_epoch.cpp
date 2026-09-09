#include "flagcx_hetero.h"
#include <gtest/gtest.h>

TEST(RmaEpochPolicy, OrdinaryOperationsRemainConcurrent) {
  EXPECT_TRUE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_PUT, false, false));
  EXPECT_TRUE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_PUT, true, false));
  EXPECT_TRUE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_GET, true, false));
  EXPECT_TRUE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_PUT_VALUE, true, false));
}

TEST(RmaEpochPolicy, ReleaseBarrierWaitsForPriorOperations) {
  EXPECT_TRUE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_PUT_SIGNAL, false, false));
  EXPECT_FALSE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_PUT_SIGNAL, true, false));
}

TEST(RmaEpochPolicy, InFlightReleaseBarrierBlocksFollowingOperations) {
  EXPECT_FALSE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_PUT, true, true));
  EXPECT_FALSE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_GET, true, true));
  EXPECT_FALSE(flagcxRmaProxyCanPostDesc(FLAGCX_RMA_PUT_SIGNAL, true, true));
}

TEST(RmaEpochPolicy, OnlyInProgressIsRetryable) {
  EXPECT_TRUE(flagcxRmaResultIsRetryable(flagcxInProgress));
  EXPECT_FALSE(flagcxRmaResultIsRetryable(flagcxSystemError));
  EXPECT_FALSE(flagcxRmaResultIsRetryable(flagcxInternalError));
  EXPECT_FALSE(flagcxRmaResultIsRetryable(flagcxRemoteError));
}

TEST(RmaEpochPolicy, FailedPeerRejectsFurtherEnqueue) {
  EXPECT_TRUE(flagcxRmaProxyCanEnqueue(flagcxSuccess, FLAGCX_RMA_PEER_ACTIVE));
  EXPECT_FALSE(
      flagcxRmaProxyCanEnqueue(flagcxRemoteError, FLAGCX_RMA_PEER_ACTIVE));
  EXPECT_FALSE(flagcxRmaProxyCanEnqueue(flagcxSuccess, FLAGCX_RMA_PEER_FAILED));
}

TEST(RmaEpochPolicy, IpcBypassRequiresAnIdleHealthyPeer) {
  EXPECT_TRUE(flagcxRmaProxyCanUseIpc(flagcxSuccess, FLAGCX_RMA_PEER_ACTIVE,
                                      false, false));
  EXPECT_FALSE(flagcxRmaProxyCanUseIpc(flagcxSuccess, FLAGCX_RMA_PEER_ACTIVE,
                                       true, false));
  EXPECT_FALSE(flagcxRmaProxyCanUseIpc(flagcxSuccess, FLAGCX_RMA_PEER_ACTIVE,
                                       false, true));
  EXPECT_FALSE(flagcxRmaProxyCanUseIpc(flagcxRemoteError,
                                       FLAGCX_RMA_PEER_ACTIVE, false, false));
  EXPECT_FALSE(flagcxRmaProxyCanUseIpc(flagcxSuccess, FLAGCX_RMA_PEER_FAILED,
                                       false, false));
}

TEST(RmaEpochPolicy, FirstAsyncErrorPoisonsPeerAndIsPreserved) {
  flagcxRmaProxyState proxy = {};
  volatile int peerState = FLAGCX_RMA_PEER_ACTIVE;
  proxy.nRanks = 1;
  proxy.peerStates = &peerState;

  flagcxRmaProxyRecordError(&proxy, 0, flagcxSystemError);
  EXPECT_EQ(flagcxRmaProxyAsyncError(&proxy), flagcxSystemError);
  EXPECT_EQ(peerState, FLAGCX_RMA_PEER_FAILED);

  flagcxRmaProxyRecordError(&proxy, 0, flagcxRemoteError);
  EXPECT_EQ(flagcxRmaProxyAsyncError(&proxy), flagcxSystemError);
}

TEST(RmaEpochPolicy, SuccessfulCompletionCannotBridgeAFailedSequence) {
  EXPECT_EQ(flagcxRmaNextCompletedSeq(0, 1, true), 1u);
  EXPECT_EQ(flagcxRmaNextCompletedSeq(1, 2, false), 1u);
  EXPECT_EQ(flagcxRmaNextCompletedSeq(1, 3, true), 1u);
}
