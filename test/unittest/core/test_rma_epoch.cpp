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
