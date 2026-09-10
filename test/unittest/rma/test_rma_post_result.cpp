/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "flagcx_hetero.h"

TEST(RmaPostResult, RetriesOnlyTransientAndLegacyBackpressure) {
  EXPECT_TRUE(flagcxRmaPostResultIsRetryable(flagcxInProgress));
  EXPECT_TRUE(flagcxRmaPostResultIsRetryable(flagcxInternalError));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxSuccess));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxSystemError));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxRemoteError));
}
