/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "flagcx_hetero.h"

TEST(RmaPostResult, RetriesOnlyExplicitBackpressure) {
  EXPECT_TRUE(flagcxRmaPostResultIsRetryable(flagcxInProgress));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxSuccess));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxInternalError));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxSystemError));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxRemoteError));
}

TEST(RmaPostResult, ClassifiesBatchPostOutcomes) {
  EXPECT_FALSE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 4, 4));
  EXPECT_FALSE(flagcxRmaBatchPostResultIsFatal(flagcxInProgress, 0, 4));
  EXPECT_FALSE(flagcxRmaBatchPostResultIsFatal(flagcxInProgress, 2, 4));

  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 2, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSystemError, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSystemError, 2, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxInternalError, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxRemoteError, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, -1, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 5, 4));
}
