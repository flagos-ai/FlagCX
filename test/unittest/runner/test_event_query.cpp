// Unit tests for UniRunner device-event query classification.

#include <gtest/gtest.h>

#include "uni_runner_impl.h"

TEST(UniRunnerEventQuery, CompletedEventIsDone) {
  bool complete = false;
  EXPECT_EQ(flagcxUniRunnerClassifyEventQuery(flagcxSuccess, &complete),
            flagcxSuccess);
  EXPECT_TRUE(complete);
}

TEST(UniRunnerEventQuery, InProgressEventRemainsPending) {
  bool complete = true;
  EXPECT_EQ(flagcxUniRunnerClassifyEventQuery(flagcxInProgress, &complete),
            flagcxSuccess);
  EXPECT_FALSE(complete);
}

TEST(UniRunnerEventQuery, HardErrorIsPropagated) {
  bool complete = true;
  EXPECT_EQ(
      flagcxUniRunnerClassifyEventQuery(flagcxUnhandledDeviceError, &complete),
      flagcxUnhandledDeviceError);
  EXPECT_FALSE(complete);
}

TEST(UniRunnerEventQuery, NullCompletionStateIsRejected) {
  EXPECT_EQ(flagcxUniRunnerClassifyEventQuery(flagcxSuccess, nullptr),
            flagcxInvalidArgument);
}
