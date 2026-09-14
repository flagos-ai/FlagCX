#include "ibvwrap.h"

#include <cerrno>
#include <gtest/gtest.h>

namespace {

int postResult = IBV_SUCCESS;
int postCalls = 0;
ibv_send_wr *firstRejected = nullptr;

int fakePostSend(ibv_qp *, ibv_send_wr *, ibv_send_wr **badWr) {
  ++postCalls;
  if (badWr != nullptr)
    *badWr = firstRejected;
  return postResult;
}

} // namespace

TEST(IbvWrapOneSidedPost, ClassifiesOnlySendQueuePressureAsRetryable) {
  EXPECT_EQ(flagcxIbOneSidedPostResult(IBV_SUCCESS), flagcxSuccess);
  EXPECT_EQ(flagcxIbOneSidedPostResult(ENOMEM), flagcxInProgress);
  EXPECT_EQ(flagcxIbOneSidedPostResult(EINVAL), flagcxSystemError);
  EXPECT_EQ(flagcxIbOneSidedPostResult(EIO), flagcxSystemError);
}

TEST(IbvWrapOneSidedPost, ReportsSendQueuePressureAfterOnePostAttempt) {
  ibv_context context = {};
  ibv_qp qp = {};
  ibv_send_wr wr = {};
  ibv_send_wr *badWr = nullptr;
  context.ops.post_send = fakePostSend;
  qp.context = &context;
  postResult = ENOMEM;
  postCalls = 0;
  firstRejected = &wr;

  EXPECT_EQ(flagcxWrapIbvPostSendOneSided(&qp, &wr, &badWr), flagcxInProgress);
  EXPECT_EQ(postCalls, 1);
  EXPECT_EQ(badWr, &wr);
}

TEST(IbvWrapOneSidedPost, ReportsPermanentFailureWithoutRetrying) {
  ibv_context context = {};
  ibv_qp qp = {};
  ibv_send_wr wrs[2] = {};
  ibv_send_wr *badWr = nullptr;
  wrs[0].next = &wrs[1];
  context.ops.post_send = fakePostSend;
  qp.context = &context;
  postResult = EINVAL;
  postCalls = 0;
  firstRejected = &wrs[1];

  EXPECT_EQ(flagcxWrapIbvPostSendOneSided(&qp, wrs, &badWr), flagcxSystemError);
  EXPECT_EQ(postCalls, 1);
  EXPECT_EQ(badWr, &wrs[1]);
}
