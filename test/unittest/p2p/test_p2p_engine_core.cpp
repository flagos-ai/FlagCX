/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "p2p_engine_core.h"

#include <algorithm>
#include <deque>
#include <type_traits>
#include <vector>

namespace {

struct MockRequest {
  int id = -1;
  int done = 0;
  flagcxResult_t pollResult = flagcxSuccess;
  flagcxResult_t result = flagcxSuccess;
};

struct PostStep {
  int accepted;
  flagcxResult_t result;
  bool immediate = false;
};

struct MockBackend {
  std::deque<MockRequest> requests;
  std::vector<PostStep> steps;
  size_t step = 0;
  int nextRequest = 0;
  int postCalls = 0;
  int active = 0;
  int maxActive = 0;
};

flagcxResult_t mockPost(void *context, const flagcxP2pTransferOp *,
                        uint32_t count, void **requests,
                        flagcxTransportPostResult *post) {
  MockBackend *backend = static_cast<MockBackend *>(context);
  backend->postCalls++;
  PostStep step = {static_cast<int>(count), flagcxSuccess, false};
  if (backend->step < backend->steps.size())
    step = backend->steps[backend->step++];
  const int accepted = std::min(step.accepted, static_cast<int>(count));
  int asyncAccepted = 0;
  for (int i = 0; i < accepted; ++i) {
    if (step.immediate) {
      requests[i] = nullptr;
      continue;
    }
    backend->requests.push_back(MockRequest());
    MockRequest &request = backend->requests.back();
    request.id = backend->nextRequest++;
    requests[i] = &request;
    asyncAccepted++;
  }
  backend->active += asyncAccepted;
  backend->maxActive = std::max(backend->maxActive, backend->active);
  return flagcxTransportPostResultInit(post, static_cast<int>(count), accepted,
                                       step.result);
}

flagcxResult_t mockTest(void *context, void *opaque, int *done) {
  MockBackend *backend = static_cast<MockBackend *>(context);
  MockRequest *request = static_cast<MockRequest *>(opaque);
  *done = request->done;
  if (!request->done)
    return request->pollResult;
  backend->active--;
  return request->result;
}

std::vector<flagcxP2pTransferOp> makeOps(uint32_t count) {
  std::vector<flagcxP2pTransferOp> ops(count);
  for (uint32_t i = 0; i < count; ++i) {
    ops[i].srcOffset = i * 4096;
    ops[i].dstOffset = i * 4096;
    ops[i].size = 4096;
    ops[i].orderingKey = i % 2;
  }
  return ops;
}

flagcxP2pTransferBackend makeBackend(MockBackend *mock) {
  flagcxP2pTransferBackend backend;
  backend.context = mock;
  backend.post = mockPost;
  backend.test = mockTest;
  return backend;
}

void markDone(MockBackend *backend, int id,
              flagcxResult_t result = flagcxSuccess) {
  for (MockRequest &request : backend->requests) {
    if (request.id == id) {
      request.result = result;
      request.done = 1;
      return;
    }
  }
  FAIL() << "request " << id << " was not posted";
}

} // namespace

static_assert(!std::is_copy_constructible<flagcxP2pTransfer>::value,
              "self-referential transfers must not be copied");
static_assert(!std::is_move_constructible<flagcxP2pTransfer>::value,
              "self-referential transfers must not be moved");

TEST(P2pEngineCoreTest, RetiresOutOfOrderCompletionsContiguously) {
  MockBackend mock;
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(3);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  3, 3, 1, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  ASSERT_EQ(status.submitted, 3u);
  markDone(&mock, 1);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.completed, 1u);
  EXPECT_EQ(status.retired, 0u);
  EXPECT_FALSE(status.done);

  markDone(&mock, 0);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.retired, 2u);
  markDone(&mock, 2);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_EQ(status.retired, 3u);
  EXPECT_TRUE(status.releaseAllowed);
  EXPECT_EQ(status.result, flagcxSuccess);
  EXPECT_EQ(flagcxP2pTransferReset(&transfer), flagcxSuccess);
}

TEST(P2pEngineCoreTest, PartialPostRespectsCreditsAndRetriesQueuedWork) {
  MockBackend mock;
  mock.steps.push_back({2, flagcxInProgress});
  mock.steps.push_back({1, flagcxSuccess});
  mock.steps.push_back({2, flagcxSuccess});
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(5);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  2, 4, 2, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.submitted, 2u);
  EXPECT_EQ(mock.maxActive, 2);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(mock.postCalls, 1);

  markDone(&mock, 0);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.submitted, 3u);
  markDone(&mock, 1);
  markDone(&mock, 2);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.submitted, 5u);
  EXPECT_LE(mock.maxActive, 2);
  markDone(&mock, 3);
  markDone(&mock, 4);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_TRUE(status.releaseAllowed);
}

TEST(P2pEngineCoreTest, ZeroAcceptanceBackpressureDoesNotConsumeCredit) {
  MockBackend mock;
  mock.steps.push_back({0, flagcxInProgress});
  mock.steps.push_back({1, flagcxSuccess});
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(1);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  1, 1, 3, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.submitted, 0u);
  EXPECT_EQ(status.pending, 1u);
  EXPECT_EQ(status.inFlight, 0u);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.submitted, 1u);
  markDone(&mock, 0);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
}

TEST(P2pEngineCoreTest, CompletionFailureStopsNewPostsAndSuppressesRelease) {
  MockBackend mock;
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(3);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  1, 1, 4, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  markDone(&mock, 0, flagcxSystemError);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_EQ(status.result, flagcxSystemError);
  EXPECT_FALSE(status.releaseAllowed);
  EXPECT_EQ(status.submitted, 1u);
  EXPECT_EQ(mock.postCalls, 1);
}

TEST(P2pEngineCoreTest, PartialFatalPostDrainsAcceptedPrefix) {
  MockBackend mock;
  mock.steps.push_back({1, flagcxSystemError});
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(3);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  3, 3, 5, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_FALSE(status.done);
  EXPECT_EQ(status.submitted, 1u);
  EXPECT_EQ(status.completed, 2u);
  EXPECT_EQ(status.result, flagcxSystemError);
  EXPECT_EQ(flagcxP2pTransferReset(&transfer), flagcxInProgress);

  markDone(&mock, 0);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_FALSE(status.releaseAllowed);
  EXPECT_EQ(status.result, flagcxSystemError);
  EXPECT_EQ(mock.postCalls, 1);
}

TEST(P2pEngineCoreTest, FullAcceptedFatalPostSuppressesRelease) {
  MockBackend mock;
  mock.steps.push_back({2, flagcxSystemError});
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(2);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  2, 2, 6, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.result, flagcxSystemError);
  EXPECT_FALSE(status.done);

  markDone(&mock, 0);
  markDone(&mock, 1);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_EQ(status.result, flagcxSystemError);
  EXPECT_FALSE(status.releaseAllowed);
}

TEST(P2pEngineCoreTest, ZeroLengthMembersCompleteWithoutBackendPost) {
  MockBackend mock;
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(2);
  ops[0].size = 0;
  ops[1].size = 0;
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  1, 1, 6, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferQuery(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_EQ(status.completed, 2u);
  EXPECT_EQ(status.submitted, 0u);
  EXPECT_TRUE(status.releaseAllowed);
  EXPECT_EQ(mock.postCalls, 0);
}

TEST(P2pEngineCoreTest, NullAcceptedRequestsCompleteSynchronously) {
  MockBackend mock;
  mock.steps.push_back({2, flagcxSuccess, true});
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(2);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  2, 2, 7, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_TRUE(status.releaseAllowed);
  EXPECT_EQ(status.submitted, 2u);
  EXPECT_EQ(status.completed, 2u);
  EXPECT_EQ(status.inFlight, 0u);
  EXPECT_EQ(mock.active, 0);
}

TEST(P2pEngineCoreTest, InProgressPollKeepsRequestAndCreditInflight) {
  MockBackend mock;
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(1);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  1, 1, 8, 0),
            flagcxSuccess);

  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  ASSERT_EQ(mock.requests.size(), 1u);
  mock.requests.front().pollResult = flagcxInProgress;
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_FALSE(status.done);
  EXPECT_EQ(status.completed, 0u);
  EXPECT_EQ(status.inFlight, 1u);
  EXPECT_EQ(status.pending, 1u);

  mock.requests.front().pollResult = flagcxSuccess;
  markDone(&mock, 0);
  ASSERT_EQ(flagcxP2pTransferProgress(&transfer, &status), flagcxSuccess);
  EXPECT_TRUE(status.done);
  EXPECT_EQ(status.result, flagcxSuccess);
}

TEST(P2pEngineCoreTest, InvalidProgressOutputHasNoSubmissionSideEffects) {
  MockBackend mock;
  flagcxP2pTransferBackend backend = makeBackend(&mock);
  std::vector<flagcxP2pTransferOp> ops = makeOps(1);
  flagcxP2pTransfer transfer;
  ASSERT_EQ(flagcxP2pTransferInit(&transfer, &backend, ops.data(), ops.size(),
                                  1, 1, 9, 0),
            flagcxSuccess);

  EXPECT_EQ(flagcxP2pTransferProgress(&transfer, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(mock.postCalls, 0);
  flagcxP2pTransferStatus status;
  ASSERT_EQ(flagcxP2pTransferQuery(&transfer, &status), flagcxSuccess);
  EXPECT_EQ(status.submitted, 0u);
  EXPECT_EQ(status.inFlight, 0u);
  EXPECT_FALSE(status.done);
}
