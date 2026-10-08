/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "flagcx_net_adaptor.h"
#include "p2p_engine_backend.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct BackendRequest {
  int done = 0;
};

struct BackendMock {
  std::deque<BackendRequest> requests;
  std::vector<flagcxNetSubmitContext> contexts;
  std::vector<uint64_t> srcOffsets;
  std::vector<uint64_t> dstOffsets;
  std::vector<size_t> sizes;
  std::vector<void *> srcMrs;
  std::vector<void *> dstMrs;
  int failAt = -1;
  flagcxResult_t failure = flagcxInProgress;
  int puts = 0;
  int gets = 0;
  std::atomic<int> activeTests{0};
  std::atomic<int> maxActiveTests{0};
  bool delayTests = false;
};

BackendMock *gMock = nullptr;

flagcxResult_t recordPost(bool write, uint64_t srcOff, uint64_t dstOff,
                          size_t size, void **srcMr, void **dstMr,
                          void **request) {
  if (gMock == nullptr || request == nullptr)
    return flagcxInvalidArgument;
  const int index = gMock->puts + gMock->gets;
  if (index == gMock->failAt)
    return gMock->failure;
  flagcxNetSubmitContext context = {};
  if (flagcxNetGetSubmitContext(&context) != flagcxSuccess)
    return flagcxInternalError;
  gMock->contexts.push_back(context);
  gMock->srcOffsets.push_back(srcOff);
  gMock->dstOffsets.push_back(dstOff);
  gMock->sizes.push_back(size);
  gMock->srcMrs.push_back(srcMr);
  gMock->dstMrs.push_back(dstMr);
  if (write)
    ++gMock->puts;
  else
    ++gMock->gets;
  gMock->requests.push_back(BackendRequest());
  *request = &gMock->requests.back();
  return flagcxSuccess;
}

flagcxResult_t mockPut(void *, uint64_t srcOff, uint64_t dstOff, size_t size,
                       int, int, void **srcMr, void **dstMr, void **request) {
  return recordPost(true, srcOff, dstOff, size, srcMr, dstMr, request);
}

flagcxResult_t mockGet(void *, uint64_t srcOff, uint64_t dstOff, size_t size,
                       int, int, void **srcMr, void **dstMr, void **request) {
  return recordPost(false, srcOff, dstOff, size, srcMr, dstMr, request);
}

flagcxResult_t mockTest(void *request, int *done, int *) {
  if (request == nullptr || done == nullptr)
    return flagcxInvalidArgument;
  const int active = gMock->activeTests.fetch_add(1) + 1;
  int observed = gMock->maxActiveTests.load();
  while (active > observed &&
         !gMock->maxActiveTests.compare_exchange_weak(observed, active)) {
  }
  if (gMock->delayTests)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  *done = static_cast<BackendRequest *>(request)->done;
  gMock->activeTests.fetch_sub(1);
  return flagcxSuccess;
}

flagcxNetAdaptor makeAdaptor() {
  flagcxNetAdaptor adaptor = {};
  adaptor.iput = mockPut;
  adaptor.iget = mockGet;
  adaptor.test = mockTest;
  return adaptor;
}

struct BackendFixture : public ::testing::Test {
  void SetUp() override { gMock = &mock; }
  void TearDown() override { gMock = nullptr; }

  BackendMock mock;
  flagcxNetAdaptor adaptor = makeAdaptor();
  int comm = 0;
  int srcMr = 0;
  int dstMr = 0;
  std::mutex progressMutex;
};

} // namespace

TEST_F(BackendFixture, WriteDispatchesOffsetsHandlesAndOrderingContext) {
  flagcxP2pNetBackendContext context;
  flagcxP2pTransferBackend backend;
  ASSERT_EQ(flagcxP2pNetBackendInit(&context, &adaptor, &comm, &progressMutex,
                                    1, &backend),
            flagcxSuccess);

  uint64_t laneMask = 0;
  flagcxP2pTransferOp op;
  op.srcOffset = 17;
  op.dstOffset = 29;
  op.size = 4096;
  op.srcMr = &srcMr;
  op.dstMr = &dstMr;
  op.orderingKey = 0x1234;
  op.groupId = 9;
  op.generation = 10;
  op.sequence = 11;
  op.submitFlags = FLAGCX_NET_SUBMIT_DATA | FLAGCX_NET_SUBMIT_INDEPENDENT;
  op.laneMask = &laneMask;
  void *request = nullptr;
  flagcxTransportPostResult post = {};
  ASSERT_EQ(backend.post(backend.context, &op, 1, &request, &post),
            flagcxSuccess);
  EXPECT_EQ(post.requested, 1);
  EXPECT_EQ(post.accepted, 1);
  EXPECT_EQ(post.result, flagcxSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(mock.puts, 1);
  EXPECT_EQ(mock.gets, 0);
  EXPECT_EQ(mock.srcOffsets[0], 17u);
  EXPECT_EQ(mock.dstOffsets[0], 29u);
  EXPECT_EQ(mock.sizes[0], 4096u);
  EXPECT_EQ(mock.srcMrs[0], &srcMr);
  EXPECT_EQ(mock.dstMrs[0], &dstMr);
  EXPECT_EQ(mock.contexts[0].orderingKey, 0x1234u);
  EXPECT_EQ(mock.contexts[0].groupId, 9u);
  EXPECT_EQ(mock.contexts[0].generation, 10u);
  EXPECT_EQ(mock.contexts[0].sequence, 11u);
  EXPECT_EQ(mock.contexts[0].flags, op.submitFlags);
  EXPECT_EQ(mock.contexts[0].laneMask, &laneMask);

  flagcxNetSubmitContext cleared = {};
  EXPECT_EQ(flagcxNetGetSubmitContext(&cleared), flagcxNotSupported);
}

TEST_F(BackendFixture, ReadStopsAtBackpressureAndReportsAcceptedPrefix) {
  flagcxP2pNetBackendContext context;
  flagcxP2pTransferBackend backend;
  ASSERT_EQ(flagcxP2pNetBackendInit(&context, &adaptor, &comm, &progressMutex,
                                    0, &backend),
            flagcxSuccess);
  mock.failAt = 1;
  mock.failure = flagcxInProgress;

  flagcxP2pTransferOp ops[3];
  for (int i = 0; i < 3; ++i) {
    ops[i].srcOffset = i;
    ops[i].dstOffset = i + 10;
    ops[i].size = 64;
    ops[i].srcMr = &srcMr;
    ops[i].dstMr = &dstMr;
    ops[i].orderingKey = i + 1;
  }
  void *requests[3] = {};
  flagcxTransportPostResult post = {};
  ASSERT_EQ(backend.post(backend.context, ops, 3, requests, &post),
            flagcxSuccess);
  EXPECT_EQ(post.requested, 3);
  EXPECT_EQ(post.accepted, 1);
  EXPECT_EQ(post.result, flagcxInProgress);
  EXPECT_EQ(mock.gets, 1);
  EXPECT_EQ(mock.puts, 0);
  EXPECT_NE(requests[0], nullptr);
  EXPECT_EQ(requests[1], nullptr);
  EXPECT_EQ(requests[2], nullptr);
}

TEST_F(BackendFixture, TestDelegatesNativeCompletion) {
  flagcxP2pNetBackendContext context;
  flagcxP2pTransferBackend backend;
  ASSERT_EQ(flagcxP2pNetBackendInit(&context, &adaptor, &comm, &progressMutex,
                                    1, &backend),
            flagcxSuccess);
  BackendRequest request;
  int done = -1;
  EXPECT_EQ(backend.test(backend.context, &request, &done), flagcxSuccess);
  EXPECT_EQ(done, 0);
  request.done = 1;
  EXPECT_EQ(backend.test(backend.context, &request, &done), flagcxSuccess);
  EXPECT_EQ(done, 1);
}

TEST_F(BackendFixture, SerializesConcurrentProgressOnOneCommunicator) {
  flagcxP2pNetBackendContext firstContext;
  flagcxP2pNetBackendContext secondContext;
  flagcxP2pTransferBackend firstBackend;
  flagcxP2pTransferBackend secondBackend;
  ASSERT_EQ(flagcxP2pNetBackendInit(&firstContext, &adaptor, &comm,
                                    &progressMutex, 1, &firstBackend),
            flagcxSuccess);
  ASSERT_EQ(flagcxP2pNetBackendInit(&secondContext, &adaptor, &comm,
                                    &progressMutex, 0, &secondBackend),
            flagcxSuccess);

  BackendRequest request;
  mock.delayTests = true;
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    flagcxP2pTransferBackend *backend =
        i % 2 == 0 ? &firstBackend : &secondBackend;
    threads.emplace_back([backend, &request]() {
      int done = 0;
      EXPECT_EQ(backend->test(backend->context, &request, &done),
                flagcxSuccess);
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  EXPECT_EQ(mock.maxActiveTests.load(), 1);
}
