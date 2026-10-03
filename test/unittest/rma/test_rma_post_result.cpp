/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <thread>

#include "adaptor.h"
#include "comm.h"
#include "dev_api_backend.h"
#include "flagcx_hetero.h"
#include "flagcx_net_adaptor.h"
#include "net_transport.h"
#include "onesided_types.h"
#include "sym_heap.h"

namespace {

flagcxResult_t unusedPut(void *, uint64_t, uint64_t, size_t, int, int, void **,
                         void **, void **) {
  return flagcxInternalError;
}

flagcxResult_t unusedGet(void *, uint64_t, uint64_t, size_t, int, int, void **,
                         void **, void **) {
  return flagcxInternalError;
}

flagcxResult_t unusedPutSignal(void *, uint64_t, uint64_t, size_t, int, int,
                               void **, void **, uint64_t, void **, uint64_t,
                               void **) {
  return flagcxInternalError;
}

int deviceMemcpyCalls = 0;

struct MockRmaRequest {
  int done;
  flagcxResult_t result;
};

MockRmaRequest mockRequests[16] = {};
int mockRequestCount = 0;
int mockDataPosts = 0;
int mockSignalPosts = 0;
int mockFlushPosts = 0;
int mockLastFlushSize = -1;
size_t mockLastSignalSize = SIZE_MAX;
uint64_t mockOrderingKeys[8] = {};
uint32_t mockSubmitFlags[8] = {};
int mockPutBackpressure = 0;
int mockFlushBackpressure = 0;
flagcxResult_t mockFlushResult = flagcxSuccess;
int mockBatchPosted = 0;
flagcxResult_t mockBatchResult = flagcxSuccess;

flagcxResult_t mockPut(void *, uint64_t, uint64_t, size_t, int, int, void **,
                       void **, void **request) {
  if (mockPutBackpressure > 0) {
    mockPutBackpressure--;
    *request = nullptr;
    return flagcxInProgress;
  }
  mockDataPosts++;
  flagcxNetSubmitContext context = {};
  if (flagcxNetGetSubmitContext(&context) == flagcxSuccess) {
    mockOrderingKeys[mockRequestCount] = context.orderingKey;
    mockSubmitFlags[mockRequestCount] = context.flags;
  }
  *request = &mockRequests[mockRequestCount++];
  return flagcxSuccess;
}

flagcxResult_t mockGet(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                       size_t size, int srcRank, int dstRank, void **srcHandles,
                       void **dstHandles, void **request) {
  return mockPut(sendComm, srcOff, dstOff, size, srcRank, dstRank, srcHandles,
                 dstHandles, request);
}

flagcxResult_t mockFlush(void *, int n, void **, int *sizes, void **,
                         void **request) {
  if (mockFlushBackpressure > 0) {
    mockFlushBackpressure--;
    *request = nullptr;
    return flagcxInProgress;
  }
  if (n != 1 || sizes == nullptr)
    return flagcxInvalidArgument;
  mockFlushPosts++;
  mockLastFlushSize = sizes[0];
  *request = nullptr;
  if (mockFlushResult != flagcxSuccess)
    return mockFlushResult;
  *request = &mockRequests[mockRequestCount++];
  return flagcxSuccess;
}

flagcxResult_t mockPutBatch(void *, int count, const uint64_t *,
                            const uint64_t *, const size_t *, int, int, void **,
                            void **, void **requests, int *posted) {
  const int accepted = mockBatchPosted < count ? mockBatchPosted : count;
  *posted = accepted;
  for (int i = 0; i < accepted; i++) {
    mockDataPosts++;
    requests[i] = &mockRequests[mockRequestCount++];
  }
  return mockBatchResult;
}

flagcxResult_t mockPutSignal(void *, uint64_t, uint64_t, size_t size, int, int,
                             void **, void **, uint64_t, void **, uint64_t,
                             void **request) {
  mockSignalPosts++;
  mockLastSignalSize = size;
  flagcxNetSubmitContext context = {};
  if (flagcxNetGetSubmitContext(&context) == flagcxSuccess) {
    mockOrderingKeys[mockRequestCount] = context.orderingKey;
    mockSubmitFlags[mockRequestCount] = context.flags;
  }
  *request = &mockRequests[mockRequestCount++];
  return flagcxSuccess;
}

flagcxResult_t mockTest(void *request, int *done, int *) {
  MockRmaRequest *mock = static_cast<MockRmaRequest *>(request);
  *done = mock->done;
  return mock->result;
}

flagcxResult_t recordDeviceMemcpy(void *, void *, size_t, flagcxMemcpyType_t,
                                  flagcxStream_t, void *) {
  deviceMemcpyCalls++;
  return flagcxSuccess;
}

struct RmaQueueFixture {
  flagcxNetAdaptor_latest net = {};
  flagcxRmaProxyState proxy = {};
  flagcxHeteroComm comm = {};
  flagcxRmaDesc *ring[2] = {};
  volatile uint32_t pi = 0;
  volatile uint32_t ci = 0;
  volatile uint64_t opSeq = 0;
  volatile uint64_t readySeq = 0;
  pthread_mutex_t producerMutex;

  RmaQueueFixture() {
    net.iput = unusedPut;
    net.iget = unusedGet;
    net.iputSignal = unusedPutSignal;

    pthread_mutex_init(&producerMutex, nullptr);
    proxy.queueSize = 2;
    proxy.queueMask = 1;
    proxy.circularBuffers = ring;
    proxy.pis = &pi;
    proxy.cis = &ci;
    proxy.opSeqs = &opSeq;
    proxy.readySeqsCpu = &readySeq;
    proxy.peerProducerMutexes = &producerMutex;

    comm.nRanks = 1;
    comm.netAdaptor = &net;
    comm.rmaProxy = &proxy;
    comm.signalHandle = reinterpret_cast<flagcxOneSideHandleInfo *>(0x1);
  }

  ~RmaQueueFixture() { pthread_mutex_destroy(&producerMutex); }
};

class ForcedNetworkFixture : public ::testing::Test {
protected:
  void SetUp() override {
    savedDeviceAdaptor_ = deviceAdaptor;
    ASSERT_EQ(setenv("FLAGCX_P2P_DISABLE", "1", 1), 0);
    testDeviceAdaptor_ = {};
    testDeviceAdaptor_.deviceMemcpy = recordDeviceMemcpy;
    deviceAdaptor = &testDeviceAdaptor_;
    deviceMemcpyCalls = 0;

    srcWindow_.localBase = reinterpret_cast<void *>(0x1000);
    srcWindow_.heapSize = 64;
    srcWindow_.mrIndex = -1;
    srcWindow_.ipcSlot = 0;
    dstWindow_.localBase = reinterpret_cast<void *>(0x2000);
    dstWindow_.heapSize = 64;
    dstWindow_.mrIndex = -1;
    dstWindow_.ipcSlot = 0;

    queue_.comm.rmaSignalBase = reinterpret_cast<void *>(0x3000);
    queue_.comm.rmaSignalSize = sizeof(uint64_t);
  }

  void TearDown() override { deviceAdaptor = savedDeviceAdaptor_; }

  RmaQueueFixture queue_;
  flagcxSymWindow srcWindow_ = {};
  flagcxSymWindow dstWindow_ = {};
  flagcxDeviceAdaptor_latest testDeviceAdaptor_ = {};
  flagcxDeviceAdaptor_latest *savedDeviceAdaptor_ = nullptr;
};

class RmaSharedTransportFixture : public ::testing::Test {
protected:
  void SetUp() override {
    memset(mockRequests, 0, sizeof(mockRequests));
    for (auto &request : mockRequests)
      request.result = flagcxSuccess;
    mockRequestCount = 0;
    mockDataPosts = 0;
    mockSignalPosts = 0;
    mockFlushPosts = 0;
    mockLastFlushSize = -1;
    mockLastSignalSize = SIZE_MAX;
    memset(mockOrderingKeys, 0, sizeof(mockOrderingKeys));
    memset(mockSubmitFlags, 0, sizeof(mockSubmitFlags));
    mockPutBackpressure = 0;
    mockFlushBackpressure = 0;
    mockFlushResult = flagcxSuccess;
    mockBatchPosted = 0;
    mockBatchResult = flagcxSuccess;

    net_.iput = mockPut;
    net_.iget = mockGet;
    net_.iputSignal = mockPutSignal;
    net_.iflush = mockFlush;
    net_.test = mockTest;
    net_.gdrFlushCaps = FLAGCX_NET_GDR_FLUSH_READ | FLAGCX_NET_GDR_FLUSH_WRITE;

    flagcxIntruQueueConstruct(&inProgress_);
    pthread_mutex_init(&producerMutex_, nullptr);
    pthread_mutex_init(&proxy_.doneMutex, nullptr);
    pthread_cond_init(&proxy_.doneCond, nullptr);

    proxy_.queueSize = 8;
    proxy_.queueMask = 7;
    proxy_.circularBuffers = ring_;
    proxy_.pis = &pi_;
    proxy_.cis = &ci_;
    proxy_.peerProducerMutexes = &producerMutex_;
    proxy_.inProgressQueues = &inProgress_;
    proxy_.opSeqs = &opSeq_;
    proxy_.doneSeqs = &doneSeq_;
    proxy_.doneSeqsCpu = &doneSeqCpu_;
    proxy_.inFlights = &inFlight_;
    proxy_.completionScoreboards = &scoreboard_;
    proxy_.completionEntries = entries_;
    proxy_.groupSeqs = &groupSeq_;
    proxy_.generation = 1;
    proxy_.nRanks = 1;
    proxy_.comm = &comm_;
    ASSERT_EQ(
        flagcxNetCompletionScoreboardInit(&scoreboard_, entries_, 8, 1, 1),
        flagcxSuccess);

    sendComms_[0] = reinterpret_cast<void *>(0x1234);
    proxy_.fullSendComms = sendComms_;
    handles_[0] = &mrInfo_;
    baseVas_[0] = 0x1000;
    regionSizes_[0] = 4096;
    mrInfo_.baseVas = baseVas_;
    mrInfo_.regionSizes = regionSizes_;
    mrInfo_.localRecvComm = reinterpret_cast<void *>(0x5678);
    mrInfo_.localMrHandle = reinterpret_cast<void *>(0x9abc);
    mrInfo_.nRanks = 1;
    comm_.rank = 0;
    comm_.nRanks = 1;
    comm_.netAdaptor = &net_;
    comm_.rmaProxy = &proxy_;
    comm_.oneSideHandles = handles_;
    comm_.oneSideHandleCount = 1;
    comm_.signalHandle = &mrInfo_;
  }

  void TearDown() override {
    pthread_cond_destroy(&proxy_.doneCond);
    pthread_mutex_destroy(&proxy_.doneMutex);
    pthread_mutex_destroy(&producerMutex_);
  }

  void Progress() {
    int madeProgress = 0;
    int outstanding = 0;
    ASSERT_EQ(flagcxHeteroRmaProxyProgressOnce(&proxy_, false, &madeProgress,
                                               &outstanding),
              flagcxSuccess);
  }

  flagcxNetAdaptor_latest net_ = {};
  flagcxRmaProxyState proxy_ = {};
  flagcxHeteroComm comm_ = {};
  flagcxOneSideHandleInfo mrInfo_ = {};
  flagcxOneSideHandleInfo *handles_[1] = {};
  flagcxRmaDesc *ring_[8] = {};
  flagcxIntruQueue<flagcxRmaDesc, &flagcxRmaDesc::next> inProgress_ = {};
  flagcxNetCompletionScoreboard scoreboard_ = {};
  flagcxNetCompletionEntry entries_[8] = {};
  void *sendComms_[1] = {};
  uintptr_t baseVas_[1] = {};
  size_t regionSizes_[1] = {};
  volatile uint32_t pi_ = 0;
  volatile uint32_t ci_ = 0;
  volatile uint32_t inFlight_ = 0;
  volatile uint64_t opSeq_ = 0;
  volatile uint64_t doneSeq_ = 0;
  volatile uint64_t doneSeqCpu_ = 0;
  volatile uint64_t groupSeq_ = 0;
  pthread_mutex_t producerMutex_;
};

} // namespace

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

TEST_F(RmaSharedTransportFixture,
       OutOfOrderCompletionsAdvanceOnlyContiguousPrefix) {
  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 8, 8, 8, 0, 0), flagcxSuccess);
  ASSERT_EQ(ring_[0]->orderingKey, 0u);
  ASSERT_EQ(ring_[0]->generation, 1u);
  ASSERT_EQ(ring_[0]->sequence, 1u);
  ASSERT_EQ(ring_[1]->sequence, 2u);

  Progress();
  ASSERT_EQ(mockDataPosts, 2);
  mockRequests[1].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 0u);

  mockRequests[0].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 2u);
  EXPECT_EQ(doneSeqCpu_, 2u);
}

TEST_F(RmaSharedTransportFixture, GetWithoutRequirementRetiresAtDataCqe) {
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);

  Progress();
  ASSERT_EQ(mockDataPosts, 1);
  mockRequests[0].done = 1;
  Progress();

  EXPECT_EQ(mockFlushPosts, 0);
  EXPECT_EQ(doneSeq_, 1u);
  EXPECT_EQ(proxy_.completionCount, 1u);
}

TEST_F(RmaSharedTransportFixture,
       GetVisibilityPolicyIsIndependentOfRegistrationRoute) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  const uint8_t routes[] = {FLAGCX_VMM_MR_ROUTE_NONE, FLAGCX_VMM_MR_ROUTE_VA,
                            FLAGCX_VMM_MR_ROUTE_DMABUF};
  for (uint8_t route : routes) {
    mrInfo_.registrationRoute = route;
    EXPECT_TRUE(flagcxOneSideGetCompletionRequiresFlush(&comm_, 0));
  }
}

TEST_F(RmaSharedTransportFixture, GetWaitsForRequiredVisibilityFlush) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);

  Progress();
  ASSERT_EQ(mockDataPosts, 1);
  mockRequests[0].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 0u);
  EXPECT_EQ(proxy_.completionCount, 0u);

  Progress();
  ASSERT_EQ(mockFlushPosts, 1);
  EXPECT_EQ(doneSeq_, 0u);
  mockRequests[1].done = 1;
  Progress();

  EXPECT_EQ(doneSeq_, 1u);
  EXPECT_EQ(proxy_.completionCount, 1u);
}

TEST_F(RmaSharedTransportFixture, LargeGetSaturatesLegacyFlushSize) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  regionSizes_[0] = static_cast<size_t>(UINT32_MAX);

  const size_t sizes[] = {static_cast<size_t>(INT_MAX) + 1,
                          static_cast<size_t>(UINT32_MAX)};
  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    SCOPED_TRACE(sizes[i]);
    ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, sizes[i], 0, 0), flagcxSuccess);

    Progress();
    const int dataRequest = mockRequestCount - 1;
    ASSERT_GE(dataRequest, 0);
    mockRequests[dataRequest].done = 1;
    Progress();
    EXPECT_EQ(doneSeq_, i);

    Progress();
    EXPECT_EQ(mockLastFlushSize, INT_MAX);
    const int flushRequest = mockRequestCount - 1;
    ASSERT_GT(flushRequest, dataRequest);
    mockRequests[flushRequest].done = 1;
    Progress();

    EXPECT_EQ(doneSeq_, i + 1);
    EXPECT_EQ(proxy_.completionCount, i + 1);
  }
}

TEST_F(RmaSharedTransportFixture, GetFlushBackpressureRetriesDescriptor) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  mockFlushBackpressure = 1;
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);

  Progress();
  mockRequests[0].done = 1;
  Progress();
  Progress();
  EXPECT_EQ(mockFlushPosts, 0);
  EXPECT_EQ(doneSeq_, 0u);

  Progress();
  ASSERT_EQ(mockFlushPosts, 1);
  mockRequests[1].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 1u);
}

TEST_F(RmaSharedTransportFixture, GetFlushFailureIsTerminal) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  mockFlushResult = flagcxRemoteError;
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);

  Progress();
  mockRequests[0].done = 1;
  Progress();
  Progress();

  EXPECT_EQ(mockFlushPosts, 1);
  EXPECT_EQ(proxy_.completionCount, 0u);
  EXPECT_NE(proxy_.rmaError, 0);
}

TEST_F(RmaSharedTransportFixture,
       RequiredGetFlushWithoutProviderCapabilityIsTerminal) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  net_.gdrFlushCaps = FLAGCX_NET_GDR_FLUSH_NONE;
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);

  Progress();
  ASSERT_EQ(mockDataPosts, 1);
  mockRequests[0].done = 1;
  Progress();
  Progress();

  EXPECT_EQ(mockFlushPosts, 0);
  EXPECT_EQ(proxy_.completionCount, 0u);
  EXPECT_NE(proxy_.rmaError, 0);
}

TEST_F(RmaSharedTransportFixture, WriteOnlyRequirementDoesNotFlushGet) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_WRITE_REQUIRES_FLUSH;
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);

  Progress();
  ASSERT_EQ(mockDataPosts, 1);
  mockRequests[0].done = 1;
  Progress();

  EXPECT_EQ(mockFlushPosts, 0);
  EXPECT_EQ(doneSeq_, 1u);
  EXPECT_EQ(proxy_.completionCount, 1u);
}

TEST_F(RmaSharedTransportFixture,
       ZeroByteGetDoesNotRequireProviderFlushCapability) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  net_.gdrFlushCaps = FLAGCX_NET_GDR_FLUSH_NONE;

  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 0, 0, 0), flagcxSuccess);
  Progress();
  EXPECT_EQ(mockDataPosts, 1);
  mockRequests[0].done = 1;
  Progress();

  EXPECT_EQ(mockFlushPosts, 0);
  EXPECT_EQ(doneSeq_, 1u);
  EXPECT_EQ(proxy_.completionCount, 1u);

  void *flushRequest = reinterpret_cast<void *>(0x1);
  EXPECT_EQ(flagcxOneSidePostGetVisibilityFlush(&comm_, 0, 0, 0, nullptr,
                                                &flushRequest),
            flagcxSuccess);
  EXPECT_EQ(flushRequest, nullptr);
}

TEST_F(RmaSharedTransportFixture,
       OutOfOrderGetFlushesStillAdvanceContiguousPrefix) {
  mrInfo_.gdrFlushRequirements = FLAGCX_GDR_READ_REQUIRES_FLUSH;
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroGet(&comm_, 0, 8, 8, 8, 0, 0), flagcxSuccess);

  Progress();
  ASSERT_EQ(mockDataPosts, 2);
  mockRequests[0].done = 1;
  mockRequests[1].done = 1;
  Progress();
  Progress();
  ASSERT_EQ(mockFlushPosts, 2);

  mockRequests[3].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 0u);
  mockRequests[2].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 2u);
  EXPECT_EQ(proxy_.completionCount, 2u);
}

TEST_F(RmaSharedTransportFixture,
       FailedOutOfOrderCompletionStillPublishesRetiredPrefix) {
  proxy_.useStreamOps = 1;
  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 8, 8, 8, 0, 0), flagcxSuccess);

  Progress();
  ASSERT_EQ(mockDataPosts, 2);
  mockRequests[1].done = 1;
  mockRequests[1].result = flagcxRemoteError;
  Progress();
  EXPECT_EQ(doneSeq_, 0u);
  EXPECT_EQ(doneSeqCpu_, 0u);
  EXPECT_EQ(proxy_.rmaError, 0);
  EXPECT_NE(proxy_.pendingError, 0);

  mockRequests[0].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 2u);
  EXPECT_EQ(doneSeqCpu_, 2u);
  EXPECT_NE(proxy_.rmaError, 0);
}

TEST_F(RmaSharedTransportFixture,
       NonzeroDomainRequiresExplicitIndependentMarker) {
  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0, false, nullptr, 5, false),
            flagcxSuccess);
  ASSERT_NE(ring_[0], nullptr);
  EXPECT_EQ(ring_[0]->orderingKey, 0u);
  EXPECT_EQ(ring_[0]->submitFlags & FLAGCX_RMA_SUBMIT_INDEPENDENT, 0u);

  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0, false, nullptr, 5, true),
            flagcxSuccess);
  ASSERT_NE(ring_[1], nullptr);
  EXPECT_EQ(ring_[1]->orderingKey, 5u);
  EXPECT_NE(ring_[1]->submitFlags & FLAGCX_RMA_SUBMIT_INDEPENDENT, 0u);

  Progress();
  mockRequests[0].done = 1;
  mockRequests[1].done = 1;
  Progress();
}

TEST_F(RmaSharedTransportFixture, SinglePostBackpressureRetriesSameSequence) {
  mockPutBackpressure = 1;
  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);

  Progress();
  EXPECT_EQ(mockDataPosts, 0);
  EXPECT_EQ(ci_, 0u);
  EXPECT_EQ(inFlight_, 0u);

  Progress();
  EXPECT_EQ(mockDataPosts, 1);
  EXPECT_EQ(ci_, 1u);
  EXPECT_EQ(inFlight_, 1u);
  mockRequests[0].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 1u);
}

TEST_F(RmaSharedTransportFixture, QuiesceRejectsNewSubmissions) {
  flagcxComm outer = {};
  outer.heteroComm = &comm_;
  ASSERT_EQ(flagcxCommQuiesce(&outer), flagcxSuccess);
  EXPECT_EQ(proxy_.quiesced, 1);
  EXPECT_EQ(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0), flagcxInvalidUsage);
  EXPECT_EQ(pi_, 0u);
  EXPECT_EQ(opSeq_, 0u);
}

TEST_F(RmaSharedTransportFixture,
       MissingSendCommPoisonsProxyWithoutPostingAndDrainsRing) {
  sendComms_[0] = nullptr;
  ASSERT_EQ(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0), flagcxSuccess);
  ASSERT_EQ(pi_, 1u);
  ASSERT_EQ(ci_, 0u);

  // The first pass discovers the missing transport and publishes a terminal
  // proxy error. It must not hand the descriptor to the network adaptor.
  Progress();
  EXPECT_EQ(mockDataPosts, 0);
  EXPECT_NE(proxy_.pendingError, 0);
  EXPECT_NE(proxy_.rmaError, 0);
  EXPECT_EQ(ci_, 0u);

  // Once poisoned, the next pass completes queued descriptors with an error
  // so producers cannot leave the ring permanently occupied.
  Progress();
  EXPECT_EQ(mockDataPosts, 0);
  EXPECT_EQ(ci_, pi_);
  EXPECT_EQ(inFlight_, 0u);
  EXPECT_EQ(doneSeq_, 1u);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&inProgress_));
}

TEST_F(RmaSharedTransportFixture, PartialBatchDrainsPrefixThenRetriesSuffix) {
  net_.iputBatch = mockPutBatch;
  mockBatchPosted = 2;
  mockBatchResult = flagcxInProgress;
  mockPutBackpressure = 1;
  const size_t offsets[] = {0, 8, 16};
  const size_t sizes[] = {8, 8, 8};
  const int mrIndexes[] = {0, 0, 0};
  ASSERT_EQ(flagcxHeteroBatchPut(&comm_, 0, offsets, offsets, sizes, mrIndexes,
                                 mrIndexes, 3),
            flagcxSuccess);

  Progress();
  EXPECT_EQ(mockDataPosts, 2);
  EXPECT_EQ(ci_, 2u);
  mockRequests[0].done = 1;
  mockRequests[1].done = 1;
  // A one-element suffix uses the single-post fallback.
  Progress();
  EXPECT_EQ(mockDataPosts, 3);
  EXPECT_EQ(ci_, 3u);
  mockRequests[2].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 3u);
}

TEST_F(RmaSharedTransportFixture, BatchLargerThanRingIsSubmittedInChunks) {
  constexpr size_t count = 9;
  size_t offsets[count] = {};
  size_t sizes[count] = {};
  int mrIndexes[count] = {};
  for (size_t i = 0; i < count; i++) {
    offsets[i] = i * 8;
    sizes[i] = 8;
  }

  std::atomic<int> batchResult{flagcxInProgress};
  std::thread submitter([&] {
    batchResult.store(flagcxHeteroBatchPut(&comm_, 0, offsets, offsets, sizes,
                                           mrIndexes, mrIndexes, count),
                      std::memory_order_release);
  });

  for (int i = 0; i < 10000 && pi_ != proxy_.queueSize; i++)
    std::this_thread::yield();
  ASSERT_EQ(pi_, proxy_.queueSize);
  Progress();
  ASSERT_EQ(mockDataPosts, static_cast<int>(proxy_.queueSize));
  for (size_t i = 0; i < proxy_.queueSize; i++)
    mockRequests[i].done = 1;
  Progress();

  submitter.join();
  EXPECT_EQ(batchResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(pi_, count);
  Progress();
  EXPECT_EQ(mockDataPosts, static_cast<int>(count));
  mockRequests[count - 1].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, count);
}

TEST_F(RmaSharedTransportFixture,
       BatchWaitsForScoreboardCapacityWithoutPublishingPrefix) {
  flagcxNetSubmitContext blocker = {};
  blocker.generation = 1;
  blocker.sequence = 1;
  blocker.flags = FLAGCX_NET_SUBMIT_DATA;
  ASSERT_EQ(flagcxNetCompletionScoreboardReset(&scoreboard_, 2, 1),
            flagcxSuccess);
  proxy_.generation = 2;
  blocker.generation = 2;
  scoreboard_.capacity = 2;
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard_, &blocker, nullptr),
            flagcxSuccess);
  opSeq_ = 1;

  const size_t offsets[] = {0, 8};
  const size_t sizes[] = {8, 8};
  const int mrIndexes[] = {0, 0};
  std::atomic<int> batchResult{flagcxInProgress};
  std::thread submitter([&] {
    batchResult.store(flagcxHeteroBatchPut(&comm_, 0, offsets, offsets, sizes,
                                           mrIndexes, mrIndexes, 2),
                      std::memory_order_release);
  });

  // Sequence 1 occupies half of the two-entry scoreboard, so reserving both
  // batch entries must retry without exposing sequence 2 in the ring.
  for (int i = 0; i < 1000 && opSeq_ == 1; i++)
    std::this_thread::yield();
  EXPECT_EQ(pi_, 0u);

  uint32_t advanced = 0;
  ASSERT_EQ(flagcxNetTrackCompletion(&scoreboard_, &blocker, flagcxSuccess,
                                     &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  submitter.join();

  EXPECT_EQ(batchResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(pi_, 2u);
  EXPECT_EQ(opSeq_, 3u);
  ASSERT_NE(ring_[0], nullptr);
  ASSERT_NE(ring_[1], nullptr);
  EXPECT_EQ(ring_[0]->sequence, 2u);
  EXPECT_EQ(ring_[1]->sequence, 3u);

  Progress();
  mockRequests[0].done = 1;
  mockRequests[1].done = 1;
  Progress();
}

TEST_F(RmaSharedTransportFixture, SinglePutWaitsForScoreboardCapacity) {
  flagcxNetSubmitContext blocker = {};
  blocker.generation = 2;
  blocker.sequence = 1;
  blocker.flags = FLAGCX_NET_SUBMIT_DATA;
  ASSERT_EQ(flagcxNetCompletionScoreboardReset(&scoreboard_, 2, 1),
            flagcxSuccess);
  scoreboard_.capacity = 1;
  proxy_.generation = 2;
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard_, &blocker, nullptr),
            flagcxSuccess);
  opSeq_ = 1;

  std::atomic<int> putResult{flagcxInProgress};
  std::thread submitter([&] {
    putResult.store(flagcxHeteroPut(&comm_, 0, 0, 0, 8, 0, 0),
                    std::memory_order_release);
  });
  for (int i = 0; i < 1000; i++)
    std::this_thread::yield();
  EXPECT_EQ(pi_, 0u);

  uint32_t advanced = 0;
  ASSERT_EQ(flagcxNetTrackCompletion(&scoreboard_, &blocker, flagcxSuccess,
                                     &advanced),
            flagcxSuccess);
  submitter.join();
  EXPECT_EQ(putResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(pi_, 1u);
  EXPECT_EQ(opSeq_, 2u);

  Progress();
  mockRequests[0].done = 1;
  Progress();
}

TEST_F(RmaSharedTransportFixture, ReleaseGroupWaitsForScoreboardCapacity) {
  flagcxNetSubmitContext blocker = {};
  blocker.generation = 2;
  blocker.sequence = 1;
  blocker.flags = FLAGCX_NET_SUBMIT_DATA;
  ASSERT_EQ(flagcxNetCompletionScoreboardReset(&scoreboard_, 2, 1),
            flagcxSuccess);
  scoreboard_.capacity = 2;
  proxy_.generation = 2;
  ASSERT_EQ(flagcxNetTrackSubmit(&scoreboard_, &blocker, nullptr),
            flagcxSuccess);
  opSeq_ = 1;

  std::atomic<int> signalResult{flagcxInProgress};
  std::thread submitter([&] {
    signalResult.store(flagcxHeteroPutSignal(&comm_, 0, 0, 0, 8, 0, 0, 0, 1),
                       std::memory_order_release);
  });
  for (int i = 0; i < 1000; i++)
    std::this_thread::yield();
  EXPECT_EQ(pi_, 0u);

  uint32_t advanced = 0;
  ASSERT_EQ(flagcxNetTrackCompletion(&scoreboard_, &blocker, flagcxSuccess,
                                     &advanced),
            flagcxSuccess);
  submitter.join();
  EXPECT_EQ(signalResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(pi_, 2u);
  EXPECT_EQ(opSeq_, 3u);

  Progress();
  mockRequests[0].done = 1;
  Progress();
  mockRequests[1].done = 1;
  Progress();
}

TEST_F(RmaSharedTransportFixture, ReleaseWaitsForAllDataCompletion) {
  uint64_t assigned = 0;
  ASSERT_EQ(
      flagcxHeteroPutSignal(&comm_, 0, 0, 0, 8, 0, 0, 0, 1, false, &assigned),
      flagcxSuccess);
  EXPECT_EQ(assigned, 2u);

  Progress();
  EXPECT_EQ(mockDataPosts, 1);
  EXPECT_EQ(mockSignalPosts, 0);

  Progress();
  EXPECT_EQ(mockSignalPosts, 0);

  mockRequests[0].done = 1;
  Progress();
  EXPECT_EQ(mockSignalPosts, 1);
  EXPECT_EQ(mockLastSignalSize, 0u);
  EXPECT_EQ(proxy_.completionCount, 0u);

  mockRequests[1].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 2u);
  EXPECT_EQ(proxy_.completionCount, 1u);
}

TEST_F(RmaSharedTransportFixture, DataFailureSuppressesReleaseSignal) {
  ASSERT_EQ(flagcxHeteroPutSignal(&comm_, 0, 0, 0, 8, 0, 0, 0, 1),
            flagcxSuccess);
  Progress();
  ASSERT_EQ(mockDataPosts, 1);

  mockRequests[0].done = 1;
  mockRequests[0].result = flagcxRemoteError;
  Progress();
  EXPECT_EQ(mockSignalPosts, 0);
  EXPECT_NE(proxy_.rmaError, 0);
}

TEST_F(RmaSharedTransportFixture,
       FailedMemberDoesNotFreeGroupHeldByOutstandingMember) {
  const size_t srcOffsets[] = {0, 8};
  const size_t dstOffsets[] = {16, 24};
  const size_t sizes[] = {8, 8};
  const int mrIndexes[] = {0, 0};
  const uint64_t orderingKeys[] = {1, 2};
  const uint8_t independent[] = {1, 1};
  ASSERT_EQ(flagcxHeteroBatchPutSignal(&comm_, 0, srcOffsets, dstOffsets, sizes,
                                       mrIndexes, mrIndexes, orderingKeys,
                                       independent, 2, 0, 1, 3, true),
            flagcxSuccess);

  Progress();
  ASSERT_EQ(mockDataPosts, 2);
  ASSERT_EQ(mockSignalPosts, 0);

  // Retire one member with an error. The progress pass drains the queued
  // release descriptor, while the second member still retains the group used
  // by its scoreboard entry.
  mockRequests[0].done = 1;
  mockRequests[0].result = flagcxRemoteError;
  Progress();
  EXPECT_EQ(mockSignalPosts, 0);
  EXPECT_EQ(ci_, 3u);
  EXPECT_EQ(inFlight_, 1u);

  // This completion used to dereference the group freed by the drained
  // release descriptor (ASan reports the old behavior as a UAF).
  mockRequests[1].done = 1;
  Progress();
  EXPECT_EQ(inFlight_, 0u);
  EXPECT_EQ(doneSeq_, 3u);
  EXPECT_EQ(mockSignalPosts, 0);
  EXPECT_NE(proxy_.rmaError, 0);
}

TEST_F(RmaSharedTransportFixture,
       MultiDomainGroupReleasesAfterEveryDataRequest) {
  const size_t srcOffsets[] = {0, 8};
  const size_t dstOffsets[] = {16, 24};
  const size_t sizes[] = {8, 8};
  const int mrIndexes[] = {0, 0};
  const uint64_t orderingKeys[] = {1, 2};
  const uint8_t independent[] = {1, 1};
  uint64_t assigned = 0;
  ASSERT_EQ(flagcxHeteroBatchPutSignal(
                &comm_, 0, srcOffsets, dstOffsets, sizes, mrIndexes, mrIndexes,
                orderingKeys, independent, 2, 0, 1, 3, true, &assigned),
            flagcxSuccess);
  EXPECT_EQ(assigned, 3u);

  Progress();
  ASSERT_EQ(mockDataPosts, 2);
  EXPECT_EQ(mockSignalPosts, 0);
  EXPECT_EQ(mockOrderingKeys[0], 1u);
  EXPECT_EQ(mockOrderingKeys[1], 2u);
  EXPECT_NE(mockSubmitFlags[0] & FLAGCX_NET_SUBMIT_INDEPENDENT, 0u);
  EXPECT_NE(mockSubmitFlags[1] & FLAGCX_NET_SUBMIT_INDEPENDENT, 0u);

  mockRequests[1].done = 1;
  Progress();
  EXPECT_EQ(mockSignalPosts, 0);

  mockRequests[0].done = 1;
  Progress();
  ASSERT_EQ(mockSignalPosts, 1);
  EXPECT_EQ(mockOrderingKeys[2], 3u);
  EXPECT_NE(mockSubmitFlags[2] & FLAGCX_NET_SUBMIT_INDEPENDENT, 0u);

  mockRequests[2].done = 1;
  Progress();
  EXPECT_EQ(doneSeq_, 3u);
}

TEST(RmaTransportSelection, MissingNetworkMrsAreRejectedBeforeEnqueue) {
  RmaQueueFixture fixture;

  const size_t offsets[] = {0};
  const size_t sizes[] = {64};
  const int mrIndexes[] = {-1};

  EXPECT_EQ(flagcxHeteroPut(&fixture.comm, 0, 0, 0, 64, -1, -1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroBatchPut(&fixture.comm, 0, offsets, offsets, sizes,
                                 mrIndexes, mrIndexes, 1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroGet(&fixture.comm, 0, 0, 0, 64, -1, -1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroPutSignal(&fixture.comm, 0, 0, 0, 64, 0, -1, -1, 1),
            flagcxNotSupported);

  EXPECT_EQ(fixture.pi, 0u);
  EXPECT_EQ(fixture.opSeq, 0u);
  EXPECT_EQ(fixture.ring[0], nullptr);
  EXPECT_EQ(fixture.ring[1], nullptr);
}

TEST(RmaTransportSelection, SignalOnlyRequiresNetworkSignalRegistration) {
  RmaQueueFixture fixture;
  fixture.comm.signalHandle = nullptr;

  EXPECT_EQ(flagcxHeteroPutSignal(&fixture.comm, 0, 0, 0, 0, 0, -1, -1, 1),
            flagcxNotSupported);
  EXPECT_EQ(fixture.pi, 0u);
  EXPECT_EQ(fixture.opSeq, 0u);
}

TEST_F(ForcedNetworkFixture,
       IpcOnlyWindowsAreRejectedWithoutCopyingOrEnqueueing) {
  flagcxStream_t stream = reinterpret_cast<flagcxStream_t>(0x1);

  EXPECT_EQ(flagcxHeteroPutStream(&queue_.comm, 0, 0, 0, 64, -1, -1,
                                  &srcWindow_, &dstWindow_, stream, nullptr),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroGet(&queue_.comm, 0, 0, 0, 64, -1, -1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroPutSignalStream(&queue_.comm, 0, 0, 0, 64, 0, -1, -1, 1,
                                        &srcWindow_, &dstWindow_, stream,
                                        nullptr),
            flagcxNotSupported);

  EXPECT_EQ(deviceMemcpyCalls, 0);
  EXPECT_EQ(queue_.pi, 0u);
  EXPECT_EQ(queue_.opSeq, 0u);
  EXPECT_EQ(queue_.ring[0], nullptr);
  EXPECT_EQ(queue_.ring[1], nullptr);
}
