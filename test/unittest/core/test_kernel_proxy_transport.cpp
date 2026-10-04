/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "device_api/completion_word.h"
#include "device_api/fifo_producer_gate.h"
#include "flagcx_kernel_core.h"
#include "flagcx_kernel_internal.h"
#include "kernel_proxy_transport.h"

#include <array>
#include <atomic>
#include <climits>
#include <thread>
#include <vector>

namespace {

struct HostAtomic {
  template <typename T>
  static T load(T *ptr, flagcxDeviceMemoryOrder_t) {
    return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
  }

  template <typename T>
  static T fetchAdd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
    return __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
  }

  template <typename T>
  static T fetchSub(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
    return __atomic_fetch_sub(ptr, value, __ATOMIC_SEQ_CST);
  }
};

flagcxNetSubmitContext track(flagcxKernelProxyTransport *transport,
                             uint32_t flags = FLAGCX_NET_SUBMIT_DATA |
                                              FLAGCX_NET_SUBMIT_INDEPENDENT) {
  flagcxNetSubmitContext submit = {};
  EXPECT_EQ(flagcxKernelProxyTrackNext(transport, flags, &submit),
            flagcxSuccess);
  return submit;
}

struct MockKernelRequest {
  int done = 0;
  flagcxResult_t result = flagcxSuccess;
};

struct MockKernelFlush {
  int backpressure = 0;
  int posts = 0;
  int dstMrIdx = -1;
  uint64_t dstOff = 0;
  size_t size = 0;
  void *recvComm = nullptr;
  flagcxResult_t result = flagcxSuccess;
  bool synchronous = false;
  MockKernelRequest requests[4] = {};
};

flagcxResult_t testKernelRequest(void *request, int *done, int *) {
  auto *mock = static_cast<MockKernelRequest *>(request);
  *done = mock->done;
  return mock->result;
}

flagcxResult_t postKernelFlush(void *context, void *recvComm, int dstMrIdx,
                               uint64_t dstOff, size_t size, void **request) {
  auto *mock = static_cast<MockKernelFlush *>(context);
  mock->dstMrIdx = dstMrIdx;
  mock->dstOff = dstOff;
  mock->size = size;
  mock->recvComm = recvComm;
  *request = nullptr;
  if (mock->backpressure > 0) {
    mock->backpressure--;
    return flagcxInProgress;
  }
  if (mock->result != flagcxSuccess)
    return mock->result;
  int post = mock->posts++;
  if (!mock->synchronous)
    *request = &mock->requests[post];
  return flagcxSuccess;
}

} // namespace

TEST(KernelProxyTransportTest, ProducerGateSerializesEntryWithClose) {
  uint64_t fifoBuffer[flagcxFifoIdxData] = {};
  auto *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  const flagcxCompletionWord_t closedMask =
      flagcxFifoProducerClosedMask<flagcxCompletionWord_t>();
  const flagcxCompletionWord_t activeMask =
      flagcxFifoProducerActiveMask<flagcxCompletionWord_t>();
  constexpr int kProducerCount = 32;
  std::atomic<int> entered{0};
  std::atomic<int> enterFailures{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> producers;
  producers.reserve(kProducerCount);

  for (int i = 0; i < kProducerCount; ++i) {
    producers.emplace_back([&] {
      bool accepted = flagcxFifoProducerTryEnter<HostAtomic>(producerState);
      if (!accepted)
        enterFailures.fetch_add(1, std::memory_order_relaxed);
      entered.fetch_add(1, std::memory_order_release);
      if (!accepted)
        return;
      while (!release.load(std::memory_order_acquire))
        std::this_thread::yield();
      flagcxFifoProducerLeave<HostAtomic>(producerState);
    });
  }
  while (entered.load(std::memory_order_acquire) != kProducerCount)
    std::this_thread::yield();
  EXPECT_EQ(enterFailures.load(std::memory_order_acquire), 0);

  std::atomic<bool> closeFinished{false};
  std::thread closer([&] {
    EXPECT_EQ(flagcxKernelProxyCloseFifoProducerGate(fifoBuffer),
              flagcxSuccess);
    closeFinished.store(true, std::memory_order_release);
  });
  while ((__atomic_load_n(producerState, __ATOMIC_ACQUIRE) & closedMask) == 0)
    std::this_thread::yield();

  EXPECT_FALSE(closeFinished.load(std::memory_order_acquire));
  EXPECT_FALSE(flagcxFifoProducerTryEnter<HostAtomic>(producerState));
  EXPECT_EQ(__atomic_load_n(producerState, __ATOMIC_ACQUIRE) & activeMask,
            static_cast<flagcxCompletionWord_t>(kProducerCount));

  release.store(true, std::memory_order_release);
  for (auto &producer : producers)
    producer.join();
  closer.join();

  EXPECT_TRUE(closeFinished.load(std::memory_order_acquire));
  EXPECT_EQ(__atomic_load_n(producerState, __ATOMIC_ACQUIRE), closedMask);
}

TEST(KernelProxyTransportTest, DequeueReturnsInProgressForEmptyFifo) {
  std::array<uint64_t, flagcxFifoIdxData + 3> fifoBuffer = {};
  fifoBuffer[flagcxFifoIdxCapacity] = 1;
  flagcxDeviceTrigger trigger = {};

  EXPECT_EQ(dequeue(fifoBuffer.data(), &trigger), flagcxInProgress);
}

TEST(KernelProxyTransportTest,
     DequeueReturnsInProgressForUnpublishedReservation) {
  std::array<uint64_t, flagcxFifoIdxData + 3> fifoBuffer = {};
  fifoBuffer[flagcxFifoIdxCapacity] = 1;
  *flagcxFifoControlPtr(fifoBuffer.data(), flagcxFifoIdxProduced) = 1;
  flagcxDeviceTrigger trigger = {};

  EXPECT_EQ(dequeue(fifoBuffer.data(), &trigger), flagcxInProgress);
  EXPECT_EQ(*flagcxFifoControlPtr(fifoBuffer.data(), flagcxFifoIdxConsumed),
            0u);
}

TEST(KernelProxyTransportTest, DequeueConsumesPublishedReservation) {
  std::array<uint64_t, flagcxFifoIdxData + 3> fifoBuffer = {};
  fifoBuffer[flagcxFifoIdxCapacity] = 1;
  *flagcxFifoControlPtr(fifoBuffer.data(), flagcxFifoIdxProduced) = 1;
  uint64_t *slot = fifoBuffer.data() + flagcxFifoIdxData;
  slot[0] = 0x1234;
  slot[1] = 0x5678;
  slot[2] =
      flagcxDeviceTriggerValidMask | (static_cast<uint64_t>(flagcxDevicePrimPut)
                                      << flagcxDeviceTriggerOffPrim);
  flagcxDeviceTrigger trigger = {};

  EXPECT_EQ(dequeue(fifoBuffer.data(), &trigger), flagcxSuccess);
  EXPECT_EQ(trigger.fst, 0x1234u);
  EXPECT_EQ(trigger.snd, 0x5678u);
  EXPECT_EQ(trigger.getPrim(), flagcxDevicePrimPut);
  EXPECT_EQ(slot[2], 0u);
}

TEST(KernelProxyTransportTest,
     TerminalFinalizeWaitsForLateProducerReservation) {
  uint64_t fifoBuffer[flagcxFifoIdxData] = {};
  auto *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  auto *produced = flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProduced);
  const flagcxCompletionWord_t closedMask =
      flagcxFifoProducerClosedMask<flagcxCompletionWord_t>();

  // Pause a producer after it has passed the gate but before it reserves a
  // sequence. This is the interleaving that made the old produced snapshot
  // stale.
  ASSERT_TRUE(flagcxFifoProducerTryEnter<HostAtomic>(producerState));

  std::atomic<bool> closeStarted{false};
  std::atomic<bool> closeFinished{false};
  std::atomic<int> closeResult{flagcxInProgress};
  std::atomic<int> finalizeResult{flagcxInProgress};
  std::thread finalizer([&] {
    closeStarted.store(true, std::memory_order_release);
    closeResult.store(flagcxKernelProxyCloseFifoProducerGate(fifoBuffer),
                      std::memory_order_release);
    finalizeResult.store(flagcxKernelProxyFinalizeTerminalFifo(fifoBuffer),
                         std::memory_order_release);
    closeFinished.store(true, std::memory_order_release);
  });

  while (!closeStarted.load(std::memory_order_acquire))
    std::this_thread::yield();
  while ((__atomic_load_n(producerState, __ATOMIC_ACQUIRE) & closedMask) == 0)
    std::this_thread::yield();
  EXPECT_FALSE(closeFinished.load(std::memory_order_acquire));

  // The pre-close producer may still advance produced. The closer must wait
  // until this reference is released before taking the final snapshot.
  __atomic_fetch_add(produced, flagcxCompletionWord_t{1}, __ATOMIC_ACQ_REL);
  flagcxFifoProducerLeave<HostAtomic>(producerState);
  finalizer.join();

  EXPECT_EQ(closeResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(finalizeResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(*flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProduced), 1u);
  EXPECT_EQ(*flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxConsumed), 1u);
  EXPECT_EQ(*flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxCompleted), 1u);
}

TEST(KernelProxyTransportTest, ClosedProducerGateRejectsNewReservations) {
  uint64_t fifoBuffer[flagcxFifoIdxData] = {};
  auto *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  auto *produced = flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProduced);

  ASSERT_EQ(flagcxKernelProxyCloseFifoProducerGate(fifoBuffer), flagcxSuccess);
  EXPECT_FALSE(flagcxFifoProducerTryEnter<HostAtomic>(producerState));
  EXPECT_EQ(__atomic_load_n(produced, __ATOMIC_ACQUIRE), 0u);
  EXPECT_EQ(flagcxKernelProxyFinalizeTerminalFifo(fifoBuffer), flagcxSuccess);
}

TEST(KernelProxyTransportTest,
     TerminalFinalizeWaitsForBackpressuredReservation) {
  uint64_t fifoBuffer[flagcxFifoIdxData] = {};
  fifoBuffer[flagcxFifoIdxCapacity] = 1;
  auto *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  auto *produced = flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProduced);
  const flagcxCompletionWord_t closedMask =
      flagcxFifoProducerClosedMask<flagcxCompletionWord_t>();

  // One queued entry fills the FIFO. A second producer has reserved sequence
  // one and is waiting for consumed to advance when terminal cleanup starts.
  __atomic_store_n(produced, flagcxCompletionWord_t{1}, __ATOMIC_RELEASE);
  ASSERT_TRUE(flagcxFifoProducerTryEnter<HostAtomic>(producerState));
  EXPECT_EQ(
      __atomic_fetch_add(produced, flagcxCompletionWord_t{1}, __ATOMIC_ACQ_REL),
      1u);

  std::atomic<bool> closeFinished{false};
  std::atomic<int> closeResult{flagcxInProgress};
  std::atomic<int> finalizeResult{flagcxInProgress};
  std::thread finalizer([&] {
    closeResult.store(flagcxKernelProxyCloseFifoProducerGate(fifoBuffer),
                      std::memory_order_release);
    finalizeResult.store(flagcxKernelProxyFinalizeTerminalFifo(fifoBuffer),
                         std::memory_order_release);
    closeFinished.store(true, std::memory_order_release);
  });

  while ((__atomic_load_n(producerState, __ATOMIC_ACQUIRE) & closedMask) == 0)
    std::this_thread::yield();
  EXPECT_FALSE(closeFinished.load(std::memory_order_acquire));

  // fifoEnqueue observes terminal status in its full-FIFO loop and releases
  // the producer reference. The unpublished reservation is then failed by the
  // stable terminal snapshot.
  flagcxFifoProducerLeave<HostAtomic>(producerState);
  finalizer.join();

  EXPECT_EQ(closeResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(finalizeResult.load(std::memory_order_acquire), flagcxSuccess);
  EXPECT_EQ(*flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProduced), 2u);
  EXPECT_EQ(*flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxConsumed), 2u);
  EXPECT_EQ(*flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxCompleted), 2u);
}

TEST(KernelProxyTransportTest, OutOfOrderRequestsAdvanceOnlyContiguousPrefix) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 2, 9, 3),
            flagcxSuccess);

  flagcxNetSubmitContext first = track(&transport);
  flagcxNetSubmitContext second = track(&transport);
  EXPECT_EQ(first.sequence, 1u);
  EXPECT_EQ(second.sequence, 2u);
  EXPECT_EQ(first.orderingKey, 3u);
  EXPECT_NE(first.flags & FLAGCX_NET_SUBMIT_INDEPENDENT, 0u);

  uint32_t firstSlot = 0;
  uint32_t secondSlot = 0;
  ASSERT_EQ(
      flagcxKernelProxyReserveRequest(&transport, &first, 0, -1, &firstSlot),
      flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, firstSlot,
                                            reinterpret_cast<void *>(0x1),
                                            flagcxSuccess),
            flagcxSuccess);
  ASSERT_EQ(
      flagcxKernelProxyReserveRequest(&transport, &second, 1, -1, &secondSlot),
      flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, secondSlot,
                                            reinterpret_cast<void *>(0x2),
                                            flagcxSuccess),
            flagcxSuccess);

  uint32_t advanced = UINT32_MAX;
  int released = -2;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(
                &transport, secondSlot, flagcxSuccess, &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 0u);
  EXPECT_EQ(released, -1);

  ASSERT_EQ(flagcxKernelProxyCompleteRequest(
                &transport, firstSlot, flagcxSuccess, &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 2u);
  EXPECT_EQ(transport.nativeInflight, 0u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, ImmediateEntryWaitsBehindNativeRequest) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext native = track(&transport);
  flagcxNetSubmitContext immediate = track(&transport, 0);

  uint32_t slot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &native, 0, -1, &slot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishRequest(
                &transport, slot, reinterpret_cast<void *>(0x1), flagcxSuccess),
            flagcxSuccess);
  uint32_t advanced = UINT32_MAX;
  ASSERT_EQ(flagcxKernelProxyCompleteImmediate(&transport, &immediate,
                                               flagcxSuccess, &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 0u);

  int released = -2;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slot, flagcxSuccess,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 2u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest,
     GetDataCompletionWaitsForFlushAndRetriesBackpressure) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submit = track(&transport);
  uint32_t slot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submit, 1, -1, &slot),
            flagcxSuccess);
  const size_t size = static_cast<size_t>(UINT32_MAX);
  void *flushRecvComm = reinterpret_cast<void *>(0x1234);
  ASSERT_EQ(flagcxKernelProxyRequireGetFlush(&transport, slot, 3, 17, size,
                                             flushRecvComm),
            flagcxSuccess);
  MockKernelRequest data = {1, flagcxSuccess};
  ASSERT_EQ(
      flagcxKernelProxyPublishRequest(&transport, slot, &data, flagcxSuccess),
      flagcxSuccess);

  MockKernelFlush flush = {};
  flush.backpressure = 1;
  int ready = -1;
  flagcxResult_t completion = flagcxInternalError;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 0);
  EXPECT_EQ(transport.nativeInflight, 1u);
  EXPECT_EQ(transport.requests[slot].completionStage,
            FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING);

  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 0);
  EXPECT_EQ(flush.posts, 0);
  EXPECT_EQ(transport.requests[slot].completionStage,
            FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING);

  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 0);
  EXPECT_EQ(flush.posts, 1);
  EXPECT_EQ(flush.dstMrIdx, 3);
  EXPECT_EQ(flush.dstOff, 17u);
  EXPECT_EQ(flush.size, size);
  EXPECT_EQ(flush.recvComm, flushRecvComm);
  EXPECT_EQ(transport.requests[slot].completionStage,
            FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING);

  flush.requests[0].done = 1;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(completion, flagcxSuccess);
  uint32_t advanced = 0;
  int released = -2;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slot, completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  EXPECT_EQ(transport.nativeInflight, 0u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, ImmediateGetStillWaitsForSynchronousFlush) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 2, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submit = track(&transport);
  uint32_t slot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submit, 0, -1, &slot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyRequireGetFlush(&transport, slot, 0, 8, 16,
                                             reinterpret_cast<void *>(0x1234)),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishGetFlushPending(&transport, slot),
            flagcxSuccess);

  MockKernelFlush flush = {};
  flush.synchronous = true;
  int ready = 0;
  flagcxResult_t completion = flagcxInternalError;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(flush.posts, 1);
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(completion, flagcxSuccess);

  uint32_t advanced = 0;
  int released = -2;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slot, completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, GetFlushFailureBecomesScoreboardError) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 2, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submit = track(&transport);
  uint32_t slot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submit, 0, -1, &slot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyRequireGetFlush(&transport, slot, 0, 0, 8,
                                             reinterpret_cast<void *>(0x1234)),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishGetFlushPending(&transport, slot),
            flagcxSuccess);

  MockKernelFlush flush = {};
  flush.result = flagcxRemoteError;
  int ready = 0;
  flagcxResult_t completion = flagcxSuccess;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(completion, flagcxRemoteError);

  uint32_t advanced = 0;
  int released = -2;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slot, completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  uint64_t nextSequence = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(
      flagcxKernelProxyQuery(&transport, &nextSequence, &inFlight, &firstError),
      flagcxSuccess);
  EXPECT_EQ(firstError, flagcxRemoteError);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, DataFailureSkipsRequiredGetFlush) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 2, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submit = track(&transport);
  uint32_t slot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submit, 0, -1, &slot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyRequireGetFlush(&transport, slot, 0, 0, 8,
                                             reinterpret_cast<void *>(0x1234)),
            flagcxSuccess);
  MockKernelRequest data = {1, flagcxRemoteError};
  ASSERT_EQ(
      flagcxKernelProxyPublishRequest(&transport, slot, &data, flagcxSuccess),
      flagcxSuccess);

  MockKernelFlush flush = {};
  int ready = 0;
  flagcxResult_t completion = flagcxSuccess;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(completion, flagcxRemoteError);
  EXPECT_EQ(flush.posts, 0);
  EXPECT_EQ(transport.requests[slot].completionStage,
            FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING);
  uint32_t advanced = 0;
  int released = -2;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slot, completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  uint64_t nextSequence = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(
      flagcxKernelProxyQuery(&transport, &nextSequence, &inFlight, &firstError),
      flagcxSuccess);
  EXPECT_EQ(inFlight, 0u);
  EXPECT_EQ(firstError, flagcxRemoteError);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, AsyncGetFlushFailureBecomesCompletionError) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 2, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submit = track(&transport);
  uint32_t slot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submit, 0, -1, &slot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyRequireGetFlush(&transport, slot, 0, 0, 8,
                                             reinterpret_cast<void *>(0x1234)),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishGetFlushPending(&transport, slot),
            flagcxSuccess);

  MockKernelFlush flush = {};
  int ready = 0;
  flagcxResult_t completion = flagcxSuccess;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 0);
  ASSERT_EQ(flush.posts, 1);
  flush.requests[0].result = flagcxRemoteError;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slot,
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(completion, flagcxRemoteError);
  uint32_t advanced = 0;
  int released = -2;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slot, completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  uint64_t nextSequence = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(
      flagcxKernelProxyQuery(&transport, &nextSequence, &inFlight, &firstError),
      flagcxSuccess);
  EXPECT_EQ(inFlight, 0u);
  EXPECT_EQ(firstError, flagcxRemoteError);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, AbortReleasesMalformedInflightRequest) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 2, 1, 1, 0),
            flagcxSuccess);
  int stagingSlot = -1;
  ASSERT_EQ(flagcxKernelProxyAcquireStagingSlot(&transport, &stagingSlot),
            flagcxSuccess);
  flagcxNetSubmitContext submit = track(&transport);
  uint32_t slot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submit, 0, stagingSlot,
                                            &slot),
            flagcxSuccess);
  MockKernelRequest request = {};
  ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, slot, &request,
                                            flagcxSuccess),
            flagcxSuccess);
  EXPECT_EQ(transport.nativeInflight, 1u);

  int released = -1;
  ASSERT_EQ(flagcxKernelProxyAbortRequest(&transport, slot, &released),
            flagcxSuccess);
  EXPECT_EQ(released, stagingSlot);
  EXPECT_EQ(transport.nativeInflight, 0u);
  EXPECT_EQ(transport.requests[slot].state, FLAGCX_KERNEL_PROXY_REQUEST_FREE);
  ASSERT_EQ(flagcxKernelProxyReleaseStagingSlot(&transport, released),
            flagcxSuccess);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest,
     OutOfOrderGetFlushesAdvanceOnlyContiguousPrefix) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submits[2] = {track(&transport), track(&transport)};
  MockKernelRequest data[2] = {{1, flagcxSuccess}, {1, flagcxSuccess}};
  uint32_t slots[2] = {};
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submits[i], i, -1,
                                              &slots[i]),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyRequireGetFlush(
                  &transport, slots[i], i, 0, 8,
                  reinterpret_cast<void *>(static_cast<uintptr_t>(i + 1))),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, slots[i], &data[i],
                                              flagcxSuccess),
              flagcxSuccess);
  }

  MockKernelFlush flush[2] = {};
  int ready = 0;
  flagcxResult_t completion = flagcxSuccess;
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(flagcxKernelProxyProgressRequest(
                  &transport, slots[i], testKernelRequest, postKernelFlush,
                  &flush[i], &ready, &completion),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyProgressRequest(
                  &transport, slots[i], testKernelRequest, postKernelFlush,
                  &flush[i], &ready, &completion),
              flagcxSuccess);
  }

  uint32_t advanced = UINT32_MAX;
  int released = -2;
  flush[1].requests[0].done = 1;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[1],
                                             testKernelRequest, postKernelFlush,
                                             &flush[1], &ready, &completion),
            flagcxSuccess);
  ASSERT_EQ(ready, 1);
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slots[1], completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 0u);

  flush[0].requests[0].done = 1;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[0],
                                             testKernelRequest, postKernelFlush,
                                             &flush[0], &ready, &completion),
            flagcxSuccess);
  ASSERT_EQ(ready, 1);
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slots[0], completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(advanced, 2u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, CompletedSamePeerGetsShareOneFlush) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submits[2] = {track(&transport), track(&transport)};
  MockKernelRequest data[2] = {{1, flagcxSuccess}, {1, flagcxSuccess}};
  uint32_t slots[2] = {};
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submits[i], 0, -1,
                                              &slots[i]),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyRequireGetFlush(
                  &transport, slots[i], i, 0, 8,
                  reinterpret_cast<void *>(static_cast<uintptr_t>(i + 1))),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, slots[i], &data[i],
                                              flagcxSuccess),
              flagcxSuccess);
  }

  MockKernelFlush flush = {};
  int ready = 0;
  flagcxResult_t completion = flagcxSuccess;
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(flagcxKernelProxyProgressRequest(
                  &transport, slots[i], testKernelRequest, postKernelFlush,
                  &flush, &ready, &completion),
              flagcxSuccess);
    EXPECT_EQ(ready, 0);
  }
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[0],
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(flush.posts, 1);
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[1],
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  EXPECT_EQ(flush.posts, 1);

  flush.requests[0].done = 1;
  uint32_t totalAdvanced = 0;
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(flagcxKernelProxyProgressRequest(
                  &transport, slots[i], testKernelRequest, postKernelFlush,
                  &flush, &ready, &completion),
              flagcxSuccess);
    ASSERT_EQ(ready, 1);
    uint32_t advanced = 0;
    int released = -2;
    ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slots[i], completion,
                                               &advanced, &released),
              flagcxSuccess);
    totalAdvanced += advanced;
  }
  EXPECT_EQ(totalAdvanced, 2u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, CancelingNonTailGetCompactsVisibilitySequence) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submits[2] = {track(&transport), track(&transport)};
  uint32_t slots[2] = {};
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submits[i], 0, -1,
                                              &slots[i]),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyRequireGetFlush(
                  &transport, slots[i], i, 0, 8,
                  reinterpret_cast<void *>(static_cast<uintptr_t>(i + 1))),
              flagcxSuccess);
  }
  const uint32_t domainIndex = transport.requests[slots[0]].getVisibilityDomain;
  ASSERT_EQ(transport.requests[slots[0]].getSequence, 1u);
  ASSERT_EQ(transport.requests[slots[1]].getSequence, 2u);

  ASSERT_EQ(flagcxKernelProxyCancelRequest(&transport, slots[0]),
            flagcxSuccess);
  EXPECT_EQ(transport.requests[slots[0]].state,
            FLAGCX_KERNEL_PROXY_REQUEST_FREE);
  EXPECT_EQ(transport.requests[slots[1]].getSequence, 1u);
  EXPECT_EQ(transport.getVisibilityDomains[domainIndex].issuedGetSequence, 1u);

  uint32_t retrySlot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submits[0], 0, -1,
                                            &retrySlot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyRequireGetFlush(
                &transport, retrySlot, 0, 0, 8,
                reinterpret_cast<void *>(static_cast<uintptr_t>(1))),
            flagcxSuccess);
  EXPECT_EQ(transport.requests[retrySlot].getSequence, 2u);

  MockKernelRequest data[2] = {{1, flagcxSuccess}, {1, flagcxSuccess}};
  ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, slots[1], &data[0],
                                            flagcxSuccess),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, retrySlot, &data[1],
                                            flagcxSuccess),
            flagcxSuccess);
  MockKernelFlush flush = {};
  int ready = 0;
  flagcxResult_t completion = flagcxSuccess;
  const uint32_t activeSlots[2] = {slots[1], retrySlot};
  for (uint32_t activeSlot : activeSlots) {
    ASSERT_EQ(flagcxKernelProxyProgressRequest(
                  &transport, activeSlot, testKernelRequest, postKernelFlush,
                  &flush, &ready, &completion),
              flagcxSuccess);
  }
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[1],
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  ASSERT_EQ(flush.posts, 1);
  flush.requests[0].done = 1;
  uint32_t totalAdvanced = 0;
  for (uint32_t activeSlot : activeSlots) {
    ASSERT_EQ(flagcxKernelProxyProgressRequest(
                  &transport, activeSlot, testKernelRequest, postKernelFlush,
                  &flush, &ready, &completion),
              flagcxSuccess);
    EXPECT_EQ(ready, 1);
    uint32_t advanced = 0;
    int released = -2;
    ASSERT_EQ(flagcxKernelProxyCompleteRequest(
                  &transport, activeSlot, completion, &advanced, &released),
              flagcxSuccess);
    totalAdvanced += advanced;
  }
  EXPECT_EQ(totalAdvanced, 2u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, AbortingPostedGetDoesNotBlockLaterGet) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext submits[2] = {track(&transport), track(&transport)};
  MockKernelRequest data[2] = {{0, flagcxSuccess}, {1, flagcxSuccess}};
  uint32_t slots[2] = {};
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &submits[i], 0, -1,
                                              &slots[i]),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyRequireGetFlush(
                  &transport, slots[i], i, 0, 8,
                  reinterpret_cast<void *>(static_cast<uintptr_t>(i + 1))),
              flagcxSuccess);
    ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, slots[i], &data[i],
                                              flagcxSuccess),
              flagcxSuccess);
  }

  int released = -2;
  ASSERT_EQ(flagcxKernelProxyAbortRequest(&transport, slots[0], &released),
            flagcxSuccess);
  EXPECT_EQ(transport.requests[slots[1]].getSequence, 1u);
  EXPECT_EQ(transport.nativeInflight, 1u);

  MockKernelFlush flush = {};
  int ready = 0;
  flagcxResult_t completion = flagcxSuccess;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[1],
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[1],
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  ASSERT_EQ(flush.posts, 1);
  flush.requests[0].done = 1;
  ASSERT_EQ(flagcxKernelProxyProgressRequest(&transport, slots[1],
                                             testKernelRequest, postKernelFlush,
                                             &flush, &ready, &completion),
            flagcxSuccess);
  ASSERT_EQ(ready, 1);
  uint32_t advanced = 0;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(&transport, slots[1], completion,
                                             &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(transport.nativeInflight, 0u);
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, StagingSlotsRemainOwnedUntilRequestCompletion) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 2, 1, 0),
            flagcxSuccess);

  int firstStaging = -1;
  int secondStaging = -1;
  int exhausted = -1;
  ASSERT_EQ(flagcxKernelProxyAcquireStagingSlot(&transport, &firstStaging),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyAcquireStagingSlot(&transport, &secondStaging),
            flagcxSuccess);
  EXPECT_NE(firstStaging, secondStaging);
  EXPECT_EQ(flagcxKernelProxyAcquireStagingSlot(&transport, &exhausted),
            flagcxInProgress);

  flagcxNetSubmitContext first = track(&transport);
  flagcxNetSubmitContext second = track(&transport);
  uint32_t firstSlot = 0;
  uint32_t secondSlot = 0;
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &first, 0, firstStaging,
                                            &firstSlot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, firstSlot,
                                            reinterpret_cast<void *>(0x1),
                                            flagcxSuccess),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyReserveRequest(&transport, &second, 0,
                                            secondStaging, &secondSlot),
            flagcxSuccess);
  ASSERT_EQ(flagcxKernelProxyPublishRequest(&transport, secondSlot,
                                            reinterpret_cast<void *>(0x2),
                                            flagcxSuccess),
            flagcxSuccess);

  uint32_t advanced = 0;
  int released = -1;
  ASSERT_EQ(flagcxKernelProxyCompleteRequest(
                &transport, secondSlot, flagcxSuccess, &advanced, &released),
            flagcxSuccess);
  EXPECT_EQ(released, secondStaging);
  ASSERT_EQ(flagcxKernelProxyReleaseStagingSlot(&transport, released),
            flagcxSuccess);
  EXPECT_EQ(flagcxKernelProxyAcquireStagingSlot(&transport, &exhausted),
            flagcxSuccess);
  EXPECT_EQ(exhausted, secondStaging);
  EXPECT_NE(transport.stagingInUse[firstStaging], 0u);

  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, SequenceWindowReportsBackpressure) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 2, 0, 1, 0),
            flagcxSuccess);
  flagcxNetSubmitContext first = track(&transport);
  flagcxNetSubmitContext second = track(&transport);
  flagcxNetSubmitContext blocked = {};
  EXPECT_EQ(flagcxKernelProxyTrackNext(&transport, 0, &blocked),
            flagcxInProgress);
  EXPECT_EQ(transport.nextSequence, 3u);

  uint32_t advanced = 0;
  ASSERT_EQ(flagcxKernelProxyCompleteImmediate(&transport, &first,
                                               flagcxSuccess, &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  ASSERT_EQ(flagcxKernelProxyTrackNext(&transport, 0, &blocked), flagcxSuccess);
  EXPECT_EQ(blocked.sequence, 3u);
  (void)second;
  flagcxKernelProxyTransportDestroy(&transport);
}

TEST(KernelProxyTransportTest, FailedDataSuppressesReleaseSubmission) {
  flagcxKernelProxyTransport transport = {};
  ASSERT_EQ(flagcxKernelProxyTransportInit(&transport, 4, 0, 3, 2),
            flagcxSuccess);
  flagcxNetSubmitContext data = track(&transport);
  flagcxNetSubmitContext release = track(
      &transport, FLAGCX_NET_SUBMIT_RELEASE | FLAGCX_NET_SUBMIT_INDEPENDENT);

  int ready = 1;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(
      flagcxKernelProxyReleaseReady(&transport, &release, &ready, &firstError),
      flagcxSuccess);
  EXPECT_EQ(ready, 0);
  EXPECT_EQ(firstError, flagcxSuccess);

  uint32_t advanced = 0;
  ASSERT_EQ(flagcxKernelProxyCompleteImmediate(&transport, &data,
                                               flagcxRemoteError, &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  ASSERT_EQ(
      flagcxKernelProxyReleaseReady(&transport, &release, &ready, &firstError),
      flagcxSuccess);
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(firstError, flagcxRemoteError);

  ASSERT_EQ(flagcxKernelProxyCompleteImmediate(&transport, &release, firstError,
                                               &advanced),
            flagcxSuccess);
  EXPECT_EQ(advanced, 1u);
  flagcxKernelProxyTransportDestroy(&transport);
}
