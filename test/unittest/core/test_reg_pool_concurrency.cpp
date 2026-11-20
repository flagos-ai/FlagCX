#include "reg_pool.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

namespace {
struct RegistryProbe {
  flagcxRegPool *pool;
  std::atomic<int> p2pCalls{0}, netCalls{0}, homoCalls{0}, failures{0};
  bool rendezvous = false, reenter = false;
  std::atomic<bool> timedOut{false};
  int arrived = 0;
  std::mutex mutex;
  std::condition_variable changed;
};
RegistryProbe *probe;
flagcxResult_t closeResult() {
  if (probe->rendezvous) {
    std::unique_lock<std::mutex> lock(probe->mutex);
    probe->arrived++;
    probe->changed.notify_all();
    if (!probe->changed.wait_for(lock, std::chrono::seconds(2),
                                 [] { return probe->arrived == 2; }))
      probe->timedOut = true;
    if (probe->timedOut)
      return flagcxSystemError;
  }
  if (probe->reenter) {
    // The metadata lock must not be held while an external close is running.
    EXPECT_GE(probe->pool->cachedP2pCount(nullptr), 0u);
  }
  int remaining = probe->failures.load();
  while (remaining > 0)
    if (probe->failures.compare_exchange_weak(remaining, remaining - 1))
      return flagcxRemoteError;
  return flagcxSuccess;
}
flagcxResult_t closeP2p(flagcxHeteroComm *, flagcxIpcRegInfo *) {
  probe->p2pCalls++;
  return closeResult();
}
flagcxResult_t closeNet(void *, flagcxProxyConnector *, void *) {
  probe->netCalls++;
  return closeResult();
}
flagcxResult_t closeHomo(void *, void *) {
  probe->homoCalls++;
  return closeResult();
}
class RegPoolConcurrencyTest : public ::testing::Test {
protected:
  RegistryProbe state;
  flagcxRegPoolOps operations{closeP2p, closeNet, closeHomo};
  flagcxRegPool pool{&operations};
  void *a = (void *)0x1000, *b = (void *)0x2000;
  void *data = (void *)0x100000;
  void SetUp() override {
    probe = &state;
    state.pool = &pool;
  }
  flagcxRegItem *registerItem(void *owner) {
    EXPECT_EQ(pool.registerBuffer(owner, data, 4096), flagcxSuccess);
    return pool.getItem(owner, data);
  }
  void addMapping(void *owner, flagcxRegItem *item, int peer) {
    auto *info = (flagcxIpcRegInfo *)calloc(1, sizeof(flagcxIpcRegInfo));
    info->peerRank = peer;
    info->allocationBase = data;
    EXPECT_EQ(pool.addP2pHandle(owner, item, info, nullptr), flagcxSuccess);
  }
};
} // namespace

TEST_F(RegPoolConcurrencyTest, SharedItemP2pClosesConcurrentlyOutsidePoolLock) {
  auto *item = registerItem(a);
  ASSERT_EQ(registerItem(b), item);
  addMapping(a, item, 0);
  addMapping(b, item, 1);
  state.rendezvous = state.reenter = true;
  flagcxResult_t results[2];
  std::thread first([&] { results[0] = pool.removeAllP2pHandles(a); });
  std::thread second([&] { results[1] = pool.removeAllP2pHandles(b); });
  first.join();
  second.join();
  EXPECT_EQ(results[0], flagcxSuccess);
  EXPECT_EQ(results[1], flagcxSuccess);
  EXPECT_EQ(state.p2pCalls, 2);
  EXPECT_EQ(pool.cachedP2pCount(nullptr), 0u);
  ASSERT_TRUE(item->handles.empty());
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(pool.deregisterBuffer(b, item), flagcxSuccess);
}

TEST_F(RegPoolConcurrencyTest,
       SharedItemNetClosesConcurrentlyAndCompactsSafely) {
  auto *item = registerItem(a);
  ASSERT_EQ(registerItem(b), item);
  ASSERT_EQ(pool.addNetHandle(a, item, (void *)1, nullptr), flagcxSuccess);
  ASSERT_EQ(pool.addNetHandle(b, item, (void *)2, nullptr), flagcxSuccess);
  state.rendezvous = state.reenter = true;
  flagcxResult_t results[2];
  std::thread first([&] { results[0] = pool.removeAllNetHandles(a); });
  std::thread second([&] { results[1] = pool.removeAllNetHandles(b); });
  first.join();
  second.join();
  EXPECT_EQ(results[0], flagcxSuccess);
  EXPECT_EQ(results[1], flagcxSuccess);
  EXPECT_EQ(state.netCalls, 2);
  EXPECT_TRUE(item->handles.empty());
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(pool.deregisterBuffer(b, item), flagcxSuccess);
}

TEST_F(RegPoolConcurrencyTest,
       FailedFinalReleasePreservesReferenceAndBlocksReuse) {
  auto *item = registerItem(a);
  ASSERT_EQ(registerItem(a), item);
  addMapping(a, item, 1);
  state.failures = 1;
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(state.p2pCalls, 0);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxRemoteError);
  EXPECT_EQ(item->refCount, 1);
  EXPECT_EQ(pool.cachedP2pCount(a), 1u);
  flagcxIpcRegInfo *borrowed = nullptr;
  EXPECT_EQ(pool.acquireP2pHandle(a, item, 1, data, &borrowed),
            flagcxInProgress);
  EXPECT_EQ(borrowed, nullptr);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(state.p2pCalls, 2);
  EXPECT_EQ(pool.cachedP2pCount(a), 0u);
  EXPECT_EQ(pool.getItem(a, data), nullptr);
}

TEST_F(RegPoolConcurrencyTest, SnapshotPinsLastOwnerUntilOperationFinishes) {
  auto *item = registerItem(a);
  flagcxRegSnapshot snapshot = {};
  ASSERT_EQ(pool.acquireItem(a, data, &snapshot), flagcxSuccess);
  EXPECT_EQ(snapshot.item, item);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxInProgress);
  EXPECT_EQ(item->refCount, 1);
  pool.releaseItem(&snapshot);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(pool.getItem(a, data), nullptr);
}

TEST_F(RegPoolConcurrencyTest, PendingRegistrationCannotBeRemovedOrReused) {
  auto *item = registerItem(a);
  void *native = nullptr;
  int create = 0;
  ASSERT_EQ(pool.beginNetHandle(a, item, nullptr, &native, &create),
            flagcxSuccess);
  EXPECT_EQ(create, 1);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxInProgress);
  EXPECT_EQ(item->refCount, 1);
  EXPECT_EQ(pool.beginNetHandle(a, item, nullptr, &native, &create),
            flagcxInProgress);
  ASSERT_EQ(pool.finishNetHandle(a, item, nullptr, (void *)1, flagcxSuccess),
            flagcxSuccess);
  state.failures = 1;
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxRemoteError);
  EXPECT_EQ(item->refCount, 1);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(state.netCalls, 2);
}

TEST_F(RegPoolConcurrencyTest,
       LocalOwnerReleaseKeepsOtherOwnersAndIsIdempotent) {
  auto *item = registerItem(a);
  ASSERT_EQ(registerItem(a), item);
  ASSERT_EQ(registerItem(b), item);
  EXPECT_EQ(pool.releaseOwnerReferences(a), flagcxSuccess);
  EXPECT_EQ(item->refCount, 1);
  EXPECT_EQ(pool.releaseOwnerReferences(a), flagcxSuccess);
  EXPECT_EQ(item->refCount, 1);
  EXPECT_EQ(pool.deregisterBuffer(b, item), flagcxSuccess);
  EXPECT_EQ(pool.getItem(b, data), nullptr);
}

TEST_F(RegPoolConcurrencyTest,
       OverlappingRegistrationAndReleaseKeepOtherCommLive) {
  auto *item = registerItem(a);
  ASSERT_EQ(registerItem(b), item);
  std::thread first([&] {
    for (int i = 0; i < 300; i++) {
      flagcxRegSnapshot snapshot = {};
      EXPECT_EQ(pool.registerAndAcquire(a, (char *)data + 128, 128, &snapshot),
                flagcxSuccess);
      EXPECT_EQ(snapshot.item, item);
      pool.releaseItem(&snapshot);
      EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
    }
  });
  std::thread second([&] {
    for (int i = 0; i < 300; i++) {
      flagcxRegSnapshot snapshot = {};
      EXPECT_EQ(pool.registerAndAcquire(b, (char *)data + 256, 128, &snapshot),
                flagcxSuccess);
      EXPECT_EQ(snapshot.item, item);
      pool.releaseItem(&snapshot);
      EXPECT_EQ(pool.deregisterBuffer(b, item), flagcxSuccess);
    }
  });
  first.join();
  second.join();
  EXPECT_EQ(item->refCount, 2);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(pool.deregisterBuffer(b, item), flagcxSuccess);
}

TEST_F(RegPoolConcurrencyTest, ConcurrentLastOwnerReleasesCloseEachHandleOnce) {
  auto *item = registerItem(a);
  ASSERT_EQ(registerItem(b), item);
  addMapping(a, item, 0);
  addMapping(b, item, 1);
  state.rendezvous = true;
  flagcxResult_t results[2];
  std::thread first([&] { results[0] = pool.deregisterBuffer(a, item); });
  std::thread second([&] { results[1] = pool.deregisterBuffer(b, item); });
  first.join();
  second.join();
  EXPECT_EQ(results[0], flagcxSuccess);
  EXPECT_EQ(results[1], flagcxSuccess);
  EXPECT_EQ(state.p2pCalls, 2);
  EXPECT_EQ(pool.cachedP2pCount(nullptr), 0u);
  EXPECT_EQ(pool.getItem(nullptr, data), nullptr);
}

TEST_F(RegPoolConcurrencyTest, SharedMappingReaderBlocksCloseUntilReleased) {
  auto *first = registerItem(a);
  void *otherData = (char *)data + 8192;
  ASSERT_EQ(pool.registerBuffer(a, otherData, 4096), flagcxSuccess);
  auto *second = pool.getItem(a, otherData);
  ASSERT_NE(first, second);
  addMapping(a, first, 1);
  flagcxIpcRegInfo *info = nullptr;
  ASSERT_EQ(pool.acquireP2pHandle(a, second, 1, data, &info), flagcxSuccess);
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->refCount, 2);
  EXPECT_EQ(pool.removeAllP2pHandles(a), flagcxInProgress);
  EXPECT_EQ(state.p2pCalls, 0);
  pool.releaseP2pHandle(info);
  EXPECT_EQ(pool.removeAllP2pHandles(a), flagcxSuccess);
  EXPECT_EQ(state.p2pCalls, 1);
  EXPECT_EQ(pool.cachedP2pCount(a), 0u);
  EXPECT_EQ(pool.releaseOwnerReferences(a), flagcxSuccess);
  EXPECT_EQ(pool.getItem(a, data), nullptr);
  EXPECT_EQ(pool.getItem(a, otherData), nullptr);
}

TEST_F(RegPoolConcurrencyTest,
       NativeReservationAndCloseFailureRetainOwnership) {
  auto *item = registerItem(a);
  void *handle = nullptr;
  int create = 0;
  ASSERT_EQ(pool.beginHomoHandle(a, item, &handle, &create), flagcxSuccess);
  EXPECT_EQ(create, 1);
  EXPECT_EQ(pool.beginHomoHandle(a, item, &handle, &create), flagcxInProgress);
  EXPECT_EQ(create, 0);
  EXPECT_EQ(pool.removeAllHomoHandles(a), flagcxInProgress);
  EXPECT_EQ(state.homoCalls, 0);
  ASSERT_EQ(pool.finishHomoHandle(a, item, (void *)1, flagcxSuccess),
            flagcxSuccess);
  state.failures = 1;
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxRemoteError);
  EXPECT_EQ(item->refCount, 1);
  EXPECT_EQ(pool.releaseOwnerReferences(a), flagcxInvalidUsage);
  EXPECT_EQ(pool.deregisterBuffer(a, item), flagcxSuccess);
  EXPECT_EQ(state.homoCalls, 2);
  EXPECT_EQ(pool.getItem(a, data), nullptr);
}
