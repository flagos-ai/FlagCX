#include "adaptor.h"
#include "device_api/flagcx_device.h"
#include "flagcx.h"
#include "flagcx_hetero.h"
#include "global_comm.h"
#include "param.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <thread>

namespace {
struct SavedEnvironment {
  const char *name;
  bool present;
  std::string value;
  SavedEnvironment(const char *key, const char *replacement) : name(key) {
    const char *old = getenv(key);
    present = old != nullptr;
    value = old ? old : "";
    setenv(key, replacement, 1);
  }
  ~SavedEnvironment() {
    if (present)
      setenv(name, value.c_str(), 1);
    else
      unsetenv(name);
  }
};
struct IndependentWork {
  std::atomic<bool> entered{false}, release{false}, expired{false};
  static void hold(void *argument) {
    auto *work = (IndependentWork *)argument;
    work->entered = true;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!work->release.load()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        work->expired = true;
        break;
      }
      std::this_thread::yield();
    }
  }
};
struct IsolationOwners {
  IndependentWork work;
  flagcxComm_t comms[2] = {};
  flagcxStream_t streams[2] = {};
  void *buffers[2] = {};
  ~IsolationOwners() {
    // Always unblock the callback, including ASSERT failures. A regression
    // also self-times out, so device-wide synchronization cannot hang CI.
    work.release = true;
    for (int i = 0; i < 2; i++) {
      if (streams[i])
        EXPECT_EQ(deviceAdaptor->streamSynchronize(streams[i]), flagcxSuccess);
      if (comms[i])
        EXPECT_EQ(flagcxCommDestroy(comms[i]), flagcxSuccess);
      if (buffers[i])
        EXPECT_EQ(
            deviceAdaptor->deviceFree(buffers[i], flagcxMemDevice, nullptr),
            flagcxSuccess);
      if (streams[i])
        EXPECT_EQ(deviceAdaptor->streamDestroy(streams[i]), flagcxSuccess);
    }
  }
};
} // namespace

// Each process owns one rank and two independent single-rank communicators.
// Uses the real adaptor/SDK in the existing Device API unit CI target.
static void runIsolation(bool withPartialOwners, bool withRma = false) {
  std::fprintf(stderr, "ISOLATION begin partial=%d rma=%d\n", withPartialOwners,
               withRma);
  std::fflush(stderr);
  int count = 0;
  if (!deviceAdaptor->getDeviceCount || !deviceAdaptor->launchHostFunc ||
      deviceAdaptor->getDeviceCount(&count) != flagcxSuccess || count == 0)
    GTEST_SKIP() << "Requires a real device and asynchronous host callbacks";
  SavedEnvironment native("FLAGCX_USE_HETERO_COMM", withRma ? "1" : "0");
  SavedEnvironment vmm("FLAGCX_VMM_ENABLE", withRma ? "0" : "1");
  if (withRma && flagcxParamVmmEnable() != 0)
    GTEST_SKIP()
        << "Run make run-unit: non-VMM isolation needs a fresh process";
  SavedEnvironment split("FLAGCX_CLUSTER_SPLIT_LIST", "1");
  ASSERT_EQ(deviceAdaptor->setDevice(0), flagcxSuccess);
  IsolationOwners owners;
  for (int i = 0; i < 2; i++) {
    flagcxUniqueId id = {};
    ASSERT_EQ(flagcxGetUniqueId(&id), flagcxSuccess);
    std::fprintf(stderr, "ISOLATION comm-init begin index=%d\n", i);
    std::fflush(stderr);
    ASSERT_EQ(flagcxCommInitRank(&owners.comms[i], 1, &id, 0), flagcxSuccess);
    std::fprintf(stderr, "ISOLATION comm-init done index=%d\n", i);
    std::fflush(stderr);
    ASSERT_EQ(deviceAdaptor->streamCreate(&owners.streams[i]), flagcxSuccess);
    ASSERT_EQ(deviceAdaptor->deviceMalloc(&owners.buffers[i], sizeof(float),
                                          flagcxMemDevice, nullptr),
              flagcxSuccess);
    ASSERT_EQ(deviceAdaptor->deviceMemset(owners.buffers[i], 0, sizeof(float),
                                          flagcxMemDevice, owners.streams[i]),
              flagcxSuccess);
  }
  if (withPartialOwners) {
    // Exercise cleanup of partially materialized Device API owners with real
    // runtime allocations. No vendor DevComm feature or device kernel needed.
    auto *comm = owners.comms[0];
    comm->activeDevComms =
        (flagcxDevComm_t)calloc(1, sizeof(flagcxDevCommInternal));
    ASSERT_NE(comm->activeDevComms, nullptr);
    auto *devComm = comm->activeDevComms;
    pthread_mutex_init(&devComm->cachedPtrMutex, nullptr);
    devComm->barrierIpcIndex = devComm->signalIpcSlot = -1;
    ASSERT_EQ(deviceAdaptor->deviceMalloc(&devComm->cachedDevicePtr, 64,
                                          flagcxMemDevice, nullptr),
              flagcxSuccess);
    comm->activeDevMems =
        (flagcxDevMem_t)calloc(1, sizeof(flagcxDevMemInternal));
    ASSERT_NE(comm->activeDevMems, nullptr);
    auto *devMem = comm->activeDevMems;
    pthread_mutex_init(&devMem->cachedPtrMutex, nullptr);
    devMem->ipcIndex = -1;
    ASSERT_EQ(deviceAdaptor->deviceMalloc(&devMem->cachedDevicePtr, 64,
                                          flagcxMemDevice, nullptr),
              flagcxSuccess);
  }
  // RMA workers are created at CommInitRank. This test checks their cleanup,
  // and must not require single-rank heterogeneous AllReduce support.
  if (!withRma) {
    ASSERT_EQ(flagcxAllReduce(owners.buffers[0], owners.buffers[0], 1,
                              flagcxFloat, flagcxSum, owners.comms[0],
                              owners.streams[0]),
              flagcxSuccess);
  }
  ASSERT_EQ(deviceAdaptor->streamSynchronize(owners.streams[0]), flagcxSuccess);
  if (withRma) {
    ASSERT_NE(owners.comms[0]->heteroComm, nullptr);
    auto *proxy = owners.comms[0]->heteroComm->rmaProxy;
    ASSERT_NE(proxy, nullptr);
    ASSERT_NE(proxy->doneSeqsDev, nullptr);
    ASSERT_NE(proxy->readySeqsDev, nullptr);
    // Complete initialization before B's independent work is blocked.
    ASSERT_EQ(deviceAdaptor->deviceSynchronize(), flagcxSuccess);
  }
  std::fprintf(stderr, "ISOLATION independent callback submit\n");
  std::fflush(stderr);
  flagcxResult_t launched = deviceAdaptor->launchHostFunc(
      owners.streams[1], IndependentWork::hold, &owners.work);
  if (launched == flagcxNotSupported)
    GTEST_SKIP() << "Adaptor does not support asynchronous host callbacks";
  ASSERT_EQ(launched, flagcxSuccess);
  if (withRma) {
    // No new allocations after the callback starts: isolate Finalize's own
    // RMA frees from unrelated submission-time allocator behavior.
    ASSERT_EQ(deviceAdaptor->deviceMemset(owners.buffers[1], 0, sizeof(float),
                                          flagcxMemDevice, owners.streams[1]),
              flagcxSuccess);
  } else {
    ASSERT_EQ(flagcxAllReduce(owners.buffers[1], owners.buffers[1], 1,
                              flagcxFloat, flagcxSum, owners.comms[1],
                              owners.streams[1]),
              flagcxSuccess);
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!owners.work.entered && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  ASSERT_TRUE(owners.work.entered);
  ASSERT_FALSE(owners.work.expired);
  std::fprintf(stderr, "ISOLATION finalize A begin\n");
  std::fflush(stderr);
  EXPECT_EQ(flagcxCommFinalize(owners.comms[0]), flagcxSuccess);
  std::fprintf(stderr, "ISOLATION finalize A done expired=%d\n",
               owners.work.expired.load());
  std::fflush(stderr);
  EXPECT_FALSE(owners.work.expired)
      << "Finalize(A) waited for independent work on communicator B";
  if (withRma) {
    auto *proxy = owners.comms[0]->heteroComm->rmaProxy;
    ASSERT_NE(proxy, nullptr);
    EXPECT_TRUE(proxy->stopComplete);
    EXPECT_NE(proxy->doneSeqsDev, nullptr);
    EXPECT_NE(proxy->readySeqsDev, nullptr);
  }
  if (withPartialOwners) {
    EXPECT_EQ(owners.comms[0]->activeDevComms, nullptr);
    EXPECT_EQ(owners.comms[0]->activeDevMems, nullptr);
    EXPECT_NE(owners.comms[0]->internalBuffers, nullptr);
  }
  owners.work.release = true;
}

TEST(CommStreamIsolation, FinalizeDoesNotDrainIndependentCommunicator) {
  runIsolation(false);
}

TEST(CommStreamIsolation, PartialOwnerCachesDoNotDrainIndependentCommunicator) {
  runIsolation(true);
}

TEST(CommStreamIsolation, HeteroRmaNonVmmFinalizeDoesNotDrainIndependentWork) {
  runIsolation(false, true);
}
