#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "flagcx_hetero.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "p2p.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

namespace {
class InternalIpcLifecycle : public ::testing::Test {
public:
  inline static InternalIpcLifecycle *current;
  flagcxDeviceAdaptor device = {};
  flagcxDeviceAdaptor *saved;
  flagcxComm comm = {};
  std::vector<void *> closed, freed;
  bool failClose = false, failFree = false;
  int syncs = 0;
  void SetUp() override {
    if (devApiBackend->kind != flagcxDevApiDefault)
      GTEST_SKIP();
    current = this;
    saved = deviceAdaptor;
    device.deviceMalloc = [](void **p, size_t bytes, flagcxMemType_t,
                             flagcxStream_t) {
      *p = calloc(1, bytes);
      return *p ? flagcxSuccess : flagcxSystemError;
    };
    device.deviceFree = [](void *p, flagcxMemType_t, flagcxStream_t) {
      if (current->failFree)
        return flagcxRemoteError;
      current->freed.push_back(p);
      free(p);
      return flagcxSuccess;
    };
    device.gdrMemFree = [](void *p, void *) {
      return current->device.deviceFree(p, flagcxMemDevice, nullptr);
    };
    device.setDevice = [](int) { return flagcxSuccess; };
    device.ipcMemHandleClose = [](void *p) {
      current->closed.push_back(p);
      return current->failClose ? flagcxInProgress : flagcxSuccess;
    };
    device.deviceSynchronize = [] {
      ++current->syncs;
      return flagcxUnhandledDeviceError;
    };
    deviceAdaptor = &device;
  }
  void TearDown() override {
    if (IsSkipped())
      return;
    failClose = failFree = false;
    comm.ipcClosingImports = true;
    EXPECT_EQ(flagcxCommCloseRetiredDevComms(&comm), flagcxSuccess);
    EXPECT_EQ(flagcxCommCleanupIpcTable(&comm), flagcxSuccess);
    comm.internalImportsConfirmed = true;
    EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSuccess);
    deviceAdaptor = saved;
  }
  flagcxDevComm_t makeInternalOwner(int slot) {
    auto *view = (flagcxDevComm_t)calloc(1, sizeof(flagcxDevCommInternal));
    pthread_mutex_init(&view->cachedPtrMutex, nullptr);
    view->barrierIpcIndex = slot;
    view->signalIpcSlot = -1;
    void *buffer = nullptr;
    EXPECT_EQ(flagcxCommAllocInternalBuffer(&comm, &buffer, 64,
                                            flagcxInternalBufferDevice),
              flagcxSuccess);
    view->localBarrierFlags = (uint64_t *)buffer;
    view->localBarrierFlagsDeviceAllocated = true;
    auto &entry = comm.ipcTable[slot];
    entry.basePtr = buffer;
    entry.ownership = flagcxBufferInternal;
    entry.inUse = entry.ownerHeld = true;
    entry.nPeers = 1;
    entry.hostPeerBasePtrs = (void **)calloc(1, sizeof(void *));
    entry.hostPeerBasePtrs[0] = (void *)(uintptr_t)(0x1000 + slot * 0x1000);
    entry.devPeerPtrs = (void **)calloc(1, sizeof(void *));
    return view;
  }
};

TEST_F(InternalIpcLifecycle,
       RetiredViewsKeepImportsAndBackingUntilCommClosure) {
  auto *first = makeInternalOwner(0);
  ASSERT_EQ(flagcxDevCommDestroy(&comm, first), flagcxSuccess);
  EXPECT_EQ(comm.retiredDevComms, first);
  EXPECT_TRUE(closed.empty());
  EXPECT_TRUE(freed.empty());
  void *unused = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(first, &unused), flagcxInvalidUsage);
  auto *second = makeInternalOwner(1);
  ASSERT_EQ(flagcxDevCommDestroy(&comm, second), flagcxSuccess);
  comm.ipcClosingImports = true;
  ASSERT_EQ(flagcxCommCloseRetiredDevComms(&comm), flagcxSuccess);
  EXPECT_EQ(comm.retiredDevComms, nullptr);
  EXPECT_EQ(closed.size(), 2u);
  EXPECT_TRUE(freed.empty());
  EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxInvalidUsage);
  comm.internalImportsConfirmed = true;
  ASSERT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed.size(), 4u); // Two exporters and two peer-pointer arrays.
  EXPECT_EQ(syncs, 0);
}

TEST_F(InternalIpcLifecycle, PendingInternalCloseRetainsRetiredOwner) {
  auto *view = makeInternalOwner(0);
  ASSERT_EQ(flagcxDevCommDestroy(&comm, view), flagcxSuccess);
  comm.ipcClosingImports = true;
  failClose = true;
  EXPECT_EQ(flagcxCommCloseRetiredDevComms(&comm), flagcxInProgress);
  EXPECT_EQ(comm.retiredDevComms, view);
  EXPECT_NE(comm.ipcTable[0].hostPeerBasePtrs[0], nullptr);
  EXPECT_TRUE(freed.empty());
  failClose = false;
  ASSERT_EQ(flagcxCommCloseRetiredDevComms(&comm), flagcxSuccess);
  EXPECT_EQ(comm.retiredDevComms, nullptr);
  EXPECT_EQ(closed.size(), 2u);
}

TEST_F(InternalIpcLifecycle, BackingFreeFailureRetainsAllocationRecord) {
  void *buffer = nullptr;
  ASSERT_EQ(flagcxCommAllocInternalBuffer(&comm, &buffer, 64,
                                          flagcxInternalBufferDevice),
            flagcxSuccess);
  comm.internalImportsConfirmed = true;
  failFree = true;
  EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxRemoteError);
  ASSERT_NE(comm.internalBuffers, nullptr);
  EXPECT_EQ(comm.internalBuffers->pointer, buffer);
  failFree = false;
  ASSERT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed, (std::vector<void *>{buffer}));
}

TEST_F(InternalIpcLifecycle, RmaStopRetainsCounterBackingForRelease) {
  flagcxHeteroComm hetero = {};
  auto *proxy = (flagcxRmaProxyState *)calloc(1, sizeof(flagcxRmaProxyState));
  hetero.rmaProxy = proxy;
  proxy->doneSeqsDev = (uint64_t *)malloc(8);
  proxy->readySeqsDev = (uint64_t *)malloc(8);
  proxy->doneSeqsCpu = (uint64_t *)calloc(1, 8);
  proxy->readySeqsCpu = (uint64_t *)calloc(1, 8);
  pthread_mutex_init(&proxy->doneMutex, nullptr);
  pthread_cond_init(&proxy->doneCond, nullptr);
  ASSERT_EQ(flagcxHeteroRmaProxyStop(&hetero), flagcxSuccess);
  EXPECT_TRUE(freed.empty());
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&hetero), flagcxInvalidUsage);
  hetero.importsClosedConfirmed = true;
  ASSERT_EQ(flagcxHeteroRmaProxyRelease(&hetero), flagcxSuccess);
  EXPECT_EQ(hetero.rmaProxy, nullptr);
  EXPECT_EQ(freed.size(), 2u);
}

TEST_F(InternalIpcLifecycle, P2pClosePhaseDoesNotReleaseFifoBacking) {
  flagcxP2pShm shm = {};
  auto *sender = (flagcxP2pResources *)calloc(1, sizeof(flagcxP2pResources));
  sender->importedRecvFifoBase = (void *)0x4000;
  sender->proxyInfo.shm = &shm;
  flagcxProxyConnection send = {};
  send.send = 1;
  send.transportResources = sender;
  ASSERT_EQ(flagcxP2pCleanupProxyConnection(&send,
                                            flagcxTransportCleanupCloseImports),
            flagcxSuccess);
  EXPECT_EQ(send.transportResources, sender);
  EXPECT_EQ(closed.size(), 1u);
  EXPECT_TRUE(freed.empty());
  ASSERT_EQ(flagcxP2pCleanupProxyConnection(
                &send, flagcxTransportCleanupReleaseResources),
            flagcxSuccess);
  EXPECT_EQ(send.transportResources, sender);
  free(sender);
  auto *receiver = (flagcxP2pResources *)calloc(1, sizeof(flagcxP2pResources));
  receiver->localRecvFifo = malloc(64);
  receiver->importsClosedConfirmed = true;
  flagcxProxyConnection recv = {};
  recv.transportResources = receiver;
  ASSERT_EQ(flagcxP2pCleanupProxyConnection(&recv,
                                            flagcxTransportCleanupCloseImports),
            flagcxSuccess);
  EXPECT_TRUE(freed.empty());
  ASSERT_EQ(flagcxP2pCleanupProxyConnection(
                &recv, flagcxTransportCleanupReleaseResources),
            flagcxSuccess);
  EXPECT_EQ(freed.size(), 1u);
  free(receiver);
}
} // namespace
