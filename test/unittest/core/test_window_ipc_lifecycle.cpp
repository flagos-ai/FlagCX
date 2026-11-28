#include "adaptor.h"
#include "global_comm.h"
#include "sym_heap.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

namespace {
class WindowIpcLifecycle : public ::testing::Test {
public:
  inline static WindowIpcLifecycle *current;
  flagcxDeviceAdaptor device = {};
  flagcxDeviceAdaptor *saved;
  flagcxComm comm = {};
  flagcxHeteroComm hetero = {};
  flagcxWindow window = {};
  char backing[64] = {};
  std::vector<void *> closed, freed;
  void *failedImport = nullptr;
  flagcxResult_t failure = flagcxSuccess;
  int deviceSyncs = 0;
  int unmaps = 0, vaFrees = 0;
  bool failVa = true;

  void SetUp() override {
    current = this;
    saved = deviceAdaptor;
    device.ipcMemHandleClose = [](void *p) {
      current->closed.push_back(p);
      return p == current->failedImport ? current->failure : flagcxSuccess;
    };
    device.deviceFree = [](void *p, flagcxMemType_t, flagcxStream_t) {
      current->freed.push_back(p);
      free(p);
      return flagcxSuccess;
    };
    device.deviceSynchronize = [] {
      ++current->deviceSyncs;
      return flagcxUnhandledDeviceError;
    };
    deviceAdaptor = &device;
    comm.commType = flagcxCommunicatorHybrid;
    comm.heteroComm = &hetero;
    hetero.nRanks = hetero.localRanks = 1;
    hetero.ipcTable = comm.ipcTable;
    hetero.ipcTableSize = comm.ipcTableCapacity;
    auto *sym = (flagcxSymWindow *)calloc(1, sizeof(flagcxSymWindow));
    sym->owner = &window;
    sym->localBase = backing;
    sym->heapSize = sizeof(backing);
    sym->mrIndex = -1;
    sym->ipcSlot = 0;
    sym->published = true;
    sym->state = flagcxSymWindowPublished;
    hetero.symWindows = sym;
    window.ownerComm = &comm;
    window.defaultBase = sym;
    window.isSymmetricDefault = 1;
    auto &entry = comm.ipcTable[0];
    entry.nPeers = 3;
    entry.basePtr = backing;
    entry.size = sizeof(backing);
    entry.inUse = entry.ownerHeld = true;
    entry.hostPeerBasePtrs = (void **)calloc(3, sizeof(void *));
    entry.hostPeerBasePtrs[1] = (void *)0x1000;
    entry.hostPeerBasePtrs[2] = (void *)0x2000;
    entry.hostPeerPtrs = (void **)calloc(3, sizeof(void *));
    entry.devPeerPtrs = (void **)calloc(3, sizeof(void *));
    backing[0] = 42;
  }
  void TearDown() override {
    failedImport = nullptr;
    failure = flagcxSuccess;
    window.devMemRefs = 0;
    if (window.defaultBase)
      EXPECT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
                flagcxSuccess);
    deviceAdaptor = saved;
  }
};

TEST_F(WindowIpcLifecycle, DeregisterClosesImportsAndLeavesUserBufferAlive) {
  EXPECT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
            flagcxSuccess);
  EXPECT_EQ(closed, (std::vector<void *>{(void *)0x1000, (void *)0x2000}));
  ASSERT_EQ(freed.size(), 1u); // Only the library-owned peer pointer array.
  EXPECT_NE(freed[0], backing);
  EXPECT_EQ(backing[0], 42);
  backing[1] = 43;
  EXPECT_EQ(backing[1], 43);
  EXPECT_EQ(window.defaultBase, nullptr);
  EXPECT_EQ(comm.internalBuffers, nullptr);
  EXPECT_EQ(deviceSyncs, 0);
}

TEST_F(WindowIpcLifecycle, FailedCloseRetainsOnlyUnfinishedImports) {
  failedImport = (void *)0x2000;
  failure = flagcxRemoteError;
  EXPECT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
            flagcxRemoteError);
  EXPECT_TRUE(window.cleanupStarted);
  EXPECT_NE(window.defaultBase, nullptr);
  EXPECT_EQ(comm.ipcTable[0].hostPeerBasePtrs[1], nullptr);
  EXPECT_EQ(comm.ipcTable[0].hostPeerBasePtrs[2], failedImport);
  EXPECT_TRUE(freed.empty());
  failure = flagcxSuccess;
  ASSERT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
            flagcxSuccess);
  EXPECT_EQ(closed, (std::vector<void *>{(void *)0x1000, (void *)0x2000,
                                         (void *)0x2000}));
}

TEST_F(WindowIpcLifecycle, InProgressDoesNotConsumeImportOwner) {
  failedImport = (void *)0x1000;
  failure = flagcxInProgress;
  EXPECT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
            flagcxInProgress);
  EXPECT_EQ(comm.ipcTable[0].hostPeerBasePtrs[1], failedImport);
  EXPECT_TRUE(freed.empty());
}

TEST_F(WindowIpcLifecycle, LiveViewPreventsClosingImports) {
  window.devMemRefs = 1;
  EXPECT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
            flagcxInvalidUsage);
  EXPECT_TRUE(closed.empty());
  EXPECT_FALSE(window.cleanupStarted);
}
TEST_F(WindowIpcLifecycle, VmmVaFailureRetainsOwnerWithoutRepeatingUnmap) {
  auto *sym = window.defaultBase;
  sym->resourcePath = flagcxSymResourceSegments;
  sym->flatBase = (void *)0x3000;
  sym->flatMappingOwned = sym->flatVaOwned = sym->hasFlatMapping = true;
  device.symFlatMappingUnmapSegments = [](void *) {
    ++current->unmaps;
    return flagcxSuccess;
  };
  device.symFlatVaFreeSegments = [](void *) {
    ++current->vaFrees;
    return current->failVa ? flagcxRemoteError : flagcxSuccess;
  };
  EXPECT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
            flagcxRemoteError);
  EXPECT_EQ(window.defaultBase, sym);
  EXPECT_FALSE(sym->flatMappingOwned);
  EXPECT_TRUE(sym->flatVaOwned);
  EXPECT_EQ(backing[0], 42);
  failVa = false;
  ASSERT_EQ(flagcxCommWindowDeregister(&comm, &window, flagcxMemCCL),
            flagcxSuccess);
  EXPECT_EQ(unmaps, 1);
  EXPECT_EQ(vaFrees, 2);
  EXPECT_EQ(deviceSyncs, 0);
}
} // namespace
