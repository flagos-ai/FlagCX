#include "comm.h"
#include "global_comm.h"
#include "sym_heap.h"
#include <gtest/gtest.h>

namespace {

struct SymWindowIpcFixture : public ::testing::Test {
  void SetUp() override {
    comm.rank = 0;
    comm.nRanks = 2;
    comm.node = 0;
    comm.localRank = 0;
    comm.localRanks = 2;
    comm.rankToNode = rankToNode;
    comm.rankToLocalRank = rankToLocalRank;
    comm.ipcTable = ipcTable;
    comm.ipcTableSize = 1;

    window.localBase = localMemory;
    window.heapSize = sizeof(localMemory);
    window.localRanks = 2;
    window.mrIndex = -1;
    window.ipcSlot = 0;
    comm.symWindows = &window;

    peerPointers[0] = localMemory;
    peerPointers[1] = peerMemory;
    ipcTable[0].hostPeerPtrs = peerPointers;
    ipcTable[0].nPeers = 2;
    ipcTable[0].basePtr = localMemory;
    ipcTable[0].inUse = true;
  }

  flagcxHeteroComm comm = {};
  flagcxSymWindow window = {};
  flagcxIpcTableEntry ipcTable[1] = {};
  int rankToNode[2] = {0, 0};
  int rankToLocalRank[2] = {0, 1};
  unsigned char localMemory[64] = {};
  unsigned char peerMemory[64] = {};
  void *peerPointers[2] = {};
};

TEST_F(SymWindowIpcFixture, FindsLocalRangeWithoutNetworkMr) {
  size_t offset = 0;
  flagcxSymWindow_t found =
      flagcxSymWindowFind(&comm, localMemory + 7, 16, &offset);

  EXPECT_EQ(found, &window);
  EXPECT_EQ(offset, 7u);
  EXPECT_EQ(window.mrIndex, -1);
}

TEST_F(SymWindowIpcFixture, ResolvesPeerRangeWithoutNetworkMr) {
  void *resolved = nullptr;
  EXPECT_EQ(
      flagcxSymWindowResolveIpcPeerPtr(&comm, &window, 1, 11, 8, &resolved),
      flagcxSuccess);
  EXPECT_EQ(resolved, peerMemory + 11);
  EXPECT_EQ(window.mrIndex, -1);
}

TEST_F(SymWindowIpcFixture, RejectsRangeOutsideWindow) {
  void *resolved = nullptr;
  EXPECT_EQ(flagcxSymWindowResolveIpcPeerPtr(
                &comm, &window, 1, sizeof(peerMemory) - 3, 4, &resolved),
            flagcxInvalidArgument);
  EXPECT_EQ(resolved, nullptr);
}

TEST_F(SymWindowIpcFixture, RejectsPeerOnAnotherNode) {
  rankToNode[1] = 1;
  void *resolved = nullptr;
  EXPECT_EQ(
      flagcxSymWindowResolveIpcPeerPtr(&comm, &window, 1, 0, 1, &resolved),
      flagcxNotSupported);
  EXPECT_EQ(resolved, nullptr);
}

} // namespace
