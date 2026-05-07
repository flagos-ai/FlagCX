/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * P2P topology manager unit test — no MPI or communicator required.
 * Initializes device and net adaptors directly, builds a standalone
 * topology graph, and verifies NIC selection for each visible GPU.
 ************************************************************************/

#include <gtest/gtest.h>
#include <iostream>

#include "adaptor.h"
#include "flagcx.h"
#include "p2p_topo.h"

extern struct flagcxNetAdaptor flagcxNetIbP2p;

class P2pTopoTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize device adaptor (sets global deviceAdaptor)
    flagcxHandleInit(&handler);

    netAdaptor = &flagcxNetIbP2p;
    ASSERT_EQ(netAdaptor->init(), flagcxSuccess)
        << "IB P2P net adaptor init failed";

    int ndev = 0;
    ASSERT_EQ(netAdaptor->devices(&ndev), flagcxSuccess);
    ASSERT_GT(ndev, 0) << "No IB P2P devices found";
    std::cout << "Net adaptor: " << netAdaptor->name << " (" << ndev
              << " devices)" << std::endl;
  }

  void TearDown() override { flagcxHandleFree(handler); }

  flagcxHandlerGroup_t handler = nullptr;
  struct flagcxNetAdaptor *netAdaptor = nullptr;
};

TEST_F(P2pTopoTest, InitAndDestroy) {
  struct flagcxP2pTopoManager *mgr = nullptr;
  ASSERT_EQ(flagcxP2pTopoInit(netAdaptor, &mgr), flagcxSuccess);
  ASSERT_NE(mgr, nullptr);
  EXPECT_GT(mgr->nGpus, 0);
  std::cout << "P2P topo manager: " << mgr->nGpus << " GPUs" << std::endl;
  EXPECT_EQ(flagcxP2pTopoDestroy(mgr), flagcxSuccess);
}

TEST_F(P2pTopoTest, GetNetDevForEachGpu) {
  struct flagcxP2pTopoManager *mgr = nullptr;
  ASSERT_EQ(flagcxP2pTopoInit(netAdaptor, &mgr), flagcxSuccess);

  for (int gpu = 0; gpu < mgr->nGpus; gpu++) {
    int netDev = -1;
    EXPECT_EQ(flagcxP2pTopoGetNetDev(mgr, gpu, &netDev), flagcxSuccess);
    EXPECT_GE(netDev, 0);
    std::cout << "  GPU " << gpu << " -> NIC " << netDev << std::endl;
  }

  EXPECT_EQ(flagcxP2pTopoDestroy(mgr), flagcxSuccess);
}

TEST_F(P2pTopoTest, OutOfRangeGpuDev) {
  struct flagcxP2pTopoManager *mgr = nullptr;
  ASSERT_EQ(flagcxP2pTopoInit(netAdaptor, &mgr), flagcxSuccess);

  int netDev = -1;
  EXPECT_NE(flagcxP2pTopoGetNetDev(mgr, -1, &netDev), flagcxSuccess);
  EXPECT_NE(flagcxP2pTopoGetNetDev(mgr, mgr->nGpus, &netDev), flagcxSuccess);

  EXPECT_EQ(flagcxP2pTopoDestroy(mgr), flagcxSuccess);
}

TEST_F(P2pTopoTest, NullArgs) {
  EXPECT_NE(flagcxP2pTopoGetNetDev(nullptr, 0, nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxP2pTopoDestroy(nullptr), flagcxSuccess);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
