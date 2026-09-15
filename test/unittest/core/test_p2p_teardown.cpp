/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "adaptor.h"
#include "p2p.h"

namespace {

enum TeardownCall {
  teardownStreamSynchronize,
  teardownSetDevice,
  teardownCloseImport,
  teardownFreeExport,
};

std::vector<TeardownCall> teardownCalls;
flagcxResult_t streamSynchronizeResult = flagcxSuccess;
flagcxResult_t closeImportResult = flagcxSuccess;
void *closedImport = nullptr;
void *freedExport = nullptr;
int selectedDevice = -1;

flagcxResult_t recordStreamSynchronize(flagcxStream_t) {
  teardownCalls.push_back(teardownStreamSynchronize);
  return streamSynchronizeResult;
}

flagcxResult_t recordSetDevice(int device) {
  teardownCalls.push_back(teardownSetDevice);
  selectedDevice = device;
  return flagcxSuccess;
}

flagcxResult_t recordCloseImport(void *ptr) {
  teardownCalls.push_back(teardownCloseImport);
  closedImport = ptr;
  return closeImportResult;
}

flagcxResult_t recordFreeExport(void *ptr, flagcxMemType_t, flagcxStream_t) {
  teardownCalls.push_back(teardownFreeExport);
  freedExport = ptr;
  return flagcxSuccess;
}

class P2pTeardownTest : public ::testing::Test {
protected:
  void SetUp() override {
    savedDeviceAdaptor_ = deviceAdaptor;
    testDeviceAdaptor_ = {};
    testDeviceAdaptor_.streamSynchronize = recordStreamSynchronize;
    testDeviceAdaptor_.setDevice = recordSetDevice;
    testDeviceAdaptor_.ipcMemHandleClose = recordCloseImport;
    testDeviceAdaptor_.deviceFree = recordFreeExport;
    deviceAdaptor = &testDeviceAdaptor_;

    teardownCalls.clear();
    streamSynchronizeResult = flagcxSuccess;
    closeImportResult = flagcxSuccess;
    closedImport = nullptr;
    freedExport = nullptr;
    selectedDevice = -1;
  }

  void TearDown() override { deviceAdaptor = savedDeviceAdaptor_; }

  struct flagcxDeviceAdaptor *savedDeviceAdaptor_ = nullptr;
  struct flagcxDeviceAdaptor testDeviceAdaptor_ = {};
};

TEST_F(P2pTeardownTest, ImportClosePublishesAckAfterStreamDrain) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.proxyInfo.shm = &shm;
  resources.proxyInfo.stream = reinterpret_cast<flagcxStream_t>(0x1000);
  resources.proxyInfo.recvFifo = reinterpret_cast<char *>(0x2080);
  resources.importedRecvFifoBase = reinterpret_cast<void *>(0x2000);
  resources.cudaDev = 3;

  ASSERT_EQ(flagcxP2pCloseImportedFifo(&resources), flagcxSuccess);

  EXPECT_EQ((std::vector<TeardownCall>{teardownStreamSynchronize,
                                       teardownSetDevice, teardownCloseImport}),
            teardownCalls);
  EXPECT_EQ(selectedDevice, 3);
  EXPECT_EQ(closedImport, reinterpret_cast<void *>(0x2000));
  EXPECT_EQ(resources.importedRecvFifoBase, nullptr);
  EXPECT_EQ(resources.proxyInfo.recvFifo, nullptr);
  EXPECT_EQ(__atomic_load_n(&shm.fifoImportClosed, __ATOMIC_ACQUIRE),
            flagcxP2pFifoImportClosed);
}

TEST_F(P2pTeardownTest, ImportCloseFailureDoesNotPublishAck) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.proxyInfo.shm = &shm;
  resources.importedRecvFifoBase = reinterpret_cast<void *>(0x3000);
  resources.proxyInfo.recvFifo = reinterpret_cast<char *>(0x3080);
  closeImportResult = flagcxSystemError;

  EXPECT_EQ(flagcxP2pCloseImportedFifo(&resources), flagcxSystemError);
  EXPECT_EQ(resources.importedRecvFifoBase, reinterpret_cast<void *>(0x3000));
  EXPECT_EQ(resources.proxyInfo.recvFifo, reinterpret_cast<char *>(0x3080));
  EXPECT_EQ(__atomic_load_n(&shm.fifoImportClosed, __ATOMIC_ACQUIRE),
            flagcxP2pFifoImportOpen);
}

TEST_F(P2pTeardownTest, MissingImportStillPublishesCloseAck) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.proxyInfo.shm = &shm;

  ASSERT_EQ(flagcxP2pCloseImportedFifo(&resources), flagcxSuccess);
  EXPECT_TRUE(teardownCalls.empty());
  EXPECT_EQ(__atomic_load_n(&shm.fifoImportClosed, __ATOMIC_ACQUIRE),
            flagcxP2pFifoImportClosed);
}

TEST_F(P2pTeardownTest, StreamFailureKeepsImportOpen) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.proxyInfo.shm = &shm;
  resources.proxyInfo.stream = reinterpret_cast<flagcxStream_t>(0x4000);
  resources.importedRecvFifoBase = reinterpret_cast<void *>(0x5000);
  streamSynchronizeResult = flagcxUnhandledDeviceError;

  EXPECT_EQ(flagcxP2pCloseImportedFifo(&resources), flagcxUnhandledDeviceError);
  EXPECT_EQ(teardownCalls,
            (std::vector<TeardownCall>{teardownStreamSynchronize}));
  EXPECT_EQ(resources.importedRecvFifoBase, reinterpret_cast<void *>(0x5000));
  EXPECT_EQ(__atomic_load_n(&shm.fifoImportClosed, __ATOMIC_ACQUIRE),
            flagcxP2pFifoImportOpen);
}

TEST_F(P2pTeardownTest, ExportIsPreservedUntilCloseAckArrives) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.shm = &shm;
  resources.localRecvFifo = reinterpret_cast<void *>(0x6000);

  EXPECT_EQ(flagcxP2pReleaseLocalFifo(&resources, 0), flagcxSystemError);
  EXPECT_EQ(freedExport, nullptr);
  EXPECT_EQ(resources.localRecvFifo, reinterpret_cast<void *>(0x6000));
}

TEST_F(P2pTeardownTest, ExportIsFreedAfterCloseAck) {
  flagcxP2pShm shm = {};
  shm.fifoImportClosed = flagcxP2pFifoImportClosed;
  flagcxP2pResources resources = {};
  resources.shm = &shm;
  resources.localRecvFifo = reinterpret_cast<void *>(0x7000);
  resources.proxyInfo.recvFifo = reinterpret_cast<char *>(0x7000);
  resources.cudaDev = 5;

  ASSERT_EQ(flagcxP2pReleaseLocalFifo(&resources, 0), flagcxSuccess);
  EXPECT_EQ(teardownCalls,
            (std::vector<TeardownCall>{teardownSetDevice, teardownFreeExport}));
  EXPECT_EQ(selectedDevice, 5);
  EXPECT_EQ(freedExport, reinterpret_cast<void *>(0x7000));
  EXPECT_EQ(resources.localRecvFifo, nullptr);
  EXPECT_EQ(resources.proxyInfo.recvFifo, nullptr);
}

TEST_F(P2pTeardownTest, ExportWaitsForDelayedCloseAck) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.shm = &shm;
  resources.localRecvFifo = reinterpret_cast<void *>(0x8000);

  std::thread importer([&shm] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    __atomic_store_n(&shm.fifoImportClosed, flagcxP2pFifoImportClosed,
                     __ATOMIC_RELEASE);
  });
  flagcxResult_t result = flagcxP2pReleaseLocalFifo(&resources, 1000);
  importer.join();

  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_EQ(freedExport, reinterpret_cast<void *>(0x8000));
}

TEST_F(P2pTeardownTest, MissingAckChannelPreservesExport) {
  flagcxP2pResources resources = {};
  resources.localRecvFifo = reinterpret_cast<void *>(0x9000);

  EXPECT_EQ(flagcxP2pReleaseLocalFifo(&resources, 1000), flagcxInternalError);
  EXPECT_EQ(freedExport, nullptr);
  EXPECT_EQ(resources.localRecvFifo, reinterpret_cast<void *>(0x9000));
}

} // namespace
