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
#include "proxy.h"

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
flagcxResult_t freeExportResult = flagcxSuccess;
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
  return freeExportResult;
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
    closeImportResult = freeExportResult = flagcxSuccess;
    closedImport = nullptr;
    freedExport = nullptr;
    selectedDevice = -1;
  }

  void TearDown() override { deviceAdaptor = savedDeviceAdaptor_; }

  struct flagcxDeviceAdaptor *savedDeviceAdaptor_ = nullptr;
  struct flagcxDeviceAdaptor testDeviceAdaptor_ = {};
};

TEST_F(P2pTeardownTest, ImportCloseUsesRawBaseAfterStreamDrain) {
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
}

TEST_F(P2pTeardownTest, ImportCloseFailureKeepsOwner) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.proxyInfo.shm = &shm;
  resources.importedRecvFifoBase = reinterpret_cast<void *>(0x3000);
  resources.proxyInfo.recvFifo = reinterpret_cast<char *>(0x3080);
  closeImportResult = flagcxSystemError;

  EXPECT_EQ(flagcxP2pCloseImportedFifo(&resources), flagcxSystemError);
  EXPECT_EQ(resources.importedRecvFifoBase, reinterpret_cast<void *>(0x3000));
  EXPECT_EQ(resources.proxyInfo.recvFifo, reinterpret_cast<char *>(0x3080));
}

TEST_F(P2pTeardownTest, MissingImportNeedsNoShmAck) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.proxyInfo.shm = &shm;

  ASSERT_EQ(flagcxP2pCloseImportedFifo(&resources), flagcxSuccess);
  EXPECT_TRUE(teardownCalls.empty());
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
}

TEST_F(P2pTeardownTest, PendingSendCleanupRetainsImportEventsAndStream) {
  struct RestoreChunks {
    int64_t saved;
    ~RestoreChunks() { flagcxP2pChunks = saved; }
  } restoreChunks{flagcxP2pChunks};
  flagcxP2pChunks = 2;
  auto *resources = (flagcxP2pResources *)calloc(1, sizeof(flagcxP2pResources));
  flagcxProxyConnection connection = {};
  connection.send = 1;
  connection.transportResources = resources;
  resources->importedRecvFifoBase = (void *)0x5000;
  resources->proxyInfo.stream = (flagcxStream_t)0x1000;
  resources->proxyInfo.events[0] = (flagcxEvent_t)0x2000;
  resources->proxyInfo.events[1] = (flagcxEvent_t)0x3000;
  static int eventCalls[2], streamCalls;
  static bool pendingEvent, pendingStream;
  eventCalls[0] = eventCalls[1] = streamCalls = 0;
  pendingEvent = pendingStream = true;
  testDeviceAdaptor_.eventDestroy = [](flagcxEvent_t event) {
    int i = event == (flagcxEvent_t)0x2000 ? 0 : 1;
    eventCalls[i]++;
    return i == 1 && pendingEvent ? flagcxInProgress : flagcxSuccess;
  };
  testDeviceAdaptor_.streamDestroy = [](flagcxStream_t) {
    streamCalls++;
    return pendingStream ? flagcxInProgress : flagcxSuccess;
  };
  auto cleanup = [&] {
    return flagcxP2pCleanupProxyConnection(
        &connection, flagcxTransportCleanupReleaseResources);
  };
  streamSynchronizeResult = flagcxInProgress;
  ASSERT_EQ(cleanup(), flagcxInProgress);
  ASSERT_EQ(connection.transportResources, resources);
  EXPECT_NE(resources->importedRecvFifoBase, nullptr);
  EXPECT_EQ(eventCalls[0] + streamCalls, 0);
  streamSynchronizeResult = flagcxSuccess;
  closeImportResult = flagcxInProgress;
  ASSERT_EQ(cleanup(), flagcxInProgress);
  ASSERT_EQ(connection.transportResources, resources);
  EXPECT_NE(resources->importedRecvFifoBase, nullptr);
  closeImportResult = flagcxSuccess;
  ASSERT_EQ(cleanup(), flagcxInProgress);
  ASSERT_EQ(connection.transportResources, resources);
  EXPECT_EQ(resources->importedRecvFifoBase, nullptr);
  EXPECT_EQ(resources->proxyInfo.events[0], nullptr);
  EXPECT_NE(resources->proxyInfo.events[1], nullptr);
  EXPECT_EQ(streamCalls, 0);
  pendingEvent = false;
  ASSERT_EQ(cleanup(), flagcxInProgress);
  ASSERT_EQ(connection.transportResources, resources);
  EXPECT_EQ(resources->proxyInfo.events[1], nullptr);
  EXPECT_NE(resources->proxyInfo.stream, nullptr);
  pendingStream = false;
  ASSERT_EQ(cleanup(), flagcxSuccess);
  EXPECT_EQ(connection.transportResources, nullptr);
  EXPECT_EQ(eventCalls[0], 1);
  EXPECT_EQ(eventCalls[1], 2);
  EXPECT_EQ(streamCalls, 2);
  EXPECT_EQ(cleanup(), flagcxSuccess);
  EXPECT_EQ(streamCalls, 2);
}

TEST_F(P2pTeardownTest, PendingExportFreeRetainsDescriptorAndAllocation) {
  auto *resources = (flagcxP2pResources *)calloc(1, sizeof(flagcxP2pResources));
  resources->localRecvFifo = (void *)0x7000;
  resources->importsClosedConfirmed = 1;
  flagcxProxyConnection connection = {};
  connection.transportResources = resources;
  static flagcxResult_t setResult;
  setResult = flagcxInProgress;
  testDeviceAdaptor_.setDevice = [](int) { return setResult; };
  auto cleanup = [&] {
    return flagcxP2pCleanupProxyConnection(
        &connection, flagcxTransportCleanupReleaseResources);
  };
  ASSERT_EQ(cleanup(), flagcxInProgress);
  ASSERT_EQ(connection.transportResources, resources);
  EXPECT_EQ(freedExport, nullptr);
  setResult = flagcxSuccess;
  freeExportResult = flagcxInProgress;
  ASSERT_EQ(cleanup(), flagcxInProgress);
  ASSERT_EQ(connection.transportResources, resources);
  EXPECT_EQ(resources->localRecvFifo, (void *)0x7000);
  freeExportResult = flagcxSuccess;
  EXPECT_EQ(cleanup(), flagcxSuccess);
  EXPECT_EQ(connection.transportResources, nullptr);
}

TEST_F(P2pTeardownTest, ExportIsPreservedUntilFinalizeConfirmation) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.shm = &shm;
  resources.localRecvFifo = reinterpret_cast<void *>(0x6000);

  EXPECT_EQ(flagcxP2pReleaseLocalFifo(&resources), flagcxInvalidUsage);
  EXPECT_EQ(freedExport, nullptr);
  EXPECT_EQ(resources.localRecvFifo, reinterpret_cast<void *>(0x6000));
}

TEST_F(P2pTeardownTest, ExportIsFreedLocallyAfterFinalize) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.shm = &shm;
  resources.localRecvFifo = reinterpret_cast<void *>(0x7000);
  resources.proxyInfo.recvFifo = reinterpret_cast<char *>(0x7000);
  resources.cudaDev = 5;
  resources.importsClosedConfirmed = 1;

  ASSERT_EQ(flagcxP2pReleaseLocalFifo(&resources), flagcxSuccess);
  EXPECT_EQ(teardownCalls,
            (std::vector<TeardownCall>{teardownSetDevice, teardownFreeExport}));
  EXPECT_EQ(selectedDevice, 5);
  EXPECT_EQ(freedExport, reinterpret_cast<void *>(0x7000));
  EXPECT_EQ(resources.localRecvFifo, nullptr);
  EXPECT_EQ(resources.proxyInfo.recvFifo, nullptr);
}

TEST_F(P2pTeardownTest, FreeFailureRetainsExportForLocalRetryWithoutShm) {
  flagcxP2pResources resources = {};
  resources.localRecvFifo = reinterpret_cast<void *>(0x8000);
  resources.importsClosedConfirmed = 1;
  freeExportResult = flagcxUnhandledDeviceError;
  EXPECT_EQ(flagcxP2pReleaseLocalFifo(&resources), flagcxUnhandledDeviceError);
  EXPECT_EQ(resources.localRecvFifo, reinterpret_cast<void *>(0x8000));
  freeExportResult = flagcxSuccess;
  EXPECT_EQ(flagcxP2pReleaseLocalFifo(&resources), flagcxSuccess);
  EXPECT_EQ(resources.localRecvFifo, nullptr);
  teardownCalls.clear();
  EXPECT_EQ(flagcxP2pReleaseLocalFifo(&resources), flagcxSuccess);
  EXPECT_TRUE(teardownCalls.empty());
}

TEST_F(P2pTeardownTest, CloseFailureRetainsStreamAndShmUntilRetry) {
  flagcxP2pShm shm = {};
  flagcxP2pResources resources = {};
  resources.proxyInfo.shm = &shm;
  resources.proxyInfo.stream = reinterpret_cast<flagcxStream_t>(0x9000);
  resources.importedRecvFifoBase = reinterpret_cast<void *>(0xa000);
  closeImportResult = flagcxSystemError;
  EXPECT_EQ(flagcxP2pSendProxyFree(&resources), flagcxSystemError);
  EXPECT_EQ(resources.proxyInfo.stream,
            reinterpret_cast<flagcxStream_t>(0x9000));
  EXPECT_EQ(resources.proxyInfo.shm, &shm);
  EXPECT_EQ(resources.importedRecvFifoBase, reinterpret_cast<void *>(0xa000));
}

TEST_F(P2pTeardownTest, ImportClosureDoesNotNeedShm) {
  flagcxP2pResources resources = {};
  resources.importedRecvFifoBase = reinterpret_cast<void *>(0xb000);
  EXPECT_EQ(flagcxP2pCloseImportedFifo(&resources), flagcxSuccess);
  EXPECT_EQ(resources.importedRecvFifoBase, nullptr);
  EXPECT_EQ(closedImport, reinterpret_cast<void *>(0xb000));
}

TEST_F(P2pTeardownTest, CachedInlineCloseFailureRetainsRawPointerForRetry) {
  flagcxHeteroComm comm = {};
  flagcxIpcRegInfo info = {};
  info.impInfo = {reinterpret_cast<void *>(0xc000), true};
  closeImportResult = flagcxSystemError;
  EXPECT_EQ(flagcxP2pDeregisterBuffer(&comm, &info), flagcxSystemError);
  EXPECT_EQ(info.impInfo.importedBase, reinterpret_cast<void *>(0xc000));
  closeImportResult = flagcxSuccess;
  EXPECT_EQ(flagcxP2pDeregisterBuffer(&comm, &info), flagcxSuccess);
  EXPECT_EQ(info.impInfo.importedBase, nullptr);
  teardownCalls.clear();
  EXPECT_EQ(flagcxP2pDeregisterBuffer(&comm, &info), flagcxSuccess);
  EXPECT_TRUE(teardownCalls.empty());
}

TEST_F(P2pTeardownTest, CachedProxyCloseCannotForgetImportWithoutLiveSocket) {
  flagcxHeteroComm comm = {};
  flagcxProxyConnector proxy = {};
  flagcxIpcRegInfo info = {};
  info.impInfo = {reinterpret_cast<void *>(0xd000), true};
  info.ipcProxyconn = &proxy;
  EXPECT_EQ(flagcxP2pDeregisterBuffer(&comm, &info), flagcxRemoteError);
  EXPECT_EQ(info.impInfo.importedBase, reinterpret_cast<void *>(0xd000));
  EXPECT_TRUE(teardownCalls.empty());
}

} // namespace
