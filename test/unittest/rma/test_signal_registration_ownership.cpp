/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "onesided.h"

namespace {

flagcxHeteroComm *signalOwnerHeteroComm = nullptr;
void *freedSignalBuffer = nullptr;
int signalBufferFreeCount = 0;
bool signalStateClearedBeforeFree = false;
void *closedIpcMappings[4] = {};
int closedIpcMappingCount = 0;
void *queriedAllocationBase = nullptr;
size_t queriedAllocationSize = 0;
flagcxResult_t addressRangeResult = flagcxSuccess;

flagcxResult_t recordSignalGdrFree(void *ptr, void *) {
  freedSignalBuffer = ptr;
  signalBufferFreeCount++;
  signalStateClearedBeforeFree =
      signalOwnerHeteroComm != nullptr &&
      signalOwnerHeteroComm->rmaSignalBase == nullptr &&
      signalOwnerHeteroComm->rmaSignalSize == 0 &&
      signalOwnerHeteroComm->rmaSignalIpcSlot == -1;
  return flagcxSuccess;
}

flagcxResult_t recordIpcMemHandleClose(void *ptr) {
  if (closedIpcMappingCount < 4)
    closedIpcMappings[closedIpcMappingCount] = ptr;
  closedIpcMappingCount++;
  return flagcxSuccess;
}

flagcxResult_t queryTestAddressRange(const void *, void **base, size_t *size) {
  if (addressRangeResult != flagcxSuccess)
    return addressRangeResult;
  *base = queriedAllocationBase;
  *size = queriedAllocationSize;
  return flagcxSuccess;
}

class RmaSignalRegistrationOwnershipTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (strcmp(devApiBackend->name, "default") != 0)
      GTEST_SKIP() << "requires the default Device API backend";

    savedDeviceAdaptor_ = deviceAdaptor;
    testDeviceAdaptor_ = *deviceAdaptor;
    testDeviceAdaptor_.gdrMemFree = recordSignalGdrFree;
    testDeviceAdaptor_.ipcMemHandleClose = recordIpcMemHandleClose;
    deviceAdaptor = &testDeviceAdaptor_;
    signalOwnerHeteroComm = nullptr;
    freedSignalBuffer = nullptr;
    signalBufferFreeCount = 0;
    signalStateClearedBeforeFree = false;
    memset(closedIpcMappings, 0, sizeof(closedIpcMappings));
    closedIpcMappingCount = 0;
    queriedAllocationBase = nullptr;
    queriedAllocationSize = 0;
    addressRangeResult = flagcxSuccess;
  }

  void TearDown() override {
    if (savedDeviceAdaptor_ != nullptr)
      deviceAdaptor = savedDeviceAdaptor_;
    signalOwnerHeteroComm = nullptr;
  }

  struct flagcxDeviceAdaptor *savedDeviceAdaptor_ = nullptr;
  struct flagcxDeviceAdaptor testDeviceAdaptor_ = {};
};

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcOnlyRegistrationIsRemovedBeforeBackingBuffer) {
  constexpr int ipcSlot = 3;
  void *signalBuffer = reinterpret_cast<void *>(0x6000);

  flagcxHeteroComm heteroComm = {};
  heteroComm.rmaSignalBase = signalBuffer;
  heteroComm.rmaSignalSize = sizeof(uint64_t);
  heteroComm.rmaSignalIpcSlot = ipcSlot;
  heteroComm.signalHandle = nullptr;
  signalOwnerHeteroComm = &heteroComm;

  flagcxComm comm = {};
  comm.heteroComm = &heteroComm;
  comm.ipcTable[ipcSlot].inUse = true;

  flagcxDevCommInternal devComm = {};
  devComm.barrierIpcIndex = -1;
  devComm.signalIpcSlot = -1;
  devComm.signalBuffer = static_cast<uint64_t *>(signalBuffer);
  devComm.ownedSignalBuffer = signalBuffer;
  devComm.ownedSignalRegistration = nullptr;

  ASSERT_EQ(devApiBackend->devCommDestroy(&comm, &devComm), flagcxSuccess);

  EXPECT_EQ(heteroComm.rmaSignalBase, nullptr);
  EXPECT_EQ(heteroComm.rmaSignalSize, 0u);
  EXPECT_EQ(heteroComm.rmaSignalIpcSlot, -1);
  EXPECT_FALSE(comm.ipcTable[ipcSlot].inUse);
  EXPECT_EQ(devComm.ownedSignalBuffer, nullptr);
  EXPECT_EQ(freedSignalBuffer, signalBuffer);
  EXPECT_EQ(signalBufferFreeCount, 1);
  EXPECT_TRUE(signalStateClearedBeforeFree);

  ASSERT_EQ(devApiBackend->devCommDestroy(&comm, &devComm), flagcxSuccess);
  EXPECT_EQ(freedSignalBuffer, signalBuffer);
  EXPECT_EQ(signalBufferFreeCount, 1);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcTableCleanupClosesRawMappingBase) {
  flagcxComm comm = {};
  struct flagcxIpcTableEntry *entry = &comm.ipcTable[0];
  entry->hostPeerPtrs = static_cast<void **>(calloc(2, sizeof(void *)));
  entry->hostPeerBasePtrs = static_cast<void **>(calloc(2, sizeof(void *)));
  ASSERT_NE(entry->hostPeerPtrs, nullptr);
  ASSERT_NE(entry->hostPeerBasePtrs, nullptr);

  entry->hostPeerPtrs[0] = reinterpret_cast<void *>(0x200000);
  entry->hostPeerPtrs[1] = reinterpret_cast<void *>(0x100400);
  entry->hostPeerBasePtrs[1] = reinterpret_cast<void *>(0x100000);
  entry->nPeers = 2;
  entry->basePtr = entry->hostPeerPtrs[0];
  entry->inUse = false;

  ASSERT_EQ(flagcxCommCleanupIpcTable(&comm), flagcxSuccess);
  ASSERT_EQ(closedIpcMappingCount, 1);
  EXPECT_EQ(closedIpcMappings[0], reinterpret_cast<void *>(0x100000));
  EXPECT_EQ(entry->hostPeerPtrs, nullptr);
  EXPECT_EQ(entry->hostPeerBasePtrs, nullptr);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       DeferredIpcCleanupClosesRawMappingBase) {
  flagcxComm comm = {};
  flagcxIntruQueueConstruct(&comm.deferredIpcQueue);
  struct flagcxIpcTableEntry *entry = &comm.ipcTable[0];
  entry->hostPeerPtrs = static_cast<void **>(calloc(2, sizeof(void *)));
  entry->hostPeerBasePtrs = static_cast<void **>(calloc(2, sizeof(void *)));
  ASSERT_NE(entry->hostPeerPtrs, nullptr);
  ASSERT_NE(entry->hostPeerBasePtrs, nullptr);

  entry->hostPeerPtrs[0] = reinterpret_cast<void *>(0x200000);
  entry->hostPeerPtrs[1] = reinterpret_cast<void *>(0x100400);
  entry->hostPeerBasePtrs[1] = reinterpret_cast<void *>(0x100000);
  entry->nPeers = 2;
  entry->basePtr = entry->hostPeerPtrs[0];
  entry->inUse = true;

  releaseIpcTableSlot(&comm, 0);
  EXPECT_EQ(entry->hostPeerPtrs, nullptr);
  EXPECT_EQ(entry->hostPeerBasePtrs, nullptr);
  ASSERT_EQ(flagcxCommDrainDeferredIpc(&comm), flagcxSuccess);
  ASSERT_EQ(closedIpcMappingCount, 1);
  EXPECT_EQ(closedIpcMappings[0], reinterpret_cast<void *>(0x100000));
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcExportRangePreservesInteriorOffset) {
  testDeviceAdaptor_.getAddressRange = queryTestAddressRange;
  queriedAllocationBase = reinterpret_cast<void *>(0x100000);
  queriedAllocationSize = 0x2000;

  void *exportBase = nullptr;
  size_t allocationSize = 0;
  size_t userOffset = 0;
  ASSERT_EQ(flagcxGetIpcExportRange(reinterpret_cast<void *>(0x100400), 0x800,
                                    &exportBase, &allocationSize, &userOffset),
            flagcxSuccess);
  EXPECT_EQ(exportBase, queriedAllocationBase);
  EXPECT_EQ(allocationSize, 0x2000u);
  EXPECT_EQ(userOffset, 0x400u);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcPeerAddressUsesAllocationRelativeOffset) {
  void *peerPtr = nullptr;
  ASSERT_EQ(flagcxResolveIpcPeerAddress(reinterpret_cast<void *>(0x800000),
                                        0x2000, 0x400, 0x800, &peerPtr),
            flagcxSuccess);
  EXPECT_EQ(peerPtr, reinterpret_cast<void *>(0x800400));
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcPeerAddressRejectsRangePastAllocation) {
  void *peerPtr = reinterpret_cast<void *>(0x1);
  EXPECT_EQ(flagcxResolveIpcPeerAddress(reinterpret_cast<void *>(0x800000),
                                        0x1000, 0xf00, 0x200, &peerPtr),
            flagcxInvalidUsage);
  EXPECT_EQ(peerPtr, nullptr);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcPeerAddressRejectsAddressOverflow) {
  void *peerPtr = reinterpret_cast<void *>(0x1);
  EXPECT_EQ(
      flagcxResolveIpcPeerAddress(reinterpret_cast<void *>(UINTPTR_MAX - 0x100),
                                  0x1000, 0x200, 0x100, &peerPtr),
      flagcxInvalidUsage);
  EXPECT_EQ(peerPtr, nullptr);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcExportRangeFallsBackWhenCallbackIsNull) {
  testDeviceAdaptor_.getAddressRange = nullptr;
  void *userPtr = reinterpret_cast<void *>(0x100400);
  void *exportBase = nullptr;
  size_t allocationSize = 0;
  size_t userOffset = 1;

  ASSERT_EQ(flagcxGetIpcExportRange(userPtr, 0x800, &exportBase,
                                    &allocationSize, &userOffset),
            flagcxSuccess);
  EXPECT_EQ(exportBase, userPtr);
  EXPECT_EQ(allocationSize, 0x800u);
  EXPECT_EQ(userOffset, 0u);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcExportRangeRejectsRangeOutsideAllocation) {
  testDeviceAdaptor_.getAddressRange = queryTestAddressRange;
  queriedAllocationBase = reinterpret_cast<void *>(0x100000);
  queriedAllocationSize = 0x1000;

  void *exportBase = nullptr;
  size_t allocationSize = 0;
  size_t userOffset = 0;
  EXPECT_EQ(flagcxGetIpcExportRange(reinterpret_cast<void *>(0x100f00), 0x200,
                                    &exportBase, &allocationSize, &userOffset),
            flagcxInvalidUsage);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       IpcExportRangeFallsBackWhenQueryIsUnsupported) {
  testDeviceAdaptor_.getAddressRange = queryTestAddressRange;
  addressRangeResult = flagcxNotSupported;
  void *userPtr = reinterpret_cast<void *>(0x100400);
  void *exportBase = nullptr;
  size_t allocationSize = 0;
  size_t userOffset = 1;

  ASSERT_EQ(flagcxGetIpcExportRange(userPtr, 0x800, &exportBase,
                                    &allocationSize, &userOffset),
            flagcxSuccess);
  EXPECT_EQ(exportBase, userPtr);
  EXPECT_EQ(allocationSize, 0x800u);
  EXPECT_EQ(userOffset, 0u);
}

TEST_F(RmaSignalRegistrationOwnershipTest, IpcExportRangePropagatesQueryError) {
  testDeviceAdaptor_.getAddressRange = queryTestAddressRange;
  addressRangeResult = flagcxSystemError;
  void *userPtr = reinterpret_cast<void *>(0x100400);
  void *exportBase = nullptr;
  size_t allocationSize = 0;
  size_t userOffset = 1;

  EXPECT_EQ(flagcxGetIpcExportRange(userPtr, 0x800, &exportBase,
                                    &allocationSize, &userOffset),
            flagcxSystemError);
  EXPECT_EQ(exportBase, userPtr);
  EXPECT_EQ(allocationSize, 0x800u);
  EXPECT_EQ(userOffset, 0u);
}

TEST(RmaSignalRegistrationOwnership,
     RejectsDifferentBufferWhileRegistrationIsActive) {
  void *registeredBuffer = reinterpret_cast<void *>(0x7000);
  void *differentBuffer = reinterpret_cast<void *>(0x8000);

  flagcxHeteroComm heteroComm = {};
  heteroComm.rmaSignalBase = registeredBuffer;
  heteroComm.rmaSignalSize = sizeof(uint64_t);
  heteroComm.rmaSignalIpcSlot = -1;

  flagcxComm comm = {};
  comm.heteroComm = &heteroComm;

  EXPECT_EQ(flagcxOneSideSignalRegister(&comm, differentBuffer,
                                        sizeof(uint64_t), FLAGCX_PTR_CUDA),
            flagcxInvalidUsage);
  EXPECT_EQ(heteroComm.rmaSignalBase, registeredBuffer);
  EXPECT_EQ(heteroComm.rmaSignalSize, sizeof(uint64_t));
}

} // namespace
