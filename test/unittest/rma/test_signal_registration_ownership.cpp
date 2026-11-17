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
#include "flagcx_net_adaptor.h"
#include "global_comm.h"
#include "onesided.h"
#include "onesided_types.h"

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
int deregMrCalls = 0;
int closeSendCalls = 0;
int closeRecvCalls = 0;
flagcxResult_t nextDeregisterResult = flagcxSuccess;

flagcxResult_t retryableDeregisterMr(void *, void *) {
  deregMrCalls++;
  flagcxResult_t result = nextDeregisterResult;
  nextDeregisterResult = flagcxSuccess;
  return result;
}

flagcxResult_t recordCloseSend(void *) {
  closeSendCalls++;
  return flagcxSuccess;
}

flagcxResult_t recordCloseRecv(void *) {
  closeRecvCalls++;
  return flagcxSuccess;
}

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
    deregMrCalls = 0;
    closeSendCalls = 0;
    closeRecvCalls = 0;
    nextDeregisterResult = flagcxSuccess;
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
       DeregisterFailureRetainsSignalOwnerAndBackingBufferForRetry) {
  void *signalBuffer = reinterpret_cast<void *>(0x7000);
  flagcxNetAdaptor_latest net = {};
  net.deregMr = retryableDeregisterMr;

  flagcxOneSideHandleInfo *info =
      static_cast<flagcxOneSideHandleInfo *>(calloc(1, sizeof(*info)));
  ASSERT_NE(info, nullptr);
  info->localMrHandle = reinterpret_cast<void *>(0x7100);
  info->localRecvComm = reinterpret_cast<void *>(0x7200);
  info->ownsLocalMr = 1;

  flagcxHeteroComm heteroComm = {};
  heteroComm.netAdaptor = &net;
  heteroComm.signalHandle = info;
  heteroComm.rmaSignalBase = signalBuffer;
  heteroComm.rmaSignalSize = sizeof(uint64_t);
  heteroComm.rmaSignalIpcSlot = -1;
  signalOwnerHeteroComm = &heteroComm;

  flagcxComm comm = {};
  comm.heteroComm = &heteroComm;

  flagcxDevCommInternal devComm = {};
  devComm.barrierIpcIndex = -1;
  devComm.signalIpcSlot = -1;
  devComm.signalBuffer = static_cast<uint64_t *>(signalBuffer);
  devComm.ownedSignalBuffer = signalBuffer;
  devComm.ownedSignalRegistration = info;

  nextDeregisterResult = flagcxRemoteError;
  EXPECT_EQ(devApiBackend->devCommDestroy(&comm, &devComm), flagcxRemoteError);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(heteroComm.signalHandle, info);
  EXPECT_EQ(heteroComm.rmaSignalBase, signalBuffer);
  EXPECT_EQ(devComm.ownedSignalBuffer, signalBuffer);
  EXPECT_EQ(devComm.ownedSignalRegistration, info);
  EXPECT_EQ(signalBufferFreeCount, 0);

  EXPECT_EQ(devApiBackend->devCommDestroy(&comm, &devComm), flagcxSuccess);
  EXPECT_EQ(deregMrCalls, 2);
  EXPECT_EQ(heteroComm.signalHandle, nullptr);
  EXPECT_EQ(heteroComm.rmaSignalBase, nullptr);
  EXPECT_EQ(devComm.ownedSignalBuffer, nullptr);
  EXPECT_EQ(devComm.ownedSignalRegistration, nullptr);
  EXPECT_EQ(signalBufferFreeCount, 1);
  EXPECT_EQ(freedSignalBuffer, signalBuffer);
  EXPECT_TRUE(signalStateClearedBeforeFree);
}

TEST_F(RmaSignalRegistrationOwnershipTest,
       DevCommDestroyPropagatesDeregisterFailureWithoutFreeingOwner) {
  void *signalBuffer = reinterpret_cast<void *>(0x7300);
  flagcxNetAdaptor_latest net = {};
  net.deregMr = retryableDeregisterMr;

  flagcxOneSideHandleInfo *info =
      static_cast<flagcxOneSideHandleInfo *>(calloc(1, sizeof(*info)));
  ASSERT_NE(info, nullptr);
  info->localMrHandle = reinterpret_cast<void *>(0x7400);
  info->localRecvComm = reinterpret_cast<void *>(0x7500);
  info->ownsLocalMr = 1;

  flagcxHeteroComm heteroComm = {};
  heteroComm.netAdaptor = &net;
  heteroComm.signalHandle = info;
  heteroComm.rmaSignalBase = signalBuffer;
  heteroComm.rmaSignalSize = sizeof(uint64_t);
  heteroComm.rmaSignalIpcSlot = -1;
  signalOwnerHeteroComm = &heteroComm;

  flagcxComm comm = {};
  comm.heteroComm = &heteroComm;

  flagcxDevCommInternal *devComm = static_cast<flagcxDevCommInternal *>(
      calloc(1, sizeof(flagcxDevCommInternal)));
  ASSERT_NE(devComm, nullptr);
  pthread_mutex_init(&devComm->cachedPtrMutex, nullptr);
  devComm->barrierIpcIndex = -1;
  devComm->signalIpcSlot = -1;
  devComm->signalBuffer = static_cast<uint64_t *>(signalBuffer);
  devComm->ownedSignalBuffer = signalBuffer;
  devComm->ownedSignalRegistration = info;
  heteroComm.devCommHandle = devComm;

  nextDeregisterResult = flagcxRemoteError;
  EXPECT_EQ(flagcxDevCommDestroy(&comm, devComm), flagcxRemoteError);
  EXPECT_EQ(heteroComm.devCommHandle, devComm);
  EXPECT_EQ(heteroComm.signalHandle, info);
  EXPECT_EQ(signalBufferFreeCount, 0);

  EXPECT_EQ(flagcxDevCommDestroy(&comm, devComm), flagcxSuccess);
  EXPECT_EQ(heteroComm.devCommHandle, nullptr);
  EXPECT_EQ(heteroComm.signalHandle, nullptr);
  EXPECT_EQ(signalBufferFreeCount, 1);
}

TEST(RmaRegistrationOwnership,
     DeregisterFailureRetainsMrAndConnectionsForRetry) {
  deregMrCalls = 0;
  closeSendCalls = 0;
  closeRecvCalls = 0;
  nextDeregisterResult = flagcxSuccess;
  flagcxNetAdaptor_latest net = {};
  net.deregMr = retryableDeregisterMr;
  net.closeSend = recordCloseSend;
  net.closeRecv = recordCloseRecv;

  flagcxOneSideHandleInfo *info =
      static_cast<flagcxOneSideHandleInfo *>(calloc(1, sizeof(*info)));
  ASSERT_NE(info, nullptr);
  info->localMrHandle = reinterpret_cast<void *>(0x1000);
  info->localRecvComm = reinterpret_cast<void *>(0x2000);
  info->ownsLocalMr = 1;
  info->ownsConnections = 1;
  info->nContexts = 1;
  info->nRanks = 1;
  info->contextSendComms = static_cast<void ***>(calloc(1, sizeof(void **)));
  info->contextRecvComms = static_cast<void ***>(calloc(1, sizeof(void **)));
  ASSERT_NE(info->contextSendComms, nullptr);
  ASSERT_NE(info->contextRecvComms, nullptr);
  info->contextSendComms[0] = static_cast<void **>(calloc(1, sizeof(void *)));
  info->contextRecvComms[0] = static_cast<void **>(calloc(1, sizeof(void *)));
  ASSERT_NE(info->contextSendComms[0], nullptr);
  ASSERT_NE(info->contextRecvComms[0], nullptr);
  info->contextSendComms[0][0] = reinterpret_cast<void *>(0x3000);
  info->contextRecvComms[0][0] = info->localRecvComm;
  info->fullSendComms = info->contextSendComms[0];
  info->fullRecvComms = info->contextRecvComms[0];

  flagcxHeteroComm heteroComm = {};
  heteroComm.netAdaptor = &net;
  heteroComm.oneSideHandles = static_cast<flagcxOneSideHandleInfo **>(
      calloc(1, sizeof(flagcxOneSideHandleInfo *)));
  ASSERT_NE(heteroComm.oneSideHandles, nullptr);
  heteroComm.oneSideHandles[0] = info;
  heteroComm.oneSideHandleCount = 1;
  heteroComm.oneSideHandleCapacity = 1;

  nextDeregisterResult = flagcxRemoteError;
  EXPECT_EQ(flagcxOneSideDeregister(&heteroComm), flagcxRemoteError);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(closeSendCalls, 0);
  EXPECT_EQ(closeRecvCalls, 0);
  EXPECT_EQ(heteroComm.oneSideHandleCount, 1);
  EXPECT_EQ(heteroComm.oneSideHandles[0], info);
  EXPECT_EQ(info->localMrHandle, reinterpret_cast<void *>(0x1000));
  EXPECT_NE(info->contextSendComms, nullptr);

  EXPECT_EQ(flagcxOneSideDeregister(&heteroComm), flagcxSuccess);
  EXPECT_EQ(deregMrCalls, 2);
  EXPECT_EQ(closeSendCalls, 1);
  EXPECT_EQ(closeRecvCalls, 1);
  EXPECT_EQ(heteroComm.oneSideHandles, nullptr);
  EXPECT_EQ(heteroComm.oneSideHandleCount, 0);
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
       IpcSlotReleaseClosesRawMappingBaseImmediately) {
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
  entry->inUse = true;

  ASSERT_EQ(releaseIpcTableSlot(&comm, 0), flagcxSuccess);
  EXPECT_EQ(entry->hostPeerPtrs, nullptr);
  EXPECT_EQ(entry->hostPeerBasePtrs, nullptr);
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
