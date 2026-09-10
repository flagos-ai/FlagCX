/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "global_comm.h"
#include "onesided.h"

namespace {

flagcxHeteroComm *signalOwnerHeteroComm = nullptr;
void *freedSignalBuffer = nullptr;
int signalBufferFreeCount = 0;
bool signalStateClearedBeforeFree = false;

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

class RmaSignalRegistrationOwnershipTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (strcmp(devApiBackend->name, "default") != 0)
      GTEST_SKIP() << "requires the default Device API backend";

    savedDeviceAdaptor_ = deviceAdaptor;
    testDeviceAdaptor_ = *deviceAdaptor;
    testDeviceAdaptor_.gdrMemFree = recordSignalGdrFree;
    deviceAdaptor = &testDeviceAdaptor_;
    signalOwnerHeteroComm = nullptr;
    freedSignalBuffer = nullptr;
    signalBufferFreeCount = 0;
    signalStateClearedBeforeFree = false;
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
