/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "flagcx_device_adaptor.h"
#include "p2p_pointer.h"

#include <cstdlib>
#include <cstring>

extern struct flagcxDeviceAdaptor *deviceAdaptor;

namespace {

flagcxResult_t queryResult = flagcxNotSupported;
int queryType = FLAGCX_PTR_HOST;
flagcxResult_t ipcGetResult = flagcxSuccess;
int queryCalls = 0;
int ipcCreateCalls = 0;
int ipcGetCalls = 0;
int ipcFreeCalls = 0;

flagcxResult_t mockPointerQuery(const void *, int *ptrType) {
  ++queryCalls;
  if (queryResult == flagcxSuccess)
    *ptrType = queryType;
  return queryResult;
}

flagcxResult_t mockIpcCreate(flagcxIpcMemHandle_t *handle, size_t *size) {
  ++ipcCreateCalls;
  *handle = reinterpret_cast<flagcxIpcMemHandle_t>(std::malloc(16));
  *size = 16;
  return *handle == nullptr ? flagcxSystemError : flagcxSuccess;
}

flagcxResult_t mockIpcGet(flagcxIpcMemHandle_t handle, void *) {
  ++ipcGetCalls;
  if (ipcGetResult == flagcxSuccess)
    std::memset(handle, 0x5a, 16);
  return ipcGetResult;
}

flagcxResult_t mockIpcFree(flagcxIpcMemHandle_t handle) {
  ++ipcFreeCalls;
  std::free(handle);
  return flagcxSuccess;
}

flagcxResult_t mockGetLastError() { return flagcxSuccess; }

class PointerClassificationTest : public ::testing::Test {
protected:
  void SetUp() override {
    savedAdaptor_ = deviceAdaptor;
    adaptor_ = {};
    std::strncpy(adaptor_.name, "MOCK", sizeof(adaptor_.name) - 1);
    adaptor_.getPointerType = mockPointerQuery;
    adaptor_.ipcMemHandleCreate = mockIpcCreate;
    adaptor_.ipcMemHandleGet = mockIpcGet;
    adaptor_.ipcMemHandleFree = mockIpcFree;
    adaptor_.getLastError = mockGetLastError;
    deviceAdaptor = &adaptor_;
    queryResult = flagcxNotSupported;
    queryType = FLAGCX_PTR_HOST;
    ipcGetResult = flagcxSuccess;
    queryCalls = 0;
    ipcCreateCalls = 0;
    ipcGetCalls = 0;
    ipcFreeCalls = 0;
  }

  void TearDown() override { deviceAdaptor = savedAdaptor_; }

  flagcxDeviceAdaptor_latest adaptor_ = {};
  flagcxDeviceAdaptor_latest *savedAdaptor_ = nullptr;
};

TEST_F(PointerClassificationTest, LatestAdaptorDoesNotInferFromIpcExport) {
  int ptrType = FLAGCX_PTR_CUDA;
  char ipcHandle[64];
  std::memset(ipcHandle, 0xff, sizeof(ipcHandle));
  uint32_t ipcHandleSize = 17;

  EXPECT_EQ(flagcxP2pDetectPointerType(reinterpret_cast<void *>(0x1000),
                                       &ptrType, ipcHandle, &ipcHandleSize),
            flagcxNotSupported);
  EXPECT_EQ(queryCalls, 1);
  EXPECT_EQ(ipcCreateCalls, 0);
  EXPECT_EQ(ipcGetCalls, 0);
  EXPECT_EQ(ipcHandleSize, 0u);
  for (char byte : ipcHandle)
    EXPECT_EQ(byte, 0);
}

TEST_F(PointerClassificationTest, AuthoritativeHostSkipsIpcExport) {
  queryResult = flagcxSuccess;
  queryType = FLAGCX_PTR_HOST;
  int ptrType = -1;

  ASSERT_EQ(flagcxP2pDetectPointerType(reinterpret_cast<void *>(0x1000),
                                       &ptrType, nullptr, nullptr),
            flagcxSuccess);
  EXPECT_EQ(ptrType, FLAGCX_PTR_HOST);
  EXPECT_EQ(ipcCreateCalls, 0);
}

TEST_F(PointerClassificationTest, GpuTypeSurvivesOptionalIpcExportFailure) {
  queryResult = flagcxSuccess;
  queryType = FLAGCX_PTR_CUDA;
  ipcGetResult = flagcxUnhandledDeviceError;
  int ptrType = -1;
  uint32_t ipcHandleSize = 17;

  ASSERT_EQ(flagcxP2pDetectPointerType(reinterpret_cast<void *>(0x1000),
                                       &ptrType, nullptr, &ipcHandleSize),
            flagcxSuccess);
  EXPECT_EQ(ptrType, FLAGCX_PTR_CUDA);
  EXPECT_EQ(ipcCreateCalls, 1);
  EXPECT_EQ(ipcGetCalls, 1);
  EXPECT_EQ(ipcFreeCalls, 1);
  EXPECT_EQ(ipcHandleSize, 0u);
}

TEST_F(PointerClassificationTest, RejectsInvalidAuthoritativeType) {
  queryResult = flagcxSuccess;
  queryType = FLAGCX_PTR_DMABUF;
  int ptrType = -1;

  EXPECT_EQ(flagcxP2pDetectPointerType(reinterpret_cast<void *>(0x1000),
                                       &ptrType, nullptr, nullptr),
            flagcxInternalError);
  EXPECT_EQ(ipcCreateCalls, 0);
}

TEST_F(PointerClassificationTest, LegacyV1RetainsWarnedIpcInference) {
  adaptor_.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1;
  adaptor_.getPointerType = nullptr;
  int ptrType = -1;
  char ipcHandle[64] = {};
  uint32_t ipcHandleSize = 0;

  ASSERT_EQ(flagcxP2pDetectPointerType(reinterpret_cast<void *>(0x1000),
                                       &ptrType, ipcHandle, &ipcHandleSize),
            flagcxSuccess);
  EXPECT_EQ(ptrType, FLAGCX_PTR_CUDA);
  EXPECT_EQ(ipcHandleSize, 16u);
  EXPECT_EQ(ipcHandle[0], 0x5a);
  EXPECT_EQ(ipcFreeCalls, 1);
}

TEST_F(PointerClassificationTest, TransitionalBuiltinRetainsIpcInference) {
  adaptor_.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_IPC_POINTER_INFERENCE;
  int ptrType = -1;

  ASSERT_EQ(flagcxP2pDetectPointerType(reinterpret_cast<void *>(0x1000),
                                       &ptrType, nullptr, nullptr),
            flagcxSuccess);
  EXPECT_EQ(queryCalls, 1);
  EXPECT_EQ(ptrType, FLAGCX_PTR_CUDA);
  EXPECT_EQ(ipcGetCalls, 1);
  EXPECT_EQ(ipcFreeCalls, 1);
}

TEST_F(PointerClassificationTest, LegacyIpcRejectionFallsBackToHost) {
  adaptor_.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1;
  adaptor_.getPointerType = nullptr;
  ipcGetResult = flagcxUnhandledDeviceError;
  int ptrType = -1;

  ASSERT_EQ(flagcxP2pDetectPointerType(reinterpret_cast<void *>(0x1000),
                                       &ptrType, nullptr, nullptr),
            flagcxSuccess);
  EXPECT_EQ(ptrType, FLAGCX_PTR_HOST);
  EXPECT_EQ(ipcFreeCalls, 1);
}

} // namespace
