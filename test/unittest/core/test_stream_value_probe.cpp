/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "launch_kernel.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct FakeDeviceState {
  int device = 0;
  flagcxResult_t getDeviceResult = flagcxSuccess;
  int allocations = 0;
  int frees = 0;
  int writes = 0;
  int waits = 0;
  int synchronizations = 0;
  flagcxResult_t writeResult = flagcxSuccess;
  flagcxResult_t waitResult = flagcxSuccess;
};

FakeDeviceState *fakeState = nullptr;
int fakeStream;

flagcxResult_t fakeGetDevice(int *device) {
  if (fakeState->getDeviceResult == flagcxSuccess)
    *device = fakeState->device;
  return fakeState->getDeviceResult;
}

flagcxResult_t fakeDeviceMalloc(void **ptr, size_t size, flagcxMemType_t,
                                flagcxStream_t) {
  *ptr = std::malloc(size);
  if (*ptr == nullptr)
    return flagcxSystemError;
  ++fakeState->allocations;
  return flagcxSuccess;
}

flagcxResult_t fakeDeviceFree(void *ptr, flagcxMemType_t, flagcxStream_t) {
  ++fakeState->frees;
  std::free(ptr);
  return flagcxSuccess;
}

flagcxResult_t fakeHostGetDevicePointer(void **devicePtr, void *hostPtr) {
  *devicePtr = hostPtr;
  return flagcxSuccess;
}

flagcxResult_t fakeStreamCreate(flagcxStream_t *stream) {
  *stream = reinterpret_cast<flagcxStream_t>(&fakeStream);
  return flagcxSuccess;
}

flagcxResult_t fakeStreamDestroy(flagcxStream_t) { return flagcxSuccess; }

flagcxResult_t fakeStreamSynchronize(flagcxStream_t) {
  ++fakeState->synchronizations;
  return flagcxSuccess;
}

flagcxResult_t fakeStreamWriteValue64(flagcxStream_t, void *ptr, uint64_t value,
                                      int) {
  ++fakeState->writes;
  if (fakeState->writeResult == flagcxSuccess)
    *static_cast<uint64_t *>(ptr) = value;
  return fakeState->writeResult;
}

flagcxResult_t fakeStreamWaitValue64(flagcxStream_t, void *ptr, uint64_t value,
                                     int) {
  ++fakeState->waits;
  if (fakeState->waitResult != flagcxSuccess)
    return fakeState->waitResult;
  return *static_cast<uint64_t *>(ptr) >= value ? flagcxSuccess
                                                : flagcxInternalError;
}

class StreamValueProbeTest : public ::testing::Test {
protected:
  flagcxDeviceAdaptor *savedAdaptor = nullptr;
  flagcxDeviceAdaptor fakeAdaptor{};
  FakeDeviceState state;

  void SetUp() override {
    savedAdaptor = deviceAdaptor;
    ASSERT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
    flagcxClearStreamValueProbeCache();
    std::strncpy(fakeAdaptor.name, "probe-test", sizeof(fakeAdaptor.name) - 1);
    fakeAdaptor.deviceMalloc = fakeDeviceMalloc;
    fakeAdaptor.deviceFree = fakeDeviceFree;
    fakeAdaptor.deviceSynchronize = []() { return flagcxSuccess; };
    fakeAdaptor.getDevice = fakeGetDevice;
    fakeAdaptor.hostGetDevicePointer = fakeHostGetDevicePointer;
    fakeAdaptor.streamCreate = fakeStreamCreate;
    fakeAdaptor.streamDestroy = fakeStreamDestroy;
    fakeAdaptor.streamSynchronize = fakeStreamSynchronize;
    fakeAdaptor.streamWriteValue64 = fakeStreamWriteValue64;
    fakeAdaptor.streamWaitValue64 = fakeStreamWaitValue64;
    fakeState = &state;
    deviceAdaptor = &fakeAdaptor;
  }

  void TearDown() override {
    EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
    flagcxClearStreamValueProbeCache();
    deviceAdaptor = savedAdaptor;
    fakeState = nullptr;
  }
};

TEST_F(StreamValueProbeTest, UnsupportedWaitFallsBackAfterDrainingWrite) {
  state.waitResult = flagcxNotSupported;
  bool supported = true;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_FALSE(supported);
  EXPECT_EQ(state.writes, 1);
  EXPECT_EQ(state.waits, 1);
  EXPECT_EQ(state.synchronizations, 1);
  EXPECT_EQ(state.allocations, state.frees);

  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_EQ(state.writes, 1); // unsupported result is cached
}

TEST_F(StreamValueProbeTest, UnsupportedWriteFallsBackWithoutWaiting) {
  state.writeResult = flagcxNotSupported;
  bool supported = true;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_FALSE(supported);
  EXPECT_EQ(state.writes, 1);
  EXPECT_EQ(state.waits, 0);
  EXPECT_EQ(state.synchronizations, 0);
  EXPECT_EQ(state.allocations, state.frees);
}

TEST_F(StreamValueProbeTest, UnsupportedDeviceQueryIsReportedWithoutProbing) {
  state.getDeviceResult = flagcxNotSupported;
  bool supported = true;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxNotSupported);
  EXPECT_FALSE(supported);
  EXPECT_EQ(state.allocations, 0);
  EXPECT_EQ(state.writes, 0);

  state.getDeviceResult = flagcxSuccess;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_TRUE(supported);
  EXPECT_EQ(state.writes, 1);
}

TEST_F(StreamValueProbeTest, SupportedResultIsCachedPerDevice) {
  bool supported = false;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_TRUE(supported);
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_EQ(state.writes, 1);

  state.device = 1;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_TRUE(supported);
  EXPECT_EQ(state.writes, 2);
  EXPECT_EQ(state.allocations, state.frees);

  flagcxClearStreamValueProbeCache();
  state.device = 0;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_EQ(state.writes, 3);
}

TEST_F(StreamValueProbeTest, OtherErrorsAreReportedAndNotCached) {
  state.writeResult = flagcxInternalError;
  bool supported = true;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxInternalError);
  EXPECT_FALSE(supported);
  EXPECT_EQ(state.synchronizations, 0);

  state.writeResult = flagcxSuccess;
  EXPECT_EQ(flagcxStreamValueSupported(&supported), flagcxSuccess);
  EXPECT_TRUE(supported);
  EXPECT_EQ(state.writes, 2);
  EXPECT_EQ(state.allocations, state.frees);
}

TEST_F(StreamValueProbeTest, PoolClearRequiresAllSlotsToBeReleased) {
  uint64_t *signals = nullptr;
  void *deviceSignals = nullptr;
  ASSERT_EQ(streamValueBufferPool.acquire(&signals, &deviceSignals),
            flagcxSuccess);
  ASSERT_NE(signals, nullptr);
  EXPECT_EQ(deviceSignals, signals);
  EXPECT_EQ(streamValueBufferPool.clear(), flagcxInvalidUsage);
  EXPECT_EQ(state.frees, 0);

  streamValueBufferPool.release(signals);
  EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
  EXPECT_EQ(state.allocations, state.frees);
}

TEST_F(StreamValueProbeTest, MissingPluginDoesNotClearAnActivePool) {
  uint64_t *signals = nullptr;
  void *deviceSignals = nullptr;
  ASSERT_EQ(streamValueBufferPool.acquire(&signals, &deviceSignals),
            flagcxSuccess);

  const char *previous = std::getenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN");
  const bool hadPrevious = previous != nullptr;
  const std::string previousValue = hadPrevious ? previous : "";
  setenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN", "/nonexistent/flagcx-device.so", 1);
  EXPECT_EQ(flagcxDeviceAdaptorPluginLoad(), flagcxSuccess);
  if (hadPrevious)
    setenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN", previousValue.c_str(), 1);
  else
    unsetenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN");

  EXPECT_EQ(state.frees, 0);
  streamValueBufferPool.release(signals);
  EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
  EXPECT_EQ(state.allocations, state.frees);
}

} // namespace
