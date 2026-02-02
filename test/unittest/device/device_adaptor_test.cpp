/*************************************************************************
 * Copyright (c) 2025. All Rights Reserved.
 * Single device adaptor test - no multi-GPU or MPI required
 ************************************************************************/

#include <cstring>
#include <gtest/gtest.h>
#include <iostream>

#include "adaptor.h"
#include "flagcx.h"

class DeviceAdaptorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize flagcx handle
    flagcxHandleInit(&handler);
    devHandle = handler->devHandle;

    // Get device count and set device 0
    int numDevices = 0;
    devHandle->getDeviceCount(&numDevices);
    ASSERT_GT(numDevices, 0) << "No devices found!";

    std::cout << "Found " << numDevices << " device(s)" << std::endl;

    devHandle->setDevice(0);

    // Create stream
    devHandle->streamCreate(&stream);
  }

  void TearDown() override {
    if (stream) {
      devHandle->streamDestroy(stream);
    }
    flagcxHandleFree(handler);
  }

  flagcxHandlerGroup_t handler = nullptr;
  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxStream_t stream = nullptr;

  static constexpr size_t TEST_SIZE = 1024 * sizeof(float); // 1K floats
  static constexpr size_t TEST_COUNT = 1024;
};

// Test: Get device count and properties
TEST_F(DeviceAdaptorTest, GetDeviceInfo) {
  int numDevices = 0;
  auto result = devHandle->getDeviceCount(&numDevices);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_GT(numDevices, 0);
  std::cout << "Device count: " << numDevices << std::endl;

  int currentDevice = -1;
  result = devHandle->getDevice(&currentDevice);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_EQ(currentDevice, 0);
  std::cout << "Current device: " << currentDevice << std::endl;

  // Get vendor name
  char vendor[128] = {0};
  result = devHandle->getVendor(vendor);
  EXPECT_EQ(result, flagcxSuccess);
  std::cout << "Vendor: " << vendor << std::endl;
}

// Test: Device memory allocation and free
TEST_F(DeviceAdaptorTest, DeviceMemoryAlloc) {
  void *devPtr = nullptr;

  // Allocate device memory
  auto result =
      devHandle->deviceMalloc(&devPtr, TEST_SIZE, flagcxMemDevice, stream);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_NE(devPtr, nullptr);

  // Free device memory
  result = devHandle->deviceFree(devPtr, flagcxMemDevice, stream);
  EXPECT_EQ(result, flagcxSuccess);
}

// Test: Host memory allocation and free
TEST_F(DeviceAdaptorTest, HostMemoryAlloc) {
  void *hostPtr = nullptr;

  // Allocate host memory
  auto result =
      devHandle->deviceMalloc(&hostPtr, TEST_SIZE, flagcxMemHost, stream);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_NE(hostPtr, nullptr);

  // Free host memory
  result = devHandle->deviceFree(hostPtr, flagcxMemHost, stream);
  EXPECT_EQ(result, flagcxSuccess);
}

// Test: Memory copy Host -> Device -> Host
TEST_F(DeviceAdaptorTest, MemoryCopy) {
  void *hostSrc = nullptr;
  void *hostDst = nullptr;
  void *devPtr = nullptr;

  // Allocate memory
  ASSERT_EQ(devHandle->deviceMalloc(&hostSrc, TEST_SIZE, flagcxMemHost, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMalloc(&hostDst, TEST_SIZE, flagcxMemHost, stream),
            flagcxSuccess);
  ASSERT_EQ(
      devHandle->deviceMalloc(&devPtr, TEST_SIZE, flagcxMemDevice, stream),
      flagcxSuccess);

  // Initialize source data
  float *srcData = static_cast<float *>(hostSrc);
  for (size_t i = 0; i < TEST_COUNT; i++) {
    srcData[i] = static_cast<float>(i);
  }

  // Clear destination
  memset(hostDst, 0, TEST_SIZE);

  // Copy: Host -> Device
  auto result = devHandle->deviceMemcpy(devPtr, hostSrc, TEST_SIZE,
                                        flagcxMemcpyHostToDevice, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Copy: Device -> Host
  result = devHandle->deviceMemcpy(hostDst, devPtr, TEST_SIZE,
                                   flagcxMemcpyDeviceToHost, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Synchronize
  result = devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Verify data
  float *dstData = static_cast<float *>(hostDst);
  for (size_t i = 0; i < TEST_COUNT; i++) {
    EXPECT_FLOAT_EQ(dstData[i], static_cast<float>(i))
        << "Mismatch at index " << i;
  }

  // Cleanup
  devHandle->deviceFree(hostSrc, flagcxMemHost, stream);
  devHandle->deviceFree(hostDst, flagcxMemHost, stream);
  devHandle->deviceFree(devPtr, flagcxMemDevice, stream);
}

// Test: Memory set
TEST_F(DeviceAdaptorTest, MemorySet) {
  void *hostPtr = nullptr;
  void *devPtr = nullptr;

  // Allocate memory
  ASSERT_EQ(devHandle->deviceMalloc(&hostPtr, TEST_SIZE, flagcxMemHost, stream),
            flagcxSuccess);
  ASSERT_EQ(
      devHandle->deviceMalloc(&devPtr, TEST_SIZE, flagcxMemDevice, stream),
      flagcxSuccess);

  // Set device memory to 0
  auto result =
      devHandle->deviceMemset(devPtr, 0, TEST_SIZE, flagcxMemDevice, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Copy back and verify
  result = devHandle->deviceMemcpy(hostPtr, devPtr, TEST_SIZE,
                                   flagcxMemcpyDeviceToHost, stream);
  EXPECT_EQ(result, flagcxSuccess);

  result = devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Verify all zeros
  unsigned char *data = static_cast<unsigned char *>(hostPtr);
  for (size_t i = 0; i < TEST_SIZE; i++) {
    EXPECT_EQ(data[i], 0) << "Non-zero at index " << i;
  }

  // Cleanup
  devHandle->deviceFree(hostPtr, flagcxMemHost, stream);
  devHandle->deviceFree(devPtr, flagcxMemDevice, stream);
}

// Test: Stream operations
TEST_F(DeviceAdaptorTest, StreamOperations) {
  flagcxStream_t newStream = nullptr;

  // Create stream
  auto result = devHandle->streamCreate(&newStream);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_NE(newStream, nullptr);

  // Query stream - result depends on implementation
  // Some backends may return flagcxSuccess, flagcxInProgress, or other values
  result = devHandle->streamQuery(newStream);
  std::cout << "streamQuery result: " << result << std::endl;

  // Synchronize stream (this should always work)
  result = devHandle->streamSynchronize(newStream);
  EXPECT_EQ(result, flagcxSuccess);

  // Destroy stream
  result = devHandle->streamDestroy(newStream);
  EXPECT_EQ(result, flagcxSuccess);
}

// Test: Event operations
TEST_F(DeviceAdaptorTest, EventOperations) {
  flagcxEvent_t event = nullptr;

  // Create event
  auto result = devHandle->eventCreate(&event, flagcxEventDefault);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_NE(event, nullptr);

  // Record event
  result = devHandle->eventRecord(event, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Synchronize event
  result = devHandle->eventSynchronize(event);
  EXPECT_EQ(result, flagcxSuccess);

  // Query event (should be completed after sync)
  result = devHandle->eventQuery(event);
  EXPECT_EQ(result, flagcxSuccess);

  // Destroy event
  result = devHandle->eventDestroy(event);
  EXPECT_EQ(result, flagcxSuccess);
}

// Test: Stream wait event
TEST_F(DeviceAdaptorTest, StreamWaitEvent) {
  flagcxStream_t stream2 = nullptr;
  flagcxEvent_t event = nullptr;

  // Create second stream and event
  ASSERT_EQ(devHandle->streamCreate(&stream2), flagcxSuccess);
  ASSERT_EQ(devHandle->eventCreate(&event, flagcxEventDefault), flagcxSuccess);

  // Record event on first stream
  auto result = devHandle->eventRecord(event, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Make second stream wait for event
  result = devHandle->streamWaitEvent(stream2, event);
  EXPECT_EQ(result, flagcxSuccess);

  // Synchronize both streams
  result = devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, flagcxSuccess);
  result = devHandle->streamSynchronize(stream2);
  EXPECT_EQ(result, flagcxSuccess);

  // Cleanup
  devHandle->eventDestroy(event);
  devHandle->streamDestroy(stream2);
}

// Test: Device synchronize
TEST_F(DeviceAdaptorTest, DeviceSynchronize) {
  auto result = devHandle->deviceSynchronize();
  EXPECT_EQ(result, flagcxSuccess);
}

// Test: Set device
TEST_F(DeviceAdaptorTest, SetDevice) {
  int numDevices = 0;
  devHandle->getDeviceCount(&numDevices);

  // Set to device 0 (always exists)
  auto result = devHandle->setDevice(0);
  EXPECT_EQ(result, flagcxSuccess);

  int currentDevice = -1;
  devHandle->getDevice(&currentDevice);
  EXPECT_EQ(currentDevice, 0);

  // If multiple devices, test switching
  if (numDevices > 1) {
    result = devHandle->setDevice(1);
    EXPECT_EQ(result, flagcxSuccess);

    devHandle->getDevice(&currentDevice);
    EXPECT_EQ(currentDevice, 1);

    // Switch back to device 0
    devHandle->setDevice(0);
  }
}

// Test: Large memory allocation
TEST_F(DeviceAdaptorTest, LargeMemoryAlloc) {
  void *devPtr = nullptr;
  const size_t largeSize = 100 * 1024 * 1024; // 100 MB

  // Allocate large device memory
  auto result =
      devHandle->deviceMalloc(&devPtr, largeSize, flagcxMemDevice, stream);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_NE(devPtr, nullptr);

  if (devPtr) {
    // Set memory to verify it's accessible
    result =
        devHandle->deviceMemset(devPtr, 0, largeSize, flagcxMemDevice, stream);
    EXPECT_EQ(result, flagcxSuccess);

    result = devHandle->streamSynchronize(stream);
    EXPECT_EQ(result, flagcxSuccess);

    // Free memory
    devHandle->deviceFree(devPtr, flagcxMemDevice, stream);
  }
}

// Test: Event timing (record and synchronize)
TEST_F(DeviceAdaptorTest, EventTiming) {
  flagcxEvent_t startEvent = nullptr;
  flagcxEvent_t endEvent = nullptr;

  ASSERT_EQ(devHandle->eventCreate(&startEvent, flagcxEventDefault),
            flagcxSuccess);
  ASSERT_EQ(devHandle->eventCreate(&endEvent, flagcxEventDefault),
            flagcxSuccess);

  // Record start event
  auto result = devHandle->eventRecord(startEvent, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Do some work (memory allocation and set)
  void *devPtr = nullptr;
  devHandle->deviceMalloc(&devPtr, TEST_SIZE, flagcxMemDevice, stream);
  devHandle->deviceMemset(devPtr, 0, TEST_SIZE, flagcxMemDevice, stream);

  // Record end event
  result = devHandle->eventRecord(endEvent, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Synchronize both events
  result = devHandle->eventSynchronize(startEvent);
  EXPECT_EQ(result, flagcxSuccess);
  result = devHandle->eventSynchronize(endEvent);
  EXPECT_EQ(result, flagcxSuccess);

  // Query events (should be completed)
  result = devHandle->eventQuery(startEvent);
  EXPECT_EQ(result, flagcxSuccess);
  result = devHandle->eventQuery(endEvent);
  EXPECT_EQ(result, flagcxSuccess);

  // Cleanup
  devHandle->deviceFree(devPtr, flagcxMemDevice, stream);
  devHandle->eventDestroy(startEvent);
  devHandle->eventDestroy(endEvent);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
