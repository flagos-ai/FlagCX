/*************************************************************************
 * Copyright (c) 2025. All Rights Reserved.
 * Single device adaptor test - no multi-GPU or MPI required
 ************************************************************************/

#include <cstring>
#include <gtest/gtest.h>
#include <iostream>

#include "adaptor.h"
#include "flagcx.h"
#include "topo.h"

class DeviceAdaptorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize flagcx handle
    flagcxDeviceHandleInit(&devHandle);

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
    flagcxDeviceHandleFree(devHandle);
  }

  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxComm_t comm = nullptr;
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

// Test: Device-to-Device memory copy
TEST_F(DeviceAdaptorTest, DeviceToDeviceMemcpy) {
  void *hostSrc = nullptr;
  void *hostDst = nullptr;
  void *devSrc = nullptr;
  void *devDst = nullptr;

  ASSERT_EQ(devHandle->deviceMalloc(&hostSrc, TEST_SIZE, flagcxMemHost, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMalloc(&hostDst, TEST_SIZE, flagcxMemHost, stream),
            flagcxSuccess);
  ASSERT_EQ(
      devHandle->deviceMalloc(&devSrc, TEST_SIZE, flagcxMemDevice, stream),
      flagcxSuccess);
  ASSERT_EQ(
      devHandle->deviceMalloc(&devDst, TEST_SIZE, flagcxMemDevice, stream),
      flagcxSuccess);

  // Fill source with known data
  float *srcData = static_cast<float *>(hostSrc);
  for (size_t i = 0; i < TEST_COUNT; i++) {
    srcData[i] = static_cast<float>(i * 2 + 1);
  }
  memset(hostDst, 0, TEST_SIZE);

  // H2D: host -> devSrc
  ASSERT_EQ(devHandle->deviceMemcpy(devSrc, hostSrc, TEST_SIZE,
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);

  // D2D: devSrc -> devDst
  auto result = devHandle->deviceMemcpy(devDst, devSrc, TEST_SIZE,
                                        flagcxMemcpyDeviceToDevice, stream);
  EXPECT_EQ(result, flagcxSuccess);

  // D2H: devDst -> hostDst
  ASSERT_EQ(devHandle->deviceMemcpy(hostDst, devDst, TEST_SIZE,
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  devHandle->streamSynchronize(stream);

  // Verify data
  float *dstData = static_cast<float *>(hostDst);
  for (size_t i = 0; i < TEST_COUNT; i++) {
    EXPECT_FLOAT_EQ(dstData[i], static_cast<float>(i * 2 + 1))
        << "Mismatch at index " << i;
  }

  devHandle->deviceFree(hostSrc, flagcxMemHost, stream);
  devHandle->deviceFree(hostDst, flagcxMemHost, stream);
  devHandle->deviceFree(devSrc, flagcxMemDevice, stream);
  devHandle->deviceFree(devDst, flagcxMemDevice, stream);
}

// Test: Stream copy and free lifecycle
TEST_F(DeviceAdaptorTest, StreamCopyAndFree) {
  if (!devHandle->streamCopy || !devHandle->streamFree) {
    GTEST_SKIP() << "streamCopy/streamFree not implemented for this backend";
  }

  // Get the raw stream pointer from our existing stream
  // streamCopy wraps a raw vendor stream into a flagcxStream_t
  // We create a new vendor stream, copy it, then free the copy
  flagcxStream_t newStream = nullptr;

  // Create a fresh stream to get a raw pointer
  flagcxStream_t tempStream = nullptr;
  ASSERT_EQ(devHandle->streamCreate(&tempStream), flagcxSuccess);

  // Copy the stream
  auto result = devHandle->streamCopy(&newStream, tempStream);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_NE(newStream, nullptr);

  // Free the copy (this should free the wrapper, not the underlying stream)
  if (newStream) {
    result = devHandle->streamFree(newStream);
    EXPECT_EQ(result, flagcxSuccess);
  }

  // Clean up the original
  devHandle->streamDestroy(tempStream);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
// Test: obtain the device-visible alias of mapped host memory.
TEST_F(DeviceAdaptorTest, HostGetDevicePointer) {
  ASSERT_NE(devHandle->hostGetDevicePointer, nullptr);
  void *hostPtr = nullptr;
  ASSERT_EQ(devHandle->deviceMalloc(&hostPtr, TEST_SIZE, flagcxMemHost, nullptr),
            flagcxSuccess);
  ASSERT_NE(hostPtr, nullptr);
  void *devicePtr = nullptr;
  EXPECT_EQ(devHandle->hostGetDevicePointer(&devicePtr, hostPtr), flagcxSuccess);
  EXPECT_NE(devicePtr, nullptr);
  EXPECT_EQ(devHandle->hostGetDevicePointer(nullptr, hostPtr),
            flagcxInvalidArgument);
  EXPECT_EQ(devHandle->hostGetDevicePointer(&devicePtr, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(devHandle->deviceFree(hostPtr, flagcxMemHost, nullptr),
            flagcxSuccess);
}

// Test: Host memory register / get device pointer / unregister
// flagcxDeviceHandle does not expose hostRegister/hostUnregister, so we
// access them through the internal deviceAdaptor pointer (from adaptor.h).
// Portable across all backends: skips if not implemented (NULL or NotSupported).
TEST_F(DeviceAdaptorTest, HostRegisterUnregister) {
  if (!deviceAdaptor->hostRegister || !deviceAdaptor->hostUnregister) {
    GTEST_SKIP() << "hostRegister/hostUnregister not implemented";
  }

  const size_t sz = 4096;
  void *hostPtr = malloc(sz);
  ASSERT_NE(hostPtr, nullptr);

  // Register host memory as pinned (page-locked) memory.
  auto result = deviceAdaptor->hostRegister(hostPtr, sz);
  if (result == flagcxNotSupported) {
    free(hostPtr);
    GTEST_SKIP() << "hostRegister not supported on this backend";
  }
  EXPECT_EQ(result, flagcxSuccess);

  if (result == flagcxSuccess) {
    // Unregister
    auto r2 = deviceAdaptor->hostUnregister(hostPtr);
    EXPECT_EQ(r2, flagcxSuccess);
  }
  free(hostPtr);

  // Verify invalid arguments are rejected (only if backend has real impl)
  auto nullResult = deviceAdaptor->hostRegister(NULL, sz);
  if (nullResult != flagcxNotSupported) {
    EXPECT_EQ(nullResult, flagcxInvalidArgument);
  }
  auto zeroResult = deviceAdaptor->hostRegister(hostPtr, 0);
  if (zeroResult != flagcxNotSupported) {
    EXPECT_EQ(zeroResult, flagcxInvalidArgument);
  }
  auto unregNullResult = deviceAdaptor->hostUnregister(NULL);
  if (unregNullResult != flagcxNotSupported) {
    EXPECT_EQ(unregNullResult, flagcxInvalidArgument);
  }
}

// Test: getDevicePciBusId — string format and invalid-arg guards
TEST_F(DeviceAdaptorTest, GetDevicePciBusId) {
  int numDevices = 0;
  ASSERT_EQ(devHandle->getDeviceCount(&numDevices), flagcxSuccess);
  ASSERT_GT(numDevices, 0);

  for (int dev = 0; dev < numDevices; dev++) {
    char pciBusId[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE] = {};
    ASSERT_EQ(deviceAdaptor->getDevicePciBusId(pciBusId, sizeof(pciBusId), dev),
              flagcxSuccess)
        << "getDevicePciBusId failed for device " << dev;
    EXPECT_NE(pciBusId[0], '\0') << "empty PCI bus ID for device " << dev;

    // Verify format: domain:bus:device.function (4 hex fields)
    unsigned int domain = 0, bus = 0, pciDevice = 0, function = 0;
    EXPECT_EQ(sscanf(pciBusId, "%x:%x:%x.%x", &domain, &bus, &pciDevice, &function),
              4)
        << "unexpected PCI bus ID format: " << pciBusId;

    std::cout << "Device " << dev << ": pci=" << pciBusId << std::endl;
  }

  // Invalid-argument guards
  char buf[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE] = {};
  EXPECT_EQ(deviceAdaptor->getDevicePciBusId(nullptr, sizeof(buf), 0),
            flagcxInvalidArgument);

  // Out-of-range device index
  EXPECT_NE(deviceAdaptor->getDevicePciBusId(buf, sizeof(buf), -1), flagcxSuccess);
  EXPECT_NE(deviceAdaptor->getDevicePciBusId(buf, sizeof(buf), numDevices),
            flagcxSuccess);
}

// Test: getDeviceProperties — name and PCI fields, invalid-arg guards
TEST_F(DeviceAdaptorTest, GetDeviceProperties) {
  int numDevices = 0;
  ASSERT_EQ(devHandle->getDeviceCount(&numDevices), flagcxSuccess);
  ASSERT_GT(numDevices, 0);

  for (int dev = 0; dev < numDevices; dev++) {
    flagcxDevProps props = {};
    ASSERT_EQ(deviceAdaptor->getDeviceProperties(&props, dev), flagcxSuccess)
        << "getDeviceProperties failed for device " << dev;
    EXPECT_NE(props.name[0], '\0') << "empty device name for device " << dev;

    // PCI fields are parsed from cudaDeviceGetPCIBusId string
    EXPECT_GE(props.pciBusId,    0) << "pciBusId out of range for device " << dev;
    EXPECT_LE(props.pciBusId,  255) << "pciBusId out of range for device " << dev;
    EXPECT_GE(props.pciDeviceId, 0) << "pciDeviceId out of range for device " << dev;
    EXPECT_LE(props.pciDeviceId, 31) << "pciDeviceId out of range for device " << dev;
    EXPECT_GE(props.pciDomainId, 0) << "pciDomainId out of range for device " << dev;
    EXPECT_LE(props.pciDomainId, 65535) << "pciDomainId out of range for device " << dev;

    std::cout << "Device " << dev << ": name=" << props.name
              << ", pciBusId=" << props.pciBusId
              << ", pciDeviceId=" << props.pciDeviceId
              << ", pciDomainId=" << props.pciDomainId << std::endl;
  }

  // Invalid-argument guards
  EXPECT_EQ(deviceAdaptor->getDeviceProperties(nullptr, 0),
            flagcxInvalidArgument);

  // Out-of-range device index
  flagcxDevProps props = {};
  EXPECT_NE(deviceAdaptor->getDeviceProperties(&props, -1), flagcxSuccess);
  EXPECT_NE(deviceAdaptor->getDeviceProperties(&props, numDevices),
            flagcxSuccess);
}

// Test: getDeviceByPciBusId — reverse lookup and invalid-arg guards
TEST_F(DeviceAdaptorTest, GetDeviceByPciBusId) {
  int numDevices = 0;
  ASSERT_EQ(devHandle->getDeviceCount(&numDevices), flagcxSuccess);
  ASSERT_GT(numDevices, 0);

  for (int dev = 0; dev < numDevices; dev++) {
    // Get PCI bus ID string directly from runtime (independent of getDevicePciBusId)
    char pciBusId[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE] = {};
    ASSERT_EQ(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), dev), cudaSuccess)
        << "cudaDeviceGetPCIBusId failed for dev=" << dev;

    // Reverse lookup: string -> device index
    int result = -1;
    ASSERT_EQ(deviceAdaptor->getDeviceByPciBusId(&result, pciBusId), flagcxSuccess)
        << "getDeviceByPciBusId failed for pci=" << pciBusId;
    EXPECT_EQ(result, dev)
        << "round-trip mismatch: pci=" << pciBusId
        << " returned dev=" << result << " expected=" << dev;

    std::cout << "Device " << dev << ": pci=" << pciBusId
              << " -> dev=" << result << std::endl;
  }

  // Invalid-argument guards
  int dev = -1;
  char pciBusId[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE] = {};
  EXPECT_EQ(deviceAdaptor->getDeviceByPciBusId(nullptr, pciBusId),
            flagcxInvalidArgument);
  EXPECT_EQ(deviceAdaptor->getDeviceByPciBusId(&dev, nullptr),
            flagcxInvalidArgument);

  // Invalid PCI string
  EXPECT_NE(deviceAdaptor->getDeviceByPciBusId(&dev, "invalid"), flagcxSuccess);
}

// Test: eventElapsedTime reports the duration between two timing events.
TEST_F(DeviceAdaptorTest, EventElapsedTime) {
  ASSERT_NE(devHandle->eventElapsedTime, nullptr);
  flagcxEvent_t startEvent = nullptr;
  flagcxEvent_t endEvent = nullptr;
  ASSERT_EQ(devHandle->eventCreate(&startEvent, flagcxEventDefault), flagcxSuccess);
  ASSERT_EQ(devHandle->eventCreate(&endEvent, flagcxEventDefault), flagcxSuccess);
  ASSERT_EQ(devHandle->eventRecord(startEvent, stream), flagcxSuccess);
  ASSERT_EQ(devHandle->eventRecord(endEvent, stream), flagcxSuccess);
  ASSERT_EQ(devHandle->eventSynchronize(endEvent), flagcxSuccess);
  float elapsedMs = -1.0f;
  EXPECT_EQ(devHandle->eventElapsedTime(&elapsedMs, startEvent, endEvent),
            flagcxSuccess);
  EXPECT_GE(elapsedMs, 0.0f);
  EXPECT_EQ(devHandle->eventElapsedTime(nullptr, startEvent, endEvent),
            flagcxInvalidArgument);
  EXPECT_EQ(devHandle->eventElapsedTime(&elapsedMs, nullptr, endEvent),
            flagcxInvalidArgument);
  EXPECT_EQ(devHandle->eventElapsedTime(&elapsedMs, startEvent, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(devHandle->eventDestroy(startEvent), flagcxSuccess);
  EXPECT_EQ(devHandle->eventDestroy(endEvent), flagcxSuccess);
}
