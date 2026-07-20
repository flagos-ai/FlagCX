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

#ifdef USE_KUNLUNXIN_ADAPTOR
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include "kunlunxin_adaptor.h"
#endif

#ifdef USE_KUNLUNXIN_ADAPTOR
namespace {
// Scenario helper: register the mapped host VA with any available IB device,
// then deregister it. This is intentionally used only by the lifecycle test.
testing::AssertionResult verifyMrRegistration(void *ptr, size_t size) {
  int count = 0;
  ibv_device **devices = ibv_get_device_list(&count);
  if (devices == nullptr || count == 0) {
    if (devices != nullptr) ibv_free_device_list(devices);
    return testing::AssertionFailure() << "no IB device is available";
  }
  for (int i = 0; i < count; ++i) {
    ibv_context *context = ibv_open_device(devices[i]);
    if (context == nullptr) continue;
    ibv_pd *pd = ibv_alloc_pd(context);
    if (pd != nullptr) {
      const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                         IBV_ACCESS_REMOTE_READ;
      ibv_mr *mr = ibv_reg_mr(pd, ptr, size, access);
      if (mr != nullptr) {
        const uint32_t lkey = mr->lkey;
        const uint32_t rkey = mr->rkey;
        const std::string deviceName = ibv_get_device_name(devices[i]);
        const int deregResult = ibv_dereg_mr(mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(context);
        ibv_free_device_list(devices);
        if (deregResult != 0) {
          return testing::AssertionFailure()
                 << "ibv_dereg_mr failed on " << deviceName
                 << ", ret=" << deregResult;
        }
        std::cout << "[GDR lifecycle] ibv_reg_mr device="
                  << deviceName << " lkey=" << lkey << " rkey=" << rkey
                  << std::endl;
        return testing::AssertionSuccess();
      }
      ibv_dealloc_pd(pd);
    }
    ibv_close_device(context);
  }
  ibv_free_device_list(devices);
  return testing::AssertionFailure()
         << "ibv_reg_mr failed on every available IB device";
}
}  // namespace
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
/*
 * KunlunXin GDR test coverage
 *
 * Local validation:
 * - GdrMemAlloc, GdrMemFree and MemHandleInitDestroy test one interface at a
 *   time without using the paired interface as the test setup or oracle.
 * - GdrMemoryLifecycle covers allocation, mapping, CPU/XPU data visibility,
 *   MR registration/deregistration and cleanup.
 *
 * Required environment (export before XCCL/runtime initialization):
 * - BKCL_USE_PEERMEM_XDR=1: enable the BKCL/XCCL peermem mapping used by
 *   xccl_mmap. Without it, xccl_mmap returned no host-mapped address in the
 *   validated environment.
 * - CUDA_ENABLE_P2P_NO_UVA=1: enable the KunlunXin no-UVA peer-memory path.
 *   This is a runtime prerequisite and does not replace xccl_mmap.
 *
 * Two-node acceptance:
 * After the upper layer propagates the same non-null memHandle, reuse the
 * existing RMA/SendRecv MPI suite on two physical nodes. Expect the RDMA path
 * to be selected, the destination XPU payload to match, and gdrMemFree plus
 * memHandleDestroy to complete. A same-node run may select IPC/SHM/P2P.
 */

// Unit test: only gdrMemAlloc is the target interface.
// Cleanup deliberately uses xccl_munmap + xpu_free, not gdrMemFree.
TEST_F(DeviceAdaptorTest, GdrMemAlloc) {
  ASSERT_NE(deviceAdaptor->gdrMemAlloc, nullptr);

  KunlunXinGdrMemHandle handle{};
  void *rawHandle = &handle;
  void *sentinel = reinterpret_cast<void *>(0x1);
  void *out = sentinel;

  EXPECT_EQ(deviceAdaptor->gdrMemAlloc(nullptr, TEST_SIZE, rawHandle),
            flagcxInvalidArgument);
  EXPECT_EQ(deviceAdaptor->gdrMemAlloc(&out, 0, rawHandle),
            flagcxInvalidArgument);
  EXPECT_EQ(out, sentinel);
  EXPECT_EQ(deviceAdaptor->gdrMemAlloc(&out, TEST_SIZE, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(out, sentinel);

  out = nullptr;
  ASSERT_EQ(deviceAdaptor->gdrMemAlloc(&out, TEST_SIZE, rawHandle),
            flagcxSuccess)
      << "BKCL_USE_PEERMEM_XDR=1 and CUDA_ENABLE_P2P_NO_UVA=1 are required";
  ASSERT_NE(out, nullptr);

  EXPECT_EQ(handle.devPtr, out);
  EXPECT_NE(handle.hostMappedPtr, nullptr);
  EXPECT_EQ(handle.size, TEST_SIZE);
  EXPECT_TRUE(handle.mapped);

  void *second = nullptr;
  EXPECT_EQ(deviceAdaptor->gdrMemAlloc(&second, TEST_SIZE, rawHandle),
            flagcxInvalidArgument);
  EXPECT_EQ(second, nullptr);

  // The returned device pointer must be usable by the device copy path.
  uint64_t source = 0x1122334455667788ULL;
  uint64_t destination = 0;
  auto copyResult = devHandle->deviceMemcpy(
      out, &source, sizeof(source), flagcxMemcpyHostToDevice, stream);
  EXPECT_EQ(copyResult, flagcxSuccess);
  if (copyResult == flagcxSuccess) {
    EXPECT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);
    copyResult = devHandle->deviceMemcpy(
        &destination, out, sizeof(destination), flagcxMemcpyDeviceToHost,
        stream);
    EXPECT_EQ(copyResult, flagcxSuccess);
    if (copyResult == flagcxSuccess) {
      EXPECT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);
      EXPECT_EQ(destination, source);
    }
  }

  // Direct cleanup keeps this test independent from gdrMemFree.
  if (handle.hostMappedPtr == nullptr) {
    xpu_free(out);
    handle = {};
    FAIL() << "gdrMemAlloc returned success without a host mapping";
  }
  EXPECT_EQ(baidu::xpu::bkcl::xccl_munmap(handle.hostMappedPtr, handle.size),
            0);
  EXPECT_EQ(xpu_free(out), 0);
  handle = {};
}

// Unit test: only gdrMemFree is the target interface.
// Its input is constructed directly, without calling gdrMemAlloc.
TEST_F(DeviceAdaptorTest, GdrMemFree) {
  ASSERT_NE(deviceAdaptor->gdrMemFree, nullptr);
  EXPECT_EQ(deviceAdaptor->gdrMemFree(nullptr, nullptr), flagcxSuccess);

  void *devPtr = nullptr;
  ASSERT_EQ(xpu_malloc(&devPtr, TEST_SIZE), 0);
  ASSERT_NE(devPtr, nullptr);

  void *hostPtr = nullptr;
  const int mmapResult =
      baidu::xpu::bkcl::xccl_mmap(&hostPtr, devPtr, TEST_SIZE);
  if (mmapResult != 0 || hostPtr == nullptr) {
    xpu_free(devPtr);
    FAIL() << "xccl_mmap failed; check the peermem environment";
  }

  KunlunXinGdrMemHandle handle{devPtr, hostPtr, TEST_SIZE, true};
  void *wrongPtr = reinterpret_cast<void *>(0xdeadbeef);
  EXPECT_EQ(deviceAdaptor->gdrMemFree(wrongPtr, &handle),
            flagcxInvalidArgument);
  EXPECT_TRUE(handle.mapped);
  EXPECT_EQ(handle.devPtr, devPtr);

  ASSERT_EQ(deviceAdaptor->gdrMemFree(devPtr, &handle), flagcxSuccess);
  EXPECT_EQ(handle.devPtr, nullptr);
  EXPECT_EQ(handle.hostMappedPtr, nullptr);
  EXPECT_EQ(handle.size, static_cast<size_t>(0));
  EXPECT_FALSE(handle.mapped);
}

// Unit test for the handle constructor/destructor contract only.
TEST_F(DeviceAdaptorTest, MemHandleInitDestroy) {
  ASSERT_NE(deviceAdaptor->memHandleInit, nullptr);
  ASSERT_NE(deviceAdaptor->memHandleDestroy, nullptr);

  EXPECT_EQ(deviceAdaptor->memHandleInit(0, nullptr), flagcxInvalidArgument);
  EXPECT_EQ(deviceAdaptor->memHandleDestroy(0, nullptr), flagcxSuccess);

  void *rawHandle = nullptr;
  ASSERT_EQ(deviceAdaptor->memHandleInit(0, &rawHandle), flagcxSuccess);
  ASSERT_NE(rawHandle, nullptr);

  auto *handle = static_cast<KunlunXinGdrMemHandle *>(rawHandle);
  EXPECT_EQ(handle->devPtr, nullptr);
  EXPECT_EQ(handle->hostMappedPtr, nullptr);
  EXPECT_EQ(handle->size, static_cast<size_t>(0));
  EXPECT_FALSE(handle->mapped);

  handle->devPtr = reinterpret_cast<void *>(0x1);
  EXPECT_EQ(deviceAdaptor->memHandleDestroy(0, rawHandle),
            flagcxInvalidArgument);
  handle->devPtr = nullptr;
  EXPECT_EQ(deviceAdaptor->memHandleDestroy(0, rawHandle), flagcxSuccess);
}

// Scenario test: public adaptor interfaces plus the downstream RDMA
// registration operation, covering the complete local GDR lifecycle.
// This local scenario validates through MR registration. The follow-up
// two-node acceptance test should verify an actual remote RDMA WRITE/READ and
// payload visibility by reusing the repository RMA/SendRecv MPI suite.
// Requires BKCL_USE_PEERMEM_XDR=1 and CUDA_ENABLE_P2P_NO_UVA=1 in the process
// environment before XCCL/runtime initialization.
TEST_F(DeviceAdaptorTest, GdrMemoryLifecycle) {
  ASSERT_NE(deviceAdaptor->memHandleInit, nullptr);
  ASSERT_NE(deviceAdaptor->memHandleDestroy, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemAlloc, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemFree, nullptr);

  void *rawHandle = nullptr;
  ASSERT_EQ(deviceAdaptor->memHandleInit(0, &rawHandle), flagcxSuccess);
  ASSERT_NE(rawHandle, nullptr);

  void *devPtr = nullptr;
  const auto allocResult =
      deviceAdaptor->gdrMemAlloc(&devPtr, TEST_SIZE, rawHandle);
  if (allocResult != flagcxSuccess || devPtr == nullptr) {
    deviceAdaptor->memHandleDestroy(0, rawHandle);
    FAIL() << "gdrMemAlloc failed; check the peermem environment";
  }

  auto *handle = static_cast<KunlunXinGdrMemHandle *>(rawHandle);
  auto cleanup = [&]() {
    if (devPtr != nullptr) {
      EXPECT_EQ(deviceAdaptor->gdrMemFree(devPtr, rawHandle), flagcxSuccess);
      devPtr = nullptr;
    }
    if (rawHandle != nullptr) {
      EXPECT_EQ(deviceAdaptor->memHandleDestroy(0, rawHandle), flagcxSuccess);
      rawHandle = nullptr;
    }
  };

  if (!handle->mapped || handle->hostMappedPtr == nullptr) {
    cleanup();
    FAIL() << "gdrMemAlloc did not create a host mapping";
  }

  // CPU mapped view -> device view.
  auto *mappedValue = static_cast<uint64_t *>(handle->hostMappedPtr);
  const uint64_t cpuValue = 0x1122334455667788ULL;
  __atomic_store_n(mappedValue, cpuValue, __ATOMIC_RELEASE);

  uint64_t deviceRead = 0;
  auto result = devHandle->deviceMemcpy(
      &deviceRead, devPtr, sizeof(deviceRead), flagcxMemcpyDeviceToHost,
      stream);
  if (result != flagcxSuccess ||
      devHandle->streamSynchronize(stream) != flagcxSuccess) {
    cleanup();
    FAIL() << "device read after mapped CPU write failed";
  }
  EXPECT_EQ(deviceRead, cpuValue);

  // Device view -> CPU mapped view.
  uint64_t deviceValue = 0x8877665544332211ULL;
  result = devHandle->deviceMemcpy(devPtr, &deviceValue, sizeof(deviceValue),
                                   flagcxMemcpyHostToDevice, stream);
  if (result != flagcxSuccess ||
      devHandle->streamSynchronize(stream) != flagcxSuccess) {
    cleanup();
    FAIL() << "device write before mapped CPU read failed";
  }
  EXPECT_EQ(__atomic_load_n(mappedValue, __ATOMIC_ACQUIRE), deviceValue);

  const auto mrResult =
      verifyMrRegistration(handle->hostMappedPtr, handle->size);
  if (!mrResult) {
    cleanup();
    FAIL() << mrResult.message();
  }

  ASSERT_EQ(deviceAdaptor->gdrMemFree(devPtr, rawHandle), flagcxSuccess);
  devPtr = nullptr;
  EXPECT_EQ(handle->devPtr, nullptr);
  EXPECT_EQ(handle->hostMappedPtr, nullptr);
  EXPECT_FALSE(handle->mapped);

  EXPECT_EQ(deviceAdaptor->memHandleDestroy(0, rawHandle), flagcxSuccess);
  rawHandle = nullptr;
}
#endif  // USE_KUNLUNXIN_ADAPTOR


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
