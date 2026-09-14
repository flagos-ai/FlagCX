/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-only tests for Device API communicator cleanup ownership.
 ************************************************************************/

#include "adaptor.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "flagcx_net_adaptor.h"
#include "global_comm.h"
#include "mem_alloc_registry.h"
#include "onesided.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

static constexpr size_t kLegacyRequirementsSize =
    offsetof(flagcxDevCommRequirements, intraScratchBytes);
static_assert(kLegacyRequirementsSize == 40,
              "flagcxDevCommRequirements legacy ABI size changed");
static_assert(offsetof(flagcxDevCommRequirements, intraMulticast) == 0,
              "flagcxDevCommRequirements legacy ABI offset changed");
static_assert(offsetof(flagcxDevCommRequirements, barrierCount) == 4,
              "flagcxDevCommRequirements legacy ABI offset changed");
static_assert(offsetof(flagcxDevCommRequirements, interForceEnable) == 24,
              "flagcxDevCommRequirements legacy ABI offset changed");
static_assert(offsetof(flagcxDevCommRequirements, interCounterCount) == 36,
              "flagcxDevCommRequirements legacy ABI offset changed");
static_assert(offsetof(flagcxDevCommRequirements, intraScratchBytes) == 40,
              "new requirements fields must follow the frozen ABI prefix");

namespace {

struct LegacyWindowTeam {};

struct LegacyWindow {
  void *pointer;

  void *getPeerPointer(size_t, const LegacyWindowTeam &, int) const {
    return pointer;
  }

  void *getIntraPointer(size_t, int) const { return pointer; }
};

TEST(WindowAccessCompatibilityTest, LegacyWindowSupportsReadWriteFallback) {
  void *expected = reinterpret_cast<void *>(0x1000);
  LegacyWindow window{expected};
  LegacyWindowTeam team;

  EXPECT_EQ(flagcxGetPeerPointerWithAccessInternal(
                window, 0, team, 0, flagcxDevPeerAccessReadWrite, 0),
            expected);
  EXPECT_EQ(flagcxGetIntraPointerWithAccessInternal(
                window, 0, 0, flagcxDevPeerAccessReadWrite, 0),
            expected);
}

TEST(WindowAccessCompatibilityTest, LegacyWindowRejectsDirectionalFallback) {
  LegacyWindow window{reinterpret_cast<void *>(0x1000)};
  LegacyWindowTeam team;

  EXPECT_EQ(flagcxGetPeerPointerWithAccessInternal(
                window, 0, team, 0, flagcxDevPeerAccessReadOnly, 0),
            nullptr);
  EXPECT_EQ(flagcxGetPeerPointerWithAccessInternal(
                window, 0, team, 0, flagcxDevPeerAccessWriteOnly, 0),
            nullptr);
  EXPECT_EQ(flagcxGetIntraPointerWithAccessInternal(
                window, 0, 0, flagcxDevPeerAccessReadOnly, 0),
            nullptr);
  EXPECT_EQ(flagcxGetIntraPointerWithAccessInternal(
                window, 0, 0, flagcxDevPeerAccessWriteOnly, 0),
            nullptr);
}

std::vector<void *> unregisteredPtrs;
std::vector<void *> deviceFreedPtrs;
std::vector<void *> allocatedPtrs;
int deviceMallocCallCount = 0;
int deviceMemsetCallCount = 0;
int failDeviceMallocCall = -1;
int failDeviceMemsetCall = -1;

enum class CleanupEventKind { Deregister, BufferFree };

struct CleanupEvent {
  CleanupEventKind kind;
  void *ptr;
};

std::vector<CleanupEvent> cleanupEvents;

flagcxResult_t recordHostUnregister(void *ptr) {
  unregisteredPtrs.push_back(ptr);
  return flagcxSuccess;
}

flagcxResult_t recordDeviceFree(void *ptr, flagcxMemType_t, flagcxStream_t) {
  deviceFreedPtrs.push_back(ptr);
  cleanupEvents.push_back({CleanupEventKind::BufferFree, ptr});
  return flagcxSuccess;
}

flagcxResult_t recordGdrFree(void *ptr, void *) {
  deviceFreedPtrs.push_back(ptr);
  cleanupEvents.push_back({CleanupEventKind::BufferFree, ptr});
  return flagcxSuccess;
}

flagcxResult_t recordMrDeregister(void *, void *mrHandle) {
  cleanupEvents.push_back({CleanupEventKind::Deregister, mrHandle});
  return flagcxSuccess;
}

flagcxResult_t allocateTestBuffer(void **ptr, size_t size) {
  *ptr = malloc(size);
  if (!*ptr)
    return flagcxSystemError;
  allocatedPtrs.push_back(*ptr);
  return flagcxSuccess;
}

flagcxResult_t recordDeviceMalloc(void **ptr, size_t size, flagcxMemType_t,
                                  flagcxStream_t) {
  return allocateTestBuffer(ptr, size);
}

flagcxResult_t injectDeviceMallocFailure(void **ptr, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  deviceMallocCallCount++;
  if (deviceMallocCallCount == failDeviceMallocCall) {
    *ptr = nullptr;
    return flagcxSystemError;
  }
  return recordDeviceMalloc(ptr, size, type, stream);
}

flagcxResult_t recordGdrMalloc(void **ptr, size_t size, void *) {
  return allocateTestBuffer(ptr, size);
}

flagcxResult_t recordDeviceMemset(void *ptr, int value, size_t size,
                                  flagcxMemType_t, flagcxStream_t) {
  memset(ptr, value, size);
  return flagcxSuccess;
}

flagcxResult_t injectDeviceMemsetFailure(void *ptr, int value, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  deviceMemsetCallCount++;
  if (deviceMemsetCallCount == failDeviceMemsetCall)
    return flagcxSystemError;
  return recordDeviceMemset(ptr, value, size, type, stream);
}

flagcxResult_t failIpcMemHandleCreate(flagcxIpcMemHandle_t *, size_t *) {
  return flagcxNotSupported;
}

flagcxOneSideHandleInfo *makeRegistration(void *buffer, void *mrHandle) {
  flagcxOneSideHandleInfo *registration =
      static_cast<flagcxOneSideHandleInfo *>(
          calloc(1, sizeof(flagcxOneSideHandleInfo)));
  if (!registration)
    return nullptr;
  registration->baseVas =
      static_cast<uintptr_t *>(calloc(1, sizeof(uintptr_t)));
  registration->regionSizes = static_cast<size_t *>(calloc(1, sizeof(size_t)));
  registration->mrInfos =
      static_cast<flagcxNetMrInfo *>(calloc(1, sizeof(flagcxNetMrInfo)));
  if (!registration->baseVas || !registration->regionSizes ||
      !registration->mrInfos) {
    free(registration->mrInfos);
    free(registration->regionSizes);
    free(registration->baseVas);
    free(registration);
    return nullptr;
  }
  registration->baseVas[0] = reinterpret_cast<uintptr_t>(buffer);
  registration->regionSizes[0] = 1;
  registration->nRanks = 1;
  registration->localMrHandle = mrHandle;
  registration->localRecvComm = reinterpret_cast<void *>(0x9000);
  return registration;
}

void freeRegistration(flagcxOneSideHandleInfo *registration) {
  if (!registration)
    return;
  free(registration->mrInfos);
  free(registration->regionSizes);
  free(registration->baseVas);
  free(registration);
}

class DevApiBackendGuard {
public:
  explicit DevApiBackendGuard(flagcxDevApiBackend *replacement)
      : saved_(devApiBackend) {
    devApiBackend = replacement;
  }
  ~DevApiBackendGuard() { devApiBackend = saved_; }

private:
  flagcxDevApiBackend *saved_;
};

flagcxResult_t (*savedDefaultDevCommDestroy)(flagcxComm_t,
                                             flagcxDevComm_t) = nullptr;
int observedDestroyCalls = 0;
bool handleClearedBeforeDestroy = false;

flagcxResult_t observeDefaultDevCommDestroy(flagcxComm_t comm,
                                            flagcxDevComm_t devComm) {
  observedDestroyCalls++;
  handleClearedBeforeDestroy = comm != nullptr && comm->heteroComm != nullptr &&
                               comm->heteroComm->devCommHandle == nullptr;
  return savedDefaultDevCommDestroy != nullptr
             ? savedDefaultDevCommDestroy(comm, devComm)
             : flagcxInternalError;
}

class DefaultDevCommCleanupTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (strcmp(devApiBackend->name, "default") != 0) {
      GTEST_SKIP() << "requires the default Device API backend";
    }
    savedDeviceAdaptor = deviceAdaptor;
    testDeviceAdaptor = *deviceAdaptor;
    testDeviceAdaptor.hostUnregister = recordHostUnregister;
    testDeviceAdaptor.deviceFree = recordDeviceFree;
    testDeviceAdaptor.gdrMemFree = recordGdrFree;
    deviceAdaptor = &testDeviceAdaptor;
    unregisteredPtrs.clear();
    deviceFreedPtrs.clear();
    allocatedPtrs.clear();
    cleanupEvents.clear();
    deviceMallocCallCount = 0;
    deviceMemsetCallCount = 0;
    failDeviceMallocCall = -1;
    failDeviceMemsetCall = -1;
    savedDefaultDevCommDestroy = nullptr;
    observedDestroyCalls = 0;
    handleClearedBeforeDestroy = false;
  }

  void TearDown() override {
    if (savedDeviceAdaptor)
      deviceAdaptor = savedDeviceAdaptor;
    for (void *ptr : allocatedPtrs)
      free(ptr);
    allocatedPtrs.clear();
  }

  struct flagcxDeviceAdaptor *savedDeviceAdaptor = nullptr;
  struct flagcxDeviceAdaptor testDeviceAdaptor = {};
};

TEST_F(DefaultDevCommCleanupTest, DevMemCreateClearsOutputOnInvalidArguments) {
  flagcxComm comm = {};
  char buffer = 0;
  auto sentinel = reinterpret_cast<flagcxDevMem_t>(0x1);

  flagcxDevMem_t devMem = sentinel;
  EXPECT_EQ(
      flagcxDevMemCreate(nullptr, &buffer, sizeof(buffer), nullptr, &devMem),
      flagcxInvalidArgument);
  EXPECT_EQ(devMem, nullptr);

  devMem = sentinel;
  EXPECT_EQ(
      flagcxDevMemCreate(&comm, nullptr, sizeof(buffer), nullptr, &devMem),
      flagcxInvalidArgument);
  EXPECT_EQ(devMem, nullptr);

  devMem = sentinel;
  EXPECT_EQ(flagcxDevMemCreate(&comm, &buffer, 0, nullptr, &devMem),
            flagcxInvalidArgument);
  EXPECT_EQ(devMem, nullptr);
}

TEST_F(DefaultDevCommCleanupTest, EarlyCreateFailureDoesNotPublishHandle) {
  testDeviceAdaptor.deviceMalloc = injectDeviceMallocFailure;
  failDeviceMallocCall = 1;

  flagcxHeteroComm heteroComm = {};
  heteroComm.nRanks = 1;
  heteroComm.nNodes = 1;

  flagcxComm comm = {};
  comm.nranks = 1;
  comm.localRanks = 1;
  comm.heteroComm = &heteroComm;

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = reinterpret_cast<flagcxDevComm_t>(0x1);
  EXPECT_EQ(flagcxDevCommCreate(&comm, &reqs, &devComm), flagcxSystemError);

  EXPECT_EQ(deviceMallocCallCount, 1);
  EXPECT_EQ(devComm, nullptr);
  EXPECT_EQ(heteroComm.devCommHandle, nullptr);
  EXPECT_TRUE(allocatedPtrs.empty());
  EXPECT_TRUE(deviceFreedPtrs.empty());
}

TEST_F(DefaultDevCommCleanupTest, PartialCreateFailureReleasesOwnedBuffers) {
  testDeviceAdaptor.deviceMalloc = injectDeviceMallocFailure;
  testDeviceAdaptor.deviceMemset = injectDeviceMemsetFailure;
  failDeviceMemsetCall = 1;

  flagcxHeteroComm heteroComm = {};
  heteroComm.nRanks = 1;
  heteroComm.nNodes = 1;

  flagcxComm comm = {};
  comm.nranks = 1;
  comm.localRanks = 1;
  comm.heteroComm = &heteroComm;

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = reinterpret_cast<flagcxDevComm_t>(0x1);
  EXPECT_EQ(flagcxDevCommCreate(&comm, &reqs, &devComm), flagcxSystemError);

  EXPECT_EQ(deviceMemsetCallCount, 1);
  ASSERT_EQ(allocatedPtrs.size(), 1u);
  ASSERT_EQ(deviceFreedPtrs.size(), 1u);
  EXPECT_EQ(deviceFreedPtrs[0], allocatedPtrs[0]);
  EXPECT_EQ(devComm, nullptr);
  EXPECT_EQ(heteroComm.devCommHandle, nullptr);
}

TEST_F(DefaultDevCommCleanupTest, UnpublishesHandleBeforeBackendDestroy) {
  auto *devComm = static_cast<flagcxDevComm_t>(
      calloc(1, sizeof(struct flagcxDevCommInternal)));
  ASSERT_NE(devComm, nullptr);
  ASSERT_EQ(pthread_mutex_init(&devComm->cachedPtrMutex, nullptr), 0);
  devComm->barrierIpcIndex = -1;
  devComm->signalIpcSlot = -1;

  flagcxHeteroComm heteroComm = {};
  heteroComm.devCommHandle = devComm;
  flagcxComm comm = {};
  comm.heteroComm = &heteroComm;

  flagcxDevApiBackend observedDefaultBackend = *devApiBackend;
  savedDefaultDevCommDestroy = observedDefaultBackend.devCommDestroy;
  observedDefaultBackend.devCommDestroy = observeDefaultDevCommDestroy;
  DevApiBackendGuard backendGuard(&observedDefaultBackend);

  EXPECT_EQ(flagcxDevCommDestroy(&comm, devComm), flagcxSuccess);
  EXPECT_EQ(observedDestroyCalls, 1);
  EXPECT_TRUE(handleClearedBeforeDestroy);
  EXPECT_EQ(heteroComm.devCommHandle, nullptr);
}

TEST_F(DefaultDevCommCleanupTest, ShmBarrierAliasesAreNotFreedTwice) {
  flagcxDevCommInternal devComm = {};
  devComm.nLocalRanks = 3;
  devComm.registeredBarrierShmCount = 2;
  devComm.peerBarrierShmPtrs =
      static_cast<void **>(calloc(devComm.nLocalRanks, sizeof(void *)));
  ASSERT_NE(devComm.peerBarrierShmPtrs, nullptr);
  devComm.peerBarrierShmPtrs[0] = reinterpret_cast<void *>(0x1000);
  devComm.peerBarrierShmPtrs[1] = reinterpret_cast<void *>(0x2000);
  devComm.peerBarrierShmPtrs[2] = reinterpret_cast<void *>(0x3000);
  devComm.localBarrierShmPtr = devComm.peerBarrierShmPtrs[0];
  devComm.localBarrierFlags =
      reinterpret_cast<uint64_t *>(devComm.localBarrierShmPtr);
  devComm.localBarrierFlagsDeviceAllocated = false;
  devComm.barrierDevPeerPtrsRaw = reinterpret_cast<uint64_t **>(0x4000);
  devComm.barrierPeers = devComm.barrierDevPeerPtrsRaw;

  ASSERT_EQ(devApiBackend->devCommDestroy(nullptr, &devComm), flagcxSuccess);

  ASSERT_EQ(unregisteredPtrs.size(), 2u);
  EXPECT_EQ(unregisteredPtrs[0], reinterpret_cast<void *>(0x1000));
  EXPECT_EQ(unregisteredPtrs[1], reinterpret_cast<void *>(0x2000));
  ASSERT_EQ(deviceFreedPtrs.size(), 1u);
  EXPECT_EQ(deviceFreedPtrs[0], reinterpret_cast<void *>(0x4000));
  EXPECT_EQ(devComm.localBarrierFlags, nullptr);
  EXPECT_EQ(devComm.localBarrierShmPtr, nullptr);
  EXPECT_EQ(devComm.registeredBarrierShmCount, 0);

  ASSERT_EQ(devApiBackend->devCommDestroy(nullptr, &devComm), flagcxSuccess);
  EXPECT_EQ(unregisteredPtrs.size(), 2u);
  EXPECT_EQ(deviceFreedPtrs.size(), 1u);
}

TEST_F(DefaultDevCommCleanupTest, IpcBarrierAllocationIsFreedOnce) {
  flagcxDevCommInternal devComm = {};
  devComm.barrierIpcIndex = -1;
  devComm.signalIpcSlot = -1;
  devComm.localBarrierFlags = reinterpret_cast<uint64_t *>(0x5000);
  devComm.localBarrierFlagsDeviceAllocated = true;

  ASSERT_EQ(devApiBackend->devCommDestroy(nullptr, &devComm), flagcxSuccess);

  EXPECT_TRUE(unregisteredPtrs.empty());
  ASSERT_EQ(deviceFreedPtrs.size(), 1u);
  EXPECT_EQ(deviceFreedPtrs[0], reinterpret_cast<void *>(0x5000));
  EXPECT_EQ(devComm.localBarrierFlags, nullptr);
  EXPECT_FALSE(devComm.localBarrierFlagsDeviceAllocated);
}

TEST_F(DefaultDevCommCleanupTest,
       OwnedRegistrationsAreRemovedBeforeBackingBuffers) {
  flagcxNetAdaptor netAdaptor = {};
  netAdaptor.name = "test";
  netAdaptor.deregMr = recordMrDeregister;

  flagcxHeteroComm heteroComm = {};
  heteroComm.rank = 0;
  heteroComm.nRanks = 1;
  heteroComm.netAdaptor = &netAdaptor;

  flagcxComm comm = {};
  comm.rank = 0;
  comm.nranks = 1;
  comm.heteroComm = &heteroComm;

  void *signalBuffer = reinterpret_cast<void *>(0x6000);
  void *stagingBuffer = reinterpret_cast<void *>(0x7000);
  void *signalMr = reinterpret_cast<void *>(0x6100);
  void *stagingMr = reinterpret_cast<void *>(0x7100);
  heteroComm.signalHandle = makeRegistration(signalBuffer, signalMr);
  heteroComm.stagingHandle = makeRegistration(stagingBuffer, stagingMr);
  ASSERT_NE(heteroComm.signalHandle, nullptr);
  ASSERT_NE(heteroComm.stagingHandle, nullptr);
  heteroComm.rmaSignalBase = signalBuffer;
  heteroComm.rmaSignalSize = 1;
  heteroComm.rmaSignalIpcSlot = -1;

  flagcxDevCommInternal devComm = {};
  devComm.barrierIpcIndex = -1;
  devComm.signalIpcSlot = -1;
  devComm.signalBuffer = static_cast<uint64_t *>(signalBuffer);
  devComm.putValueStagingBuffer = stagingBuffer;
  devComm.ownedSignalBuffer = signalBuffer;
  devComm.ownedSignalRegistration = heteroComm.signalHandle;
  devComm.ownedStagingRegistration = heteroComm.stagingHandle;

  ASSERT_EQ(devApiBackend->devCommDestroy(&comm, &devComm), flagcxSuccess);

  EXPECT_EQ(heteroComm.signalHandle, nullptr);
  EXPECT_EQ(heteroComm.rmaSignalBase, nullptr);
  EXPECT_EQ(heteroComm.stagingHandle, nullptr);
  EXPECT_EQ(devComm.ownedSignalBuffer, nullptr);
  EXPECT_EQ(devComm.ownedSignalRegistration, nullptr);
  EXPECT_EQ(devComm.ownedStagingRegistration, nullptr);
  ASSERT_EQ(cleanupEvents.size(), 4u);
  EXPECT_EQ(cleanupEvents[0].kind, CleanupEventKind::Deregister);
  EXPECT_EQ(cleanupEvents[0].ptr, stagingMr);
  EXPECT_EQ(cleanupEvents[1].kind, CleanupEventKind::Deregister);
  EXPECT_EQ(cleanupEvents[1].ptr, signalMr);
  EXPECT_EQ(cleanupEvents[2].kind, CleanupEventKind::BufferFree);
  EXPECT_EQ(cleanupEvents[2].ptr, signalBuffer);
  EXPECT_EQ(cleanupEvents[3].kind, CleanupEventKind::BufferFree);
  EXPECT_EQ(cleanupEvents[3].ptr, stagingBuffer);

  ASSERT_EQ(devApiBackend->devCommDestroy(&comm, &devComm), flagcxSuccess);
  EXPECT_EQ(cleanupEvents.size(), 4u);
}

TEST_F(DefaultDevCommCleanupTest,
       PreexistingRegistrationsAreNotRemovedByDestroy) {
  flagcxNetAdaptor netAdaptor = {};
  netAdaptor.name = "test";
  netAdaptor.deregMr = recordMrDeregister;

  flagcxHeteroComm heteroComm = {};
  heteroComm.rank = 0;
  heteroComm.nRanks = 1;
  heteroComm.netAdaptor = &netAdaptor;

  flagcxComm comm = {};
  comm.rank = 0;
  comm.nranks = 1;
  comm.heteroComm = &heteroComm;

  flagcxOneSideHandleInfo *existingSignal = makeRegistration(
      reinterpret_cast<void *>(0x8000), reinterpret_cast<void *>(0x8100));
  flagcxOneSideHandleInfo *existingStaging = makeRegistration(
      reinterpret_cast<void *>(0x8200), reinterpret_cast<void *>(0x8300));
  ASSERT_NE(existingSignal, nullptr);
  ASSERT_NE(existingStaging, nullptr);
  heteroComm.signalHandle = existingSignal;
  heteroComm.stagingHandle = existingStaging;

  flagcxDevCommInternal devComm = {};
  devComm.barrierIpcIndex = -1;
  devComm.signalIpcSlot = -1;
  devComm.signalBuffer = reinterpret_cast<uint64_t *>(0x8400);
  devComm.putValueStagingBuffer = reinterpret_cast<void *>(0x8500);

  ASSERT_EQ(devApiBackend->devCommDestroy(&comm, &devComm), flagcxSuccess);

  EXPECT_EQ(heteroComm.signalHandle, existingSignal);
  EXPECT_EQ(heteroComm.stagingHandle, existingStaging);
  ASSERT_EQ(cleanupEvents.size(), 2u);
  EXPECT_EQ(cleanupEvents[0].kind, CleanupEventKind::BufferFree);
  EXPECT_EQ(cleanupEvents[0].ptr, reinterpret_cast<void *>(0x8400));
  EXPECT_EQ(cleanupEvents[1].kind, CleanupEventKind::BufferFree);
  EXPECT_EQ(cleanupEvents[1].ptr, reinterpret_cast<void *>(0x8500));

  heteroComm.signalHandle = nullptr;
  heteroComm.stagingHandle = nullptr;
  freeRegistration(existingSignal);
  freeRegistration(existingStaging);
}

TEST_F(DefaultDevCommCleanupTest,
       MismatchedPreexistingRegistrationsAreNotReusedOnCreate) {
  testDeviceAdaptor.deviceMalloc = recordDeviceMalloc;
  testDeviceAdaptor.gdrMemAlloc = recordGdrMalloc;
  testDeviceAdaptor.deviceMemset = recordDeviceMemset;
  testDeviceAdaptor.ipcMemHandleCreate = failIpcMemHandleCreate;

  flagcxNetAdaptor netAdaptor = {};
  netAdaptor.name = "test";
  netAdaptor.deregMr = recordMrDeregister;

  flagcxHeteroComm heteroComm = {};
  heteroComm.rank = 0;
  heteroComm.nRanks = 1;
  heteroComm.nNodes = 1;
  heteroComm.netAdaptor = &netAdaptor;

  flagcxComm comm = {};
  comm.rank = 0;
  comm.nranks = 1;
  comm.localRank = 0;
  comm.localRanks = 1;
  comm.heteroComm = &heteroComm;

  flagcxOneSideHandleInfo *existingSignal = makeRegistration(
      reinterpret_cast<void *>(0xa000), reinterpret_cast<void *>(0xa100));
  flagcxOneSideHandleInfo *existingStaging = makeRegistration(
      reinterpret_cast<void *>(0xa200), reinterpret_cast<void *>(0xa300));
  ASSERT_NE(existingSignal, nullptr);
  ASSERT_NE(existingStaging, nullptr);
  heteroComm.signalHandle = existingSignal;
  heteroComm.stagingHandle = existingStaging;

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interSignalCount = 1;
  flagcxDevComm_t devComm = reinterpret_cast<flagcxDevComm_t>(0x1);
  EXPECT_EQ(flagcxDevCommCreate(&comm, &reqs, &devComm), flagcxNotSupported);

  EXPECT_EQ(devComm, nullptr);
  EXPECT_EQ(heteroComm.devCommHandle, nullptr);
  EXPECT_EQ(heteroComm.signalHandle, existingSignal);
  EXPECT_EQ(heteroComm.stagingHandle, existingStaging);
  EXPECT_EQ(cleanupEvents.size(), allocatedPtrs.size());
  for (const CleanupEvent &event : cleanupEvents)
    EXPECT_EQ(event.kind, CleanupEventKind::BufferFree);

  heteroComm.signalHandle = nullptr;
  heteroComm.stagingHandle = nullptr;
  freeRegistration(existingSignal);
  freeRegistration(existingStaging);
}

int rollbackCallCount = 0;
flagcxResult_t rollbackResult = flagcxSuccess;

flagcxResult_t failDevCommCreate(flagcxComm_t,
                                 const flagcxDevCommRequirements *, size_t,
                                 flagcxDevComm_t) {
  return flagcxSystemError;
}

size_t capturedRequirementsSize = 0;
size_t capturedScratchBytes = 0;

flagcxResult_t captureDevCommCreate(flagcxComm_t,
                                    const flagcxDevCommRequirements *reqs,
                                    size_t reqsSize, flagcxDevComm_t) {
  capturedRequirementsSize = reqsSize;
  capturedScratchBytes = 0;
  if (reqsSize >= offsetof(flagcxDevCommRequirements, intraScratchBytes) +
                      sizeof(reqs->intraScratchBytes)) {
    capturedScratchBytes = reqs->intraScratchBytes;
  }
  return flagcxSuccess;
}

flagcxResult_t captureDevCommDestroy(flagcxComm_t, flagcxDevComm_t) {
  return flagcxSuccess;
}

TEST(DevCommRequirementsTest, PublicCallPassesCurrentStructureSize) {
  flagcxDevApiBackend captureBackend = {};
  captureBackend.name = "capture-config";
  captureBackend.devCommCreate = captureDevCommCreate;
  captureBackend.devCommDestroy = captureDevCommDestroy;
  DevApiBackendGuard guard(&captureBackend);

  flagcxComm comm = {};
  comm.rank = 0;
  comm.nranks = 1;
  comm.localRank = 0;
  comm.localRanks = 1;

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraScratchBytes = 4096;

  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(&comm, &reqs, &devComm), flagcxSuccess);
  ASSERT_NE(devComm, nullptr);
  EXPECT_EQ(capturedRequirementsSize, sizeof(reqs));
  EXPECT_EQ(capturedScratchBytes, 4096u);
  EXPECT_EQ(flagcxDevCommDestroy(&comm, devComm), flagcxSuccess);
}

TEST(DevCommRequirementsTest, LegacySymbolReadsOnlyFrozenPrefix) {
  flagcxDevApiBackend captureBackend = {};
  captureBackend.name = "capture-config";
  captureBackend.devCommCreate = captureDevCommCreate;
  captureBackend.devCommDestroy = captureDevCommDestroy;
  DevApiBackendGuard guard(&captureBackend);

  flagcxComm comm = {};
  comm.rank = 0;
  comm.nranks = 1;
  comm.localRank = 0;
  comm.localRanks = 1;

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraScratchBytes = 4096;

  auto legacyCreate = &flagcxDevCommCreate;
  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(legacyCreate(&comm, &reqs, &devComm), flagcxSuccess);
  ASSERT_NE(devComm, nullptr);
  EXPECT_EQ(capturedRequirementsSize, kLegacyRequirementsSize);
  EXPECT_EQ(capturedScratchBytes, 0u);
  EXPECT_EQ(flagcxDevCommDestroy(&comm, devComm), flagcxSuccess);
}

TEST(DevCommRequirementsTest, RejectsAnIncompleteLegacyPrefix) {
  flagcxComm comm = {};
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = reinterpret_cast<flagcxDevComm_t>(0x1);

  EXPECT_EQ(flagcxDevCommCreateSized(&comm, &reqs, kLegacyRequirementsSize - 1,
                                     &devComm),
            flagcxInvalidArgument);
  EXPECT_EQ(devComm, nullptr);
}

TEST(DevCommRequirementsTest, SizedEntryAcceptsFutureTail) {
  flagcxDevApiBackend captureBackend = {};
  captureBackend.name = "capture-config";
  captureBackend.devCommCreate = captureDevCommCreate;
  captureBackend.devCommDestroy = captureDevCommDestroy;
  DevApiBackendGuard guard(&captureBackend);

  flagcxComm comm = {};
  comm.rank = 0;
  comm.nranks = 1;
  comm.localRank = 0;
  comm.localRanks = 1;

  struct FutureRequirements {
    flagcxDevCommRequirements current;
    uint64_t futureField;
  } reqs = {FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER, 17};
  reqs.current.intraScratchBytes = 8192;

  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(
      flagcxDevCommCreateSized(&comm, &reqs.current, sizeof(reqs), &devComm),
      flagcxSuccess);
  ASSERT_NE(devComm, nullptr);
  EXPECT_EQ(capturedRequirementsSize, sizeof(reqs));
  EXPECT_EQ(capturedScratchBytes, 8192u);
  EXPECT_EQ(flagcxDevCommDestroy(&comm, devComm), flagcxSuccess);
}

flagcxResult_t recordDevCommRollback(flagcxComm_t, flagcxDevComm_t) {
  rollbackCallCount++;
  return rollbackResult;
}

TEST(DevCommCreateTest, PreservesCreateErrorWhenRollbackFails) {
  struct flagcxDevApiBackend failingBackend = {};
  failingBackend.name = "failing-test-backend";
  failingBackend.devCommCreate = failDevCommCreate;
  failingBackend.devCommDestroy = recordDevCommRollback;

  struct flagcxDevApiBackend *savedBackend = devApiBackend;
  devApiBackend = &failingBackend;
  rollbackCallCount = 0;
  rollbackResult = flagcxInternalError;

  flagcxComm comm = {};
  comm.rank = 0;
  comm.nranks = 1;
  comm.localRank = 0;
  comm.localRanks = 1;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = nullptr;
  flagcxResult_t result = flagcxDevCommCreate(&comm, &reqs, &devComm);

  devApiBackend = savedBackend;
  EXPECT_EQ(result, flagcxSystemError);
  EXPECT_EQ(rollbackCallCount, 1);
  EXPECT_EQ(devComm, nullptr);
}

int devMemCreateCalls = 0;
flagcxMemAllocationInfo capturedAllocation = {};
bool capturedTracked = false;

flagcxResult_t captureDevMemCreate(flagcxComm_t, void *, size_t, flagcxWindow_t,
                                   flagcxDevMem_t devMem) {
  devMemCreateCalls++;
  capturedTracked = devMem->allocationTracked;
  if (capturedTracked) {
    capturedAllocation = {devMem->allocationBase, devMem->allocationSize,
                          devMem->allocator, devMem->allocBackend};
  }
  return flagcxSuccess;
}

flagcxResult_t captureDevMemDestroy(flagcxComm_t, flagcxDevMem_t) {
  return flagcxSuccess;
}

class TrackedAllocationGuard {
public:
  explicit TrackedAllocationGuard(const flagcxMemAllocationInfo &info)
      : base_(info.base) {
    EXPECT_EQ(globalMemAllocRegistry.insert(info), flagcxSuccess);
  }
  ~TrackedAllocationGuard() { globalMemAllocRegistry.erase(base_); }

private:
  void *base_;
};

TEST(DevMemProvenanceTest, PropagatesNativeAllocationProvenance) {
  void *allocation = malloc(256);
  ASSERT_NE(allocation, nullptr);
  const flagcxMemAllocator_t allocator = flagcxMemCCL;
  const flagcxMemAllocBackend_t backend = flagcxMemAllocBackendNative;
  TrackedAllocationGuard allocationGuard({allocation, 256, allocator, backend});

  flagcxDevApiBackend captureBackend = {};
  captureBackend.name = "capture-provenance";
  captureBackend.devMemCreate = captureDevMemCreate;
  captureBackend.devMemDestroy = captureDevMemDestroy;
  DevApiBackendGuard backendGuard(&captureBackend);
  devMemCreateCalls = 0;
  capturedTracked = false;

  flagcxComm comm = {};
  flagcxDevMem_t devMem = nullptr;
  EXPECT_EQ(flagcxDevMemCreate(&comm, static_cast<char *>(allocation) + 32, 64,
                               nullptr, &devMem),
            flagcxSuccess);
  ASSERT_NE(devMem, nullptr);
  EXPECT_EQ(devMemCreateCalls, 1);
  EXPECT_TRUE(capturedTracked);
  EXPECT_EQ(capturedAllocation.base, allocation);
  EXPECT_EQ(capturedAllocation.size, 256u);
  EXPECT_EQ(capturedAllocation.allocator, allocator);
  EXPECT_EQ(capturedAllocation.backend, backend);
  EXPECT_EQ(flagcxDevMemDestroy(&comm, devMem), flagcxSuccess);
  free(allocation);
}

TEST(DevMemProvenanceTest, RejectsRangeBeyondTrackedAllocation) {
  void *allocation = malloc(256);
  ASSERT_NE(allocation, nullptr);
  const flagcxMemAllocationInfo info = {allocation, 256, flagcxMemCCL,
                                        flagcxMemAllocBackendNative};
  TrackedAllocationGuard allocationGuard(info);

  flagcxDevApiBackend captureBackend = {};
  captureBackend.name = "capture-provenance";
  captureBackend.devMemCreate = captureDevMemCreate;
  captureBackend.devMemDestroy = captureDevMemDestroy;
  DevApiBackendGuard backendGuard(&captureBackend);
  devMemCreateCalls = 0;

  flagcxComm comm = {};
  flagcxDevMem_t devMem = nullptr;
  EXPECT_EQ(flagcxDevMemCreate(&comm, static_cast<char *>(allocation) + 192, 65,
                               nullptr, &devMem),
            flagcxInvalidUsage);
  EXPECT_EQ(devMem, nullptr);
  EXPECT_EQ(devMemCreateCalls, 0);
  free(allocation);
}

#ifndef FLAGCX_TEST_ALLOCATOR_SHMEM
TEST(DevMemProvenanceTest, DefaultBackendAllowsExternalCclUserBuffer) {
  void *allocation = malloc(64);
  ASSERT_NE(allocation, nullptr);

  flagcxDevApiBackend captureBackend = {};
  captureBackend.name = "capture-provenance";
  captureBackend.devMemCreate = captureDevMemCreate;
  captureBackend.devMemDestroy = captureDevMemDestroy;
  DevApiBackendGuard backendGuard(&captureBackend);
  devMemCreateCalls = 0;
  capturedTracked = true;

  flagcxComm comm = {};
  flagcxDevMem_t devMem = nullptr;
  EXPECT_EQ(flagcxDevMemCreate(&comm, allocation, 64, nullptr, &devMem),
            flagcxSuccess);
  ASSERT_NE(devMem, nullptr);
  EXPECT_EQ(devMemCreateCalls, 1);
  EXPECT_FALSE(capturedTracked);
  EXPECT_EQ(flagcxDevMemDestroy(&comm, devMem), flagcxSuccess);
  free(allocation);
}
#endif

} // namespace
