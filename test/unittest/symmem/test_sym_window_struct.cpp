// Unit tests for symmetric memory structs and parameter validation.
// No MPI or GPU required — runs locally.
// Links against libflagcx.

#include <gtest/gtest.h>

#include "adaptor.h"
#include "flagcx.h"
#include "flagcx_net_adaptor.h"
#include "mem_alloc_registry.h"
#include "onesided.h"
#include "sym_heap.h"
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>

namespace {

int unmapCalls = 0;
int vaFreeCalls = 0;
int physFreeCalls = 0;
int multicastTeardownCalls = 0;
int multicastFreeCalls = 0;
int cleanupStep = 0;
int multicastFreeStep = 0;
int flatUnmapStep = 0;
int physFreeStep = 0;
int vaRegCalls = 0;
void *vaRegBuffer = nullptr;
size_t vaRegSize = 0;
int dmaBufRegCalls = 0;
int deregMrCalls = 0;
int dmaSupportCalls = 0;
int dmaBufRegPtrType = -1;
uint64_t dmaBufRegOffset = UINT64_MAX;
void *dmaBufExportBase = nullptr;
size_t dmaBufExportSize = 0;
void *mockRangeBase = nullptr;
size_t mockRangeSize = 0;
void *mockSegmentRangeBase = nullptr;
size_t mockSegmentRangeSize = 0;
int multicastProviderRefs = 0;
int multicastObjectDestroyCalls = 0;

struct flagcxNetAdaptor makeVmmCapableTestNet() {
  struct flagcxNetAdaptor net = {};
  net.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  return net;
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    if (old != nullptr) {
      hadOldValue_ = true;
      oldValue_ = old;
    }
    set(value);
  }

  void set(const char *value) {
    if (value != nullptr)
      setenv(name_, value, 1);
    else
      unsetenv(name_);
  }

  ~ScopedEnvVar() {
    if (hadOldValue_)
      setenv(name_, oldValue_.c_str(), 1);
    else
      unsetenv(name_);
  }

private:
  const char *name_;
  bool hadOldValue_ = false;
  std::string oldValue_;
};

class ScopedTwoPageRange {
public:
  ScopedTwoPageRange() {
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
      return;
    pageSize_ = static_cast<size_t>(page);
    if (posix_memalign(&allocation_, pageSize_, pageSize_ * 2) != 0)
      return;
    mockSegmentRangeBase = allocation_;
    mockSegmentRangeSize = pageSize_ * 2;
  }

  ~ScopedTwoPageRange() {
    mockSegmentRangeBase = nullptr;
    mockSegmentRangeSize = 0;
    free(allocation_);
  }

  bool valid() const { return allocation_ != nullptr; }
  void *crossingBuffer() const {
    return static_cast<char *>(allocation_) + pageSize_ - 32;
  }
  void *firstPageBuffer() const {
    return static_cast<char *>(allocation_) + 123;
  }
  void *base() const { return allocation_; }
  size_t pageSize() const { return pageSize_; }

private:
  void *allocation_ = nullptr;
  size_t pageSize_ = 0;
};

flagcxResult_t failUnmapOnce(void *, size_t, int) {
  unmapCalls++;
  return unmapCalls == 1 ? flagcxRemoteError : flagcxSuccess;
}

flagcxResult_t failVaFreeOnce(void *, size_t, int) {
  vaFreeCalls++;
  return vaFreeCalls == 1 ? flagcxRemoteError : flagcxSuccess;
}

flagcxResult_t countFlatVaFree(void *, size_t, int) {
  vaFreeCalls++;
  return flagcxSuccess;
}

flagcxResult_t failSegmentUnmapOnce(void *) {
  unmapCalls++;
  return unmapCalls == 1 ? flagcxRemoteError : flagcxSuccess;
}

flagcxResult_t failSegmentVaFreeOnce(void *) {
  vaFreeCalls++;
  return vaFreeCalls == 1 ? flagcxRemoteError : flagcxSuccess;
}

flagcxResult_t countSegmentBundleFree(void *) {
  physFreeCalls++;
  return flagcxSuccess;
}

flagcxResult_t countMulticastVaFree(void *, size_t) {
  vaFreeCalls++;
  return flagcxSuccess;
}

flagcxResult_t countPhysFree(void *) {
  physFreeCalls++;
  physFreeStep = ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t failMulticastTeardownOnce(void *, size_t) {
  multicastTeardownCalls++;
  if (multicastTeardownCalls == 1)
    return flagcxRemoteError;
  ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t countMulticastFree(void *) {
  multicastFreeCalls++;
  multicastFreeStep = ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t retainMockMulticastHandle(int, void **handle) {
  if (handle == nullptr)
    return flagcxInvalidArgument;
  *handle = malloc(1);
  if (*handle == nullptr)
    return flagcxSystemError;
  multicastProviderRefs++;
  return flagcxSuccess;
}

flagcxResult_t releaseMockMulticastHandle(void *handle) {
  if (handle == nullptr)
    return flagcxSuccess;
  free(handle);
  multicastProviderRefs--;
  if (multicastProviderRefs == 0)
    multicastObjectDestroyCalls++;
  return flagcxSuccess;
}

flagcxResult_t teardownMockMulticastMapping(void *, size_t) {
  multicastTeardownCalls++;
  return flagcxSuccess;
}

flagcxResult_t countFlatUnmap(void *, size_t, int) {
  unmapCalls++;
  flatUnmapStep = ++cleanupStep;
  return flagcxSuccess;
}

flagcxResult_t mockNetProperties(int, void *properties) {
  auto *props = static_cast<flagcxNetProperties_t *>(properties);
  memset(props, 0, sizeof(*props));
  props->ptrSupport = FLAGCX_PTR_CUDA | FLAGCX_PTR_DMABUF;
  return flagcxSuccess;
}

flagcxResult_t mockNetPropertiesUnsupported(int, void *) {
  return flagcxNotSupported;
}

flagcxResult_t mockNetPropertiesSystemError(int, void *) {
  return flagcxSystemError;
}

flagcxResult_t mockNetPropertiesDeviceError(int, void *) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t mockDmaSupport(bool *supported) {
  dmaSupportCalls++;
  if (supported == nullptr)
    return flagcxInvalidArgument;
  *supported = true;
  return flagcxSuccess;
}

flagcxResult_t mockDmaSupportFailure(bool *) {
  dmaSupportCalls++;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t mockDmaBufExportUnsupported(void *, void *, size_t,
                                           unsigned long long) {
  return flagcxNotSupported;
}

flagcxResult_t mockDmaBufExportFailure(void *, void *, size_t,
                                       unsigned long long) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t mockDmaBufExportSuccess(void *handleOut, void *buffer,
                                       size_t size, unsigned long long) {
  if (handleOut == nullptr)
    return flagcxInvalidArgument;
  int fd = open("/dev/null", O_RDONLY);
  if (fd < 0)
    return flagcxSystemError;
  *static_cast<int *>(handleOut) = fd;
  dmaBufExportBase = buffer;
  dmaBufExportSize = size;
  return flagcxSuccess;
}

flagcxResult_t mockAddressRange(const void *ptr, void **base, size_t *size) {
  if (ptr == nullptr || base == nullptr || size == nullptr)
    return flagcxInvalidArgument;
  if (mockRangeBase != nullptr) {
    *base = mockRangeBase;
    *size = mockRangeSize;
    return flagcxSuccess;
  }
  long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0)
    return flagcxSystemError;
  uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  *base = reinterpret_cast<void *>(address - address % pageSize);
  *size = static_cast<size_t>(pageSize) * 2;
  return flagcxSuccess;
}

flagcxResult_t mockPhysicalPageRange(const void *ptr, void **base,
                                     size_t *size) {
  if (ptr == nullptr || base == nullptr || size == nullptr)
    return flagcxInvalidArgument;
  if (mockSegmentRangeBase == nullptr)
    return flagcxNotSupported;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t rangeBase = reinterpret_cast<uintptr_t>(mockSegmentRangeBase);
  if (address < rangeBase || address - rangeBase >= mockSegmentRangeSize)
    return flagcxNotSupported;
  const size_t page = mockSegmentRangeSize / 2;
  *base = reinterpret_cast<void *>(rangeBase +
                                   ((address - rangeBase) / page) * page);
  *size = page;
  return flagcxSuccess;
}

flagcxResult_t mockVaRegMr(void *, void *buffer, size_t size, int, int,
                           void **handle) {
  vaRegCalls++;
  vaRegBuffer = buffer;
  vaRegSize = size;
  if (handle != nullptr)
    *handle = reinterpret_cast<void *>(0x1234);
  return flagcxSuccess;
}

flagcxResult_t mockCountPhysicalPages(const void *ptr, size_t size,
                                      int *count) {
  if (ptr == nullptr || size == 0 || count == nullptr ||
      mockSegmentRangeBase == nullptr)
    return flagcxInvalidArgument;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t base = reinterpret_cast<uintptr_t>(mockSegmentRangeBase);
  if (address < base || address - base >= mockSegmentRangeSize ||
      size > mockSegmentRangeSize - (address - base))
    return flagcxInvalidArgument;
  const size_t page = mockSegmentRangeSize / 2;
  const size_t start = (address - base) / page;
  const size_t end = (address - base + size - 1) / page;
  *count = static_cast<int>(end - start + 1);
  return flagcxSuccess;
}

flagcxResult_t mockVaRegMrFailure(void *, void *, size_t, int, int,
                                  void **handle) {
  vaRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxSystemError;
}

flagcxResult_t mockVaRegMrSuccessWithoutHandle(void *, void *, size_t, int, int,
                                               void **handle) {
  vaRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxSuccess;
}

flagcxResult_t mockDmaBufRegMr(void *, void *, size_t, int ptrType,
                               uint64_t offset, int, int, void **handle) {
  dmaBufRegCalls++;
  dmaBufRegPtrType = ptrType;
  dmaBufRegOffset = offset;
  if (handle != nullptr)
    *handle = reinterpret_cast<void *>(0x5678);
  return flagcxSuccess;
}

flagcxResult_t mockDmaBufRegMrUnsupported(void *, void *, size_t, int, uint64_t,
                                          int, int, void **handle) {
  dmaBufRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxNotSupported;
}

flagcxResult_t mockDmaBufRegMrPartialUnsupported(void *, void *, size_t, int,
                                                 uint64_t, int, int,
                                                 void **handle) {
  dmaBufRegCalls++;
  if (handle != nullptr)
    *handle = reinterpret_cast<void *>(0x5678);
  return flagcxNotSupported;
}

flagcxResult_t mockDeregMr(void *, void *handle) {
  deregMrCalls++;
  return handle == reinterpret_cast<void *>(0x5678) ? flagcxSuccess
                                                    : flagcxInvalidArgument;
}

flagcxResult_t mockDeregMrFailure(void *, void *) {
  deregMrCalls++;
  return flagcxRemoteError;
}

flagcxResult_t mockDmaBufRegMrFailure(void *, void *, size_t, int, uint64_t,
                                      int, int, void **handle) {
  dmaBufRegCalls++;
  if (handle != nullptr)
    *handle = nullptr;
  return flagcxSystemError;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Struct layout tests — verify fields exist and have expected types
// ---------------------------------------------------------------------------

TEST(SymWindowStruct, WindowStructLayout) {
  struct flagcxWindow win;
  memset(&win, 0, sizeof(win));

  // Verify fields exist and are accessible
  win.vendorBase = nullptr;
  win.defaultBase = nullptr;
  win.isSymmetricDefault = 0;

  EXPECT_EQ(win.vendorBase, nullptr);
  EXPECT_EQ(win.defaultBase, nullptr);
  EXPECT_EQ(win.isSymmetricDefault, 0);
}

TEST(SymWindowStruct, SymWindowStructLayout) {
  struct flagcxSymWindow sw;
  memset(&sw, 0, sizeof(sw));

  // Verify all fields exist
  sw.flatBase = nullptr;
  sw.mcBase = nullptr;
  sw.mrIndex = -1;
  sw.mrBase = 0;
  sw.heapSize = 1024;
  sw.allocSize = 2048;
  sw.localRanks = 2;
  sw.physHandle = nullptr;
  sw.mcHandle = nullptr;
  sw.isVmmAllocation = true;
  sw.hasFlatMapping = false;

  EXPECT_EQ(sw.mrIndex, -1);
  EXPECT_EQ(sw.heapSize, 1024u);
  EXPECT_EQ(sw.allocSize, 2048u);
  EXPECT_EQ(sw.localRanks, 2);
  EXPECT_TRUE(sw.isVmmAllocation);
  EXPECT_FALSE(sw.hasFlatMapping);
}

// ---------------------------------------------------------------------------
// 2. Flag constants
// ---------------------------------------------------------------------------

TEST(SymWindowStruct, WindowFlagConstants) {
  EXPECT_EQ(FLAGCX_WIN_DEFAULT, 0x00);
  EXPECT_EQ(FLAGCX_WIN_COLL_SYMMETRIC, 0x01);
}

TEST(SymWindowStruct, LegacyIpcRequiresNonVmmAllocationWithoutFlatMapping) {
  flagcxSymWindow window = {};

  EXPECT_TRUE(flagcxSymWindowCanUseLegacyIpc(&window));
  window.isVmmAllocation = true;
  EXPECT_FALSE(flagcxSymWindowCanUseLegacyIpc(&window));
  window.isVmmAllocation = false;
  window.hasFlatMapping = true;
  EXPECT_FALSE(flagcxSymWindowCanUseLegacyIpc(&window));
  EXPECT_FALSE(flagcxSymWindowCanUseLegacyIpc(nullptr));
}

TEST(SymWindowMrRoute, PrefersDmaBufWhenEveryLayerSupportsIt) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF |
                                              FLAGCX_VMM_MR_CAP_VA,
                                          FLAGCX_PTR_DMABUF | FLAGCX_PTR_CUDA,
                                          /*dmaBufExportSupported=*/true,
                                          /*hasDmaBufRegistration=*/true,
                                          /*hasVaRegistration=*/true,
                                          /*dmaBufEnabled=*/true),
            FLAGCX_VMM_MR_ROUTE_DMABUF);
}

TEST(SymWindowMrRoute, FallsBackToVaWhenDmaBufExportIsUnavailable) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF |
                                              FLAGCX_VMM_MR_CAP_VA,
                                          FLAGCX_PTR_DMABUF | FLAGCX_PTR_CUDA,
                                          /*dmaBufExportSupported=*/false,
                                          /*hasDmaBufRegistration=*/true,
                                          /*hasVaRegistration=*/true),
            FLAGCX_VMM_MR_ROUTE_VA);
}

TEST(SymWindowMrRoute, FallsBackToVaWhenProviderHasNoDmaBufEntryPoint) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF |
                                              FLAGCX_VMM_MR_CAP_VA,
                                          FLAGCX_PTR_DMABUF | FLAGCX_PTR_CUDA,
                                          /*dmaBufExportSupported=*/true,
                                          /*hasDmaBufRegistration=*/false,
                                          /*hasVaRegistration=*/true),
            FLAGCX_VMM_MR_ROUTE_VA);
}

TEST(SymWindowMrRoute, RejectsVmmWhenNeitherRouteIsUsable) {
  EXPECT_EQ(flagcxOneSideSelectVmmMrRoute(FLAGCX_VMM_MR_CAP_DMABUF,
                                          FLAGCX_PTR_DMABUF,
                                          /*dmaBufExportSupported=*/false,
                                          /*hasDmaBufRegistration=*/true,
                                          /*hasVaRegistration=*/true),
            FLAGCX_VMM_MR_ROUTE_NONE);
}

TEST(SymWindowMrRoute, OrdinaryDeviceAllocationPrefersDmaBufWhenEnabled) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;
  testDevice.symPhysSegmentCount = nullptr;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = dmaSupportCalls = 0;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    false, FLAGCX_NET_MR_FLAG_NONE, &mrHandle,
                                    &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, OrdinaryDeviceAllocationFallsBackToVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;
  testDevice.symPhysSegmentCount = nullptr;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportUnsupported;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = 0;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    false, FLAGCX_NET_MR_FLAG_NONE, &mrHandle,
                                    &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, OrdinaryDmaBufExportErrorDoesNotFallBackToVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;
  testDevice.symPhysSegmentCount = nullptr;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = 0;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    false, FLAGCX_NET_MR_FLAG_NONE, &mrHandle,
                                    &route),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 0);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, OrdinaryDmaBufPartialMrIsCleanedBeforeVaFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;
  testDevice.symPhysSegmentCount = nullptr;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrPartialUnsupported;
  testNet.deregMr = mockDeregMr;
  comm.netAdaptor = &testNet;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = deregMrCalls = 0;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    false, FLAGCX_NET_MR_FLAG_NONE, &mrHandle,
                                    &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(vaRegCalls, 1);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, PreservesNetPropertyQueryResults) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  deviceAdaptor = &testDevice;

  using GetPropertiesFn = flagcxResult_t (*)(int, void *);
  struct TestCase {
    const char *name;
    GetPropertiesFn getProperties;
    flagcxResult_t expected;
  };
  const TestCase cases[] = {
      {"missing", nullptr, flagcxNotSupported},
      {"unsupported", mockNetPropertiesUnsupported, flagcxNotSupported},
      {"system-error", mockNetPropertiesSystemError, flagcxSystemError},
      {"device-error", mockNetPropertiesDeviceError,
       flagcxUnhandledDeviceError},
  };

  char buffer[64] = {};
  for (const auto &testCase : cases) {
    SCOPED_TRACE(testCase.name);
    struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
    struct flagcxHeteroComm comm = {};
    testNet.getProperties = testCase.getProperties;
    testNet.regMr = mockVaRegMr;
    testNet.regMrDmaBuf = mockDmaBufRegMr;
    comm.netAdaptor = &testNet;
    comm.netDev = 0;
    void *mrHandle = reinterpret_cast<void *>(0x1);
    flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_VA;
    vaRegCalls = 0;
    dmaBufRegCalls = 0;
    dmaSupportCalls = 0;

    EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                      buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                      /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                      &mrHandle, &route,
                                      /*mrRangeSegmentCount=*/1),
              testCase.expected);
    EXPECT_EQ(mrHandle, nullptr);
    EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
    EXPECT_EQ(vaRegCalls, 0);
    EXPECT_EQ(dmaBufRegCalls, 0);
    EXPECT_EQ(dmaSupportCalls, 0);
  }

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, ProviderCapabilitiesSuppressUnvalidatedVmmRoutes) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = reinterpret_cast<void *>(0x1);
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_VA;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.vmmMrCaps = FLAGCX_VMM_MR_CAP_NONE;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxNotSupported);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, DisabledDmaBufSkipsProbeAndRegistration) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "0");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, DeviceCapabilitiesCanRejectUnsafeVaRoute) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "0");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = reinterpret_cast<void *>(0x1);
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_DMABUF;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxNotSupported);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, DmaBufUnsupportedWithoutVaDoesNotFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrUnsupported;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxNotSupported);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, DmaBufProviderErrorDoesNotFallbackToVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrFailure;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSystemError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, PartialDmaBufMrIsReleasedBeforeVaFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrPartialUnsupported;
  testNet.deregMr = mockDeregMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  deregMrCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(vaRegCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, PartialDmaBufMrCleanupFailureSuppressesVaFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrPartialUnsupported;
  testNet.deregMr = mockDeregMrFailure;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  deregMrCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxRemoteError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(vaRegCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, DmaBufSubrangeExportsAllocationAndUsesPageOffset) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  long pageSizeResult = sysconf(_SC_PAGESIZE);
  ASSERT_GT(pageSizeResult, 0);
  const size_t pageSize = static_cast<size_t>(pageSizeResult);
  void *allocation = nullptr;
  ASSERT_EQ(posix_memalign(&allocation, pageSize, pageSize * 3), 0);

  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  auto *subrange = static_cast<char *>(allocation) + pageSize + 123;
  constexpr size_t subrangeSize = 257;
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufRegOffset = UINT64_MAX;
  dmaBufExportBase = nullptr;
  dmaBufExportSize = 0;
  mockRangeBase = allocation;
  mockRangeSize = pageSize * 3;
  flagcxMemAllocationInfo tracked = {allocation, pageSize * 2 + 333,
                                     flagcxMemCCL, flagcxMemAllocBackendNative,
                                     true};
  ASSERT_EQ(globalMemAllocRegistry.insert(tracked), flagcxSuccess);
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    subrange, subrangeSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(dmaBufExportBase, allocation);
  EXPECT_EQ(dmaBufExportSize, pageSize * 3);
  EXPECT_EQ(dmaBufRegOffset, pageSize);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);

  deviceAdaptor = savedDevice;
  EXPECT_EQ(globalMemAllocRegistry.erase(allocation), flagcxSuccess);
  mockRangeBase = nullptr;
  mockRangeSize = 0;
  free(allocation);
}

TEST(SymWindowMrRoute, VaProviderErrorIsPreserved) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "0");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMrFailure;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSystemError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, ProviderSuccessWithoutHandleIsInternalError) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "0");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = reinterpret_cast<void *>(0x1);
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_VA;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMrSuccessWithoutHandle;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxInternalError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, EnabledDmaBufUsesDmaBufRoute) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, ExportNotSupportedFallsBackToValidatedVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportUnsupported;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufRegPtrType = -1;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 1);
  EXPECT_EQ(dmaBufRegPtrType, -1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, SegmentedVmmUsesWholeRangeDmaBuf) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedTwoPageRange allocation;
  ASSERT_TRUE(allocation.valid());
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  void *buffer = allocation.crossingBuffer();
  const size_t bufferSize = 64;
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockPhysicalPageRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testDevice.symPhysSegmentCount = mockCountPhysicalPages;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufExportBase = nullptr;
  dmaBufExportSize = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);
  EXPECT_EQ(dmaBufExportSize, allocation.pageSize() * 2);
  EXPECT_EQ(dmaBufExportBase, allocation.base());
  EXPECT_EQ(dmaBufRegOffset, 0u);

  routeMode.set("1");
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSuccess);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 2);

  routeMode.set("0");
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  vaRegBuffer = nullptr;
  vaRegSize = 0;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSuccess);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(vaRegBuffer, buffer);
  EXPECT_EQ(vaRegSize, bufferSize);
  EXPECT_EQ(dmaBufRegCalls, 2);
  EXPECT_EQ(dmaSupportCalls, 2);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, SegmentedVmmFallsBackToWholeVaRange) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedTwoPageRange allocation;
  ASSERT_TRUE(allocation.valid());
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  void *buffer = allocation.crossingBuffer();
  const size_t bufferSize = 64;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.symPhysSegmentCount = mockCountPhysicalPages;
  testDevice.getAddressRange = mockPhysicalPageRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportUnsupported;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = 0;
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSuccess);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);

  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.regMrDmaBuf = mockDmaBufRegMrUnsupported;
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSuccess);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(vaRegCalls, 2);
  EXPECT_EQ(dmaBufRegCalls, 1);

  testNet.regMrDmaBuf = mockDmaBufRegMrFailure;
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSystemError);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(vaRegCalls, 2);
  EXPECT_EQ(dmaBufRegCalls, 2);

  testNet.regMrDmaBuf = nullptr;
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSuccess);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(vaRegCalls, 3);
  EXPECT_EQ(dmaBufRegCalls, 2);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, SegmentedVmmCleansPartialDmaBufMrBeforeVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedTwoPageRange allocation;
  ASSERT_TRUE(allocation.valid());
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  void *buffer = allocation.crossingBuffer();
  const size_t bufferSize = 64;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.symPhysSegmentCount = mockCountPhysicalPages;
  testDevice.getAddressRange = mockPhysicalPageRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMrPartialUnsupported;
  testNet.deregMr = mockDeregMr;
  comm.netAdaptor = &testNet;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = deregMrCalls = 0;
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA, true,
                                    FLAGCX_NET_MR_FLAG_NONE, &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(deregMrCalls, 1);
  EXPECT_EQ(vaRegCalls, 1);

  testNet.deregMr = mockDeregMrFailure;
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA, true,
                                    FLAGCX_NET_MR_FLAG_NONE, &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxRemoteError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(dmaBufRegCalls, 2);
  EXPECT_EQ(deregMrCalls, 2);
  EXPECT_EQ(vaRegCalls, 1);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, SegmentedVmmVaErrorAndCapabilityGate) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedTwoPageRange allocation;
  ASSERT_TRUE(allocation.valid());
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "0");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  void *buffer = allocation.crossingBuffer();
  const size_t bufferSize = 64;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.symPhysSegmentCount = mockCountPhysicalPages;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMrFailure;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = dmaSupportCalls = 0;
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA, true,
                                    FLAGCX_NET_MR_FLAG_NONE, &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxSystemError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  testNet.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA, true,
                                    FLAGCX_NET_MR_FLAG_NONE, &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxNotSupported);
  EXPECT_EQ(vaRegCalls, 1);
  testNet.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA, true,
                                    FLAGCX_NET_MR_FLAG_NONE, &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/2),
            flagcxNotSupported);
  EXPECT_EQ(vaRegCalls, 1);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, SegmentedVmmRejectsLegacyV1Adaptor) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};

  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  deviceAdaptor = &testDevice;

  for (int legacyKind = 0; legacyKind < 2; ++legacyKind) {
    testDevice.internalFlags = legacyKind == 0
                                   ? FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1
                                   : FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
    testNet.internalFlags = legacyKind == 1
                                ? FLAGCX_NET_ADAPTOR_INTERNAL_LEGACY_V1
                                : FLAGCX_NET_ADAPTOR_INTERNAL_NONE;
    vaRegCalls = 0;
    dmaBufRegCalls = 0;
    void *mrHandle = nullptr;
    flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

    EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                      buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                      /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                      &mrHandle, &route,
                                      /*mrRangeSegmentCount=*/2),
              flagcxNotSupported);
    EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
    EXPECT_EQ(mrHandle, nullptr);
    EXPECT_EQ(vaRegCalls, 0);
    EXPECT_EQ(dmaBufRegCalls, 0);

    routeMode.set("1");
    mrHandle = nullptr;
    route = FLAGCX_VMM_MR_ROUTE_NONE;
    EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                      buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                      /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                      &mrHandle, &route,
                                      /*mrRangeSegmentCount=*/2),
              flagcxNotSupported);
    EXPECT_EQ(mrHandle, nullptr);
    EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
    EXPECT_EQ(vaRegCalls, 0);
    EXPECT_EQ(dmaBufRegCalls, 0);
    routeMode.set("1");
  }

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, PublicRegistrationDetectsSegmentedAllocation) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedTwoPageRange allocation;
  ASSERT_TRUE(allocation.valid());
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  void *buffer = allocation.crossingBuffer();
  const size_t bufferSize = 64;
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.symPhysSegmentCount = mockCountPhysicalPages;
  testDevice.getAddressRange = mockPhysicalPageRange;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = dmaBufRegCalls = dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/false, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);

  routeMode.set("0");
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/false, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 1);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, OneSegmentSubrangeOfTwoSegmentAllocation) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedTwoPageRange allocation;
  ASSERT_TRUE(allocation.valid());
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  void *buffer = allocation.firstPageBuffer();
  const size_t bufferSize = 257;
  int rangeCount = 0;
  ASSERT_EQ(mockCountPhysicalPages(buffer, bufferSize, &rangeCount),
            flagcxSuccess);
  ASSERT_EQ(rangeCount, 1);

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.symPhysSegmentCount = mockCountPhysicalPages;
  testDevice.getAddressRange = mockPhysicalPageRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testDevice.dmaSupport = mockDmaSupport;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  deviceAdaptor = &testDevice;
  vaRegCalls = dmaBufRegCalls = 0;
  dmaBufExportBase = nullptr;
  dmaBufExportSize = 0;

  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/false, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(dmaBufExportBase, allocation.base());
  EXPECT_EQ(dmaBufExportSize, allocation.pageSize());
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(vaRegCalls, 0);

  dmaBufEnabled.set("0");
  mrHandle = nullptr;
  route = FLAGCX_VMM_MR_ROUTE_NONE;
  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, bufferSize, FLAGCX_PTR_CUDA,
                                    /*isVmm=*/false, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(vaRegCalls, 1);
  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, LegacyV1ExportSuccessUsesDmaBuf) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_NONE;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufRegPtrType = -1;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 0);
  EXPECT_EQ(dmaBufRegPtrType, FLAGCX_PTR_CUDA);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, LegacyV1ExportFailureFallsBackToVa) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_NONE;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_VA);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x1234));
  EXPECT_EQ(vaRegCalls, 1);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 0);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, LegacyNetV1PreservesHistoricalRouteSelection) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_NONE;
  testDevice.dmaSupport = mockDmaSupportFailure;
  testDevice.getHandleForAddressRange = mockDmaBufExportSuccess;
  testNet.internalFlags = FLAGCX_NET_ADAPTOR_INTERNAL_LEGACY_V1;
  testNet.vmmMrCaps = FLAGCX_VMM_MR_CAP_NONE;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  dmaBufRegPtrType = -1;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxSuccess);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_DMABUF);
  EXPECT_EQ(mrHandle, reinterpret_cast<void *>(0x5678));
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 1);
  EXPECT_EQ(dmaSupportCalls, 0);
  EXPECT_EQ(dmaBufRegPtrType, FLAGCX_PTR_CUDA);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrRoute, LatestExportDeviceErrorDoesNotFallback) {
  ASSERT_NE(deviceAdaptor, nullptr);
  ScopedEnvVar dmaBufEnabled("FLAGCX_DMABUF_ENABLE", "1");
  struct flagcxDeviceAdaptor testDevice = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  struct flagcxNetAdaptor testNet = makeVmmCapableTestNet();
  struct flagcxHeteroComm comm = {};
  char buffer[64] = {};
  void *mrHandle = nullptr;
  flagcxVmmMrRoute_t route = FLAGCX_VMM_MR_ROUTE_NONE;

  testDevice.internalFlags = FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE;
  testDevice.vmmMrCaps = FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA;
  testDevice.dmaSupport = mockDmaSupport;
  testDevice.getAddressRange = mockAddressRange;
  testDevice.getHandleForAddressRange = mockDmaBufExportFailure;
  testNet.getProperties = mockNetProperties;
  testNet.regMr = mockVaRegMr;
  testNet.regMrDmaBuf = mockDmaBufRegMr;
  comm.netAdaptor = &testNet;
  comm.netDev = 0;
  vaRegCalls = 0;
  dmaBufRegCalls = 0;
  dmaSupportCalls = 0;
  deviceAdaptor = &testDevice;

  EXPECT_EQ(flagcxOneSideRegisterMr(&comm, reinterpret_cast<void *>(0x1),
                                    buffer, sizeof(buffer), FLAGCX_PTR_CUDA,
                                    /*isVmm=*/true, FLAGCX_NET_MR_FLAG_NONE,
                                    &mrHandle, &route,
                                    /*mrRangeSegmentCount=*/1),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(route, FLAGCX_VMM_MR_ROUTE_NONE);
  EXPECT_EQ(mrHandle, nullptr);
  EXPECT_EQ(vaRegCalls, 0);
  EXPECT_EQ(dmaBufRegCalls, 0);
  EXPECT_EQ(dmaSupportCalls, 1);

  deviceAdaptor = savedDevice;
}

TEST(SymWindowMrSlots, SelectsOnlySlotFreeOnEveryRank) {
  const uint8_t occupancy[] = {
      1, 0, 0, // rank 0 released slot 1
      1, 1, 0, // rank 1 retained slot 1 after deregMr failed
  };
  int slot = -1;
  EXPECT_EQ(flagcxOneSideSelectCommonPublishSlot(occupancy, 2, 3, &slot),
            flagcxSuccess);
  EXPECT_EQ(slot, 2);
}

TEST(SymWindowMrSlots, AppendsWhenNoCommonHoleExists) {
  const uint8_t occupancy[] = {
      1, 0, 1, // rank 0
      1, 1, 0, // rank 1
  };
  int slot = -1;
  EXPECT_EQ(flagcxOneSideSelectCommonPublishSlot(occupancy, 2, 3, &slot),
            flagcxSuccess);
  EXPECT_EQ(slot, 3);
}

TEST(SymWindowMrSlots, RejectsDivergentConnectionOwner) {
  const uint8_t occupancy[] = {
      0, 0, // rank 0
      1, 0, // rank 1
  };
  int slot = -1;
  EXPECT_EQ(flagcxOneSideSelectCommonPublishSlot(occupancy, 2, 2, &slot),
            flagcxInvalidUsage);
}

TEST(SymWindowRoutes, RequiresAUsablePathForEveryPeer) {
  char allocation[256] = {};
  uintptr_t baseVas[] = {reinterpret_cast<uintptr_t>(allocation), 0, 0, 0};
  size_t regionSizes[] = {sizeof(allocation), 0, 0, 0};
  struct flagcxOneSideHandleInfo handle = {};
  handle.baseVas = baseVas;
  handle.regionSizes = regionSizes;
  struct flagcxOneSideHandleInfo *handles[] = {&handle};

  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 4;
  comm.localRanks = 2;
  comm.oneSideHandles = handles;
  comm.oneSideHandleCount = 1;

  struct flagcxSymWindow window = {};
  window.localBase = allocation;
  window.heapSize = sizeof(allocation);
  window.mrIndex = -1;
  window.ipcSlot = -1;

  // Flat mappings are node-local and cannot satisfy remote peers.
  window.hasFlatMapping = true;
  window.flatBase = reinterpret_cast<void *>(0x1000);
  EXPECT_EQ(flagcxSymWindowValidateDataRoutes(&comm, &window),
            flagcxNotSupported);

  window.hasNetworkMrRef = true;
  window.mrIndex = 0;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutes(&comm, &window), flagcxSuccess);

  // A local-only communicator can use flat or IPC mappings without an MR.
  comm.nRanks = 2;
  comm.localRanks = 2;
  window.hasNetworkMrRef = false;
  window.mrIndex = -1;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, true),
            flagcxSuccess);
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxNotSupported);
  window.hasFlatMapping = false;
  window.flatBase = nullptr;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, true),
            flagcxNotSupported);
  window.ipcSlot = 3;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, true),
            flagcxSuccess);
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxNotSupported);

  // A valid NET MR remains usable when local P2P transports are disabled.
  window.hasNetworkMrRef = true;
  window.mrIndex = 0;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxSuccess);

  // With no peer, the local allocation itself is sufficient.
  comm.nRanks = 1;
  comm.localRanks = 1;
  window.ipcSlot = -1;
  window.hasNetworkMrRef = false;
  window.mrIndex = -1;
  EXPECT_EQ(flagcxSymWindowValidateDataRoutesForMode(&comm, &window, false),
            flagcxSuccess);
}

TEST(SymWindowMrRoute, UsesRecordedVmmProvenanceForPublicRegistration) {
  char allocation[256] = {};
  flagcxMemAllocationInfo info = {allocation, sizeof(allocation), flagcxMemCCL,
                                  flagcxMemAllocBackendNative, true};
  ASSERT_EQ(globalMemAllocRegistry.insert(info), flagcxSuccess);
  EXPECT_TRUE(flagcxOneSideRegistryRangeIsVmm(allocation + 32, 64));
  EXPECT_FALSE(flagcxOneSideRegistryRangeIsVmm(allocation + 240, 32));
  EXPECT_EQ(globalMemAllocRegistry.erase(allocation), flagcxSuccess);
  EXPECT_FALSE(flagcxOneSideRegistryRangeIsVmm(allocation, sizeof(allocation)));
}

// ---------------------------------------------------------------------------
// 3. Parameter validation — NULL / zero args
// ---------------------------------------------------------------------------

TEST(SymWindowValidation, RegisterNullComm) {
  flagcxWindow_t win = nullptr;
  char dummy[64] = {};
  flagcxResult_t res =
      flagcxSymWindowRegister(nullptr, dummy, sizeof(dummy), &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterNullBuff) {
  // Pass a non-null comm placeholder so the test exercises the buff-null check
  // specifically, not the comm-null check. The function checks all args in a
  // single compound condition before dereferencing comm.
  flagcxHeteroComm_t fakeComm = reinterpret_cast<flagcxHeteroComm_t>(0x1);
  flagcxWindow_t win = nullptr;
  flagcxResult_t res =
      flagcxSymWindowRegister(fakeComm, nullptr, 1024, &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterZeroSize) {
  flagcxHeteroComm_t fakeComm = reinterpret_cast<flagcxHeteroComm_t>(0x1);
  flagcxWindow_t win = nullptr;
  char dummy[64] = {};
  flagcxResult_t res = flagcxSymWindowRegister(fakeComm, dummy, 0, &win, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, RegisterNullWinPtr) {
  flagcxHeteroComm_t fakeComm = reinterpret_cast<flagcxHeteroComm_t>(0x1);
  char dummy[64] = {};
  flagcxResult_t res =
      flagcxSymWindowRegister(fakeComm, dummy, sizeof(dummy), nullptr, 0);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(SymWindowValidation, DeregisterNull) {
  // Deregistering a NULL window should be a safe no-op
  flagcxResult_t res = flagcxSymWindowDeregister(nullptr, nullptr);
  EXPECT_EQ(res, flagcxSuccess);
}

TEST(SymWindowValidation, CommWindowDeregisterNull) {
  // Public API: deregistering NULL window should succeed
  flagcxResult_t res = flagcxCommWindowDeregister(nullptr, nullptr);
  EXPECT_EQ(res, flagcxSuccess);
}

TEST(SymWindowOwnership, TeardownFailureRetainsWindowForRetry) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symFlatMappingUnmap = failUnmapOnce;
  testAdaptor.symFlatVaFree = countFlatVaFree;
  testAdaptor.symPhysFree = countPhysFree;
  unmapCalls = 0;
  physFreeCalls = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->hasFlatMapping = true;
  sym->flatBase = reinterpret_cast<void *>(0x1000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x2000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;
  comm->symWindows = sym;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_EQ(comm->symWindows, sym);
  EXPECT_EQ(win->defaultBase, sym);
  EXPECT_EQ(sym->flatBase, reinterpret_cast<void *>(0x1000));
  EXPECT_EQ(sym->physHandle, reinterpret_cast<void *>(0x2000));
  EXPECT_EQ(physFreeCalls, 0);

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(unmapCalls, 2);
  EXPECT_EQ(physFreeCalls, 1);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership, SegmentedCleanupRetainsEachStageForRetry) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symFlatMappingUnmapSegments = failSegmentUnmapOnce;
  testAdaptor.symFlatVaFreeSegments = failSegmentVaFreeOnce;
  testAdaptor.symPhysFreeSegments = countSegmentBundleFree;
  unmapCalls = vaFreeCalls = physFreeCalls = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;
  win->defaultBase = sym;
  sym->owner = win;
  sym->resourcePath = flagcxSymResourceSegments;
  sym->hasFlatMapping = true;
  sym->flatBase = reinterpret_cast<void *>(0x1000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x2000);
  sym->localRanks = 2;
  sym->published = true;
  comm->symWindows = sym;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(vaFreeCalls, 0);
  EXPECT_EQ(physFreeCalls, 0);
  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_EQ(unmapCalls, 2);
  EXPECT_EQ(vaFreeCalls, 1);
  EXPECT_EQ(physFreeCalls, 0);
  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(unmapCalls, 2);
  EXPECT_EQ(vaFreeCalls, 2);
  EXPECT_EQ(physFreeCalls, 1);
  EXPECT_EQ(comm->symWindows, nullptr);
  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership, FlatUnmapIsNotRepeatedWhenVaFreeRetryIsRequired) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symFlatMappingUnmap = countFlatUnmap;
  testAdaptor.symFlatVaFree = failVaFreeOnce;
  testAdaptor.symPhysFree = countPhysFree;
  unmapCalls = 0;
  vaFreeCalls = 0;
  physFreeCalls = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->hasFlatMapping = true;
  sym->flatBase = reinterpret_cast<void *>(0x1000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x2000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;
  comm->symWindows = sym;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_FALSE(sym->flatMappingOwned);
  EXPECT_TRUE(sym->flatVaOwned);
  EXPECT_EQ(sym->flatBase, reinterpret_cast<void *>(0x1000));
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(vaFreeCalls, 1);
  EXPECT_EQ(physFreeCalls, 0);

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(vaFreeCalls, 2);
  EXPECT_EQ(physFreeCalls, 1);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership, LocalTeardownSkipsBootstrapRendezvous) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symFlatMappingUnmap = countFlatUnmap;
  testAdaptor.symFlatVaFree = countFlatVaFree;
  testAdaptor.symPhysFree = countPhysFree;
  unmapCalls = 0;
  physFreeCalls = 0;
  cleanupStep = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;

  // A non-null sentinel proves local teardown does not attempt to use the
  // bootstrap object even though this window describes multiple local ranks.
  comm->bootstrap = reinterpret_cast<bootstrapState *>(0x1);
  comm->symWindows = sym;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->hasFlatMapping = true;
  sym->flatBase = reinterpret_cast<void *>(0x1000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x2000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(physFreeCalls, 1);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership, MulticastTeardownCompletesBeforeObjectRelease) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symMulticastMappingUnmap = failMulticastTeardownOnce;
  testAdaptor.symMulticastVaFree = countMulticastVaFree;
  testAdaptor.symMulticastFree = countMulticastFree;
  testAdaptor.symFlatMappingUnmap = countFlatUnmap;
  testAdaptor.symFlatVaFree = countFlatVaFree;
  testAdaptor.symPhysFree = countPhysFree;
  multicastTeardownCalls = 0;
  multicastFreeCalls = 0;
  unmapCalls = 0;
  physFreeCalls = 0;
  cleanupStep = 0;
  multicastFreeStep = 0;
  flatUnmapStep = 0;
  physFreeStep = 0;

  auto *comm = static_cast<flagcxHeteroComm *>(
      calloc(1, sizeof(struct flagcxHeteroComm)));
  auto *win =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *sym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(sym, nullptr);
  deviceAdaptor = &testAdaptor;
  win->defaultBase = sym;
  win->isSymmetricDefault = 1;
  sym->owner = win;
  sym->hasFlatMapping = true;
  sym->mcBase = reinterpret_cast<void *>(0x1000);
  sym->mcMapSize = 4096;
  sym->multicastMappingOwned = true;
  sym->multicastVaOwned = true;
  sym->mcHandle = reinterpret_cast<void *>(0x2000);
  sym->flatBase = reinterpret_cast<void *>(0x3000);
  sym->flatMappingOwned = true;
  sym->flatVaOwned = true;
  sym->physHandle = reinterpret_cast<void *>(0x4000);
  sym->allocSize = 4096;
  sym->localRanks = 2;
  sym->published = true;
  comm->symWindows = sym;

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxRemoteError);
  EXPECT_EQ(comm->symWindows, sym);
  EXPECT_EQ(multicastTeardownCalls, 1);
  EXPECT_EQ(multicastFreeCalls, 0);
  EXPECT_EQ(unmapCalls, 0);
  EXPECT_EQ(physFreeCalls, 0);

  EXPECT_EQ(flagcxSymWindowDeregister(comm, win, flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(comm->symWindows, nullptr);
  EXPECT_EQ(multicastTeardownCalls, 2);
  EXPECT_EQ(multicastFreeCalls, 1);
  EXPECT_EQ(unmapCalls, 1);
  EXPECT_EQ(physFreeCalls, 1);
  EXPECT_LT(multicastFreeStep, flatUnmapStep);
  EXPECT_LT(flatUnmapStep, physFreeStep);

  deviceAdaptor = savedAdaptor;
  free(comm);
}

TEST(SymWindowOwnership,
     PerRankMulticastReferencesPermitCreatorFirstLocalDestroy) {
  ASSERT_NE(deviceAdaptor, nullptr);
  struct flagcxDeviceAdaptor testAdaptor = *deviceAdaptor;
  struct flagcxDeviceAdaptor *savedAdaptor = deviceAdaptor;
  testAdaptor.symMulticastImport = retainMockMulticastHandle;
  testAdaptor.symMulticastMappingUnmap = teardownMockMulticastMapping;
  testAdaptor.symMulticastVaFree = countMulticastVaFree;
  testAdaptor.symMulticastFree = releaseMockMulticastHandle;
  multicastProviderRefs = 0;
  multicastObjectDestroyCalls = 0;
  multicastTeardownCalls = 0;

  struct flagcxHeteroComm creatorComm = {};
  struct flagcxHeteroComm peerComm = {};
  auto *creatorWin =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *peerWin =
      static_cast<flagcxWindow *>(calloc(1, sizeof(struct flagcxWindow)));
  auto *creatorSym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  auto *peerSym =
      static_cast<flagcxSymWindow *>(calloc(1, sizeof(struct flagcxSymWindow)));
  ASSERT_NE(creatorWin, nullptr);
  ASSERT_NE(peerWin, nullptr);
  ASSERT_NE(creatorSym, nullptr);
  ASSERT_NE(peerSym, nullptr);
  deviceAdaptor = &testAdaptor;

  ASSERT_EQ(testAdaptor.symMulticastImport(7, &creatorSym->mcHandle),
            flagcxSuccess);
  ASSERT_EQ(testAdaptor.symMulticastImport(7, &peerSym->mcHandle),
            flagcxSuccess);
  ASSERT_EQ(multicastProviderRefs, 2);

  creatorWin->defaultBase = creatorSym;
  creatorWin->isSymmetricDefault = 1;
  creatorSym->owner = creatorWin;
  creatorSym->hasFlatMapping = true;
  creatorSym->mcBase = reinterpret_cast<void *>(0x1000);
  creatorSym->mcMapSize = 4096;
  creatorSym->multicastMappingOwned = true;
  creatorSym->multicastVaOwned = true;
  creatorSym->published = true;
  creatorComm.symWindows = creatorSym;

  peerWin->defaultBase = peerSym;
  peerWin->isSymmetricDefault = 1;
  peerSym->owner = peerWin;
  peerSym->hasFlatMapping = true;
  peerSym->mcBase = reinterpret_cast<void *>(0x2000);
  peerSym->mcMapSize = 4096;
  peerSym->multicastMappingOwned = true;
  peerSym->multicastVaOwned = true;
  peerSym->published = true;
  peerComm.symWindows = peerSym;

  EXPECT_EQ(flagcxSymWindowDeregister(&creatorComm, creatorWin,
                                      flagcxSymCleanupLocal),
            flagcxSuccess);
  EXPECT_EQ(multicastTeardownCalls, 1);
  EXPECT_EQ(multicastProviderRefs, 1);
  EXPECT_EQ(multicastObjectDestroyCalls, 0);
  EXPECT_NE(peerSym->mcBase, nullptr);
  EXPECT_NE(peerSym->mcHandle, nullptr);

  EXPECT_EQ(
      flagcxSymWindowDeregister(&peerComm, peerWin, flagcxSymCleanupLocal),
      flagcxSuccess);
  EXPECT_EQ(multicastTeardownCalls, 2);
  EXPECT_EQ(multicastProviderRefs, 0);
  EXPECT_EQ(multicastObjectDestroyCalls, 1);

  deviceAdaptor = savedAdaptor;
}

TEST(SymWindowLookup, PublishThenFindPreservesNonZeroOffsetAndBounds) {
  char allocation[64] = {};
  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 1;
  comm.localRanks = 1;
  struct flagcxWindow win = {};
  struct flagcxSymWindow sym = {};
  win.defaultBase = &sym;
  sym.owner = &win;
  sym.localBase = allocation;
  sym.heapSize = sizeof(allocation);

  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 7, 8, nullptr), nullptr);
  ASSERT_EQ(flagcxSymWindowPublish(&comm, &win), flagcxSuccess);
  EXPECT_TRUE(sym.published);
  EXPECT_EQ(comm.symWindows, &sym);

  size_t offset = 0;
  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 7, 8, &offset), &sym);
  EXPECT_EQ(offset, 7u);
  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 60, 4, &offset), &sym);
  EXPECT_EQ(offset, 60u);
  EXPECT_EQ(flagcxSymWindowFind(&comm, allocation + 60, 5, &offset), nullptr);

  // Publishing is idempotent and must not create a self-referential list.
  EXPECT_EQ(flagcxSymWindowPublish(&comm, &win), flagcxSuccess);
  EXPECT_EQ(sym.next, nullptr);
}

TEST(SymWindowLayout, DifferentPhysicalLayoutsShareOneStride) {
  flagcxSymSegmentMeta segments[] = {{0, 4096}, {4096, 8192}, {0, 8192}};
  flagcxSymRankLayout layouts[] = {{12288, 1024, 10000, 0, 2},
                                   {8192, 512, 7000, 2, 1}};
  size_t stride = 0;
  EXPECT_EQ(flagcxSymValidateLayouts(2, layouts, 3, segments, 4096, &stride),
            flagcxSuccess);
  EXPECT_EQ(stride, 12288u);
}

TEST(SymWindowLayout, MrCountsOnlySegmentsTouchedByUserRange) {
  flagcxSymSegmentMeta segments[] = {{0, 4096}, {4096, 4096}, {8192, 4096}};
  flagcxSymRankLayout layout = {12288, 512, 1024, 0, 3};
  size_t stride = 0;
  ASSERT_EQ(flagcxSymValidateLayouts(1, &layout, 3, segments, 4096, &stride),
            flagcxSuccess);
  int count = 0;
  EXPECT_EQ(flagcxSymCountUserRangeSegments(&layout, segments, &count),
            flagcxSuccess);
  EXPECT_EQ(count, 1);
  layout.userOffset = 3584;
  layout.userSize = 1024;
  EXPECT_EQ(flagcxSymCountUserRangeSegments(&layout, segments, &count),
            flagcxSuccess);
  EXPECT_EQ(count, 2);
}

TEST(SymWindowLayout, RejectsUnmappedUserGap) {
  flagcxSymSegmentMeta segments[] = {{0, 4096}, {8192, 4096}};
  flagcxSymRankLayout layout = {12288, 2048, 8192, 0, 2};
  size_t stride = 0;
  EXPECT_EQ(flagcxSymValidateLayouts(1, &layout, 2, segments, 4096, &stride),
            flagcxInvalidArgument);
  layout.userOffset = 8192;
  layout.userSize = 4096;
  EXPECT_EQ(flagcxSymValidateLayouts(1, &layout, 2, segments, 4096, &stride),
            flagcxSuccess);
}

TEST(SymWindowLayout, RejectsOverflowAndBadSegmentIndex) {
  flagcxSymSegmentMeta segment = {0, 4096};
  flagcxSymRankLayout layout = {4096, 0, 4096, 0, 1};
  size_t stride = 0;
  EXPECT_EQ(flagcxSymValidateLayouts(1, &layout, 1, &segment, 8192, &stride),
            flagcxSuccess);
  EXPECT_EQ(stride, 8192u);
  EXPECT_EQ(flagcxSymValidateLayouts(1, &layout, 1, &segment, 4096, &stride),
            flagcxSuccess);
  layout.firstSegment = 1;
  EXPECT_EQ(flagcxSymValidateLayouts(1, &layout, 1, &segment, 4096, &stride),
            flagcxInvalidArgument);
  layout.firstSegment = 0;
  layout.userOffset = SIZE_MAX;
  EXPECT_EQ(flagcxSymValidateLayouts(1, &layout, 1, &segment, 4096, &stride),
            flagcxInvalidArgument);
}

TEST(SymWindowLayout, LsaPeerAddressUsesWindowAndRankOffsets) {
  flagcxSymRankLayout layout = {12288, 4096, 2048, 0, 2};
  void *ptr = nullptr;
  void *base = reinterpret_cast<void *>(static_cast<uintptr_t>(0x10000000));
  EXPECT_EQ(
      flagcxSymLsaPeerPtr(base, 65536, 16384, 3, 2, &layout, 128, 256, &ptr),
      flagcxSuccess);
  EXPECT_EQ(
      reinterpret_cast<uintptr_t>(ptr),
      static_cast<uintptr_t>(0x10000000 + 2 * 65536 + 16384 + 4096 + 128));
  EXPECT_EQ(flagcxSymLsaPeerPtr(base, 65536, 60000, 3, 2, &layout, 0, 1, &ptr),
            flagcxInvalidArgument);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_EQ(
      flagcxSymLsaPeerPtr(base, 65536, 16384, 3, 2, &layout, 2048, 1, &ptr),
      flagcxInvalidArgument);
  EXPECT_EQ(ptr, nullptr);
}

TEST(SymWindowLookup, ResolveSelfPeerValidatesRangeBeforePointerArithmetic) {
  char allocation[64] = {};
  int rankToNode[] = {0};
  int rankToLocalRank[] = {0};
  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 1;
  comm.node = 0;
  comm.rankToNode = rankToNode;
  comm.rankToLocalRank = rankToLocalRank;

  struct flagcxSymWindow sym = {};
  sym.localBase = allocation;
  sym.heapSize = sizeof(allocation);
  sym.localRanks = 1;

  void *peer = nullptr;
  EXPECT_EQ(flagcxSymWindowResolveIpcPeerPtr(&comm, &sym, 0, 9, 8, &peer),
            flagcxSuccess);
  EXPECT_EQ(peer, static_cast<void *>(allocation + 9));
  peer = reinterpret_cast<void *>(0x1);
  EXPECT_EQ(flagcxSymWindowResolveIpcPeerPtr(&comm, &sym, 0, 60, 5, &peer),
            flagcxInvalidArgument);
  EXPECT_EQ(peer, nullptr);
}

TEST(SymWindowLookup, SegmentedPeerUsesPeerOffsetAndBounds) {
  int rankToNode[] = {0, 0};
  int rankToLocalRank[] = {0, 1};
  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 2;
  comm.node = 0;
  comm.rankToNode = rankToNode;
  comm.rankToLocalRank = rankToLocalRank;

  flagcxSymRankLayout layouts[] = {{8192, 0, 4096, 0, 1},
                                   {12288, 1024, 10000, 1, 2}};
  struct flagcxSymWindow sym = {};
  sym.localBase = reinterpret_cast<void *>(0x100000);
  sym.heapSize = 4096;
  sym.localRanks = 2;
  sym.flatBase = reinterpret_cast<void *>(0x200000);
  sym.hasFlatMapping = true;
  sym.slotStride = 12288;
  sym.rankLayouts = layouts;

  void *peer = nullptr;
  EXPECT_EQ(flagcxSymWindowResolveIpcPeerPtr(&comm, &sym, 1, 9000, 1000, &peer),
            flagcxSuccess);
  EXPECT_EQ(peer, reinterpret_cast<void *>(0x200000 + 12288 + 1024 + 9000));
  EXPECT_EQ(flagcxSymWindowResolveIpcPeerPtr(&comm, &sym, 1, 9000, 1001, &peer),
            flagcxInvalidArgument);
  EXPECT_EQ(peer, nullptr);
}

TEST(SymWindowLookup, PeerUserSizeUsesLocalLayoutAndRemoteMr) {
  char local[64] = {};
  int rankToNode[] = {0, 0};
  int rankToLocalRank[] = {0, 1};
  struct flagcxHeteroComm comm = {};
  comm.rank = 0;
  comm.nRanks = 2;
  comm.node = 0;
  comm.rankToNode = rankToNode;
  comm.rankToLocalRank = rankToLocalRank;
  flagcxSymRankLayout layouts[] = {{4096, 0, 64, 0, 1},
                                   {8192, 128, 7000, 1, 2}};
  struct flagcxSymWindow window = {};
  window.localBase = local;
  window.heapSize = sizeof(local);
  window.localRanks = 2;
  window.rankLayouts = layouts;

  size_t peerSize = 0;
  EXPECT_EQ(flagcxSymWindowPeerUserSize(&comm, &window, 1, &peerSize),
            flagcxSuccess);
  EXPECT_EQ(peerSize, 7000u);

  rankToNode[1] = 1;
  uintptr_t baseVas[] = {(uintptr_t)local, 0};
  size_t regionSizes[] = {sizeof(local), 9000};
  struct flagcxOneSideHandleInfo handle = {};
  handle.baseVas = baseVas;
  handle.regionSizes = regionSizes;
  struct flagcxOneSideHandleInfo *handles[] = {&handle};
  comm.oneSideHandles = handles;
  comm.oneSideHandleCount = 1;
  window.hasNetworkMrRef = true;
  window.mrIndex = 0;
  EXPECT_EQ(flagcxSymWindowPeerUserSize(&comm, &window, 1, &peerSize),
            flagcxSuccess);
  EXPECT_EQ(peerSize, 9000u);
}
