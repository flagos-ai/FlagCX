#include "flagcx.h"
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>

namespace DeviceEventMock {
static int allocations, frees, creates, destroys;
static unsigned int flags;
static bool failAllocation;
static int createResult, destroyResult;
static void reset() {
  allocations = frees = creates = destroys = 0;
  flags = 0;
  failAllocation = false;
  createResult = destroyResult = 0;
}
static void *allocate(size_t n, size_t bytes) {
  if (failAllocation)
    return nullptr;
  allocations++;
  return std::calloc(n, bytes);
}
static void release(void *p) {
  frees++;
  std::free(p);
}
static int create(uintptr_t *p, unsigned int inputFlags) {
  creates++;
  flags = inputFlags;
  *p = createResult == 0 ? 42 : 99;
  return createResult;
}
static int destroy(uintptr_t p) {
  EXPECT_EQ(p, 42u) << "destroying an invalid native event";
  destroys++;
  return destroyResult;
}
} // namespace DeviceEventMock

struct DeviceEventContract {
  const char *name;
  unsigned int defaultFlags, disableTimingFlags;
  flagcxResult_t pendingResult;
  flagcxResult_t (*create)(void **, flagcxEventType_t);
  flagcxResult_t (*destroy)(void *);
};
#include "adaptor-contracts/device_events.h"

TEST(DeviceEventContract, AllocationFailurePublishesNullAndSkipsSdk) {
  for (const auto &ops : eventContracts) {
    SCOPED_TRACE(ops.name);
    DeviceEventMock::reset();
    EXPECT_EQ(ops.create(nullptr, flagcxEventDefault), flagcxInvalidArgument);
    DeviceEventMock::failAllocation = true;
    void *owner = (void *)1;
    EXPECT_EQ(ops.create(&owner, flagcxEventDefault), flagcxSystemError);
    EXPECT_EQ(owner, nullptr);
    EXPECT_EQ(DeviceEventMock::creates + DeviceEventMock::frees, 0);
  }
}

TEST(DeviceEventContract, NativeFailureFreesWrapperAndCanRetry) {
  for (const auto &ops : eventContracts) {
    SCOPED_TRACE(ops.name);
    DeviceEventMock::reset();
    DeviceEventMock::createResult = 99;
    void *owner = (void *)1;
    EXPECT_EQ(ops.create(&owner, flagcxEventDisableTiming),
              flagcxUnhandledDeviceError);
    EXPECT_EQ(owner, nullptr);
    EXPECT_EQ(DeviceEventMock::allocations, 1);
    EXPECT_EQ(DeviceEventMock::frees, 1);
    EXPECT_EQ(DeviceEventMock::destroys, 0);
    DeviceEventMock::createResult = 0;
    ASSERT_EQ(ops.create(&owner, flagcxEventDisableTiming), flagcxSuccess);
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(DeviceEventMock::flags, ops.disableTimingFlags);
    EXPECT_EQ(ops.destroy(owner), flagcxSuccess);
    EXPECT_EQ(DeviceEventMock::allocations, DeviceEventMock::frees);
    EXPECT_EQ(DeviceEventMock::destroys, 1);
  }
}

TEST(DeviceEventContract, NonSuccessDoesNotPublishAnUncreatedEvent) {
  for (const auto &ops : eventContracts) {
    SCOPED_TRACE(ops.name);
    DeviceEventMock::reset();
    DeviceEventMock::createResult = 70;
    void *owner = nullptr;
    EXPECT_EQ(ops.create(&owner, flagcxEventDefault), ops.pendingResult);
    EXPECT_EQ(owner, nullptr);
    EXPECT_EQ(DeviceEventMock::allocations, DeviceEventMock::frees);
    EXPECT_EQ(DeviceEventMock::destroys, 0);
  }
}

TEST(DeviceEventContract, DestroyFailureRetainsWrapperForRetry) {
  for (const auto &ops : eventContracts) {
    SCOPED_TRACE(ops.name);
    DeviceEventMock::reset();
    void *owner = nullptr;
    ASSERT_EQ(ops.create(&owner, flagcxEventDefault), flagcxSuccess);
    EXPECT_EQ(DeviceEventMock::flags, ops.defaultFlags);
    DeviceEventMock::destroyResult = 99;
    EXPECT_EQ(ops.destroy(owner), flagcxUnhandledDeviceError);
    EXPECT_EQ(DeviceEventMock::frees, 0);
    DeviceEventMock::destroyResult = 0;
    EXPECT_EQ(ops.destroy(owner), flagcxSuccess);
    EXPECT_EQ(DeviceEventMock::frees, 1);
    EXPECT_EQ(DeviceEventMock::destroys, 2);
  }
}
