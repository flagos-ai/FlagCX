#include "uni_runner_stream_flags.h"
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <set>

namespace {
struct AllocationState {
  std::set<void *> live;
  int allocations = 0;
  int originalCalls = 0;
  int policyCalls = 0;
  int freeFailures = 0;
  bool failAllocate = false;
  bool failInitialize = false;
  flagcxMemType_t type = flagcxMemDevice;
  flagcxStream_t stream = nullptr;
  unsigned int flags = 0;
} * state;

flagcxResult_t allocate(void **ptr, size_t size, flagcxMemType_t type,
                        flagcxStream_t stream) {
  ++state->originalCalls;
  state->type = type;
  state->stream = stream;
  if (state->failAllocate)
    return flagcxSystemError;
  *ptr = std::malloc(size);
  if (*ptr == nullptr)
    return flagcxSystemError;
  state->live.insert(*ptr);
  ++state->allocations;
  return flagcxSuccess;
}
flagcxResult_t allocatePolicy(void **ptr, size_t size, flagcxMemType_t type,
                              flagcxStream_t stream, unsigned int flags) {
  ++state->policyCalls;
  state->flags = flags;
  return allocate(ptr, size, type, stream);
}
flagcxResult_t release(void *ptr, flagcxMemType_t type, flagcxStream_t stream) {
  EXPECT_EQ(type, flagcxMemDevice);
  EXPECT_EQ(stream, nullptr);
  if (state->freeFailures > 0) {
    --state->freeFailures;
    return flagcxUnhandledDeviceError;
  }
  EXPECT_EQ(state->live.erase(ptr), 1u);
  std::free(ptr);
  return flagcxSuccess;
}
flagcxResult_t initialize(void *ptr, int value, size_t size, flagcxMemType_t,
                          flagcxStream_t) {
  if (state->failInitialize)
    return flagcxUnhandledDeviceError;
  std::memset(ptr, value, size);
  return flagcxSuccess;
}

class UniRunnerStreamFlagPoolTest : public ::testing::Test {
protected:
  AllocationState allocations;
  flagcxDeviceAdaptor adaptor{};
  flagcxUniRunnerStreamFlagPool pool{};
  void SetUp() override {
    state = &allocations;
    adaptor.deviceMalloc = allocate;
    adaptor.deviceMallocWithFlags = allocatePolicy;
    adaptor.deviceFree = release;
    adaptor.deviceMemset = initialize;
  }
  void TearDown() override {
    allocations.freeFailures = 0;
    EXPECT_EQ(flagcxUniRunnerReleaseStreamFlags(&adaptor, &pool),
              flagcxSuccess);
    EXPECT_TRUE(allocations.live.empty());
    state = nullptr;
  }
};

TEST_F(UniRunnerStreamFlagPoolTest, LegacyPluginKeepsAllocationArguments) {
  flagcxDeviceAdaptor_v1 legacy{};
  legacy.deviceMalloc = allocate;
  flagcxDeviceAdaptorUpgradeV1(&legacy, &adaptor);
  ASSERT_EQ(adaptor.deviceMallocWithFlags, nullptr);
  auto stream = reinterpret_cast<flagcxStream_t>(&allocations);
  void *ptr = nullptr;
  ASSERT_EQ(flagcxDeviceAllocWithFlags(&adaptor, &ptr, 16, flagcxMemManaged,
                                       stream, FLAGCX_DEVICE_ALLOC_UNIRUNNER),
            flagcxSuccess);
  EXPECT_EQ(allocations.type, flagcxMemManaged);
  EXPECT_EQ(allocations.stream, stream);
  EXPECT_EQ(allocations.policyCalls, 0);
  allocations.live.erase(ptr);
  std::free(ptr);
}

TEST_F(UniRunnerStreamFlagPoolTest, DefaultBypassesPolicyAndUnknownFlagsFail) {
  void *ptr = nullptr;
  ASSERT_EQ(flagcxDeviceAllocWithFlags(&adaptor, &ptr, 8, flagcxMemDevice,
                                       nullptr, FLAGCX_DEVICE_ALLOC_DEFAULT),
            flagcxSuccess);
  EXPECT_EQ(allocations.policyCalls, 0);
  EXPECT_EQ(flagcxDeviceAllocWithFlags(&adaptor, &ptr, 8, flagcxMemDevice,
                                       nullptr, 2u),
            flagcxInvalidArgument);
  EXPECT_EQ(allocations.originalCalls, 1);
  EXPECT_EQ(release(ptr, flagcxMemDevice, nullptr), flagcxSuccess);
}

TEST_F(UniRunnerStreamFlagPoolTest, ReusesAndGrowsDeviceControlStorage) {
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 3),
            flagcxSuccess);
  EXPECT_EQ(pool.capacity, 4u);
  EXPECT_EQ(allocations.flags, FLAGCX_DEVICE_ALLOC_UNIRUNNER);
  EXPECT_EQ(allocations.type, flagcxMemDevice);
  EXPECT_EQ(allocations.stream, nullptr);
  void *original = pool.buffer;
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 2),
            flagcxSuccess);
  EXPECT_EQ(pool.buffer, original);
  EXPECT_EQ(allocations.allocations, 1);
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 5),
            flagcxSuccess);
  EXPECT_EQ(pool.capacity, 8u);
  for (size_t i = 0; i < pool.capacity; ++i) {
    EXPECT_EQ(pool.addresses[i], static_cast<char *>(pool.buffer) + i * 8);
    EXPECT_EQ(*static_cast<uint64_t *>(pool.addresses[i]), 0u);
  }
  EXPECT_EQ(allocations.live.size(), 1u);
}

TEST_F(UniRunnerStreamFlagPoolTest,
       AllocationErrorDoesNotFallBackOrReplacePool) {
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 1),
            flagcxSuccess);
  void *original = pool.buffer;
  void **addresses = pool.addresses;
  allocations.failAllocate = true;
  EXPECT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 2),
            flagcxSystemError);
  EXPECT_EQ(pool.buffer, original);
  EXPECT_EQ(pool.addresses, addresses);
  EXPECT_EQ(pool.capacity, 1u);
  EXPECT_EQ(allocations.originalCalls, 2);
}

TEST_F(UniRunnerStreamFlagPoolTest, InitializationErrorKeepsPreviousStorage) {
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 1),
            flagcxSuccess);
  void *original = pool.buffer;
  allocations.failInitialize = true;
  EXPECT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 2),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(pool.buffer, original);
  EXPECT_EQ(pool.capacity, 1u);
  EXPECT_EQ(allocations.live.size(), 1u);
}

TEST_F(UniRunnerStreamFlagPoolTest, FailedRollbackRetainsBothAllocations) {
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 1),
            flagcxSuccess);
  void *original = pool.buffer;
  allocations.freeFailures = 2;
  EXPECT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 2),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(pool.buffer, original);
  EXPECT_NE(pool.pendingFree, nullptr);
  EXPECT_EQ(allocations.live.size(), 2u);
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 2),
            flagcxSuccess);
  EXPECT_EQ(pool.pendingFree, nullptr);
  EXPECT_EQ(pool.capacity, 2u);
  EXPECT_EQ(allocations.live.size(), 1u);
}

TEST_F(UniRunnerStreamFlagPoolTest, ReleaseErrorRetainsOwnershipForRetry) {
  ASSERT_EQ(flagcxUniRunnerReserveStreamFlags(&adaptor, &pool, 1),
            flagcxSuccess);
  void *original = pool.buffer;
  allocations.freeFailures = 1;
  EXPECT_EQ(flagcxUniRunnerReleaseStreamFlags(&adaptor, &pool),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(pool.buffer, original);
  EXPECT_NE(pool.addresses, nullptr);
  ASSERT_EQ(flagcxUniRunnerReleaseStreamFlags(&adaptor, &pool), flagcxSuccess);
  EXPECT_EQ(pool.buffer, nullptr);
  EXPECT_EQ(pool.addresses, nullptr);
  EXPECT_EQ(pool.capacity, 0u);
}

TEST_F(UniRunnerStreamFlagPoolTest, OverflowDoesNotAllocate) {
  EXPECT_EQ(flagcxUniRunnerReserveStreamFlags(
                &adaptor, &pool, std::numeric_limits<size_t>::max()),
            flagcxInvalidArgument);
  EXPECT_EQ(allocations.allocations, 0);
}
} // namespace
