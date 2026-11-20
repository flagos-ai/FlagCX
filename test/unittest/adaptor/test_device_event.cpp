#include "device_event.h"
#include <gtest/gtest.h>

namespace {
struct NativeEvent {
  int handle;
};
class DeviceEventTransaction : public ::testing::Test {
protected:
  inline static DeviceEventTransaction *current;
  int allocations = 0, frees = 0, creates = 0, destroys = 0;
  bool failAllocation = false;
  flagcxResult_t nativeResult = flagcxSuccess;
  flagcxEventCreateOps ops = {};
  void *owner = nullptr;
  void SetUp() override {
    current = this;
    ops.allocate = [](size_t bytes) -> void * {
      if (current->failAllocation)
        return nullptr;
      current->allocations++;
      return calloc(1, bytes);
    };
    ops.release = [](void *wrapper) {
      current->frees++;
      free(wrapper);
    };
    ops.nativeCreate = [](void *wrapper, unsigned int flags) {
      current->creates++;
      EXPECT_EQ(flags, 7u);
      if (current->nativeResult != flagcxSuccess)
        return current->nativeResult;
      ((NativeEvent *)wrapper)->handle = 42;
      return flagcxSuccess;
    };
  }
  flagcxResult_t create() {
    return flagcxEventCreateTransactional(&owner, sizeof(NativeEvent), 7, &ops);
  }
  void TearDown() override {
    if (owner != nullptr) {
      EXPECT_EQ(((NativeEvent *)owner)->handle, 42);
      destroys++;
      ops.release(owner);
    }
    EXPECT_EQ(allocations, frees);
  }
};
} // namespace

TEST_F(DeviceEventTransaction, AllocationFailurePublishesNullAndSkipsSdk) {
  failAllocation = true;
  owner = (void *)1;
  EXPECT_EQ(create(), flagcxSystemError);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(creates + frees + destroys, 0);
}

TEST_F(DeviceEventTransaction, NativeFailureFreesWrapperAndCanRetry) {
  nativeResult = flagcxUnhandledDeviceError;
  EXPECT_EQ(create(), flagcxUnhandledDeviceError);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(allocations, 1);
  EXPECT_EQ(frees, 1);
  EXPECT_EQ(destroys, 0);
  nativeResult = flagcxSuccess;
  EXPECT_EQ(create(), flagcxSuccess);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(((NativeEvent *)owner)->handle, 42);
  EXPECT_EQ(allocations, 2);
}

TEST_F(DeviceEventTransaction, InProgressDoesNotPublishAnUncreatedEvent) {
  nativeResult = flagcxInProgress;
  EXPECT_EQ(create(), flagcxInProgress);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(allocations, frees);
  EXPECT_EQ(destroys, 0);
}
