#include <gtest/gtest.h>

#include "adaptor.h"
#include "launch_kernel.h"

#include <cstdlib>

namespace {

flagcxStreamValueSemaphore *recordingSemaphore = nullptr;
int eventQueries = 0;
bool failRecord = false;
char eventStorage;

flagcxResult_t hostAlloc(void **ptr, size_t bytes, flagcxMemType_t,
                         flagcxStream_t) {
  *ptr = std::malloc(bytes);
  return *ptr == nullptr ? flagcxSystemError : flagcxSuccess;
}

flagcxResult_t hostFree(void *ptr, flagcxMemType_t, flagcxStream_t) {
  std::free(ptr);
  return flagcxSuccess;
}

flagcxResult_t hostAlias(void **devicePtr, void *hostPtr) {
  *devicePtr = hostPtr;
  return flagcxSuccess;
}

flagcxResult_t createEvent(flagcxEvent_t *event, flagcxEventType_t) {
  *event = reinterpret_cast<flagcxEvent_t>(&eventStorage);
  return flagcxSuccess;
}

flagcxResult_t destroyEvent(flagcxEvent_t) { return flagcxSuccess; }

flagcxResult_t writeOrWait(flagcxStream_t, void *, uint64_t, int) {
  return flagcxSuccess;
}

flagcxResult_t queryEvent(flagcxEvent_t) {
  ++eventQueries;
  return flagcxSuccess;
}

flagcxResult_t recordEvent(flagcxEvent_t, flagcxStream_t) {
  // CUDA would report success for an event queried before its first record.
  EXPECT_EQ(recordingSemaphore->pollEnd(), 0);
  EXPECT_EQ(eventQueries, 0);
  return failRecord ? flagcxUnhandledDeviceError : flagcxSuccess;
}

class StreamValueCompletionTest : public ::testing::Test {
protected:
  flagcxDeviceAdaptor_latest *savedAdaptor = nullptr;
  flagcxDeviceAdaptor_latest fakeAdaptor{};

  void SetUp() override {
    savedAdaptor = deviceAdaptor;
    ASSERT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
    fakeAdaptor.deviceMalloc = hostAlloc;
    fakeAdaptor.deviceFree = hostFree;
    fakeAdaptor.hostGetDevicePointer = hostAlias;
    fakeAdaptor.eventCreate = createEvent;
    fakeAdaptor.eventDestroy = destroyEvent;
    fakeAdaptor.eventRecord = recordEvent;
    fakeAdaptor.eventQuery = queryEvent;
    fakeAdaptor.streamWriteValue64 = writeOrWait;
    fakeAdaptor.streamWaitValue64 = writeOrWait;
    deviceAdaptor = &fakeAdaptor;
    eventQueries = 0;
    failRecord = false;
  }

  void TearDown() override {
    recordingSemaphore = nullptr;
    EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
    deviceAdaptor = savedAdaptor;
  }
};

TEST_F(StreamValueCompletionTest, EventIsQueriedOnlyAfterRecordReturns) {
  flagcxStreamValueSemaphore semaphore;
  ASSERT_EQ(semaphore.initResult, flagcxSuccess);
  recordingSemaphore = &semaphore;
  EXPECT_EQ(semaphore.enqueueCompletion(nullptr), flagcxSuccess);
  EXPECT_EQ(semaphore.pollEnd(), 1);
  EXPECT_EQ(eventQueries, 1);
}

TEST_F(StreamValueCompletionTest, FailedRecordCannotCompleteUntilDrained) {
  flagcxStreamValueSemaphore semaphore;
  ASSERT_EQ(semaphore.initResult, flagcxSuccess);
  recordingSemaphore = &semaphore;
  failRecord = true;
  EXPECT_EQ(semaphore.enqueueCompletion(nullptr), flagcxUnhandledDeviceError);
  EXPECT_EQ(semaphore.pollEnd(), 0);
  EXPECT_EQ(eventQueries, 0);
  semaphore.markCompletionDrained();
  EXPECT_EQ(semaphore.pollEnd(), 1);
  EXPECT_EQ(eventQueries, 0);
}

} // namespace
