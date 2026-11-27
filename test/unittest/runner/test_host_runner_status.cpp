#include "adaptor.h"
#include "global_comm.h"
#include "runner.h"
#include <cstring>
#include <gtest/gtest.h>

class HostRunnerStatus : public ::testing::Test {
public:
  inline static HostRunnerStatus *active;
  flagcxDeviceAdaptor device = {}, *savedDevice;
  flagcxCCLAdaptor host = {}, *savedHost;
  flagcxComm comm = {};
  flagcxResult_t status = flagcxSuccess;
  int copies = 0;
  int staged = 42, output = 0;
  void SetUp() override {
    active = this;
    savedDevice = deviceAdaptor;
    savedHost = cclAdaptors[flagcxCCLAdaptorHost];
    deviceAdaptor = &device;
    cclAdaptors[flagcxCCLAdaptorHost] = &host;
    host.groupStart = host.groupEnd = [] { return active->status; };
    host.getStagedBuffer = [](flagcxInnerComm_t, void **out, size_t, int) {
      *out = &active->staged;
      return flagcxSuccess;
    };
    host.recv = [](void *, size_t, flagcxDataType_t, int, flagcxInnerComm_t,
                   flagcxStream_t) { return flagcxSuccess; };
    device.deviceMemcpy = [](void *dst, void *src, size_t bytes,
                             flagcxMemcpyType_t, flagcxStream_t, void *) {
      ++active->copies;
      memcpy(dst, src, bytes);
      return flagcxSuccess;
    };
  }
  void TearDown() override {
    deviceAdaptor = savedDevice;
    cclAdaptors[flagcxCCLAdaptorHost] = savedHost;
  }
  void receive() {
    EXPECT_EQ(hostRunner.recv(&output, 1, flagcxInt, 0, &comm, nullptr),
              flagcxSuccess);
  }
};

TEST_F(HostRunnerStatus, PendingAndFailedGroupEndKeepReceivesUntilSuccess) {
  ASSERT_EQ(hostRunner.groupStart(), flagcxSuccess);
  receive();
  for (auto result : {flagcxInProgress, flagcxRemoteError}) {
    status = result;
    EXPECT_EQ(hostRunner.groupEnd(), result);
    EXPECT_EQ(copies, 0);
    EXPECT_EQ(output, 0);
  }
  status = flagcxSuccess;
  EXPECT_EQ(hostRunner.groupEnd(), flagcxSuccess);
  EXPECT_EQ(copies, 1);
  EXPECT_EQ(output, staged);
  EXPECT_EQ(hostRunner.groupStart(), flagcxSuccess);
  EXPECT_EQ(hostRunner.groupEnd(), flagcxSuccess);
  EXPECT_EQ(copies, 1);
}

TEST_F(HostRunnerStatus, NestedPendingEndDoesNotConsumeOuterGroup) {
  ASSERT_EQ(hostRunner.groupStart(), flagcxSuccess);
  ASSERT_EQ(hostRunner.groupStart(), flagcxSuccess);
  receive();
  EXPECT_EQ(hostRunner.groupEnd(), flagcxSuccess);
  EXPECT_EQ(copies, 0);
  status = flagcxInProgress;
  EXPECT_EQ(hostRunner.groupEnd(), flagcxInProgress);
  EXPECT_EQ(copies, 0);
  status = flagcxSuccess;
  EXPECT_EQ(hostRunner.groupEnd(), flagcxSuccess);
  EXPECT_EQ(copies, 1);
}

TEST_F(HostRunnerStatus, PendingOrFailedStartDoesNotOpenGroup) {
  for (auto result : {flagcxInProgress, flagcxRemoteError}) {
    status = result;
    EXPECT_EQ(hostRunner.groupStart(), result);
    receive();
  }
  EXPECT_EQ(copies, 2);
}
