#include "comm.h"
#include "flagcx_hetero.h"
#include "work_completion.hpp"

namespace {
class RmaWorkCompletion : public WorkCompletionTest {
protected:
  void SetUp() override {
    WorkCompletionTest::SetUp();
    device.streamWaitValue64 = [](flagcxStream_t input, void *, uint64_t, int) {
      ((WorkTestStream *)input)->submitted++;
      return flagcxSuccess;
    };
  }
};
} // namespace

TEST_F(RmaWorkCompletion, StreamCompletionOutlivesTheProxyCounterSubmission) {
  flagcxHeteroComm comm = {};
  flagcxRmaProxyState proxy = {};
  uint64_t counter = 0;
  comm.nRanks = proxy.nRanks = 1;
  comm.rmaProxy = &proxy;
  proxy.useStreamOps = 1;
  proxy.doneSeqsDev = &counter;
  ASSERT_EQ(flagcxHeteroFlushRmaStream(&comm, 0, 1, (flagcxStream_t)&stream),
            flagcxSuccess);
  EXPECT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxInProgress);
  EXPECT_EQ(destroyed, 0);
  stream.completed = 1;
  EXPECT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
  EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
}
