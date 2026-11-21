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

TEST_F(RmaWorkCompletion, FailedOperationNeedsClosedAdmissionAndRetiredWork) {
  flagcxHeteroComm comm = {};
  flagcxRmaProxyState proxy = {};
  uint32_t pi = 1, ci = 0, inFlight = 0;
  uint64_t submitted = 1, retired = 0;
  comm.rmaProxy = &proxy;
  proxy.nRanks = 1;
  proxy.pis = &pi;
  proxy.cis = &ci;
  proxy.inFlights = &inFlight;
  proxy.opSeqs = &submitted;
  proxy.doneSeqs = &retired;
  proxy.rmaError = 1;
  EXPECT_EQ(flagcxHeteroDrainProducers(&comm), flagcxRemoteError);
  comm.finalizeState.started = 1;
  EXPECT_EQ(flagcxHeteroDrainProducers(&comm), flagcxRemoteError);
  ci = pi;
  inFlight = 1;
  EXPECT_EQ(flagcxHeteroDrainProducers(&comm), flagcxRemoteError);
  inFlight = 0;
  // Consuming the last ring entry precedes retiring its completion record.
  EXPECT_EQ(flagcxHeteroDrainProducers(&comm), flagcxRemoteError);
  retired = submitted;
  EXPECT_EQ(flagcxHeteroDrainProducers(&comm), flagcxSuccess);
  EXPECT_EQ(proxy.rmaError, 1);
  comm.finalizeState.started = 0;
  EXPECT_EQ(flagcxHeteroDrainProducers(&comm), flagcxRemoteError);
}
