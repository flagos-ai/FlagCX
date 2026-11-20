#include "comm.h"
#include "dev_api_backend.h"
#include "device_api/completion_word.h"
#include "flagcx_hetero.h"
#include "global_comm.h"
#include "group.h"
#include "launch_kernel.h"
#include "proxy.h"
#include "work_completion.hpp"
#include <string>

namespace {
class FifoFinalizeAdmission : public WorkCompletionTest {
protected:
  inline static FifoFinalizeAdmission *active;
  flagcxComm_t owner = nullptr;
  flagcxHeteroComm hetero = {};
  flagcxProxyState proxy = {};
  flagcxTasks::Peer taskPeer = {};
  flagcxChannelPeer peer = {};
  flagcxChannelPeer *peerPointer = &peer;
  int order[2] = {}, submissions = 0;
  bool consumed = false, hadEnv;
  std::string previousEnv;
  flagcxDevApiBackend *savedBackend;
  flagcxDevApiBackend backend = {};
  flagcxLaunchFunc_t savedKernel;
  void SetUp() override {
    WorkCompletionTest::SetUp();
    active = this;
    hadEnv = getenv("FLAGCX_USE_HETERO_COMM") != nullptr;
    previousEnv = hadEnv ? getenv("FLAGCX_USE_HETERO_COMM") : "";
    setenv("FLAGCX_USE_HETERO_COMM", "1", 1);
    savedKernel = deviceAsyncKernel;
    deviceAsyncKernel = nullptr;
    savedBackend = devApiBackend;
    backend = *savedBackend;
    devApiBackend = &backend;
    owner = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    owner->commType = flagcxCommunicatorHomo;
    owner->nranks = owner->localRanks = 1;
    owner->heteroComm = &hetero;
    hetero.proxyState = &proxy;
    hetero.nRanks = hetero.localRanks = 1;
    hetero.tasks.peers = &taskPeer;
    hetero.tasks.p2pOrder = order;
    hetero.groupNext = hetero.preconnectNext = (flagcxHeteroComm *)0x1;
    peer.send[0].connected = peer.recv[0].connected = 1;
    for (auto &channel : hetero.channels)
      channel.peers = &peerPointer;
    backend.commQuiesce = [](flagcxComm_t comm) {
      auto *h = comm->heteroComm;
      EXPECT_EQ(h->finalizeState.started, 1);
      EXPECT_EQ(h->submittedWork.closed, 1);
      flagcxResult_t status = flagcxHeteroDrainProducers(h);
      if (status != flagcxSuccess)
        return status;
      if (!active->consumed) {
        EXPECT_EQ(h->proxyWork.closed, 0);
        EXPECT_EQ(flagcxHeteroSend(nullptr, 0, flagcxFloat, 0, h, nullptr),
                  flagcxInvalidUsage);
        EXPECT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
        auto stream = (flagcxStream_t)&active->stream;
        EXPECT_EQ(flagcxHeteroProxySend(nullptr, 0, flagcxFloat, 0, h, stream),
                  flagcxSuccess);
        EXPECT_EQ(flagcxHeteroProxyRecv(nullptr, 0, flagcxFloat, 0, h, stream),
                  flagcxSuccess);
        EXPECT_EQ(flagcxHeteroGroupEnd(), flagcxSuccess);
        active->submissions++;
        active->consumed = true;
      }
      return flagcxSuccess;
    };
    // Stop after Drain: this fixture borrows its Hetero/proxy scaffolding.
    backend.commCleanup = [](flagcxComm_t) { return flagcxRemoteError; };
  }
  void TearDown() override {
    stream.completed = stream.submitted;
    releaseFake(&hetero.submittedWork);
    releaseFake(&hetero.proxyWork);
    owner->heteroComm = nullptr;
    owner->finalizeState.commonPhase = flagcxFinalizeComplete;
    EXPECT_EQ(flagcxCommDestroy(owner), flagcxSuccess);
    devApiBackend = savedBackend;
    deviceAsyncKernel = savedKernel;
    if (hadEnv)
      setenv("FLAGCX_USE_HETERO_COMM", previousEnv.c_str(), 1);
    else
      unsetenv("FLAGCX_USE_HETERO_COMM");
    WorkCompletionTest::TearDown();
  }
};
} // namespace

TEST_F(FifoFinalizeAdmission,
       AcceptedFifoSubmissionDrainsAfterHostTrackerCloses) {
  stream.submitted = 1;
  EXPECT_EQ(flagcxCommFinalize(owner), flagcxInProgress);
  EXPECT_TRUE(consumed);
  EXPECT_EQ(owner->finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_NE(hetero.proxyWork.head, nullptr);
  EXPECT_EQ(submissions, 1);
  stream.completed = 1;
  EXPECT_EQ(flagcxCommFinalize(owner), flagcxRemoteError);
  EXPECT_EQ(owner->finalizeState.commonPhase, flagcxFinalizeTransport);
  EXPECT_EQ(submissions, 1);
  EXPECT_EQ(hetero.proxyWork.head->state, flagcxWorkComplete);
}

TEST_F(FifoFinalizeAdmission, EqualFifoCountersCannotHideTerminalErrors) {
  uint64_t buffer[flagcxFifoIdxData] = {};
  flagcxFifo fifo;
  fifo.buffer = buffer;
  proxy.kernelState.contextCount = 1;
  proxy.kernelState.fifos[0] = &fifo;
  proxy.kernelState.terminalResult = flagcxRemoteError;
  EXPECT_EQ(flagcxHeteroDrainProducers(&hetero), flagcxRemoteError);
  EXPECT_EQ(flagcxCommFinalize(owner), flagcxRemoteError);
  EXPECT_EQ(owner->finalizeState.commonPhase, flagcxFinalizeDrain);
  proxy.kernelState.fifos[0] = nullptr;
}
