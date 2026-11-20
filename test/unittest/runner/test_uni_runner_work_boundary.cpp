#include "comm.h"
#include "flagcx_hetero.h"
#include "group.h"
#include "launch_kernel.h"
#include "uni_runner_impl.h"
#include "work_completion.hpp"

namespace {
class UniRunnerWorkBoundary : public WorkCompletionTest {
protected:
  flagcxHeteroComm comm = {};
  flagcxTasks::Peer taskPeer = {};
  flagcxChannelPeer channelPeer = {};
  flagcxChannelPeer *peerPtr = &channelPeer;
  int p2pOrder[2] = {};
  uniRunnerDagNode node = {};
  uniRunnerP2pOpData ops[2] = {};
  flagcxUniRunnerState state = {};
  int topoOrder = 0;
  void *flag = &topoOrder;
  flagcxLaunchFunc_t savedKernel;
  void SetUp() override {
    WorkCompletionTest::SetUp();
    savedKernel = deviceAsyncKernel;
    deviceAsyncKernel = nullptr;
    // The test's paired self operations share an address, so the real group
    // scheduler retires their tasks without needing a GPU/proxy thread.
    comm.nRanks = comm.localRanks = 1;
    comm.groupNext = (flagcxHeteroComm *)0x1;
    comm.preconnectNext = (flagcxHeteroComm *)0x1;
    comm.tasks.peers = &taskPeer;
    comm.tasks.p2pOrder = p2pOrder;
    for (auto &channel : comm.channels)
      channel.peers = &peerPtr;
    channelPeer.send[0].connected = channelPeer.recv[0].connected = 1;
    node.nodeIdx = 0;
    node.nodeType = uniRunnerDagNodeTypeP2p;
    node.commDomain = uniRunnerDagCommNet;
    node.nodeData.p2p.numOps = 2;
    node.nodeData.p2p.ops = ops;
    ops[0].type = flagcxDevicePrimSend;
    ops[1].type = flagcxDevicePrimRecv;
    ops[0].datatype = ops[1].datatype = flagcxFloat;
    state.dagNodes = &node;
    state.numDagNodes = 1;
    state.dagPlan.numNodes = 1;
    state.dagPlan.topoOrder = &topoOrder;
    state.streamFlags = &flag;
    state.streamFlagsSize = 1;
    state.netCommStream = (flagcxStream_t)&stream;
    device.hostGetDevicePointer = nullptr; // No stream-value capability probe.
    device.streamWriteValue64 = [](flagcxStream_t input, void *, uint64_t,
                                   int) {
      ((WorkTestStream *)input)->submitted++;
      return flagcxSuccess;
    };
  }
  void TearDown() override {
    releaseFake(&comm.submittedWork);
    releaseFake(&comm.proxyWork);
    deviceAsyncKernel = savedKernel;
    EXPECT_EQ(flagcxGetGroupDepth(), 0);
    WorkCompletionTest::TearDown();
  }
};
} // namespace

TEST_F(UniRunnerWorkBoundary, HostDagRetiresEveryDeferredSendRecvMarker) {
  for (int invocation = 0; invocation < 2; invocation++) {
    int before = recorded;
    ASSERT_EQ(flagcxUniRunnerSubmitHostDag(&state, &comm), flagcxSuccess);
    EXPECT_EQ(recorded, before + 2);
    for (auto *record = comm.submittedWork.head; record; record = record->next)
      EXPECT_EQ(record->state, flagcxWorkRecorded);
    ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
    ASSERT_EQ(flagcxHeteroGroupEnd(), flagcxSuccess);
    EXPECT_EQ(recorded, before + 2) << "stale invocation stream reused";
  }
  stream.completed = stream.submitted;
  EXPECT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
}

TEST_F(UniRunnerWorkBoundary, ProxySubmissionSurvivesExternalAdmissionClosure) {
  comm.finalizeState.started = 1;
  ASSERT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroSend(nullptr, 0, flagcxFloat, 0, &comm,
                             (flagcxStream_t)&stream),
            flagcxInvalidUsage);
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  stream.submitted = 1;
  EXPECT_EQ(flagcxHeteroProxySend(nullptr, 0, flagcxFloat, 0, &comm,
                                  (flagcxStream_t)&stream),
            flagcxSuccess);
  EXPECT_EQ(flagcxHeteroProxyRecv(nullptr, 0, flagcxFloat, 0, &comm,
                                  (flagcxStream_t)&stream),
            flagcxSuccess);
  ASSERT_EQ(flagcxHeteroGroupEnd(), flagcxSuccess);
  ASSERT_NE(comm.proxyWork.head, nullptr);
  EXPECT_EQ(flagcxWorkWait(&comm.proxyWork), flagcxInProgress);
  waitFailures = 1;
  EXPECT_EQ(flagcxWorkWait(&comm.proxyWork), flagcxUnhandledDeviceError);
  stream.completed = 1;
  EXPECT_EQ(flagcxWorkWait(&comm.proxyWork), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroProxySend(nullptr, 0, flagcxFloat, 0, &comm,
                                  (flagcxStream_t)&stream),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
  destroyFailures = 1;
  EXPECT_EQ(flagcxWorkRelease(&comm.proxyWork), flagcxSystemError);
  EXPECT_EQ(flagcxWorkRelease(&comm.proxyWork), flagcxSuccess);
  EXPECT_EQ(destroyed, 2);
}
