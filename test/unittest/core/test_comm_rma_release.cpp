#include "dev_api_backend.h"
#include "rma_init_rollback.hpp"

namespace {
class CommRmaReleaseTest : public RmaInitRollbackTest {
protected:
  flagcxDevApiBackend *savedBackend;
  flagcxDevApiBackend backend = {};
  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    savedBackend = devApiBackend;
    backend = *savedBackend;
    backend.commQuiesce = [](flagcxComm_t comm) {
      return flagcxHeteroDrainProducers(comm->heteroComm);
    };
    backend.commCleanup = [](flagcxComm_t) { return flagcxSuccess; };
    devApiBackend = &backend;
    makeOwner(false);
    ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSuccess);
    attachFallibleTransport();
    outer = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    outer->commType = flagcxCommunicatorHybrid;
    outer->nranks = outer->localRanks = 1;
    outer->ipcTable = outer->ipcTableInline;
    outer->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
    outer->heteroComm = owner;
    owner->lifetimeOwner = outer;
    owner = nullptr;
  }
  void TearDown() override {
    RmaInitRollbackTest::TearDown();
    devApiBackend = savedBackend;
  }
};
} // namespace

TEST_F(CommRmaReleaseTest, FinalizeRetainsRmaUntilRetryableLocalDestroy) {
  auto *inner = outer->heteroComm;
  auto *proxy = inner->rmaProxy;
  ASSERT_EQ(flagcxCommFinalize(outer), flagcxSuccess);
  EXPECT_EQ(inner->rmaProxy, proxy);
  EXPECT_TRUE(proxy->stopComplete);
  EXPECT_EQ(proxy->worker, nullptr);
  EXPECT_NE(proxy->doneSeqsDev, nullptr);
  EXPECT_NE(proxy->readySeqsDev, nullptr);
  EXPECT_EQ(doneFrees + readyFrees + drains, 0);
  freeFailures = 1;
  freeFailureResult = flagcxInProgress;
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxInProgress);
  EXPECT_EQ(outer->heteroComm, inner);
  EXPECT_EQ(outer->destroyPhase, flagcxDestroyHetero);
  EXPECT_EQ(outer->finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(inner->rmaProxy, proxy);
  EXPECT_EQ(proxy->doneSeqsDev, nullptr);
  EXPECT_NE(proxy->readySeqsDev, nullptr);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 2);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(releases, 1);
}
