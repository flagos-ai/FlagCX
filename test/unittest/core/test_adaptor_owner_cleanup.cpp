#include "adaptor.h"
#include "global_comm.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <new>
#include <set>

namespace {
class AdaptorOwnerCleanup : public ::testing::Test {
protected:
  static AdaptorOwnerCleanup *current;
  flagcxCCLAdaptor *saved;
  flagcxCCLAdaptor sdk = {};
  flagcxComm_t comm = nullptr;
  void *pending = nullptr;
  int calls = 0, pendingCalls = 0;
  std::set<void *> live;
  void SetUp() override {
    current = this;
    saved = cclAdaptors[flagcxCCLAdaptorDevice];
    sdk = *saved;
    sdk.commDestroy = [](flagcxInnerComm_t p) {
      current->calls++;
      if ((void *)p == current->pending) {
        int attempt = ++current->pendingCalls;
        if (attempt == 1)
          return flagcxInProgress;
        if (attempt == 2)
          return flagcxRemoteError;
      }
      EXPECT_EQ(current->live.erase(p), 1u);
      free(p);
      return flagcxSuccess;
    };
    cclAdaptors[flagcxCCLAdaptorDevice] = &sdk;
    comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    // The normal initializer builds these maps when tuning is enabled.
    new (&comm->homoCommMap) decltype(comm->homoCommMap);
    new (&comm->homoBestCommMap) decltype(comm->homoBestCommMap);
    new (&comm->commMap) decltype(comm->commMap);
    comm->commType = flagcxCommunicatorHomo;
    comm->destroyPhase = 3;
    comm->nranks = comm->localRanks = 1;
    comm->ipcTable = comm->ipcTableInline;
    comm->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
  }
  void TearDown() override {
    if (comm) {
      pendingCalls = 2;
      EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    }
    EXPECT_TRUE(live.empty());
    cclAdaptors[flagcxCCLAdaptorDevice] = saved;
    current = nullptr;
  }
  flagcxInnerComm_t owner() {
    void *p = calloc(1, 16);
    live.insert(p);
    return (flagcxInnerComm_t)p;
  }
};
AdaptorOwnerCleanup *AdaptorOwnerCleanup::current;
} // namespace

TEST_F(AdaptorOwnerCleanup, OldDestroyRetainsPendingOwnerAndClearsEveryAlias) {
  flagcxInnerComm_t first = owner(), second = owner();
  pending = second;
  comm->homoComm = comm->tunerInnerComm = first;
  comm->homoInterComm = second;
  comm->homoCommMap[TunerCollCategory{}] = first;
  comm->homoBestCommMap[TunerCollCategory{}] = second;
  comm->commMap[flagcxCommTag{}] = first;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxInProgress);
  EXPECT_EQ(comm->destroyPhase, 3);
  EXPECT_EQ(comm->homoComm, nullptr);
  EXPECT_EQ(comm->tunerInnerComm, nullptr);
  EXPECT_EQ(comm->commMap.begin()->second, nullptr);
  EXPECT_EQ(comm->homoInterComm, second);
  EXPECT_EQ(live.size(), 1u);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxRemoteError);
  EXPECT_EQ(comm->homoInterComm, second);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(calls, 4);
}

TEST_F(AdaptorOwnerCleanup, TunerOwnerRetainsAllAliasesUntilStrictSuccess) {
  auto inner = owner();
  pending = inner;
  comm->homoComm = comm->tunerInnerComm = inner;
  comm->homoCommMap[TunerCollCategory{}] = inner;
  comm->homoBestCommMap[TunerCollCategory{}] = inner;
  comm->commMap[flagcxCommTag{}] = inner;
  EXPECT_EQ(flagcxDestroyHomoCommOwner(comm, inner), flagcxInProgress);
  EXPECT_EQ(comm->homoComm, inner);
  EXPECT_EQ(comm->homoCommMap.begin()->second, inner);
  EXPECT_EQ(flagcxDestroyHomoCommOwner(comm, inner), flagcxRemoteError);
  EXPECT_EQ(comm->tunerInnerComm, inner);
  EXPECT_EQ(comm->commMap.begin()->second, inner);
  EXPECT_EQ(flagcxDestroyHomoCommOwner(comm, inner), flagcxSuccess);
  EXPECT_EQ(comm->homoComm, nullptr);
  EXPECT_EQ(comm->tunerInnerComm, nullptr);
  EXPECT_EQ(comm->homoCommMap.begin()->second, nullptr);
  EXPECT_EQ(comm->homoBestCommMap.begin()->second, nullptr);
  EXPECT_EQ(comm->commMap.begin()->second, nullptr);
  EXPECT_EQ(calls, 3);
}
