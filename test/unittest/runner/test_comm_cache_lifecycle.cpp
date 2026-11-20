#include "comm_lifecycle.h"
#include "dev_api_backend.h"
#include "global_comm.h"
#include "proxy.h"
#include "reg_pool.h"
#include <gtest/gtest.h>

TEST(CommCacheLifecycleTest,
     IndependentCommunicatorsReleaseOnlyTheirRegistryCache) {
  auto *saved = devApiBackend;
  flagcxDevApiBackend backend = *saved;
  backend.commQuiesce = [](flagcxComm_t) { return flagcxSuccess; };
  backend.commCleanup = [](flagcxComm_t) { return flagcxSuccess; };
  devApiBackend = &backend;
  struct Restore {
    flagcxDevApiBackend *saved;
    ~Restore() { devApiBackend = saved; }
  } restore{saved};
  for (int iteration = 0; iteration < 12; iteration++) {
    flagcxComm_t members[2] = {};
    void *data = (void *)(uintptr_t)(0x700000 + 4096 * iteration);
    for (int rank = 0; rank < 2; rank++) {
      auto *comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
      members[rank] = comm;
      comm->commType = flagcxCommunicatorHybrid;
      comm->localRanks = comm->nranks = 1;
      comm->ipcTable = comm->ipcTableInline;
      comm->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
      auto *hetero = (flagcxHeteroComm *)calloc(1, sizeof(flagcxHeteroComm));
      comm->heteroComm = hetero;
      hetero->rmaSignalIpcSlot = -1;
      auto *proxy = (flagcxProxyState *)calloc(1, sizeof(flagcxProxyState));
      hetero->proxyState = proxy;
      pthread_mutex_init(&proxy->mutex, nullptr);
      pthread_mutex_init(&proxy->rpcMutex, nullptr);
      pthread_mutex_init(&proxy->relayImportMutex, nullptr);
      pthread_cond_init(&proxy->cond, nullptr);
      for (int i = 0; i < MAXCHANNELS; i++)
        pthread_mutex_init(&proxy->proxyOps[i].mutex, nullptr);
      flagcxRegSnapshot snapshot = {};
      ASSERT_EQ(globalRegPool.registerAndAcquire(hetero, data, 64, &snapshot),
                flagcxSuccess);
      auto *info = (flagcxIpcRegInfo *)calloc(1, sizeof(flagcxIpcRegInfo));
      info->peerRank = 1 - rank;
      info->allocationBase = info->impInfo.importedBase = data;
      info->allocationSize = 64;
      info->sameProcess = info->handleReady = info->impInfo.legacyIpcCap = true;
      ASSERT_EQ(
          globalRegPool.addP2pHandle(hetero, snapshot.item, info, nullptr),
          flagcxSuccess);
      globalRegPool.releaseItem(&snapshot);
    }
    ASSERT_EQ(globalRegPool.cachedP2pCount(nullptr), 2u);
    ASSERT_EQ(flagcxCommDestroy(members[0]), flagcxSuccess);
    EXPECT_EQ(globalRegPool.cachedP2pCount(nullptr), 1u);
    ASSERT_EQ(flagcxCommDestroy(members[1]), flagcxSuccess);
    EXPECT_EQ(globalRegPool.cachedP2pCount(nullptr), 0u);
    flagcxRegSnapshot snapshot = {};
    ASSERT_EQ(globalRegPool.acquireItem(nullptr, data, &snapshot),
              flagcxSuccess);
    EXPECT_EQ(snapshot.item, nullptr);
    globalRegPool.releaseItem(&snapshot);
  }
}
