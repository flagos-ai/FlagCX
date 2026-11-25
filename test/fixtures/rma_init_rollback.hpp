#ifndef FLAGCX_TEST_RMA_INIT_ROLLBACK_HPP_
#define FLAGCX_TEST_RMA_INIT_ROLLBACK_HPP_

#include "adaptor.h"
#include "comm.h"
#include "flagcx_hetero.h"
#include "global_comm.h"
#include "proxy.h"
#include "transport.h"
#include <gtest/gtest.h>

namespace {
class RmaInitRollbackTest : public ::testing::Test {
protected:
  inline static RmaInitRollbackTest *current;
  flagcxDeviceAdaptor *savedDevice = deviceAdaptor;
  flagcxDeviceAdaptor device = {};
  flagcxCCLAdaptor *savedCcl = cclAdaptors[flagcxCCLAdaptorDevice];
  flagcxCCLAdaptor ccl = {};
  flagcxHeteroComm_t owner = nullptr;
  flagcxComm_t outer = nullptr;
  flagcxTransportComm transport = {};
  int allocations = 0, unmaps = 0, doneFrees = 0, readyFrees = 0;
  int unmapFailures = 0, freeFailures = 0, releaseFailures = 0;
  flagcxResult_t unmapFailureResult = flagcxSystemError;
  flagcxResult_t freeFailureResult = flagcxUnhandledDeviceError;
  int nativeFailures = 0, nativeDestroys = 0, nativeFinalizes = 0, drains = 0;
  int closes = 0, releases = 0;
  bool failDevice = false;

  void SetUp() override {
    current = this;
    deviceAdaptor = &device;
    cclAdaptors[flagcxCCLAdaptorDevice] = &ccl;
    device.setDevice = [](int) {
      return current->failDevice ? flagcxRemoteError : flagcxSuccess;
    };
    device.deviceSynchronize = [] {
      current->drains++;
      return flagcxRemoteError;
    };
    device.gdrMemAlloc = [](void **pointer, size_t, void *) {
      *pointer = (void *)(uintptr_t)(++current->allocations * 0x10000);
      return flagcxSuccess;
    };
    device.gdrPtrMmap = [](void **pointer, void *allocation, size_t) {
      *pointer = allocation;
      return flagcxSuccess;
    };
    device.deviceMemset = [](void *, int, size_t, flagcxMemType_t,
                             flagcxStream_t) { return flagcxSuccess; };
    device.gdrPtrMunmap = [](void *, size_t) {
      current->unmaps++;
      if (current->unmapFailures > 0) {
        current->unmapFailures--;
        return current->unmapFailureResult;
      }
      return flagcxSuccess;
    };
    device.gdrMemFree = [](void *pointer, void *) {
      if (pointer == (void *)0x10000)
        current->doneFrees++;
      else {
        current->readyFrees++;
        if (current->freeFailures > 0) {
          current->freeFailures--;
          return current->freeFailureResult;
        }
      }
      return flagcxSuccess;
    };
    ccl.commFinalize = [](flagcxInnerComm_t) {
      current->nativeFinalizes++;
      return flagcxRemoteError;
    };
    ccl.commDestroy = [](flagcxInnerComm_t) {
      current->nativeDestroys++;
      if (current->nativeFailures > 0) {
        current->nativeFailures--;
        return flagcxRemoteError;
      }
      return flagcxSuccess;
    };
  }

  void TearDown() override {
    unmapFailures = freeFailures = releaseFailures = nativeFailures = 0;
    failDevice = false;
    if (outer != nullptr) {
      EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
    }
    if (owner != nullptr) {
      EXPECT_EQ(flagcxHeteroCommRollbackInit(&owner), flagcxSuccess);
    }
    cclAdaptors[flagcxCCLAdaptorDevice] = savedCcl;
    deviceAdaptor = savedDevice;
  }

  void makeOwner(bool rejectStart = true) {
    owner = (flagcxHeteroComm_t)calloc(1, sizeof(flagcxHeteroComm));
    owner->nRanks = 1;
    owner->cudaDev = 7;
    owner->rmaSignalIpcSlot = -1;
    owner->workerPool = new flagcxWorkerPool;
    // Rollback tests reject worker registration; direct-destroy tests also
    // exercise successful startup with a live worker.
    if (rejectStart)
      owner->workerPool->requestStopAll();
    auto *proxy = (flagcxProxyState *)calloc(1, sizeof(flagcxProxyState));
    owner->proxyState = proxy;
    pthread_mutex_init(&proxy->mutex, nullptr);
    pthread_mutex_init(&proxy->rpcMutex, nullptr);
    pthread_mutex_init(&proxy->relayImportMutex, nullptr);
    pthread_cond_init(&proxy->cond, nullptr);
    for (int i = 0; i < MAXCHANNELS; i++) {
      pthread_mutex_init(&proxy->proxyOps[i].mutex, nullptr);
      owner->channels[i].peers =
          (flagcxChannelPeer **)calloc(1, sizeof(flagcxChannelPeer *));
    }
  }

  void transferToOuter() {
    outer = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    outer->commType = flagcxCommunicatorHybrid;
    outer->initRollbackPending = 1;
    outer->finalizeState.lifecycle = flagcxCommDestroying;
    outer->heteroComm = owner;
    owner = nullptr;
    outer->homoComm = (flagcxInnerComm_t)1;
    outer->ipcTable = outer->ipcTableInline;
    outer->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
    // No bootstrap peers exist. Retrying rollback must not enter Finalize.
    outer->nranks = outer->localRanks = 3;
    outer->nativeFinalizeScope = flagcxNativeFinalizeComm;
  }

  void attachFallibleTransport() {
    transport.cleanupProxyConnection = [](flagcxProxyConnection *connection,
                                          int phase) {
      if (phase == flagcxTransportCleanupCloseImports) {
        current->closes++;
        return flagcxSuccess;
      }
      current->releases++;
      if (current->releaseFailures > 0) {
        current->releaseFailures--;
        return flagcxRemoteError;
      }
      connection->transportResources = nullptr;
      return flagcxSuccess;
    };
    auto *pool = (flagcxProxyConnectionPool *)calloc(
        1, sizeof(flagcxProxyConnectionPool));
    pool->banks = 1;
    pool->offset = 1;
    pool->pools = (flagcxProxyConnection **)calloc(1, sizeof(*pool->pools));
    pool->pools[0] =
        (flagcxProxyConnection *)calloc(1, sizeof(flagcxProxyConnection));
    pool->pools[0][0].state = connConnected;
    pool->pools[0][0].tcomm = &transport;
    pool->pools[0][0].transportResources = (void *)1;
    owner->proxyState->connectionPool = pool;
  }
};
} // namespace

#endif
