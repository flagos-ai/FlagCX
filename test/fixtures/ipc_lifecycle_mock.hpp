#pragma once
#include "adaptor.h"
#include "bootstrap.h"
#include "dev_api_backend.h"
#include "device_api/flagcx_device.h"
#include "flagcx_hetero.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "proxy.h"
#include "transport.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <new>
#include <vector>

// Runs production Finalize/Destroy with deterministic allocator/SDK failures.
// MPI cases supply real bootstrap state; no accelerator is emulated as
// hardware.
class IpcLifecycleMock : public ::testing::Test {
public:
  inline static IpcLifecycleMock *active;
  flagcxComm_t comm = nullptr;
  flagcxDeviceAdaptor device = {};
  flagcxCCLAdaptor sdk = {}, host = {};
  flagcxDevApiBackend backend = {};
  flagcxDeviceAdaptor *savedDevice;
  flagcxCCLAdaptor *savedSdk, *savedHost;
  flagcxDevApiBackend *savedBackend;
  flagcxTransportComm transport = {};
  flagcxResult_t nativeResult = flagcxSuccess, asyncResult = flagcxSuccess;
  flagcxResult_t freeError = flagcxRemoteError;
  bool failFree = false, failImport = false, failDevice = false;
  bool failTransportClose = false, failTransportFree = false;
  int finalizeCalls = 0, destroyCalls = 0, asyncPolls = 0, syncs = 0;
  int backendCleanups = 0, transportCloses = 0, transportFrees = 0;
  int pendingPolls = 0;
  std::vector<void *> closes, frees;
  struct TransportResource {
    bool imported;
  };

  void SetUp() override {
    active = this;
    savedDevice = deviceAdaptor;
    savedSdk = cclAdaptors[flagcxCCLAdaptorDevice];
    savedHost = cclAdaptors[flagcxCCLAdaptorHost];
    savedBackend = devApiBackend;
    device.setDevice = [](int) {
      return active->failDevice ? flagcxRemoteError : flagcxSuccess;
    };
    device.deviceMalloc = [](void **p, size_t bytes, flagcxMemType_t,
                             flagcxStream_t) {
      *p = calloc(1, bytes);
      return *p ? flagcxSuccess : flagcxSystemError;
    };
    device.deviceFree = [](void *p, flagcxMemType_t, flagcxStream_t) {
      if (active->failFree)
        return active->freeError;
      active->frees.push_back(p);
      free(p);
      return flagcxSuccess;
    };
    device.ipcMemHandleClose = [](void *p) {
      active->closes.push_back(p);
      return active->failImport ? flagcxRemoteError : flagcxSuccess;
    };
    device.deviceSynchronize = [] {
      ++active->syncs;
      return flagcxUnhandledDeviceError;
    };
    sdk.name = host.name = "MOCK";
    sdk.commFinalize = host.commFinalize = [](flagcxInnerComm_t) {
      ++active->finalizeCalls;
      return active->nativeResult;
    };
    sdk.commGetAsyncError = host.commGetAsyncError = [](flagcxInnerComm_t,
                                                        flagcxResult_t *r) {
      ++active->asyncPolls;
      *r = active->pendingPolls-- > 0 ? flagcxInProgress : active->asyncResult;
      return flagcxSuccess;
    };
    sdk.commDestroy = host.commDestroy = [](flagcxInnerComm_t) {
      ++active->destroyCalls;
      return flagcxSuccess;
    };
    deviceAdaptor = &device;
    cclAdaptors[flagcxCCLAdaptorDevice] = &sdk;
    cclAdaptors[flagcxCCLAdaptorHost] = &host;
    backend = *savedBackend;
    backend.commQuiesce = [](flagcxComm_t) { return flagcxSuccess; };
    backend.commCleanup = [](flagcxComm_t) {
      ++active->backendCleanups;
      return flagcxSuccess;
    };
    devApiBackend = &backend;
    comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    ASSERT_NE(comm, nullptr);
    new (comm) flagcxComm{};
    comm->commType = flagcxCommunicatorHomo;
    comm->rank = 0;
    comm->nranks = comm->localRanks = 1;
    comm->deviceOrdinalValid = true;
  }
  void TearDown() override {
    failFree = failImport = failDevice = failTransportClose =
        failTransportFree = false;
    nativeResult = asyncResult = flagcxSuccess;
    pendingPolls = 0;
    sdk.commGetAsyncError =
        host.commGetAsyncError = [](flagcxInnerComm_t, flagcxResult_t *r) {
          *r = flagcxSuccess;
          return flagcxSuccess;
        };
    if (comm)
      EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    comm = nullptr;
    devApiBackend = savedBackend;
    deviceAdaptor = savedDevice;
    cclAdaptors[flagcxCCLAdaptorDevice] = savedSdk;
    cclAdaptors[flagcxCCLAdaptorHost] = savedHost;
  }
  void addBackingAndImport() {
    void *buffer = nullptr;
    ASSERT_EQ(flagcxCommAllocInternalBuffer(comm, &buffer, 64,
                                            flagcxInternalBufferDevice),
              flagcxSuccess);
    auto &entry = comm->ipcTable[0];
    entry.ownership = flagcxBufferInternal;
    entry.inUse = entry.ownerHeld = true;
    entry.basePtr = buffer;
    entry.nPeers = 1;
    entry.hostPeerBasePtrs = (void **)calloc(1, sizeof(void *));
    entry.hostPeerBasePtrs[0] = (void *)0x55000;
    entry.devPeerPtrs = (void **)calloc(1, sizeof(void *));
  }
  void addTransport() {
    comm->commType = flagcxCommunicatorHybrid;
    auto *child = (flagcxHeteroComm *)calloc(1, sizeof(flagcxHeteroComm));
    ASSERT_NE(child, nullptr);
    comm->heteroComm = child;
    child->ownerComm = comm;
    child->nRanks = 1;
    child->proxyState = (flagcxProxyState *)calloc(1, sizeof(flagcxProxyState));
    auto *state = child->proxyState;
    pthread_mutex_init(&state->mutex, nullptr);
    pthread_mutex_init(&state->rpcMutex, nullptr);
    pthread_mutex_init(&state->relayImportMutex, nullptr);
    pthread_cond_init(&state->cond, nullptr);
    for (int i = 0; i < MAXCHANNELS; ++i)
      pthread_mutex_init(&state->proxyOps[i].mutex, nullptr);
    state->connectionPool = (flagcxProxyConnectionPool *)calloc(
        1, sizeof(flagcxProxyConnectionPool));
    auto *pool = state->connectionPool;
    pool->banks = pool->offset = 1;
    pool->pools =
        (flagcxProxyConnection **)calloc(1, sizeof(flagcxProxyConnection *));
    pool->pools[0] =
        (flagcxProxyConnection *)calloc(1, sizeof(flagcxProxyConnection));
    auto *connection = pool->pools[0];
    connection->transport = TRANSPORT_NET;
    connection->state = connConnected;
    connection->tcomm = &transport;
    auto *resource = (TransportResource *)calloc(1, sizeof(TransportResource));
    resource->imported = true;
    connection->transportResources = resource;
    transport.cleanupProxyConnection = [](flagcxProxyConnection *c, int phase) {
      auto *r = (TransportResource *)c->transportResources;
      if (!r)
        return flagcxSuccess;
      if (phase == flagcxTransportCleanupCloseImports) {
        if (!r->imported)
          return flagcxSuccess;
        ++active->transportCloses;
        if (active->failTransportClose)
          return flagcxRemoteError;
        r->imported = false;
      } else {
        if (r->imported)
          return flagcxInvalidUsage;
        ++active->transportFrees;
        if (active->failTransportFree)
          return flagcxRemoteError;
        free(r);
        c->transportResources = nullptr;
      }
      return flagcxSuccess;
    };
  }
  void destroySuccessfully() {
    ASSERT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    comm = nullptr;
  }
};
