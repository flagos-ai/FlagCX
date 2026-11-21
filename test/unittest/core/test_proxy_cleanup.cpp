#include "adaptor.h"
#include "comm.h"
#include "net.h"
#include "proxy.h"
#include "transport.h"
#include <gtest/gtest.h>
#include <tuple>

TEST(ProxyCleanupTest, FailedCloseAndReleaseKeepPoolAndSuccessfulPrefix) {
  flagcxHeteroComm comm = {};
  flagcxProxyState proxy = {};
  comm.proxyState = &proxy;
  auto *pool =
      (flagcxProxyConnectionPool *)calloc(1, sizeof(flagcxProxyConnectionPool));
  proxy.connectionPool = pool;
  pool->banks = 1;
  pool->offset = 2;
  pool->pools = (flagcxProxyConnection **)calloc(1, sizeof(*pool->pools));
  pool->pools[0] =
      (flagcxProxyConnection *)calloc(2, sizeof(flagcxProxyConnection));
  static int closes, releases;
  static bool failClose, failRelease;
  closes = releases = 0;
  failClose = failRelease = true;
  flagcxTransportComm transport = {};
  transport.cleanupProxyConnection = [](flagcxProxyConnection *conn,
                                        int phase) {
    if (phase == flagcxTransportCleanupCloseImports) {
      closes++;
      return failClose ? flagcxSystemError : flagcxSuccess;
    }
    if (conn->transportResources == nullptr)
      return flagcxSuccess;
    releases++;
    if (conn->send && failRelease)
      return flagcxUnhandledDeviceError;
    conn->transportResources = nullptr;
    return flagcxSuccess;
  };
  for (int i = 0; i < 2; i++) {
    pool->pools[0][i].state = connConnected;
    pool->pools[0][i].tcomm = &transport;
    pool->pools[0][i].send = i;
    pool->pools[0][i].transportResources = (void *)1;
  }
  EXPECT_EQ(flagcxProxyFinalize(&comm), flagcxSystemError);
  EXPECT_EQ(proxy.connectionPool, pool);
  EXPECT_EQ(releases, 0);
  failClose = false;
  EXPECT_EQ(flagcxProxyFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(flagcxProxyDestroy(&comm), flagcxUnhandledDeviceError);
  EXPECT_EQ(proxy.connectionPool, pool);
  EXPECT_EQ(releases, 2);
  failRelease = false;
  EXPECT_EQ(flagcxProxyDestroy(&comm), flagcxSuccess);
  EXPECT_EQ(proxy.connectionPool, nullptr);
  EXPECT_EQ(releases, 3);
}

TEST(ProxyCleanupTest, NetCloseAndFreeFailuresRetainDescriptorForLocalRetry) {
  auto *saved = deviceAdaptor;
  flagcxDeviceAdaptor device = *saved;
  deviceAdaptor = &device;
  struct Restore {
    flagcxDeviceAdaptor *saved;
    ~Restore() { deviceAdaptor = saved; }
  } restore{saved};
  static int deregisters, closes, frees;
  static bool failClose, failFree;
  deregisters = closes = frees = 0;
  failClose = failFree = true;
  flagcxNetAdaptor net = {};
  net.deregMr = [](void *, void *) {
    deregisters++;
    return flagcxSuccess;
  };
  net.closeSend = [](void *) {
    closes++;
    return failClose ? flagcxSystemError : flagcxSuccess;
  };
  device.gdrMemFree = [](void *, void *) {
    frees++;
    return failFree ? flagcxUnhandledDeviceError : flagcxSuccess;
  };
  auto *resources = (sendNetResources *)calloc(1, sizeof(sendNetResources));
  resources->netAdaptor = &net;
  resources->netSendComm = (void *)1;
  resources->mhandles[0] = (void *)2;
  resources->buffers[0] = (char *)3;
  resources->ptrSupport = FLAGCX_PTR_CUDA;
  flagcxProxyConnection conn = {};
  conn.send = 1;
  conn.transportResources = resources;
  EXPECT_EQ(flagcxNetCleanupProxyConnection(
                &conn, flagcxTransportCleanupReleaseResources),
            flagcxSystemError);
  EXPECT_EQ(conn.transportResources, resources);
  EXPECT_EQ(resources->mhandles[0], nullptr);
  EXPECT_EQ(resources->netSendComm, (void *)1);
  EXPECT_EQ(frees, 0);
  failClose = false;
  EXPECT_EQ(flagcxNetCleanupProxyConnection(
                &conn, flagcxTransportCleanupReleaseResources),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(conn.transportResources, resources);
  EXPECT_EQ(resources->netSendComm, nullptr);
  EXPECT_EQ(resources->buffers[0], (void *)3);
  failFree = false;
  EXPECT_EQ(flagcxNetCleanupProxyConnection(
                &conn, flagcxTransportCleanupReleaseResources),
            flagcxSuccess);
  EXPECT_EQ(conn.transportResources, nullptr);
  EXPECT_EQ(deregisters, 1);
  EXPECT_EQ(closes, 2);
  EXPECT_EQ(frees, 2);
}

namespace {
class ProxyPendingCleanupTest
    : public ::testing::TestWithParam<std::tuple<int, bool, flagcxResult_t>> {};

TEST_P(ProxyPendingCleanupTest, RetainsPoolAndShadowsUntilSuccess) {
  const int phase = std::get<0>(GetParam());
  const bool throughOwner = std::get<1>(GetParam());
  const flagcxResult_t failure = std::get<2>(GetParam());
  flagcxHeteroComm comm = {};
  flagcxProxyState proxy = {};
  comm.proxyState = &proxy;
  flagcxProxyConnector shadow = {};
  comm.nRanks = 1;
  comm.gproxyConn = &shadow;
  shadow.connection =
      (flagcxProxyConnection *)calloc(1, sizeof(flagcxProxyConnection));
  auto *pool =
      (flagcxProxyConnectionPool *)calloc(1, sizeof(flagcxProxyConnectionPool));
  proxy.connectionPool = pool;
  pool->banks = 1;
  pool->offset = 2;
  pool->pools = (flagcxProxyConnection **)calloc(1, sizeof(*pool->pools));
  pool->pools[0] =
      (flagcxProxyConnection *)calloc(2, sizeof(flagcxProxyConnection));
  static flagcxResult_t pendingResult;
  static int pendingPhase, closes[2], releases[2];
  static bool closed[2];
  pendingResult = failure;
  pendingPhase = phase;
  for (int i = 0; i < 2; i++) {
    closes[i] = releases[i] = 0;
    closed[i] = false;
  }
  flagcxTransportComm transport = {};
  transport.cleanupProxyConnection = [](flagcxProxyConnection *conn,
                                        int nextPhase) {
    int i = conn->send;
    if (nextPhase == flagcxTransportCleanupCloseImports) {
      if (closed[i])
        return flagcxSuccess;
      closes[i]++;
      if (i == 1 && pendingPhase == nextPhase && pendingResult != flagcxSuccess)
        return pendingResult;
      closed[i] = true;
    } else if (conn->transportResources != nullptr) {
      releases[i]++;
      if (i == 1 && pendingPhase == nextPhase && pendingResult != flagcxSuccess)
        return pendingResult;
      conn->transportResources = nullptr;
    }
    return flagcxSuccess;
  };
  for (int i = 0; i < 2; i++) {
    pool->pools[0][i].state = connConnected;
    pool->pools[0][i].tcomm = &transport;
    pool->pools[0][i].send = i;
    pool->pools[0][i].transportResources = (void *)1;
  }
  auto cleanup = [&] {
    if (!throughOwner)
      return flagcxProxyCleanupConnections(&comm, phase);
    return phase == flagcxTransportCleanupCloseImports
               ? flagcxProxyFinalize(&comm)
               : flagcxProxyDestroy(&comm);
  };
  ASSERT_EQ(cleanup(), failure);
  ASSERT_EQ(proxy.connectionPool, pool);
  EXPECT_NE(shadow.connection, nullptr);
  EXPECT_NE(pool->pools[0][1].transportResources, nullptr);
  EXPECT_FALSE(comm.destroyProxyComplete);
  if (phase == flagcxTransportCleanupCloseImports) {
    EXPECT_EQ(releases[0] + releases[1], 0);
    EXPECT_FALSE(closed[1]);
  } else {
    EXPECT_EQ(pool->pools[0][0].transportResources, nullptr);
  }
  pendingResult = flagcxSuccess;
  ASSERT_EQ(cleanup(), flagcxSuccess);
  if (phase == flagcxTransportCleanupCloseImports) {
    EXPECT_EQ(closes[0], 1);
    EXPECT_EQ(closes[1], 2);
    EXPECT_EQ(proxy.connectionPool, pool);
  }
  EXPECT_EQ(flagcxProxyDestroy(&comm), flagcxSuccess);
  EXPECT_EQ(proxy.connectionPool, nullptr);
  EXPECT_EQ(shadow.connection, nullptr);
  EXPECT_EQ(releases[0], 1);
  EXPECT_EQ(releases[1],
            phase == flagcxTransportCleanupReleaseResources ? 2 : 1);
}

INSTANTIATE_TEST_SUITE_P(
    PendingAndErrors, ProxyPendingCleanupTest,
    ::testing::Combine(
        ::testing::Values(flagcxTransportCleanupCloseImports,
                          flagcxTransportCleanupReleaseResources),
        ::testing::Bool(),
        ::testing::Values(flagcxInProgress, flagcxSystemError)));
} // namespace

namespace {
class NetPendingCleanupTest : public ::testing::TestWithParam<bool> {};

TEST_P(NetPendingCleanupTest,
       RetainsEveryPendingResourceAndSkipsReleasedPrefix) {
  auto *saved = deviceAdaptor;
  flagcxDeviceAdaptor device = *saved;
  deviceAdaptor = &device;
  struct Restore {
    flagcxDeviceAdaptor *saved;
    int64_t chunks;
    ~Restore() {
      deviceAdaptor = saved;
      flagcxNetChunks = chunks;
    }
  } restore{saved, flagcxNetChunks};
  flagcxNetChunks = 2;
  const bool send = GetParam();
  auto *sender =
      send ? (sendNetResources *)calloc(1, sizeof(sendNetResources)) : nullptr;
  auto *receiver =
      send ? nullptr : (recvNetResources *)calloc(1, sizeof(recvNetResources));
  flagcxProxyConnection connection = {};
  connection.send = send;
  connection.transportResources = send ? (void *)sender : (void *)receiver;
  flagcxEvent_t *events = send ? sender->cpEvents : receiver->cpEvents;
  flagcxStream_t *stream = send ? &sender->cpStream : &receiver->cpStream;
  void **native = send ? &sender->netSendComm : &receiver->netRecvComm;
  void **mr = send ? sender->mhandles : receiver->mhandles;
  char **buffers = send ? sender->buffers : receiver->buffers;
  static int pendingStage, calls[6], firstEventCalls;
  pendingStage = -1;
  firstEventCalls = 0;
  for (int &call : calls)
    call = 0;
  device.eventDestroy = [](flagcxEvent_t event) {
    if (event == (flagcxEvent_t)0x1000) {
      firstEventCalls++;
      return flagcxSuccess;
    }
    calls[0]++;
    return pendingStage == 0 ? flagcxInProgress : flagcxSuccess;
  };
  device.streamDestroy = [](flagcxStream_t) {
    calls[1]++;
    return pendingStage == 1 ? flagcxInProgress : flagcxSuccess;
  };
  device.gdrMemFree = [](void *, void *) {
    calls[5]++;
    return pendingStage == 5 ? flagcxInProgress : flagcxSuccess;
  };
  flagcxNetAdaptor net = {};
  net.deregMr = [](void *, void *) {
    calls[2]++;
    return pendingStage == 2 ? flagcxInProgress : flagcxSuccess;
  };
  net.closeSend = net.closeRecv = [](void *) {
    calls[3]++;
    return pendingStage == 3 ? flagcxInProgress : flagcxSuccess;
  };
  net.closeListen = [](void *) {
    calls[4]++;
    return pendingStage == 4 ? flagcxInProgress : flagcxSuccess;
  };
  if (send) {
    sender->netAdaptor = &net;
    sender->ptrSupport = FLAGCX_PTR_CUDA;
  } else {
    receiver->netAdaptor = &net;
    receiver->ptrSupport = FLAGCX_PTR_CUDA;
    receiver->netListenComm = (void *)0x3000;
  }
  events[0] = (flagcxEvent_t)0x1000;
  events[1] = (flagcxEvent_t)0x2000;
  *stream = (flagcxStream_t)0x4000;
  *native = (void *)0x5000;
  mr[0] = (void *)0x6000;
  buffers[0] = (char *)0x7000;
  auto cleanup = [&] {
    return flagcxNetCleanupProxyConnection(
        &connection, flagcxTransportCleanupReleaseResources);
  };
  for (int stage = 0; stage < 6; stage++) {
    if (send && stage == 4)
      continue;
    SCOPED_TRACE(stage);
    pendingStage = stage;
    ASSERT_EQ(cleanup(), flagcxInProgress);
    ASSERT_NE(connection.transportResources, nullptr);
    EXPECT_EQ(events[0], nullptr);
    if (stage == 0) {
      EXPECT_NE(events[1], nullptr);
    }
    if (stage == 1) {
      EXPECT_NE(*stream, nullptr);
    }
    if (stage == 2) {
      EXPECT_NE(mr[0], nullptr);
    }
    if (stage == 3) {
      EXPECT_NE(*native, nullptr);
    }
    if (stage == 4) {
      EXPECT_NE(receiver->netListenComm, nullptr);
    }
    if (stage == 5) {
      EXPECT_NE(buffers[0], nullptr);
    }
    EXPECT_EQ(firstEventCalls, 1);
  }
  pendingStage = -1;
  EXPECT_EQ(cleanup(), flagcxSuccess);
  EXPECT_EQ(connection.transportResources, nullptr);
  EXPECT_EQ(cleanup(), flagcxSuccess);
  for (int stage = 0; stage < 6; stage++)
    EXPECT_EQ(calls[stage], send && stage == 4 ? 0 : 2);
  EXPECT_EQ(firstEventCalls, 1);
}

INSTANTIATE_TEST_SUITE_P(SendAndRecv, NetPendingCleanupTest, ::testing::Bool());
} // namespace
