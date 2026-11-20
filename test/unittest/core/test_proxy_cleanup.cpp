#include "adaptor.h"
#include "comm.h"
#include "net.h"
#include "proxy.h"
#include "transport.h"
#include <gtest/gtest.h>

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
