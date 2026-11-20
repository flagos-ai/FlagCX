#include "adaptor.h"
#include "comm.h"
#include "flagcx_hetero.h"
#include <gtest/gtest.h>

TEST(RmaCleanupTest, RmaPartialCpuMappingAndDeviceFreeRetryKeepOwners) {
  auto *saved = deviceAdaptor;
  flagcxDeviceAdaptor provider = *saved;
  deviceAdaptor = &provider;
  struct Restore {
    flagcxDeviceAdaptor *saved;
    ~Restore() { deviceAdaptor = saved; }
  } restore{saved};
  static int unmaps, doneFrees, readyFrees;
  unmaps = doneFrees = readyFrees = 0;
  provider.gdrPtrMunmap = [](void *, size_t) {
    return ++unmaps == 1 ? flagcxSystemError : flagcxSuccess;
  };
  provider.gdrMemFree = [](void *pointer, void *) {
    if (pointer == (void *)0x2022) {
      doneFrees++;
      return flagcxSuccess;
    }
    return ++readyFrees == 1 ? flagcxUnhandledDeviceError : flagcxSuccess;
  };
  flagcxHeteroComm comm = {};
  auto *proxy = (flagcxRmaProxyState *)calloc(1, sizeof(flagcxRmaProxyState));
  comm.rmaProxy = proxy;
  proxy->nRanks = 1;
  proxy->doneSeqsCpu = (uint64_t *)0x1011;
  proxy->doneSeqsCpuMapped = 1;
  proxy->readySeqsCpu = (uint64_t *)calloc(1, sizeof(uint64_t));
  proxy->doneSeqsDev = (uint64_t *)0x2022;
  proxy->readySeqsDev = (uint64_t *)0x3033;
  proxy->peerProducerMutexes =
      (pthread_mutex_t *)calloc(1, sizeof(pthread_mutex_t));
  pthread_mutex_init(&proxy->peerProducerMutexes[0], nullptr);
  pthread_mutex_init(&proxy->doneMutex, nullptr);
  pthread_cond_init(&proxy->doneCond, nullptr);
  // HOST_FUNC is selected; only one mapping succeeded during initialization.
  EXPECT_EQ(proxy->useStreamOps, 0);
  EXPECT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxSystemError);
  EXPECT_EQ(comm.rmaProxy, proxy);
  EXPECT_EQ(proxy->doneSeqsCpu, (uint64_t *)0x1011);
  EXPECT_EQ(doneFrees, 0);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxSuccess);
  EXPECT_TRUE(proxy->stopComplete);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxSuccess);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxUnhandledDeviceError);
  EXPECT_EQ(comm.rmaProxy, proxy);
  EXPECT_EQ(proxy->doneSeqsCpu, nullptr);
  EXPECT_EQ(proxy->doneSeqsDev, nullptr);
  EXPECT_EQ(proxy->readySeqsDev, (uint64_t *)0x3033);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxSuccess);
  EXPECT_EQ(comm.rmaProxy, nullptr);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 2);
}
