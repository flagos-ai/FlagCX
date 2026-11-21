#include "comm.h"
#include "flagcx_kernel_internal.h"
#include "proxy.h"
#include <gtest/gtest.h>

namespace {
class ProxyFifoCleanupTest : public ::testing::Test {
protected:
  flagcxDeviceAdaptor *saved = nullptr;
  flagcxDeviceAdaptor device = {};
  inline static int frees;
  inline static flagcxResult_t freeResult;
  void SetUp() override {
    saved = deviceAdaptor;
    device = *saved;
    deviceAdaptor = &device;
    frees = 0;
    freeResult = flagcxInProgress;
    device.deviceFree = [](void *, flagcxMemType_t type, flagcxStream_t) {
      EXPECT_EQ(type, flagcxMemHost);
      frees++;
      return freeResult;
    };
  }
  void TearDown() override { deviceAdaptor = saved; }
};

TEST_F(ProxyFifoCleanupTest, BufferOwnerSurvivesPendingAndFailedFree) {
  flagcxFifo fifo;
  fifo.buffer = (uint64_t *)0x1000;
  EXPECT_EQ(fifo.flagcxFifoDestroy(), flagcxInProgress);
  EXPECT_EQ(fifo.buffer, (uint64_t *)0x1000);
  freeResult = flagcxSystemError;
  EXPECT_EQ(fifo.flagcxFifoDestroy(), flagcxSystemError);
  EXPECT_EQ(fifo.buffer, (uint64_t *)0x1000);
  freeResult = flagcxSuccess;
  EXPECT_EQ(fifo.flagcxFifoDestroy(), flagcxSuccess);
  EXPECT_EQ(fifo.buffer, nullptr);
  EXPECT_EQ(fifo.flagcxFifoDestroy(), flagcxSuccess);
  EXPECT_EQ(frees, 3);
}

#ifdef COMPILE_KERNEL_HOST
TEST_F(ProxyFifoCleanupTest, FinalizeDefersFifoReleaseToLocalDestroy) {
  flagcxHeteroComm comm = {};
  flagcxProxyState proxy = {};
  comm.proxyState = &proxy;
  proxy.kernelState.contextCount = 1;
  auto *fifo = new flagcxFifo;
  fifo->buffer = (uint64_t *)0x1000;
  proxy.kernelState.fifos[0] = fifo;
  comm.fifoBuffers[0] = fifo->buffer;
  ASSERT_EQ(flagcxProxyFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(frees, 0);
  ASSERT_EQ(flagcxProxyDestroy(&comm), flagcxInProgress);
  EXPECT_EQ(proxy.kernelState.fifos[0], fifo);
  EXPECT_EQ(fifo->buffer, (uint64_t *)0x1000);
  EXPECT_EQ(comm.fifoBuffers[0], (void *)0x1000);
  freeResult = flagcxSuccess;
  EXPECT_EQ(flagcxProxyDestroy(&comm), flagcxSuccess);
  EXPECT_EQ(proxy.kernelState.fifos[0], nullptr);
  EXPECT_EQ(comm.fifoBuffers[0], nullptr);
  EXPECT_EQ(flagcxProxyFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(frees, 2);
}
#endif
} // namespace
