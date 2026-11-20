#include "ib_common.h"
#include "ibvsymbols.h"
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <map>
#include <string>

flagcxResult_t flagcxIbCloseSend(void *);
flagcxResult_t flagcxIbCloseRecv(void *);
flagcxResult_t flagcxIbDestroyBase(flagcxIbNetCommDevBase *);

namespace {
struct CloseCase {
  bool recv;
  const char *failure;
};

// Exercise the real provider close functions, replacing only the verbs
// boundary.
class IbrcCloseRetry : public ::testing::TestWithParam<CloseCase> {
protected:
  inline static IbrcCloseRetry *current;
  flagcxIbvSymbols saved = ibvSymbols;
  ibv_context context = {};
  std::array<ibv_qp, 4> qps = {};
  std::array<ibv_mr, 8> mrs = {};
  std::array<ibv_cq, 3> cqs = {};
  ibv_srq srq = {};
  ibv_ah ah = {};
  ibv_pd pd = {};
  std::map<void *, int> successes;
  std::map<void *, int> attempts;
  void *fail = nullptr;
  bool failed = false;
  flagcxIbSendComm *send = nullptr;
  flagcxIbRecvComm *recv = nullptr;
  static constexpr int dev = MAX_IB_DEVS - 1;
  ibv_pd *savedPd = nullptr;
  int savedRefs = 0;

  static int release(void *pointer) {
    current->attempts[pointer]++;
    if (pointer == current->fail && !current->failed) {
      current->failed = true;
      return EBUSY;
    }
    EXPECT_EQ(current->successes[pointer]++, 0) << "duplicate release";
    return 0;
  }
  static int destroyQp(ibv_qp *p) { return release(p); }
  static int deregMr(ibv_mr *p) {
    EXPECT_EQ(current->successes[&current->qps[0]], 1);
    EXPECT_EQ(current->successes[&current->qps[1]], 1);
#ifndef USE_SHCA
    EXPECT_EQ(current->successes[&current->qps[2]], 1);
#endif
    if (p == &current->mrs[6])
      EXPECT_EQ(current->successes[&current->qps[3]], 1);
    return release(p);
  }
  static int destroyAh(ibv_ah *p) {
    EXPECT_EQ(current->successes[&current->qps[2]], 1);
    return release(p);
  }
  static int destroyCq(ibv_cq *p) { return release(p); }
  static int deallocPd(ibv_pd *p) { return release(p); }
  static int destroySrq(ibv_srq *p) {
    EXPECT_EQ(current->successes[&current->qps[0]], 1);
    EXPECT_EQ(current->successes[&current->qps[1]], 1);
    return release(p);
  }
  static int poll(ibv_cq *, int, ibv_wc *) { return 0; }

  template <typename T>
  T *allocate() {
    void *pointer = nullptr;
    EXPECT_EQ(posix_memalign(&pointer, alignof(T), sizeof(T)), 0);
    memset(pointer, 0, sizeof(T));
    return static_cast<T *>(pointer);
  }
  flagcxIbNetCommBase *base() { return recv ? &recv->base : &send->base; }
  flagcxResult_t close() {
    flagcxResult_t result =
        recv ? flagcxIbCloseRecv(recv) : flagcxIbCloseSend(send);
    if (result == flagcxSuccess) {
      recv = nullptr;
      send = nullptr;
    }
    return result;
  }
  void SetUp() override {
    current = this;
    ibvSymbols.ibv_internal_destroy_qp = destroyQp;
    ibvSymbols.ibv_internal_dereg_mr = deregMr;
    ibvSymbols.ibv_internal_destroy_cq = destroyCq;
    ibvSymbols.ibv_internal_dealloc_pd = deallocPd;
    ibvSymbols.ibv_internal_destroy_srq = destroySrq;
    ibvSymbols.ibv_internal_destroy_ah = destroyAh;
    context.ops.poll_cq = poll;
    for (auto &cq : cqs)
      cq.context = &context;
    savedPd = flagcxIbDevs[dev].pd;
    savedRefs = flagcxIbDevs[dev].pdRefs;
    ASSERT_EQ(savedRefs, 0);
    pthread_mutex_init(&flagcxIbDevs[dev].lock, nullptr);
    flagcxIbDevs[dev].pd = &pd;
    flagcxIbDevs[dev].pdRefs = 1;
    flagcxIbNetCommDevBase *device;
    flagcxIbCtrlQp *ctrl;
    if (GetParam().recv) {
      recv = allocate<flagcxIbRecvComm>();
      device = &recv->devs[0].base;
      ctrl = &recv->devs[0].ctrlQp;
      recv->devs[0].fifoMr = &mrs[0];
      recv->devs[0].sizesFifoMr = &mrs[1];
      recv->devs[0].ackMr = &mrs[2];
      recv->devs[0].ackBuffer = malloc(16);
      recv->devs[0].gpuFlush.qp.qp = &qps[3];
      recv->devs[0].gpuFlush.hostMr = &mrs[6];
#ifndef USE_SHCA
      recv->srqMgr.srq = &srq;
      recv->srqMgr.bufCount = 1;
      recv->srqMgr.bufs[0].mr = &mrs[7];
      recv->srqMgr.bufs[0].buffer = malloc(16);
#endif
    } else {
      send = allocate<flagcxIbSendComm>();
      device = &send->devs[0].base;
      ctrl = &send->devs[0].ctrlQp;
      send->devs[0].fifoMr = &mrs[0];
      send->devs[0].putSignalScratchpadMr = &mrs[1];
      send->devs[0].ackMr = &mrs[2];
      send->devs[0].ackBuffer = malloc(16);
      send->remSizesFifo.mrs[0] = &mrs[3];
      send->retransHdrMr = &mrs[4];
    }
#ifndef USE_SHCA
    ctrl->qp = &qps[2];
    ctrl->ah = &ah;
    ctrl->cq = &cqs[1];
#endif
    device->cq = &cqs[0];
    device->pd = &pd;
    device->ibDevN = dev;
    base()->ndevs = 1;
    base()->nqps = 2;
    base()->qps[0].qp = &qps[0];
    base()->qps[1].qp = &qps[1];
    base()->sock.fd = base()->sock.acceptFd = -1;
    std::string stage = GetParam().failure;
    if (stage == "qp")
      fail = &qps[1];
    if (stage == "mr")
      fail = &mrs[1];
    if (stage == "ack")
      fail = &mrs[2];
    if (stage == "cq")
      fail = &cqs[0];
    if (stage == "pd")
      fail = &pd;
    if (stage == "ah")
      fail = &ah;
    if (stage == "device") {
      base()->ndevs = 2;
      flagcxIbDevs[dev].pdRefs = 2;
      auto *other = recv ? &recv->devs[1].base : &send->devs[1].base;
      other->pd = &pd;
      other->ibDevN = dev;
      other->cq = &cqs[2];
      if (recv)
        recv->devs[1].fifoMr = &mrs[5];
      else
        send->devs[1].fifoMr = &mrs[5];
      fail = &mrs[5];
    }
    if (stage == "ctrl")
      fail = &qps[2];
    if (stage == "srq")
      fail = &srq;
    if (stage == "flush")
      fail = &qps[3];
  }
  void TearDown() override {
    fail = nullptr;
    if (recv || send)
      EXPECT_EQ(close(), flagcxSuccess);
    EXPECT_EQ(flagcxIbDevs[dev].pdRefs, 0);
    flagcxIbDevs[dev].pd = savedPd;
    flagcxIbDevs[dev].pdRefs = savedRefs;
    pthread_mutex_destroy(&flagcxIbDevs[dev].lock);
    ibvSymbols = saved;
  }
};

TEST_P(IbrcCloseRetry, PreservesOwnerAndCommitsSuccessfulPrefixes) {
#ifdef USE_SHCA
  if (std::string(GetParam().failure) == "ctrl" ||
      std::string(GetParam().failure) == "srq" ||
      std::string(GetParam().failure) == "ah")
    GTEST_SKIP() << "UD control resources are unavailable on SHCA";
#endif
  ASSERT_NE(fail, nullptr);
  EXPECT_EQ(close(), flagcxSystemError);
  ASSERT_TRUE(failed);
  ASSERT_TRUE(recv || send);
  EXPECT_EQ(successes[fail], 0);
  if (std::string(GetParam().failure) == "device") {
    EXPECT_EQ(flagcxIbDevs[dev].pdRefs, 2);
    EXPECT_EQ(successes[&mrs[0]], 1);
    EXPECT_EQ(attempts[&cqs[0]], 0);
  } else {
    EXPECT_EQ(flagcxIbDevs[dev].pdRefs, 1);
  }
  if (std::string(GetParam().failure) == "ack")
    EXPECT_NE(recv ? recv->devs[0].ackBuffer : send->devs[0].ackBuffer,
              nullptr);
  EXPECT_EQ(close(), flagcxSuccess);
  EXPECT_EQ(attempts[fail], 2);
  EXPECT_EQ(successes[fail], 1);
  EXPECT_EQ(successes[&pd], 1);
  EXPECT_EQ(successes[&cqs[0]], 1);
  for (int i = 0; i < 2; i++)
    EXPECT_EQ(successes[&qps[i]], 1);
  for (int i = 0; i < (GetParam().recv ? 3 : 5); i++)
    EXPECT_EQ(successes[&mrs[i]], 1);
  if (GetParam().recv) {
    EXPECT_EQ(successes[&qps[3]], 1);
    EXPECT_EQ(successes[&mrs[6]], 1);
#ifndef USE_SHCA
    EXPECT_EQ(successes[&srq], 1);
    EXPECT_EQ(successes[&mrs[7]], 1);
#endif
  }
#ifndef USE_SHCA
  EXPECT_EQ(successes[&qps[2]], 1);
  EXPECT_EQ(successes[&ah], 1);
  EXPECT_EQ(successes[&cqs[1]], 1);
#endif
  for (const auto &entry : successes)
    EXPECT_EQ(entry.second, 1);
}

INSTANTIATE_TEST_SUITE_P(
    Provider, IbrcCloseRetry,
    ::testing::Values(CloseCase{false, "qp"}, CloseCase{false, "mr"},
                      CloseCase{false, "ack"}, CloseCase{false, "cq"},
                      CloseCase{false, "pd"}, CloseCase{false, "device"},
                      CloseCase{false, "ah"}, CloseCase{false, "ctrl"},
                      CloseCase{true, "qp"}, CloseCase{true, "mr"},
                      CloseCase{true, "ack"}, CloseCase{true, "cq"},
                      CloseCase{true, "pd"}, CloseCase{true, "device"},
                      CloseCase{true, "ah"}, CloseCase{true, "ctrl"},
                      CloseCase{true, "srq"}, CloseCase{true, "flush"}),
    [](const ::testing::TestParamInfo<CloseCase> &info) {
      return std::string(info.param.recv ? "Recv_" : "Send_") +
             info.param.failure;
    });

TEST(IbrcSharedPdRetry, LastOwnerFailureRetainsExactlyOneReference) {
  auto saved = ibvSymbols.ibv_internal_dealloc_pd;
  static int calls;
  calls = 0;
  ibvSymbols.ibv_internal_dealloc_pd = [](ibv_pd *) {
    return ++calls == 1 ? EBUSY : 0;
  };
  constexpr int dev = MAX_IB_DEVS - 1;
  ibv_pd pd = {};
  auto *ibDev = &flagcxIbDevs[dev];
  auto *savedPd = ibDev->pd;
  int savedRefs = ibDev->pdRefs;
  ASSERT_EQ(savedRefs, 0);
  pthread_mutex_init(&ibDev->lock, nullptr);
  ibDev->pd = &pd;
  ibDev->pdRefs = 2;
  flagcxIbNetCommDevBase first = {}, last = {};
  first.ibDevN = last.ibDevN = dev;
  first.pd = last.pd = &pd;
  EXPECT_EQ(flagcxIbDestroyBase(&first), flagcxSuccess);
  EXPECT_EQ(first.pd, nullptr);
  EXPECT_EQ(ibDev->pdRefs, 1);
  EXPECT_EQ(flagcxIbDestroyBase(&last), flagcxSystemError);
  EXPECT_EQ(last.pd, &pd);
  EXPECT_EQ(ibDev->pdRefs, 1);
  EXPECT_EQ(flagcxIbDestroyBase(&first), flagcxSuccess);
  EXPECT_EQ(flagcxIbDestroyBase(&last), flagcxSuccess);
  EXPECT_EQ(last.pd, nullptr);
  EXPECT_EQ(ibDev->pd, nullptr);
  EXPECT_EQ(ibDev->pdRefs, 0);
  EXPECT_EQ(calls, 2);
  ibDev->pd = savedPd;
  ibDev->pdRefs = savedRefs;
  pthread_mutex_destroy(&ibDev->lock);
  ibvSymbols.ibv_internal_dealloc_pd = saved;
}
} // namespace
