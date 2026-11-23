/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "device_api/comm_traits.h"
#include "device_api/flagcx_device_internal.h"
#include "kernel_proxy_completion.h"
#include "trigger_accessors.inc"

#include <cassert>
#include <cstdio>
#include <limits>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifdef FLAGCX_FIFO_TEST_WORD32
struct FifoWord32Platform {};
template <>
struct PlatformTraits<FifoWord32Platform> : PlatformTraits<DefaultPlatform> {};
template <>
struct PlatformCompletionWord<FifoWord32Platform> {
  using type = uint32_t;
};
using API = CommTraits<DefaultBackend<FifoWord32Platform>>;
#else
using API = CommTraits<DefaultBackend<DefaultPlatform>>;
#endif

struct flagcxHeteroComm {
  int rank;
  flagcxDevComm_t devCommHandle;
  void *signalHandle;
};

struct PostedSignal {
  int peer, context, srcMr, dstMr;
  size_t src, dst, bytes;
  uint64_t signalOffset, value;
  bool posted = false;
} postedSignal;

static constexpr int FLAGCX_RMA_PUT_SIGNAL = 1;
static flagcxResult_t flagcxKernelProxyValidatePeer(flagcxHeteroComm *, int,
                                                    int) {
  return flagcxSuccess;
}
static flagcxResult_t
flagcxKernelProxyPost(void *, flagcxHeteroComm *, int *, int peer, int op,
                      int context, size_t src, size_t dst, size_t bytes,
                      int srcMr, int dstMr, uint64_t signalOffset,
                      uint64_t value, uint64_t, bool *posted) {
  assert(op == FLAGCX_RMA_PUT_SIGNAL);
  postedSignal.peer = peer;
  postedSignal.context = context;
  postedSignal.src = src;
  postedSignal.dst = dst;
  postedSignal.bytes = bytes;
  postedSignal.srcMr = srcMr;
  postedSignal.dstMr = dstMr;
  postedSignal.signalOffset = signalOffset;
  postedSignal.value = value;
  postedSignal.posted = true;
  *posted = true;
  return flagcxSuccess;
}

// Actual Signal/SignalValue/PutSignal worker bodies. Logging and the RMA
// submission are replaced; index validation and local counter updates are real.
static flagcxResult_t consume(flagcxDeviceTrigger pending,
                              flagcxDevCommInternal *dc, int contextId) {
  flagcxHeteroComm outer = {0, dc, reinterpret_cast<void *>(1)};
  flagcxHeteroComm *comm = &outer;
  const int ctx = contextId + 1;
  void *kproxyState = nullptr;
  int submit = 0;
  bool postedIB = false, retryPending = false;
  postedSignal = PostedSignal{};
  flagcxResult_t res = flagcxSuccess;
#define WARN(...) ((void)0)
#define TRACE(...) ((void)0)
#include "completion_consumer.inc"
#undef TRACE
#undef WARN
  (void)postedIB;
  (void)retryPending;
  return res;
}

// One real FIFO slot. The producer still reserves/publishes it with the real
// producer gate and atomics; the host consumer captures the three wire words.
struct Fixture {
  std::vector<uint64_t> fifo;
  API::Comm comm;
  Fixture(int count = 65537) : fifo(flagcxFifoIdxData + 3, 0), comm{} {
    fifo[flagcxFifoIdxCapacity] = 1;
    comm.contextCount = 36;
    comm.signalCount = count;
    comm.counterCount = count;
    for (int i = 0; i < 36; ++i)
      comm.fifoBuffers[i] = fifo.data();
  }
  flagcxDeviceTrigger take() {
    assert(fifo[flagcxFifoIdxProduced] == fifo[flagcxFifoIdxConsumed] + 1);
    flagcxDeviceTrigger trigger = {fifo[flagcxFifoIdxData],
                                   fifo[flagcxFifoIdxData + 1],
                                   fifo[flagcxFifoIdxData + 2]};
    assert(trigger.trd & flagcxDeviceTriggerValidMask);
    ++fifo[flagcxFifoIdxConsumed];
    ++fifo[flagcxFifoIdxCompleted];
    fifo[flagcxFifoIdxData + 2] = 0;
    return trigger;
  }
};

template <typename F>
void rejectsProducer(F call) {
  pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    struct rlimit limit = {0, 0};
    setrlimit(RLIMIT_CORE, &limit);
    // Sanitizers may intercept the trap; either a signal or nonzero exit must
    // occur before the operation can return and publish a truncated ID.
    call();
    _exit(0);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFSIGNALED(status) ||
         (WIFEXITED(status) && WEXITSTATUS(status) != 0));
}

void testWireAndContextIsolation() {
  Fixture f;
  f.comm.counterCount = 65539; // A different stride catches buffer mixups.
  const int ids[] = {0, 288, 4095, 4096, 65535, 65536};
  const int contexts[] = {0, 12, 35};
  std::vector<uint64_t> signals(36 * f.comm.signalCount, 0);
  std::vector<uint64_t> counters(36 * f.comm.counterCount, 0);
  flagcxDevCommInternal dc = {};
  dc.contextCount = 36;
  dc.signalCount = f.comm.signalCount;
  dc.counterCount = f.comm.counterCount;
  dc.counterBuffer = counters.data();
  for (int context : contexts) {
    API::Net net(f.comm, context);
    for (int id : ids) {
      const uint64_t values[] = {1, 65535, 65536, UINT64_C(0xfedcba9876543210)};
      for (uint64_t value : values) {
        for (int type = 0; type < 2; ++type) {
          if (value <= 65535)
            assert(net.enqueueFifoSignal(id, uint32_t(value), 17, type) ==
                   flagcxSuccess);
          else
            assert(net.enqueueFifoSignalValue(id, value, 17, type) ==
                   flagcxSuccess);
          auto trigger = f.take();
          assert(trigger.getPrim() ==
                 uint64_t(value <= 65535 ? flagcxDevicePrimSignal
                                         : flagcxDevicePrimSignalValue));
          assert(trigger.getFormatVersion() == 1);
          assert(trigger.getPeerRank() == 17);
          assert(trigger.getSlotIdx() == 0);
          assert(trigger.getBufferType() == uint64_t(type));
          assert(trigger.getSignalIdx() == uint64_t(id));
          assert(trigger.getSignalValue() == value);
          assert(trigger.fst == 0 && trigger.snd == value);
          size_t offset = 0;
          assert(flagcxKernelProxyCompletionOffset(
                     trigger, context, 36, f.comm.signalCount,
                     f.comm.counterCount, &offset) == flagcxSuccess);
          int stride = type == 0 ? f.comm.signalCount : f.comm.counterCount;
          assert(offset == (size_t(context) * stride + id) * 8);
          assert(consume(trigger, &dc, context) == flagcxSuccess);
          if (type == 0) {
            assert(postedSignal.posted && postedSignal.context == context);
            assert(postedSignal.peer == 17 && postedSignal.value == value);
            assert(postedSignal.signalOffset == offset);
            signals[postedSignal.signalOffset / 8] += postedSignal.value;
          } else {
            assert(!postedSignal.posted); // actual branch updates counters
          }
        }
      }
      assert(signals[context * f.comm.signalCount + id] ==
             counters[context * f.comm.counterCount + id]);
      assert(signals[context * f.comm.signalCount + id] ==
             UINT64_C(0xfedcba9876543210) + 1 + 65535 + 65536);
    }
  }
  for (int context = 0; context < 36; ++context)
    for (int id : ids)
      if (context != 0 && context != 12 && context != 35)
        assert(signals[context * f.comm.signalCount + id] == 0 &&
               counters[context * f.comm.counterCount + id] == 0);

  // Original failure: 12 * 361 + 288 = 4620 is legal in memory. Only 288 is
  // on the wire, so the adjacent bufferType remains zero.
  Fixture legacy(361);
  API::Net net(legacy.comm, 12);
  assert(net.enqueueFifoSignalRaw(288, 1) == flagcxSuccess);
  auto trigger = legacy.take();
  size_t offset = 0;
  assert(trigger.getSignalIdx() == 288 && trigger.getBufferType() == 0);
  assert(flagcxKernelProxyCompletionOffset(trigger, 12, 36, 361, 36, &offset) ==
         flagcxSuccess);
  assert(offset == 4620 * 8);
  dc.signalCount = 361;
  assert(consume(trigger, &dc, 12) == flagcxSuccess);
  assert(postedSignal.signalOffset == 4620 * 8);
}

void testTransfersAndUnchangedPrimitives() {
  Fixture f;
  API::Net net(f.comm, 35);
  for (int prim :
       {flagcxDevicePrimPut, flagcxDevicePrimGet, flagcxDevicePrimPutSignal}) {
    for (int mr : {0, 1, 127}) {
      const uint32_t src = 0x89abcdef, dst = 0xfedcba98, bytes = UINT32_MAX;
      flagcxResult_t result =
          prim == flagcxDevicePrimPut
              ? net.enqueueFifoPut(src, dst, bytes, 1048575, mr, 127 - mr)
          : prim == flagcxDevicePrimGet
              ? net.enqueueFifoGet(src, dst, bytes, 1048575, mr, 127 - mr)
              : net.enqueueFifoPutSignal(src, dst, bytes, 65536, 65535, 1048575,
                                         mr, 127 - mr);
      assert(result == flagcxSuccess);
      auto trigger = f.take();
      assert(trigger.getPrim() == uint64_t(prim));
      assert(flagcxKernelProxyValidateFormat(trigger) == flagcxSuccess);
      assert(trigger.getPeerRank() == 1048575);
      assert(trigger.getSrcOffset() == src && trigger.getDstOffset() == dst);
      assert(trigger.getSize() == bytes);
      assert(trigger.getSrcMrIdx() == uint64_t(mr));
      assert(trigger.getDstMrIdx() == uint64_t(127 - mr));
      assert((trigger.snd & (UINT64_C(3) << 16)) == 0);
      assert(trigger.getBufferType() == 0);
      if (prim == flagcxDevicePrimPutSignal) {
        assert(trigger.getSignalIdx() == 65536);
        assert(trigger.getSignalValue() == 65535);
        size_t offset = 0;
        assert(flagcxKernelProxyCompletionOffset(trigger, 35, 36, 65537, 65537,
                                                 &offset) == flagcxSuccess);
        assert(offset == (size_t(35) * 65537 + 65536) * 8);
        flagcxDevCommInternal dc = {};
        dc.contextCount = 36;
        dc.signalCount = 65537;
        assert(consume(trigger, &dc, 35) == flagcxSuccess);
        assert(postedSignal.signalOffset == offset);
        assert(postedSignal.src == src && postedSignal.dst == dst);
        assert(postedSignal.bytes == bytes && postedSignal.value == 65535);
        assert(postedSignal.srcMr == mr && postedSignal.dstMr == 127 - mr);
      } else {
        assert(trigger.getSignalIdx() == 0);
        assert((trigger.snd & 0xffff) == 0);
      }
    }
  }
  assert(net.enqueueFifoPutValue(0xfedcba98, UINT64_MAX, 23, 127) ==
         flagcxSuccess);
  auto putValue = f.take();
  assert(putValue.getPrim() == flagcxDevicePrimPutValue);
  assert(putValue.getDstOffset() == 0xfedcba98 &&
         putValue.getDstMrIdx() == 127);
  assert(putValue.getValue() == UINT64_MAX);
  assert(flagcxKernelProxyValidateFormat(putValue) == flagcxSuccess);
  assert(net.enqueueFifoTerm(36) == flagcxSuccess);
  auto term = f.take();
  assert(term.getTotalCoops() == 36);
  assert(flagcxKernelProxyValidateFormat(term) == flagcxSuccess);
}

void testCapacityAndRejections() {
  int count = 0;
  size_t signalBytes = 0, counterBytes = 0;
  for (int nodes : {2, 128, 1024, 8192}) {
    assert(flagcxKernelCompletionLayout(289, 36, nodes, 36, 36, &count,
                                        &signalBytes, &counterBytes));
    assert(count == 289 + 36 * nodes);
    assert(signalBytes == size_t(36) * count * 8);
    assert(counterBytes == size_t(36) * 36 * 8);
  }
  assert(flagcxKernelCompletionLayout(1 << 24, 0, 1, 1 << 24, 36, &count,
                                      &signalBytes, &counterBytes));
  assert(!flagcxKernelCompletionLayout(1 << 24, 1, 1, 1, 36, &count,
                                       &signalBytes, &counterBytes));
  assert(!flagcxKernelCompletionLayout(1, 0, 1, (1 << 24) + 1, 36, &count,
                                       &signalBytes, &counterBytes));
  assert(!flagcxKernelCompletionLayout(-1, 0, 1, 1, 36, &count, &signalBytes,
                                       &counterBytes));
  assert(!flagcxKernelCompletionLayout(1, 1, 0, 1, 36, &count, &signalBytes,
                                       &counterBytes));
  assert(!flagcxKernelCompletionBytes(SIZE_MAX / 8 + 1, 1, &signalBytes));
  assert(!flagcxKernelCompletionBytes(SIZE_MAX / 16, 3, &signalBytes));

  Fixture f(1 << 24); // Counts only; the FIFO itself still occupies one slot.
  API::Net net(f.comm, 35);
  assert(net.enqueueFifoSignal((1 << 24) - 1, 1, 1, 0) == flagcxSuccess);
  auto trigger = f.take();
  assert(trigger.getSignalIdx() == (1 << 24) - 1);
  size_t offset = 0;
  assert(flagcxKernelProxyCompletionOffset(trigger, 35, 36, 1 << 24, 1 << 24,
                                           &offset) == flagcxSuccess);
  assert(offset == (size_t(36) * (1 << 24) - 1) * 8);
  flagcxDevCommInternal dc = {};
  dc.contextCount = 36;
  dc.signalCount = 1 << 24;
  dc.counterCount = 1 << 24;
  assert(net.enqueueFifoPutSignal(0, 4, 4, (1 << 24) - 1, 65535, 1, 127, 127) ==
         flagcxSuccess);
  auto maxFused = f.take();
  assert(maxFused.getSignalIdx() == (1 << 24) - 1);
  assert(consume(maxFused, &dc, 35) == flagcxSuccess);
  assert(postedSignal.signalOffset == offset && postedSignal.value == 65535);
  assert(consume(trigger, nullptr, 0) == flagcxInternalError);
  for (int context : {-1, 36})
    assert(flagcxKernelProxyCompletionOffset(trigger, context, 36, 1 << 24,
                                             1 << 24,
                                             &offset) != flagcxSuccess);
  assert(flagcxKernelProxyCompletionOffset(trigger, 0, 36, 1, 1, &offset) !=
         flagcxSuccess);
  for (int version : {0, 2, 15}) {
    auto bad = trigger;
    bad.trd &= ~(UINT64_C(15) << 32);
    bad.trd |= uint64_t(version) << 32;
    assert(flagcxKernelProxyValidateFormat(bad) != flagcxSuccess);
    assert(consume(bad, &dc, 0) == flagcxInvalidArgument);
    assert(!postedSignal.posted);
    assert(flagcxKernelProxyCompletionOffset(bad, 0, 36, 1 << 24, 1 << 24,
                                             &offset) != flagcxSuccess);
  }
  for (int type : {2, 3}) {
    auto bad = trigger;
    bad.trd |= uint64_t(type) << 24;
    assert(consume(bad, &dc, 0) == flagcxInvalidArgument);
    assert(!postedSignal.posted);
    assert(flagcxKernelProxyCompletionOffset(bad, 0, 36, 1 << 24, 1 << 24,
                                             &offset) != flagcxSuccess);
  }
  auto fusedCounter = trigger;
  fusedCounter.trd &= ~(UINT64_C(15) << flagcxDeviceTriggerOffPrim);
  fusedCounter.trd |= uint64_t(flagcxDevicePrimPutSignal)
                      << flagcxDeviceTriggerOffPrim;
  fusedCounter.trd |= UINT64_C(1) << flagcxDeviceTriggerOffBufferType;
  assert(consume(fusedCounter, &dc, 0) == flagcxInvalidArgument);
  assert(!postedSignal.posted);
  rejectsProducer([&] { net.enqueueFifoSignal(1 << 24, 1, 1, 0); });
  rejectsProducer([&] { net.enqueueFifoSignalValue(-1, UINT64_MAX, 1, 0); });
  rejectsProducer([&] { net.enqueueFifoSignal(0, 1, 1, 2); });
  rejectsProducer(
      [&] { net.enqueueFifoPutSignal(0, 0, 4, 1 << 24, 1, 1, 0, 0); });
  rejectsProducer(
      [&] { net.enqueueFifoPutSignal(0, 0, 4, 0, 65536, 1, 0, 0); });
  rejectsProducer([&] { net.enqueueFifoPut(0, 0, 4, 1, 128, 0); });
}

int main() {
  static_assert(sizeof(flagcxDeviceTrigger) == 24,
                "FIFO entry must stay 24 bytes");
  testWireAndContextIsolation();
  testTransfersAndUnchangedPrimitives();
  testCapacityAndRejections();
  std::printf("FIFO v1: 24-bit IDs, 36 contexts, bounds/isolation, %zu-bit "
              "controls PASS\n",
              sizeof(API::CompletionWord) * 8);
}
