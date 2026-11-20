// Host protocol simulation of the actual test kernels extracted by
// check_signal_kernels.py. This does not emulate GPU memory ordering or NICs.
#include "device_signal_test_layout.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

constexpr int ranks = 8, ctas = 4, words = 8;
constexpr int slots = 1024, rowWords = 72 * words;
thread_local int simRank, simCta;
const char *stage = "legacy";
#define FLAGCX_BLOCK_IDX_X simCta
#define FLAGCX_THREAD_IDX_X 0
#define FLAGCX_BLOCK_DIM_X 1
#define FLAGCX_GRID_DIM_X ctas
#define FLAGCX_IR_GLOBAL_PTR
struct flagcxCoopThread {};
struct flagcxCoopBlock {
  void sync() {}
};
struct flagcxTeamTagWorld {};
struct flagcxTeamTagIntra {};
using flagcxDevSignal_t = unsigned;
using flagcxDevCounter_t = unsigned;
using flagcxDataType_t = int;
constexpr int flagcxFloat = 0;
constexpr int flagcxDeviceMemoryOrderRelaxed = 0;
struct flagcxDevNet_None {};
struct flagcxDevNet_SignalInc {
  unsigned signal;
};
struct flagcxDevNet_SignalAdd {
  unsigned signal;
  uint64_t value;
};
struct flagcxDevNet_CounterInc {
  unsigned counter;
};
struct flagcxDevComm {
  int _nInterPeers = 3, _contextCount = 1;
  int getContextCount() const { return _contextCount; }
  int getSize() const { return ranks; }
  int getRank() const { return simRank; }
  int getIntraSize() const { return 2; }
  int getIntraRank() const { return simRank % 2; }
};
struct flagcxTeam {
  bool intra;
};
flagcxTeam flagcxTeamWorld(flagcxDevComm) { return {false}; }
flagcxTeam flagcxTeamIntra(flagcxDevComm) { return {true}; }
struct flagcxDevMem {
  float (*data)[rowWords];
};
float sendData[ranks][rowWords], recvData[ranks][rowWords];
std::atomic<uint64_t> signals[ranks][4][slots], counters[ranks][4][slots];
uint64_t shadows[ranks][4][slots];
void *flagcxGetLocalPointer(flagcxDevMem mem, size_t off) {
  return (char *)mem.data[simRank] + off;
}
void *flagcxGetPeerPointer(flagcxDevMem mem, size_t off, flagcxTeam team,
                           int peer) {
  int rank = team.intra ? simRank - simRank % 2 + peer : peer;
  return (char *)mem.data[rank] + off;
}
size_t getFlagcxDataTypeSizeDevice(int) { return sizeof(float); }

struct Channel {
  std::mutex mutex;
  std::condition_variable ready;
  int arrived = 0, generation = 0;
  void sync(int participants = ranks) {
    std::unique_lock<std::mutex> lock(mutex);
    int old = generation;
    if (++arrived == participants) {
      arrived = 0;
      ++generation;
      ready.notify_all();
    } else {
      bool done = ready.wait_for(lock, std::chrono::seconds(5),
                                 [&] { return generation != old; });
      assert(done && "per-channel rank barrier timed out");
    }
  }
} channels[ctas];
template <class Team, class Coop>
struct flagcxDevBarrier {
  int index;
  template <class Tag, class... Args>
  flagcxDevBarrier(Coop, Tag, const struct flagcxDevNet &, int b, Args...)
      : index(b) {}
  void sync(int) { channels[index].sync(); }
};
struct flagcxDevNet {
  bool isValid() const { return true; }
  int ctx;
  flagcxDevNet(flagcxDevComm, int context) : ctx(context) {}
  uint64_t readSignal(int id, int = 64) {
    return signals[simRank][ctx][id].load();
  }
  uint64_t readCounter(int id) { return counters[simRank][ctx][id].load(); }
  uint64_t *getSignalShadowPtr(int id) { return &shadows[simRank][ctx][id]; }
  void resetSignal(int id) {
    signals[simRank][ctx][id] = 0;
    shadows[simRank][ctx][id] = 0;
  }
  void resetCounter(int id) { counters[simRank][ctx][id] = 0; }
  void waitSignal(flagcxCoopBlock, int id, uint64_t target) {
    auto start = std::chrono::steady_clock::now();
    while (signals[simRank][ctx][id].load() < target) {
      if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(5)) {
        std::fprintf(
            stderr, "%s rank=%d cta=%d ctx=%d sig=%d got=%llu target=%llu\n",
            stage, simRank, simCta, ctx, id, (unsigned long long)readSignal(id),
            (unsigned long long)target);
        std::abort();
      }
      std::this_thread::yield();
    }
  }
  void waitCounter(flagcxCoopBlock, int id, uint64_t target) {
    assert(counters[simRank][ctx][id].load() >= target);
  }
  void flush(flagcxCoopBlock) {}
  void remote(int peer, flagcxDevNet_None) {}
  void remote(int peer, flagcxDevNet_SignalInc op) {
    ++signals[peer][ctx][op.signal];
  }
  void remote(int peer, flagcxDevNet_SignalAdd op) {
    signals[peer][ctx][op.signal] += op.value;
  }
  void local(flagcxDevNet_None) {}
  void local(flagcxDevNet_CounterInc op) {
    ++counters[simRank][ctx][op.counter];
  }
  template <class Remote, class Local>
  void put(flagcxTeam, int peer, flagcxDevMem dst, size_t dstOff,
           flagcxDevMem src, size_t srcOff, size_t bytes, Remote r, Local l,
           flagcxCoopThread) {
    std::memcpy((char *)dst.data[peer] + dstOff,
                (char *)src.data[simRank] + srcOff, bytes);
    remote(peer, r);
    local(l);
  }
  template <class Remote>
  void putValue(flagcxTeam, int peer, flagcxDevMem dst, size_t off,
                uint64_t value, Remote r, flagcxCoopThread) {
    std::memcpy((char *)dst.data[peer] + off, &value, sizeof(value));
    remote(peer, r);
  }
  template <class Remote>
  void signal(flagcxTeam, int peer, Remote r, flagcxCoopThread) {
    remote(peer, r);
  }
};
void flagcxTestSignalDelay(int rank, int cta, unsigned skew) {
  std::this_thread::sleep_for(
      std::chrono::microseconds(signalSkewMultiplier(rank, cta) * skew));
}
unsigned int atomicAdd(unsigned int *p, unsigned int value) {
  return __atomic_fetch_add(p, value, __ATOMIC_SEQ_CST);
}
#include "signal_ir_sim.h"
#include "signal_test_kernels.inc"

uint64_t results[ranks][4 * ctas];
template <class F>
void run(F fn) {
  std::vector<std::thread> tasks;
  for (int r = 0; r < ranks; ++r)
    for (int b = 0; b < ctas; ++b)
      tasks.emplace_back([=] {
        simRank = r;
        simCta = b;
        fn();
      });
  for (auto &task : tasks)
    task.join();
}
void prepare() {
  for (int r = 0; r < ranks; ++r) {
    for (int p = 0; p < ranks; ++p)
      for (int w = 0; w < words; ++w) {
        sendData[r][p * words + w] = float(r * 100 + p);
        recvData[r][p * words + w] = -1;
      }
    for (auto &v : results[r])
      v = UINT64_MAX;
  }
}
void checkPayload(bool interOnly) {
  for (int r = 0; r < ranks; ++r)
    for (int p = 0; p < ranks; ++p) {
      if (interOnly && p / 2 == r / 2)
        continue;
      for (int w = 0; w < words; ++w)
        assert(recvData[r][p * words + w] == float(p * 100 + r));
    }
}
int main() {
  flagcxDevComm dc;
  flagcxDevMem send{sendData}, recv{recvData};
  for (int round = 0; round < 20; ++round) {
    prepare();
    run([&] { flagcxInterTestResetKernel(dc, results[simRank]); });
    for (auto &r : results)
      for (auto v : r)
        assert(v == 0);
    prepare();
    run([&] {
      flagcxInterTestPutSignalIncKernel(send, recv, words, 0, dc, 100);
    });
    checkPayload(false);
    prepare();
    run([&] {
      flagcxInterTestPutSignalAddDecoupledKernel(send, recv, words, 0, dc, 100);
    });
    checkPayload(false);
    prepare();
    run([&] {
      flagcxInterTestCounterPipelineKernel(send, recv, words, 0, dc,
                                           results[simRank], 100);
    });
    for (int r = 0; r < ranks; ++r) {
      for (int p = 0; p < ranks; ++p)
        assert(recvData[r][p * words] == 999.0f);
      for (int b = 0; b < ctas; ++b) {
        int sent = countInterPeersAssignedTo(b, ctas, 1, ranks, r - r % 2, 2);
        assert(results[r][4 * b] == uint64_t(2 * sent));
        assert(results[r][4 * b + 1] == uint64_t(sent + 1));
        assert(results[r][4 * b + 2] ==
               (b == signalOwnerCta(r, ctas, 1) ? 12u : 0u));
      }
    }
    prepare();
    run([&] {
      flagcxInterTestFlushDecoupleKernel(send, recv, words, 0, dc, 100);
    });
    checkPayload(false);
    prepare();
    run([&] { flagcxInterTestPutValueKernel(recv, dc, 0, 100); });
    for (int r = 0; r < ranks; ++r)
      for (int p = 0; p < ranks; ++p) {
        uint64_t value;
        std::memcpy(&value, (char *)recvData[r] + p * sizeof(value),
                    sizeof(value));
        assert(value == uint64_t(p * 1000 + r));
      }
    run([&] { flagcxInterTestSignalKernel(dc, 100); });
    prepare();
    std::memset(results, 0, sizeof(results));
    run([&] {
      flagcxInterTestWaitSignalFlushKernel(dc, results[simRank], 1000);
    });
    for (auto &r : results)
      for (int b = 0; b < ctas; ++b) {
        assert(r[kFlagcxTestWaitResultWords * b] == 0 &&
               r[kFlagcxTestWaitResultWords * b + 1] == 6 &&
               r[kFlagcxTestWaitResultWords * b + 2] == 1);
      }
    prepare();
    run([&] {
      flagcxInterOneSidedAlltoAllKernel(send, recv, words, 0, dc, 100);
    });
    checkPayload(false);
  }
  runIrTests(dc, send, recv);
  std::puts(
      "actual legacy kernels: PASS (8 ranks, 4 CTAs/context, skew, 20 rounds)");
}

void runIrTests(flagcxDevComm &dc, flagcxDevMem send, flagcxDevMem recv) {
  const void *comm = &dc, *src = &send, *dst = &recv;
  prepare();
  int resetResults[ranks][ctas * 3] = {};
  stage = "kernelNetResetS";
  run([&] { kernelNetResetS(comm, resetResults[simRank]); });
  for (auto &r : resetResults)
    for (int v : r)
      assert(v == 1);
  stage = "kernelNetPutSignalIncS";
  run([&] { kernelNetPutSignalIncS(comm, src, dst, words, 100); });
  checkPayload(true);
  prepare();
  stage = "kernelNetPutSignalAddS";
  run([&] { kernelNetPutSignalAddS(comm, src, dst, words, 100); });
  checkPayload(true);
  prepare();
  stage = "kernelNetCounterPipelineS";
  run([&] {
    kernelNetCounterPipelineS(comm, src, dst, words, results[simRank], 100);
  });
  for (int r = 0; r < ranks; ++r) {
    for (int p = 0; p < ranks; ++p)
      if (p / 2 != r / 2)
        assert(recvData[r][p * words] == 999.0f);
    for (int b = 0; b < ctas; ++b) {
      int sent = countInterPeersAssignedTo(b, ctas, 1, ranks, r - r % 2, 2);
      assert(results[r][4 * b] == uint64_t(2 * sent));
      assert(results[r][4 * b + 1] == uint64_t(sent + 1));
    }
  }
  prepare();
  stage = "kernelNetFlushDecoupleS";
  run([&] { kernelNetFlushDecoupleS(comm, src, dst, words, 100); });
  checkPayload(true);
  prepare();
  stage = "kernelNetPutValueS";
  run([&] { kernelNetPutValueS(comm, dst, 0, 100); });
  for (int r = 0; r < ranks; ++r)
    for (int p = 0; p < ranks; ++p) {
      if (p / 2 == r / 2)
        continue;
      uint64_t value;
      std::memcpy(&value, (char *)recvData[r] + p * sizeof(value),
                  sizeof(value));
      assert(value == uint64_t(p * 1000 + r));
    }
  stage = "kernelNetSignalS";
  run([&] { kernelNetSignalS(comm, 100); });
  prepare();
  stage = "kernelNetWaitSignalFlushS";
  std::memset(results, 0, sizeof(results));
  run([&] { kernelNetWaitSignalFlushS(comm, results[simRank], 1000); });
  for (auto &r : results)
    for (int b = 0; b < ctas; ++b)
      assert(r[kFlagcxTestWaitResultWords * b] == 0 &&
             r[kFlagcxTestWaitResultWords * b + 1] == 6 &&
             r[kFlagcxTestWaitResultWords * b + 2] == 1);
  bool observed = false;
  for (auto &r : results)
    for (int b = 0; b < ctas; ++b)
      observed |= r[kFlagcxTestWaitResultWords * b + 3] != 0;
  assert(observed);
  prepare();
  stage = "kernelNetOneSidedAlltoAllS";
  run([&] { kernelNetOneSidedAlltoAllS(comm, src, dst, words, 100); });
  checkPayload(false);

  int verdict[ranks][ctas];
  for (int contexts = 1; contexts <= 4; ++contexts) {
    dc._contextCount = contexts;
    for (int round = 0; round < 5; ++round)
      for (bool inter : {false, true}) {
        auto scenario = [&](auto fn, int combos, bool payload, bool scalar) {
          for (int r = 0; r < ranks; ++r) {
            for (int b = 0; b < ctas; ++b)
              verdict[r][b] = 0;
            for (int i = 0; i < rowWords; ++i) {
              sendData[r][i] = float(r * 1000 + i);
              recvData[r][i] = -1;
            }
          }
          stage = inter ? "unified inter" : "unified intra";
          run(fn);
          for (int r = 0; r < ranks; ++r) {
            for (int b = 0; b < ctas; ++b)
              assert(verdict[r][b] == 1);
            if (payload)
              for (int index = 0; index < ctas * combos; ++index) {
                int combo = index % combos;
                int previous =
                    combo % 2 ? (r + ranks - 1) % ranks
                    : inter
                        ? ((r / 2 + ranks / 2 - 1) % (ranks / 2)) * 2 + r % 2
                        : r - r % 2 + (r % 2 + 1) % 2;
                if (scalar) {
                  uint64_t value;
                  std::memcpy(&value,
                              (char *)recvData[r] + index * sizeof(value),
                              sizeof(value));
                  assert(value == uint64_t(previous * 100 + index));
                } else {
                  for (int i = 0; i < words; ++i)
                    assert(recvData[r][index * words + i] ==
                           float(previous * 1000 + index * words + i));
                }
              }
          }
        };
#define CHECK_UNIFIED(Name, Combos, Payload, Scalar, ...)                      \
  scenario(                                                                    \
      [&] {                                                                    \
        if (inter)                                                             \
          kernelDev##Name##InterWorldS(comm, __VA_ARGS__ verdict[simRank],     \
                                       100);                                   \
        else                                                                   \
          kernelDev##Name##IntraWorldS(comm, __VA_ARGS__ verdict[simRank],     \
                                       100);                                   \
      },                                                                       \
      Combos, Payload, Scalar)
        CHECK_UNIFIED(SignalStandalone, 12, false, false, );
#undef CHECK_UNIFIED
        scenario(
            [&] {
              if (inter)
                kernelDevPutSignalWaitInterWorldS(comm, dst, src,
                                                  verdict[simRank],
                                                  words * sizeof(float), 100);
              else
                kernelDevPutSignalWaitIntraWorldS(comm, dst, src,
                                                  verdict[simRank],
                                                  words * sizeof(float), 100);
            },
            12, true, false);
        scenario(
            [&] {
              if (inter)
                kernelDevPutRSigInterWorldS(comm, dst, src, verdict[simRank],
                                            words * sizeof(float), 100);
              else
                kernelDevPutRSigIntraWorldS(comm, dst, src, verdict[simRank],
                                            words * sizeof(float), 100);
            },
            12, true, false);
        scenario(
            [&] {
              if (inter)
                kernelDevPutCounterInterWorldS(comm, dst, src, verdict[simRank],
                                               words * sizeof(float), 100);
              else
                kernelDevPutCounterIntraWorldS(comm, dst, src, verdict[simRank],
                                               words * sizeof(float), 100);
            },
            18, true, false);
        scenario(
            [&] {
              if (inter)
                kernelDevPutValueRSigInterWorldS(comm, dst, verdict[simRank],
                                                 words * sizeof(float), 100);
              else
                kernelDevPutValueRSigIntraWorldS(comm, dst, verdict[simRank],
                                                 words * sizeof(float), 100);
            },
            12, true, true);
        scenario(
            [&] {
              if (inter)
                kernelDevSignalShadowFlushInterWorldS(comm, verdict[simRank],
                                                      100);
              else
                kernelDevSignalShadowFlushIntraWorldS(comm, verdict[simRank],
                                                      100);
            },
            6, false, false);
      }
  }
  std::puts("actual scalar/unified kernels: PASS (1-4 contexts, "
            "INTRA/INTER/WORLD, skew)");
}
