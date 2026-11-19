/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * PPU SDK contract tests. No communicator or network provider is required.
 ************************************************************************/
#include "device_api/comm_traits.h"

#include <cstdio>
#include <cstring>
#include <hggc_runtime.h>
#include <type_traits>

using Traits = PlatformTraits<PpuPlatform>;
using Atomic = Traits::Atomic;
using Ops = Traits::CoopOps;
static_assert(!std::is_same<Traits::CoopAny, PlatformCoop>::value,
              "PPU CoopAny must use direct synchronization dispatch");
static_assert(
    std::is_same<DeviceAPI, CommTraits<DefaultBackend<PpuPlatform>>>::value,
    "PPU must select its own default backend");
static_assert(sizeof(Traits::CoopAny) == sizeof(PlatformCoop) &&
                  alignof(Traits::CoopAny) == alignof(PlatformCoop),
              "PPU CoopAny layout changed");
static_assert(std::is_standard_layout<Traits::CoopAny>::value &&
                  std::is_trivially_copyable<Traits::CoopAny>::value,
              "PPU CoopAny must retain value/opaque-copy semantics");
static_assert(sizeof(PlatformCompletionWord<PpuPlatform>::type) == 8,
              "PPU completion protocol must stay 64-bit");

struct Sum {
  FLAGCX_DEVICE_DECORATOR uint32_t operator()(uint32_t a, uint32_t b) const {
    return a + b;
  }
};

template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR bool coopTileContract(uint32_t *scratch,
                                                     int tid) {
  Traits::CoopTile<N> group;
  Traits::CoopAny original(group), copied(original), assigned;
  assigned = copied;
  scratch[tid] = tid + 1;
  assigned.sync();
  bool ok = assigned.threadRank() == tid % N && assigned.size() == N &&
            scratch[(tid / N) * N + N - 1] == uint32_t((tid / N) * N + N);
  assigned.sync();
  return ok;
}

// Instantiate every native HGGC atomic operation for both protocol widths
// and all four scopes, including CAS success and failure semantics.
template <typename T, flagcxDeviceScope_t Scope>
FLAGCX_DEVICE_DECORATOR bool atomicContract(T *word) {
  Atomic::store<T, Scope>(word, T{4}, flagcxDeviceMemoryOrderRelease);
  bool ok =
      Atomic::load<T, Scope>(word, flagcxDeviceMemoryOrderAcquire) == T{4};
  ok &= Atomic::fetchAdd<T, Scope>(word, T{3}, flagcxDeviceMemoryOrderAcqRel) ==
        T{4};
  ok &= Atomic::fetchSub<T, Scope>(word, T{2},
                                   flagcxDeviceMemoryOrderRelaxed) == T{7};
  ok &= Atomic::fetchOr<T, Scope>(word, T{8}, flagcxDeviceMemoryOrderSeqCst) ==
        T{5};
  ok &= Atomic::fetchAnd<T, Scope>(word, T{7}, flagcxDeviceMemoryOrderSeqCst) ==
        T{13};
  ok &= Atomic::exchange<T, Scope>(word, T{9}, flagcxDeviceMemoryOrderAcqRel) ==
        T{5};
  T expected = 9;
  ok &= Atomic::compareExchange<T, Scope>(word, expected, T{11},
                                          flagcxDeviceMemoryOrderAcqRel);
  expected = 0;
  ok &= !Atomic::compareExchange<T, Scope>(word, expected, T{12},
                                           flagcxDeviceMemoryOrderAcqRel);
  ok &= expected == T{11};
  ok &= Atomic::fetchMax<T, Scope>(word, T{15},
                                   flagcxDeviceMemoryOrderAcqRel) == T{11};
  ok &= Atomic::fetchMax<T, Scope>(word, T{10},
                                   flagcxDeviceMemoryOrderRelaxed) == T{15};
  return ok &&
         Atomic::load<T, Scope>(word, flagcxDeviceMemoryOrderAcquire) == T{15};
}

FLAGCX_GLOBAL_DECORATOR void testPpuTraits(uint32_t *result, uint32_t *word32,
                                           uint64_t *word64) {
  const int tid = threadIdx.x;
  const int lane = Traits::Intrin::lane();
  Traits::CoopBlock block;
  Traits::CoopWarp warp;
  Traits::CoopTile<8> tile;
  Traits::CoopThread thread;
  FLAGCX_SHARED_DECORATOR uint32_t scratch[128];
  bool ok = lane == (tid % 32) && block.threadRank() == tid &&
            block.size() == 128 && warp.threadRank() == lane &&
            warp.size() == 32;
  ok &= Traits::Intrin::lanemaskLt() == ((uint64_t{1} << lane) - 1);
  ok &= Traits::Intrin::activemask() == Traits::Intrin::fullMask();
  ok &= Traits::Intrin::popc(Traits::Intrin::fullMask()) == 32;
  ok &= tile.threadRank() == lane % 8 && tile.size() == 8 &&
        tile.laneMask() == (uint64_t{0xff} << (lane & ~7));
  scratch[tid] = tid + 1;
  tile.sync();
  ok &= scratch[(tid & ~7) + 7] == uint32_t((tid & ~7) + 8);
  Traits::CoopAny anyBlock(block), anyWarp(warp), anyTile(tile),
      anyThread(thread);
  anyBlock.sync();
  anyWarp.sync();
  anyTile.sync();
  anyThread.sync();
  ok &= anyBlock.threadRank() == tid && anyBlock.size() == 128 &&
        anyWarp.threadRank() == lane && anyWarp.size() == 32 &&
        anyTile.threadRank() == lane % 8 && anyTile.size() == 8 &&
        anyThread.threadRank() == 0 && anyThread.size() == 1;
  Traits::CoopAny noop;
  noop.sync();
  ok &= noop.threadRank() == 0 && noop.size() == 1;
  ok &= coopTileContract<1>(scratch, tid);
  ok &= coopTileContract<2>(scratch, tid);
  ok &= coopTileContract<4>(scratch, tid);
  ok &= coopTileContract<8>(scratch, tid);
  ok &= coopTileContract<16>(scratch, tid);
  ok &= coopTileContract<32>(scratch, tid);
  block.sync();

  const flagcxLaneMask_t masks[] = {0xffffu, 0x55555555u, 0x80000001u, 1u,
                                    Traits::Intrin::fullMask()};
  for (unsigned i = 0; i < sizeof(masks) / sizeof(masks[0]); ++i) {
    const flagcxLaneMask_t mask = masks[i];
    if (mask & (flagcxLaneMask_t{1} << lane)) {
      Traits::CoopLanes lanes(mask);
      Traits::CoopAny original(lanes), copied(original);
      scratch[tid] = tid + 1;
      copied.sync();
      int last = 31;
      while ((mask & (flagcxLaneMask_t{1} << last)) == 0)
        --last;
      ok &= copied.threadRank() ==
                Traits::Intrin::popc(mask & Traits::Intrin::lanemaskLt()) &&
            copied.size() == Traits::Intrin::popc(mask) &&
            scratch[(tid / 32) * 32 + last] ==
                uint32_t((tid / 32) * 32 + last + 1);
      copied.sync();
    }
    warp.sync();
  }
  block.sync();
  if (tid < 32) {
    Traits::CoopTileSpan span(0, 1, 0);
    Traits::CoopAny anySpan(span);
    anySpan.sync();
    ok &= anySpan.threadRank() == tid && anySpan.size() == 32;
  }
  block.sync();
  for (int id = 0; id < 2; ++id) {
    if (tid >= 32 && tid < 96) {
      Traits::CoopTileSpan span(1, 2, id);
      Traits::CoopAny original(span), copied(original), assigned;
      assigned = copied;
      scratch[tid] = tid + 1;
      assigned.sync();
      ok &= assigned.threadRank() == tid - 32 && assigned.size() == 64 &&
            scratch[95] == 96;
      assigned.sync();
    }
    block.sync();
  }

  // IR objects can be copied through opaque storage. Queries must describe
  // the calling thread even when thread 0 constructed the stored group.
  FLAGCX_SHARED_DECORATOR uint64_t coopStorage[3];
  if (tid == 0) {
    Traits::CoopAny source(block);
    const unsigned char *srcBytes = (const unsigned char *)&source;
    unsigned char *dstBytes = (unsigned char *)coopStorage;
    for (unsigned i = 0; i < sizeof(source); ++i)
      dstBytes[i] = srcBytes[i];
  }
  block.sync();
  Traits::CoopAny fromStorage;
  const unsigned char *srcBytes = (const unsigned char *)coopStorage;
  unsigned char *dstBytes = (unsigned char *)&fromStorage;
  for (unsigned i = 0; i < sizeof(fromStorage); ++i)
    dstBytes[i] = srcBytes[i];
  ok &= fromStorage.threadRank() == tid && fromStorage.size() == 128;
  fromStorage.sync();
  ok &= Ops::broadcast(warp, uint32_t(lane), 3) == 3;
  ok &= Ops::reduce(warp, uint32_t(lane + 1), Sum{}) == 528;
  Traits::Intrin::threadfenceDevice();
  Traits::Intrin::threadfenceSystem();
  Traits::Intrin::spinBackoff(1);
  const uint64_t t0 = Traits::Clock::now();
  Traits::Intrin::spinBackoff(2);
  ok &= Traits::Clock::elapsedTicks(t0, Traits::Clock::now()) > 0;

  if (tid == 0) {
    ok &= atomicContract<uint32_t, flagcxDeviceScopeSystem>(word32);
    ok &= atomicContract<uint64_t, flagcxDeviceScopeSystem>(word64);
    ok &= atomicContract<uint32_t, flagcxDeviceScopeDevice>(word32);
    ok &= atomicContract<uint64_t, flagcxDeviceScopeDevice>(word64);
    ok &= atomicContract<uint32_t, flagcxDeviceScopeBlock>(word32);
    ok &= atomicContract<uint64_t, flagcxDeviceScopeBlock>(word64);
    ok &= atomicContract<uint32_t, flagcxDeviceScopeThread>(word32);
    ok &= atomicContract<uint64_t, flagcxDeviceScopeThread>(word64);
    Atomic::store(word32, uint32_t{0}, flagcxDeviceMemoryOrderRelease);
    Atomic::store(word64, uint64_t{0}, flagcxDeviceMemoryOrderRelease);
  }
  block.sync();
  Atomic::fetchAdd(word32, uint32_t{1}, flagcxDeviceMemoryOrderAcqRel);
  Atomic::fetchAdd(word64, uint64_t{1}, flagcxDeviceMemoryOrderAcqRel);
  block.sync();
  ok &= Atomic::load(word32, flagcxDeviceMemoryOrderAcquire) == 128;
  ok &= Atomic::load(word64, flagcxDeviceMemoryOrderAcquire) == 128;

  // Exercise aligned, unaligned and non-word-sized cooperative copies.
  FLAGCX_SHARED_DECORATOR unsigned char src[67], dst[67];
  for (int i = tid; i < 67; i += 128)
    src[i] = static_cast<unsigned char>(i + 1);
  block.sync();
  Ops::copy(block, dst, src, 67);
  block.sync();
  for (int i = tid; i < 67; i += 128)
    ok &= dst[i] == src[i];
  block.sync();
  for (int i = tid; i < 67; i += 128)
    dst[i] = 0;
  block.sync();
  Ops::copy(block, dst + 1, src + 1, 63);
  block.sync();
  for (int i = tid; i < 63; i += 128)
    ok &= dst[i + 1] == src[i + 1];
  if (tid == 0)
    ok &= dst[0] == 0 && dst[64] == 0;
  if (tid == 0) {
    uint32_t local = 0;
    Ops::copy<uint32_t, Scalar, Scalar, flagcxDeviceMemoryScratchpad,
              flagcxDeviceMemoryLocal>(thread, &local, scratch, 1);
    ok &= local == 1;
  }
  result[tid] = ok ? 0 : 1;
}

enum class PpuAtomicCase { Load, Store, Add, Sub, Cas, Exchange, Max, Or, And };

template <typename T>
FLAGCX_GLOBAL_DECORATOR void testPpuAtomicCase(T *word, uint32_t *result,
                                               PpuAtomicCase operation) {
  Atomic::store(word, T{5}, flagcxDeviceMemoryOrderRelease);
  bool ok = true;
  T finalValue = 5;
  switch (operation) {
    case PpuAtomicCase::Load:
      ok = Atomic::load(word, flagcxDeviceMemoryOrderAcquire) == T{5};
      break;
    case PpuAtomicCase::Store:
      Atomic::store(word, T{9}, flagcxDeviceMemoryOrderRelease);
      finalValue = 9;
      break;
    case PpuAtomicCase::Add:
      ok = Atomic::fetchAdd(word, T{3}, flagcxDeviceMemoryOrderAcqRel) == T{5};
      finalValue = 8;
      break;
    case PpuAtomicCase::Sub:
      ok = Atomic::fetchSub(word, T{2}, flagcxDeviceMemoryOrderRelaxed) == T{5};
      finalValue = 3;
      break;
    case PpuAtomicCase::Cas: {
      T expected = 5;
      ok = Atomic::compareExchange(word, expected, T{9},
                                   flagcxDeviceMemoryOrderAcqRel);
      expected = 0;
      ok &= !Atomic::compareExchange(word, expected, T{7},
                                     flagcxDeviceMemoryOrderAcqRel);
      ok &= expected == T{9};
      finalValue = 9;
      break;
    }
    case PpuAtomicCase::Exchange:
      ok = Atomic::exchange(word, T{9}, flagcxDeviceMemoryOrderAcqRel) == T{5};
      finalValue = 9;
      break;
    case PpuAtomicCase::Max:
      ok = Atomic::fetchMax(word, T{9}, flagcxDeviceMemoryOrderAcqRel) == T{5};
      ok &=
          Atomic::fetchMax(word, T{7}, flagcxDeviceMemoryOrderRelaxed) == T{9};
      finalValue = 9;
      break;
    case PpuAtomicCase::Or:
      ok = Atomic::fetchOr(word, T{8}, flagcxDeviceMemoryOrderSeqCst) == T{5};
      finalValue = 13;
      break;
    case PpuAtomicCase::And:
      ok = Atomic::fetchAnd(word, T{3}, flagcxDeviceMemoryOrderSeqCst) == T{5};
      finalValue = 1;
      break;
  }
  ok &= Atomic::load(word, flagcxDeviceMemoryOrderAcquire) == finalValue;
  result[0] = ok ? 0 : 1;
  result[1] = static_cast<uint32_t>(finalValue);
}

#define SDK_CHECK(call)                                                        \
  do {                                                                         \
    hggcError_t err = (call);                                                  \
    if (err != hggcSuccess) {                                                  \
      fprintf(stderr, "%s: %s\n", #call, hggcGetErrorString(err));             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

template <typename T>
int runPpuAtomicCase(bool hostMapped, PpuAtomicCase operation) {
  T *word = nullptr, *deviceWord = nullptr;
  uint32_t *result = nullptr;
  SDK_CHECK(hggcSetDevice(0));
  if (hostMapped) {
    SDK_CHECK(hggcHostAlloc(reinterpret_cast<void **>(&word), sizeof(T),
                            hggcHostAllocMapped));
    SDK_CHECK(hggcHostGetDevicePointer(reinterpret_cast<void **>(&deviceWord),
                                       word, 0));
  } else {
    SDK_CHECK(hggcMalloc(reinterpret_cast<void **>(&word), sizeof(T)));
    deviceWord = word;
  }
  SDK_CHECK(
      hggcMalloc(reinterpret_cast<void **>(&result), 2 * sizeof(uint32_t)));
  testPpuAtomicCase<T><<<1, 1>>>(deviceWord, result, operation);
  SDK_CHECK(hggcGetLastError());
  SDK_CHECK(hggcDeviceSynchronize());
  uint32_t outcome[2] = {};
  SDK_CHECK(
      hggcMemcpy(outcome, result, sizeof(outcome), hggcMemcpyDeviceToHost));
  const bool ok = outcome[0] == 0 &&
                  (!hostMapped || __atomic_load_n(word, __ATOMIC_ACQUIRE) ==
                                      static_cast<T>(outcome[1]));
  SDK_CHECK(hggcFree(result));
  if (hostMapped) {
    SDK_CHECK(hggcFreeHost(word));
  } else {
    SDK_CHECK(hggcFree(word));
  }
  printf("PPU atomic case: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc != 1) {
    const char *operations[] = {"load",     "store", "add", "sub", "cas",
                                "exchange", "max",   "or",  "and"};
    if (argc == 5 && strcmp(argv[1], "--atomic") == 0 &&
        (strcmp(argv[2], "host") == 0 || strcmp(argv[2], "device") == 0) &&
        (strcmp(argv[3], "32") == 0 || strcmp(argv[3], "64") == 0)) {
      for (unsigned i = 0; i < sizeof(operations) / sizeof(operations[0]);
           ++i) {
        if (strcmp(argv[4], operations[i]) != 0)
          continue;
        printf("PPU atomic case: memory=%s bits=%s operation=%s\n", argv[2],
               argv[3], argv[4]);
        fflush(stdout);
        const bool hostMapped = strcmp(argv[2], "host") == 0;
        return strcmp(argv[3], "32") == 0
                   ? runPpuAtomicCase<uint32_t>(hostMapped,
                                                static_cast<PpuAtomicCase>(i))
                   : runPpuAtomicCase<uint64_t>(hostMapped,
                                                static_cast<PpuAtomicCase>(i));
      }
    }
    fprintf(stderr,
            "Usage: %s [--atomic host|device 32|64 "
            "load|store|add|sub|cas|exchange|max|or|and]\n",
            argv[0]);
    return 1;
  }
  uint32_t *results = nullptr, *word32 = nullptr;
  uint64_t *word64 = nullptr;
  SDK_CHECK(hggcSetDevice(0));
  // The three-argument API uses the device's current pool. The four-argument
  // overload forwards to hggcMallocFromPoolAsync and requires an explicit pool.
  hggcStream_t allocationStream = nullptr;
  void *asyncBuffer = nullptr;
  SDK_CHECK(
      hggcStreamCreateWithFlags(&allocationStream, hggcStreamNonBlocking));
  SDK_CHECK(hggcMallocAsync(&asyncBuffer, 64, allocationStream));
  SDK_CHECK(hggcMemsetAsync(asyncBuffer, 0, 64, allocationStream));
  SDK_CHECK(hggcFreeAsync(asyncBuffer, allocationStream));
  SDK_CHECK(hggcStreamSynchronize(allocationStream));
  SDK_CHECK(hggcStreamDestroy(allocationStream));
  SDK_CHECK(
      hggcMalloc(reinterpret_cast<void **>(&results), 128 * sizeof(uint32_t)));
  SDK_CHECK(hggcMalloc(reinterpret_cast<void **>(&word32), sizeof(uint32_t)));
  SDK_CHECK(hggcMalloc(reinterpret_cast<void **>(&word64), sizeof(uint64_t)));
  testPpuTraits<<<1, 128>>>(results, word32, word64);
  SDK_CHECK(hggcGetLastError());
  SDK_CHECK(hggcDeviceSynchronize());
  uint32_t host[128];
  SDK_CHECK(hggcMemcpy(host, results, sizeof(host), hggcMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < 128; ++i) {
    if (host[i] != 0) {
      fprintf(stderr, "PPU traits contract failed at thread %d\n", i);
      ok = false;
    }
  }
  ok &= Traits::Clock::elapsedNs(100, 1600, 1500000) == 1000;
  SDK_CHECK(hggcFree(results));
  SDK_CHECK(hggcFree(word32));
  SDK_CHECK(hggcFree(word64));
  printf("PPU SDK platform traits: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
