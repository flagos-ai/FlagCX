/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * PPU SDK contract tests. No communicator or network provider is required.
 ************************************************************************/
#include "device_api/comm_traits.h"

#include <cstdio>
#include <hggc_runtime.h>
#include <type_traits>

using Traits = PlatformTraits<PpuPlatform>;
using Atomic = Traits::Atomic;
using Ops = Traits::CoopOps;
static_assert(std::is_same<Traits::CoopAny, PlatformCoop>::value,
              "PPU must retain the public PlatformCoop implementation");
static_assert(
    std::is_same<DeviceAPI, CommTraits<DefaultBackend<PpuPlatform>>>::value,
    "PPU must select its own default backend");
static_assert(sizeof(Traits::CoopAny) == sizeof(PlatformCoop) &&
                  alignof(Traits::CoopAny) == alignof(PlatformCoop),
              "PPU CoopAny layout changed");
static_assert(sizeof(PlatformCompletionWord<PpuPlatform>::type) == 8,
              "PPU completion protocol must stay 64-bit");

struct Sum {
  FLAGCX_DEVICE_DECORATOR uint32_t operator()(uint32_t a, uint32_t b) const {
    return a + b;
  }
};

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
                                           uint64_t *word64, uint64_t *mapped) {
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
  if (lane < 16) {
    Traits::CoopLanes lanes(0xffffu);
    Traits::CoopAny anyLanes(lanes);
    anyLanes.sync();
    ok &= anyLanes.threadRank() == lane && anyLanes.size() == 16;
  }
  if (tid < 32) {
    Traits::CoopTileSpan span(0, 1, 0);
    Traits::CoopAny anySpan(span);
    anySpan.sync();
    ok &= anySpan.threadRank() == tid && anySpan.size() == 32;
  }
  block.sync();
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
    ok &= atomicContract<uint64_t, flagcxDeviceScopeSystem>(mapped);
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

#define SDK_CHECK(call)                                                        \
  do {                                                                         \
    hggcError_t err = (call);                                                  \
    if (err != hggcSuccess) {                                                  \
      fprintf(stderr, "%s: %s\n", #call, hggcGetErrorString(err));             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  uint32_t *results = nullptr, *word32 = nullptr;
  uint64_t *word64 = nullptr, *mapped = nullptr, *mappedDevice = nullptr;
  SDK_CHECK(hggcSetDevice(0));
  // The three-argument API uses the device's current pool. The four-argument
  // overload forwards to hggcMallocFromPoolAsync and requires an explicit pool.
  hggcStream_t allocationStream = nullptr;
  void *asyncBuffer = nullptr;
  SDK_CHECK(hggcStreamCreateWithFlags(&allocationStream, hggcStreamNonBlocking));
  SDK_CHECK(hggcMallocAsync(&asyncBuffer, 64, allocationStream));
  SDK_CHECK(hggcMemsetAsync(asyncBuffer, 0, 64, allocationStream));
  SDK_CHECK(hggcFreeAsync(asyncBuffer, allocationStream));
  SDK_CHECK(hggcStreamSynchronize(allocationStream));
  SDK_CHECK(hggcStreamDestroy(allocationStream));
  SDK_CHECK(
      hggcMalloc(reinterpret_cast<void **>(&results), 128 * sizeof(uint32_t)));
  SDK_CHECK(hggcMalloc(reinterpret_cast<void **>(&word32), sizeof(uint32_t)));
  SDK_CHECK(hggcMalloc(reinterpret_cast<void **>(&word64), sizeof(uint64_t)));
  SDK_CHECK(hggcHostAlloc(reinterpret_cast<void **>(&mapped), sizeof(uint64_t),
                          hggcHostAllocMapped));
  SDK_CHECK(hggcHostGetDevicePointer(reinterpret_cast<void **>(&mappedDevice),
                                     mapped, 0));
  __atomic_store_n(mapped, uint64_t{0}, __ATOMIC_RELEASE);
  testPpuTraits<<<1, 128>>>(results, word32, word64, mappedDevice);
  SDK_CHECK(hggcGetLastError());
  SDK_CHECK(hggcDeviceSynchronize());
  uint32_t host[128];
  SDK_CHECK(hggcMemcpy(host, results, sizeof(host), hggcMemcpyDeviceToHost));
  bool ok = __atomic_load_n(mapped, __ATOMIC_ACQUIRE) == 15;
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
  SDK_CHECK(hggcFreeHost(mapped));
  printf("PPU SDK platform traits: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
