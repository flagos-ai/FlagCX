/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Single-GPU PlatformTraits smoke test. Build with FLAGCX_FORCE_DEFAULT_PATH;
 * no communicator, network provider, or NIC mapping is used here.
 ************************************************************************/
#include "device_api/comm_traits.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <type_traits>

using NvidiaTraits = PlatformTraits<NvidiaPlatform>;
using Ops = DeviceAPI::CoopOps;
static_assert(FLAGCX_DEVICE_IR_ABI_VERSION == 1, "Device IR ABI version");
static_assert(flagcxDeviceMemoryOrderWeak == 0 &&
                  flagcxDeviceMemoryOrderRelaxed == 1,
              "Device memory order values");
static_assert(flagcxDeviceMemoryLocal == 0, "Device Local space value");
static_assert(
    std::is_same<DeviceAPI, CommTraits<DefaultBackend<NvidiaPlatform>>>::value,
    "this test must use the NVIDIA default backend");
static_assert(std::is_same<Ops, NvidiaTraits::CoopOps>::value, "CoopOps alias");
static_assert(std::is_same<DeviceAPI::Clock, NvidiaTraits::Clock>::value,
              "Clock alias");
static_assert(defaultNetSelectProvider<NvidiaPlatform>(
                  0, DefaultNetGdaRequiredCapabilities, true) ==
                  DefaultNetProviderProxy,
              "GDA implementation must be compiled before selection");
static_assert(defaultNetSelectProvider<NvidiaPlatform>(
                  0, DefaultNetGdaRequiredCapabilities & ~DefaultNetGdaGet,
                  true) == DefaultNetProviderProxy,
              "incomplete GDA operations must select Proxy");
static_assert(defaultNetSelectProvider<NvidiaPlatform>(
                  0, DefaultNetGdaRequiredCapabilities, false) ==
                  DefaultNetProviderProxy,
              "missing GDA resources must select Proxy");

// Compile every NVIDIA default Net dispatch entry and the Inter Barrier
// path without requiring an NIC in this single-GPU test executable.
FLAGCX_GLOBAL_DECORATOR void
compileDefaultNetProvider(DeviceAPI::Comm comm, DeviceAPI::Window window) {
  DeviceAPI::Net net(comm, 0);
  (void)net.isValid();
  DeviceAPI::Team team{1, 0, 1};
  NvidiaTraits::CoopThread thread;
  DeviceAPI::DescriptorSmem descriptor;
  net.put(team, 0, window, 0, window, 0, 0, flagcxDevNet_None{},
          flagcxDevNet_None{}, thread, descriptor, flagcxDeviceScopeThread,
          flagcxDeviceScopeDevice);
  net.get(team, 0, window, 0, window, 0, 0, thread);
  net.putValue(team, 0, window, 0, uint64_t{1}, flagcxDevNet_None{}, thread,
               descriptor, flagcxDeviceScopeThread, flagcxDeviceScopeDevice);
  net.signal(team, 0, flagcxDevNet_SignalInc{(flagcxDevSignal_t)0}, thread,
             descriptor, flagcxDeviceScopeThread, flagcxDeviceScopeDevice);
  net.flush(thread, flagcxDeviceMemoryOrderAcquire);
  net.waitSignal(thread, (flagcxDevSignal_t)0, 1, DeviceAPI::completionBits,
                 flagcxDeviceMemoryOrderAcquire);
  net.readSignal((flagcxDevSignal_t)0, DeviceAPI::completionBits,
                 flagcxDeviceMemoryOrderAcquire);
  net.resetSignal((flagcxDevSignal_t)0);
  net.waitCounter(thread, (flagcxDevCounter_t)0, 1,
                  DeviceAPI::defaultCounterBits,
                  flagcxDeviceMemoryOrderAcquire);
  net.readCounter((flagcxDevCounter_t)0, DeviceAPI::defaultCounterBits,
                  flagcxDeviceMemoryOrderAcquire);
  net.resetCounter((flagcxDevCounter_t)0);
  DeviceAPI::Barrier<flagcxTeamTagInter, NvidiaTraits::CoopBlock> barrier(
      NvidiaTraits::CoopBlock{}, net, comm, team, 0, 1);
  barrier.arrive();
  barrier.wait();
}

struct Sum {
  FLAGCX_DEVICE_DECORATOR uint32_t operator()(uint32_t a, uint32_t b) const {
    return a + b;
  }
};

FLAGCX_GLOBAL_DECORATOR void
testTraits(uint32_t *src, uint32_t *dst, uint32_t *dstWarp, uint32_t *dstThread,
           uint32_t *dstSingle, uint32_t *dstLegacy, uint32_t *results,
           uint32_t *atomicWord, uint64_t *elapsed) {
  const int lane = NvidiaTraits::Intrin::lane();
  NvidiaTraits::CoopBlock block;
  NvidiaTraits::CoopWarp warp;
  NvidiaTraits::CoopThread thread;
  NvidiaTraits::CoopTile<8> tile;
  NvidiaTraits::CoopTileSpan span(0, 1, 0);

  if (block.threadRank() != lane || block.size() != 32 ||
      warp.threadRank() != lane || warp.size() != 32 ||
      tile.threadRank() != lane % 8 || tile.size() != 8 ||
      tile.laneMask() != (flagcxLaneMask_t{0xff} << (lane & ~7)) ||
      thread.size() != 1 || span.threadRank() != lane || span.size() != 32)
    results[lane] |= 1u;

  const flagcxLaneMask_t expectedLower = (flagcxLaneMask_t{1} << lane) - 1;
  if (NvidiaTraits::Intrin::lanemaskLt() != expectedLower ||
      NvidiaTraits::Intrin::activemask() != NvidiaTraits::Intrin::fullMask() ||
      NvidiaTraits::Intrin::popc(NvidiaTraits::Intrin::fullMask()) != 32)
    results[lane] |= 2u;
  NvidiaTraits::Intrin::validateMask(NvidiaTraits::Intrin::fullMask());
  NvidiaTraits::Intrin::syncwarp();
  NvidiaTraits::Intrin::namedBarrierSync(1, 32);
  NvidiaTraits::Intrin::spinBackoff(1);
  NvidiaTraits::Intrin::threadfenceDevice();
  NvidiaTraits::Intrin::threadfenceSystem();

  warp.sync();
  tile.sync();
  span.sync();
  block.sync();
  NvidiaTraits::CoopAny any(warp);
  NvidiaTraits::CoopAny defaultAny;
  if (any.threadRank() != lane || any.size() != 32)
    results[lane] |= 4u;
  any.sync();
  if (defaultAny.threadRank() != 0 || defaultAny.size() != 1)
    results[lane] |= 4u;
  defaultAny.sync();

  const uint32_t bcast = Ops::broadcast(warp, uint32_t(lane + 1), 3);
  const uint32_t sum = Ops::reduce(warp, uint32_t(lane + 1), Sum{});
  if (bcast != 4 || sum != 528 ||
      Ops::broadcast(tile, uint32_t(lane), 0) != uint32_t(lane & ~7))
    results[lane] |= 8u;
  if (lane < 16) {
    NvidiaTraits::CoopLanes lanes(0xffffu);
    if (lanes.threadRank() != lane || lanes.size() != 16 ||
        lanes.getLmask() != 0xffffu ||
        Ops::broadcast(lanes, uint32_t(lane), 5) != 5u ||
        Ops::reduce(lanes, uint32_t(lane + 1), Sum{}) != 136u)
      results[lane] |= 16u;
    lanes.sync();
  }

  Ops::copy<uint32_t, Vector4, Vector4, flagcxDeviceMemoryGlobal,
            flagcxDeviceMemoryGlobal>(block, dst, src, 67u);
  Ops::copy(warp, dstWarp, src, 67u);
  NvidiaTraits::Intrin::coopCopyBytes(dstLegacy, src, 67u * sizeof(uint32_t),
                                      lane, 32);
  if (lane == 0) {
    Ops::copy(thread, dstThread, src, 67u);
    NvidiaTraits::Intrin::copyBytes(dstSingle, src, 67u * sizeof(uint32_t));
  }
  block.sync();
  if (lane < 32 &&
      (dst[lane] != src[lane] || dstWarp[lane] != src[lane] ||
       dstThread[lane] != src[lane] || dstSingle[lane] != src[lane] ||
       dstLegacy[lane] != src[lane]))
    results[lane] |= 32u;

  uint32_t localCopy = src[lane];
  Ops::copy<uint32_t, Scalar, Scalar, flagcxDeviceMemoryLocal,
            flagcxDeviceMemoryGlobal>(thread, &dstThread[lane], &localCopy, 1u);
  if (dstThread[lane] != src[lane])
    results[lane] |= 64u;

  FLAGCX_SHARED_DECORATOR uint32_t scratch[32];
  scratch[lane] = uint32_t(lane);
  block.sync();
  const uint32_t local =
      Ops::load<uint32_t, Scalar, flagcxDeviceMemoryScratchpad,
                flagcxDeviceMemoryOrderRelaxed, flagcxDeviceScopeBlock>(
          thread, &scratch[lane]);
  Ops::store<uint32_t, Scalar, flagcxDeviceMemoryScratchpad,
             flagcxDeviceMemoryOrderRelaxed, flagcxDeviceScopeBlock>(
      thread, &scratch[lane], local + 1);
  Ops::store<uint32_t, Scalar, flagcxDeviceMemoryScratchpad,
             flagcxDeviceMemoryOrderRelease, flagcxDeviceScopeBlock>(
      thread, &scratch[lane], local + 1);
  const uint32_t sharedAcquire =
      Ops::load<uint32_t, Scalar, flagcxDeviceMemoryScratchpad,
                flagcxDeviceMemoryOrderAcquire, flagcxDeviceScopeBlock>(
          thread, &scratch[lane]);
  uint32_t privateValue = local;
  Ops::store<uint32_t, Scalar, flagcxDeviceMemoryLocal,
             flagcxDeviceMemoryOrderWeak, flagcxDeviceScopeThread>(
      thread, &privateValue, local + 2);
  const uint32_t privateRead =
      Ops::load<uint32_t, Scalar, flagcxDeviceMemoryLocal,
                flagcxDeviceMemoryOrderWeak, flagcxDeviceScopeThread>(
          thread, &privateValue);
  if (scratch[lane] != uint32_t(lane + 1) ||
      sharedAcquire != uint32_t(lane + 1) || privateRead != uint32_t(lane + 2))
    results[lane] |= 64u;

  if (lane == 0) {
    const uint64_t start = DeviceAPI::Clock::now();
    NvidiaTraits::Intrin::spinBackoff(100);
    const uint64_t end = DeviceAPI::Clock::now();
    *elapsed = DeviceAPI::Clock::elapsedNs(start, end);

    if (NvidiaTraits::Intrin::byteSwap<uint32_t>(0x12345678u) != 0x78563412u ||
        NvidiaTraits::Intrin::byteSwap<uint64_t>(0x0102030405060708ull) !=
            0x0807060504030201ull ||
        NvidiaTraits::Intrin::funnelShiftRight<uint32_t>(
            0x12345678u, 0xabcdef01u, 4) != 0x11234567u ||
        NvidiaTraits::Intrin::funnelShiftRight<uint64_t,
                                               flagcxDeviceFunnelShiftClamp>(
            0x1122334455667788ull, 0x99aabbccddeeff00ull, 64) !=
            0x99aabbccddeeff00ull)
      results[0] |= 128u;

    const uint32_t immutable =
        Ops::load<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
                  flagcxDeviceMemoryOrderWeak, flagcxDeviceScopeSystem,
                  flagcxDeviceMemoryAccessKindNormal,
                  flagcxDeviceLoadIntentImmutable>(thread, src);
    const uint32_t vectorBytes =
        Ops::load<uint32_t, Vector4, flagcxDeviceMemoryGlobal>(thread, src);
    const uint32_t noAllocate =
        Ops::load<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
                  flagcxDeviceMemoryOrderRelaxed, flagcxDeviceScopeSystem,
                  flagcxDeviceMemoryAccessKindNormal,
                  flagcxDeviceLoadIntentOrdinary,
                  flagcxDeviceCacheHintNoAllocate>(thread, src);
    if (immutable != src[0] || vectorBytes != src[0] || noAllocate != src[0])
      results[0] |= 256u;

    Ops::store<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryOrderRelaxed>(thread, atomicWord, uint32_t{1});
    if (Ops::load<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
                  flagcxDeviceMemoryOrderAcquire>(thread, atomicWord) != 1u)
      results[0] |= 512u;
    Ops::store<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryOrderRelease, flagcxDeviceScopeSystem,
               flagcxDeviceMemoryAccessKindNormal,
               flagcxDeviceCacheHintNoAllocate>(thread, atomicWord,
                                                uint32_t{1});
    if (Ops::load<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
                  flagcxDeviceMemoryOrderAcquire>(thread, atomicWord) != 1u)
      results[0] |= 512u;
    if (DeviceAPI::Atomic::fetchAdd(atomicWord, uint32_t{2},
                                    flagcxDeviceMemoryOrderAcqRel) != 1u ||
        DeviceAPI::Atomic::fetchSub(atomicWord, uint32_t{1},
                                    flagcxDeviceMemoryOrderAcqRel) != 3u ||
        DeviceAPI::Atomic::fetchOr(atomicWord, uint32_t{4},
                                   flagcxDeviceMemoryOrderAcqRel) != 2u ||
        DeviceAPI::Atomic::fetchAnd(atomicWord, uint32_t{3},
                                    flagcxDeviceMemoryOrderAcqRel) != 6u ||
        DeviceAPI::Atomic::exchange(atomicWord, uint32_t{5},
                                    flagcxDeviceMemoryOrderAcqRel) != 2u)
      results[0] |= 1024u;
    uint32_t expected = 5;
    if (!DeviceAPI::Atomic::compareExchange(atomicWord, expected, uint32_t{7},
                                            flagcxDeviceMemoryOrderAcqRel) ||
        DeviceAPI::Atomic::fetchMax(atomicWord, uint32_t{9},
                                    flagcxDeviceMemoryOrderAcqRel) != 7u ||
        DeviceAPI::Atomic::fetchMax(atomicWord, uint32_t{8},
                                    flagcxDeviceMemoryOrderAcquire) != 9u ||
        DeviceAPI::Atomic::load(atomicWord, flagcxDeviceMemoryOrderAcquire) !=
            9u)
      results[0] |= 2048u;
    Ops::store<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryOrderSeqCst>(thread, atomicWord, uint32_t{9});
    if (Ops::load<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
                  flagcxDeviceMemoryOrderSeqCst>(thread, atomicWord) != 9u)
      results[0] |= 4096u;
    (void)DeviceAPI::Atomic::toNativeOrder(flagcxDeviceMemoryOrderAcquire);
    (void)DeviceAPI::Atomic::toNativeScope(flagcxDeviceScopeSystem);
    Ops::fence(thread, flagcxDeviceMemoryOrderRelease, flagcxDeviceScopeSystem);
  }
}

FLAGCX_GLOBAL_DECORATOR void testBulkCopy(uint32_t *dst, int *ran) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  alignas(16) FLAGCX_SHARED_DECORATOR uint32_t scratch[64];
  const int lane = threadIdx.x;
  scratch[lane] = uint32_t(lane + 1);
  scratch[lane + 32] = uint32_t(lane + 33);
  NvidiaTraits::CoopBlock block;
  block.sync();
  Ops::AsyncCopyBatch batch;
  Ops::copyAsync(block, batch, dst, scratch, 64u);
  Ops::commitAsync(block, batch);
  Ops::waitAsync(block, batch);
  block.sync();
  if (threadIdx.x == 0)
    *ran = 1;
#else
  (void)dst;
  if (threadIdx.x == 0)
    *ran = 0;
#endif
}

FLAGCX_GLOBAL_DECORATOR void testAsyncStore(uint32_t *dst, int *ran) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 &&                         \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
  if (threadIdx.x == 0)
    Ops::storeAsync(NvidiaTraits::CoopThread{}, dst, uint32_t{77},
                    flagcxDeviceMemoryOrderRelease);
  if (threadIdx.x == 0)
    *ran = 1;
#else
  (void)dst;
  if (threadIdx.x == 0)
    *ran = 0;
#endif
}

// Compile-only probe. A real NIC BAR mapping is required before either MMIO
// branch may be executed; this kernel is deliberately never launched.
FLAGCX_GLOBAL_DECORATOR void compileMmio(uint32_t *mappedBar, bool enabled) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 &&                          \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2))
  if (enabled && threadIdx.x == 0) {
    Ops::store<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryOrderRelease, flagcxDeviceScopeSystem,
               flagcxDeviceMemoryAccessKindMmio>(NvidiaTraits::CoopThread{},
                                                 mappedBar, uint32_t{1});
    (void)Ops::load<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
                    flagcxDeviceMemoryOrderRelaxed, flagcxDeviceScopeSystem,
                    flagcxDeviceMemoryAccessKindMmio>(
        NvidiaTraits::CoopThread{}, mappedBar);
#if __CUDA_ARCH__ >= 750 &&                                                    \
    (__CUDACC_VER_MAJOR__ > 13 ||                                              \
     (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 3))
    (void)Ops::load<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
                    flagcxDeviceMemoryOrderAcquire, flagcxDeviceScopeSystem,
                    flagcxDeviceMemoryAccessKindMmio>(
        NvidiaTraits::CoopThread{}, mappedBar);
#endif
  }
#else
  (void)mappedBar;
  (void)enabled;
#endif
}

// Exercise each supported value format with real device memory. The 16/32-byte
// paths are guarded by the target and toolkit that can assemble their PTX.
FLAGCX_GLOBAL_DECORATOR void testPtxShapes(uint8_t *bytes, uint16_t *halves,
                                           uint64_t *wide, float *floats,
                                           Bulk16 *b16, Bulk32 *b32, int *ran) {
  if (threadIdx.x != 0)
    return;
  NvidiaTraits::CoopThread thread;
  uint8_t byte =
      NvidiaTraits::Intrin::load<uint8_t, Scalar, flagcxDeviceMemoryGlobal>(
          bytes);
  NvidiaTraits::Intrin::store<uint8_t, Scalar, flagcxDeviceMemoryGlobal>(
      bytes + 1, byte);
  Ops::store<uint8_t, Scalar, flagcxDeviceMemoryGlobal>(
      thread, bytes + 2,
      Ops::load<uint8_t, Scalar, flagcxDeviceMemoryGlobal>(thread, bytes));
  uint16_t half =
      NvidiaTraits::Intrin::load<uint16_t, Vector2, flagcxDeviceMemoryGlobal>(
          halves);
  NvidiaTraits::Intrin::store<uint16_t, Vector2, flagcxDeviceMemoryGlobal>(
      halves + 1, half);
  Ops::store<uint16_t, Vector2, flagcxDeviceMemoryGlobal>(
      thread, halves + 2,
      Ops::load<uint16_t, Vector2, flagcxDeviceMemoryGlobal>(thread, halves));
  uint64_t word =
      NvidiaTraits::Intrin::load<uint64_t, Scalar, flagcxDeviceMemoryGlobal>(
          wide);
  NvidiaTraits::Intrin::store<uint64_t, Scalar, flagcxDeviceMemoryGlobal>(
      wide + 1, word);
  NvidiaTraits::Intrin::copy<uint64_t, Scalar, Scalar, flagcxDeviceMemoryGlobal,
                             flagcxDeviceMemoryGlobal>(wide + 2, wide);
  float value =
      NvidiaTraits::Intrin::load<float, Vector4, flagcxDeviceMemoryGlobal>(
          floats);
  NvidiaTraits::Intrin::store<float, Vector4, flagcxDeviceMemoryGlobal>(
      floats + 1, value);
  Ops::store<float, Vector4, flagcxDeviceMemoryGlobal>(
      thread, floats + 2,
      Ops::load<float, Vector4, flagcxDeviceMemoryGlobal>(thread, floats));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 &&                          \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
  Bulk16 block =
      NvidiaTraits::Intrin::load<Bulk16, Scalar, flagcxDeviceMemoryGlobal>(b16);
  NvidiaTraits::Intrin::store<Bulk16, Scalar, flagcxDeviceMemoryGlobal>(b16 + 1,
                                                                        block);
  NvidiaTraits::Intrin::copy<Bulk16, Scalar, Vector4, flagcxDeviceMemoryGlobal,
                             flagcxDeviceMemoryGlobal>(b16 + 2, b16);
  Ops::copy<Bulk16, Vector4, Scalar, flagcxDeviceMemoryGlobal,
            flagcxDeviceMemoryGlobal>(thread, b16 + 3, b16, 1u);
  Ops::store<Bulk16, Scalar, flagcxDeviceMemoryGlobal>(
      thread, b16 + 4,
      Ops::load<Bulk16, Scalar, flagcxDeviceMemoryGlobal>(thread, b16));
  FLAGCX_SHARED_DECORATOR Bulk16 sharedBlock;
  NvidiaTraits::Intrin::copy<Bulk16, Scalar, Scalar, flagcxDeviceMemoryGlobal,
                             flagcxDeviceMemoryScratchpad>(&sharedBlock, b16);
  NvidiaTraits::Intrin::copy<Bulk16, Scalar, Scalar,
                             flagcxDeviceMemoryScratchpad,
                             flagcxDeviceMemoryGlobal>(b16 + 5, &sharedBlock);
  Bulk16 localBlock = block;
  NvidiaTraits::Intrin::copy<Bulk16, Scalar, Scalar, flagcxDeviceMemoryGlobal,
                             flagcxDeviceMemoryLocal>(&localBlock, b16);
  NvidiaTraits::Intrin::copy<Bulk16, Scalar, Scalar, flagcxDeviceMemoryLocal,
                             flagcxDeviceMemoryGlobal>(b16 + 6, &localBlock);
  ran[0] = 1;
#else
  ran[0] = 0;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 &&                         \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
  Bulk32 wideBlock =
      NvidiaTraits::Intrin::load<Bulk32, Vector8, flagcxDeviceMemoryGlobal>(
          b32);
  NvidiaTraits::Intrin::store<Bulk32, Vector4, flagcxDeviceMemoryGlobal>(
      b32 + 1, wideBlock);
  Ops::copy<Bulk32, Vector8, Vector4, flagcxDeviceMemoryGlobal,
            flagcxDeviceMemoryGlobal>(thread, b32 + 2, b32, 1u);
  Ops::store<Bulk32, Vector4, flagcxDeviceMemoryGlobal>(
      thread, b32 + 3,
      Ops::load<Bulk32, Vector8, flagcxDeviceMemoryGlobal>(thread, b32));
  ran[1] = 1;
#else
  (void)b32;
  ran[1] = 0;
#endif
}

FLAGCX_GLOBAL_DECORATOR void testFetchMaxContention(uint32_t *word) {
  DeviceAPI::Atomic::fetchMax(word, uint32_t(threadIdx.x),
                              flagcxDeviceMemoryOrderAcqRel);
}

static bool cudaOk(cudaError_t result, const char *operation) {
  if (result == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
  return false;
}

int main() {
  int count = 0;
  if (!cudaOk(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count == 0)
    return 77;
  cudaDeviceProp prop;
  if (!cudaOk(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties"))
    return 1;

  uint32_t *src = nullptr, *dst = nullptr, *dstWarp = nullptr;
  uint32_t *dstThread = nullptr, *dstSingle = nullptr;
  uint32_t *dstLegacy = nullptr, *results = nullptr, *atomicWord = nullptr;
  uint64_t *elapsed = nullptr;
  int *bulkRan = nullptr;
  uint8_t *bytes8 = nullptr;
  uint16_t *halves16 = nullptr;
  uint64_t *wide64 = nullptr;
  float *floats32 = nullptr;
  Bulk16 *blocks16 = nullptr;
  Bulk32 *blocks32 = nullptr;
  int *formatRan = nullptr;
  const size_t bytes = 67 * sizeof(uint32_t);
  if (!cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&src), bytes),
              "src allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&dst), bytes),
              "dst allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&dstWarp), bytes),
              "warp allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&dstThread), bytes),
              "thread allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&dstSingle), bytes),
              "single allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&dstLegacy), bytes),
              "legacy allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&results),
                                32 * sizeof(uint32_t)),
              "results allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&atomicWord),
                                sizeof(uint32_t)),
              "atomic allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&elapsed),
                                sizeof(uint64_t)),
              "clock allocation") ||
      !cudaOk(
          cudaMallocManaged(reinterpret_cast<void **>(&bulkRan), sizeof(int)),
          "bulk flag allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&bytes8),
                                3 * sizeof(uint8_t)),
              "byte format allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&halves16),
                                3 * sizeof(uint16_t)),
              "half format allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&wide64),
                                3 * sizeof(uint64_t)),
              "wide format allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&floats32),
                                3 * sizeof(float)),
              "float format allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&blocks16),
                                7 * sizeof(Bulk16)),
              "Bulk16 allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&blocks32),
                                4 * sizeof(Bulk32)),
              "Bulk32 allocation") ||
      !cudaOk(cudaMallocManaged(reinterpret_cast<void **>(&formatRan),
                                2 * sizeof(int)),
              "format capability allocation"))
    return 1;

  for (int i = 0; i < 67; ++i)
    src[i] = uint32_t(i * 17 + 3);
  std::memset(dst, 0, bytes);
  std::memset(dstWarp, 0, bytes);
  std::memset(dstThread, 0, bytes);
  std::memset(dstSingle, 0, bytes);
  std::memset(dstLegacy, 0, bytes);
  std::memset(results, 0, 32 * sizeof(uint32_t));
  *atomicWord = 0;
  *elapsed = 0;
  *bulkRan = 0;
  bytes8[0] = 0xa5;
  halves16[0] = 0xb53d;
  wide64[0] = 0x0123456789abcdefull;
  floats32[0] = 3.25f;
  std::memset(bytes8 + 1, 0, 2 * sizeof(uint8_t));
  std::memset(halves16 + 1, 0, 2 * sizeof(uint16_t));
  std::memset(wide64 + 1, 0, 2 * sizeof(uint64_t));
  std::memset(floats32 + 1, 0, 2 * sizeof(float));
  std::memset(blocks16, 0, 7 * sizeof(Bulk16));
  std::memset(blocks32, 0, 4 * sizeof(Bulk32));
  for (int i = 0; i < 4; ++i)
    blocks16[0].words[i] = uint32_t(0x10203040u + i * 0x11111111u);
  for (int i = 0; i < 8; ++i)
    blocks32[0].words[i] = uint32_t(0x01020304u + i * 0x10101010u);
  formatRan[0] = formatRan[1] = 0;

  testTraits<<<1, 32>>>(src, dst, dstWarp, dstThread, dstSingle, dstLegacy,
                        results, atomicWord, elapsed);
  if (!cudaOk(cudaDeviceSynchronize(), "testTraits"))
    return 1;
  for (int i = 0; i < 32; ++i) {
    if (results[i] != 0) {
      std::fprintf(stderr, "lane %d failed: 0x%x\n", i, results[i]);
      return 1;
    }
  }
  for (int i = 0; i < 67; ++i) {
    if (dst[i] != src[i] || dstWarp[i] != src[i] || dstThread[i] != src[i] ||
        dstSingle[i] != src[i] || dstLegacy[i] != src[i]) {
      std::fprintf(stderr, "copy failed at element %d\n", i);
      return 1;
    }
  }
  if (*atomicWord != 9u)
    return 1;
  *atomicWord = 0;
  testFetchMaxContention<<<1, 32>>>(atomicWord);
  if (!cudaOk(cudaDeviceSynchronize(), "testFetchMaxContention") ||
      *atomicWord != 31u)
    return 1;
  if (*elapsed == 0) {
    std::fputs("Clock did not advance during the test\n", stderr);
    return 1;
  }

  testPtxShapes<<<1, 1>>>(bytes8, halves16, wide64, floats32, blocks16,
                          blocks32, formatRan);
  if (!cudaOk(cudaDeviceSynchronize(), "testPtxShapes"))
    return 1;
  if (bytes8[1] != bytes8[0] || bytes8[2] != bytes8[0] ||
      halves16[1] != halves16[0] || halves16[2] != halves16[0] ||
      wide64[1] != wide64[0] || wide64[2] != wide64[0] ||
      floats32[1] != floats32[0] || floats32[2] != floats32[0]) {
    std::fputs("scalar/vector load-store or Intrin copy failed\n", stderr);
    return 1;
  }
#if __CUDACC_VER_MAJOR__ > 12 ||                                               \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
  if (prop.major >= 7 && !formatRan[0]) {
    std::fputs("Bulk16 PTX path unexpectedly skipped\n", stderr);
    return 1;
  }
#endif
  if (formatRan[0]) {
    for (int slot = 1; slot < 7; ++slot)
      for (int word = 0; word < 4; ++word)
        if (blocks16[slot].words[word] != blocks16[0].words[word]) {
          std::fprintf(stderr, "Bulk16 copy failed at slot %d word %d\n", slot,
                       word);
          return 1;
        }
  } else {
    std::puts("Bulk16 PTX test skipped: target or toolkit lacks b128");
  }
#if __CUDACC_VER_MAJOR__ > 12 ||                                               \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
  if (prop.major >= 10 && !formatRan[1]) {
    std::fputs("Bulk32 PTX path unexpectedly skipped\n", stderr);
    return 1;
  }
#endif
  if (formatRan[1]) {
    for (int slot = 1; slot < 4; ++slot)
      for (int word = 0; word < 8; ++word)
        if (blocks32[slot].words[word] != blocks32[0].words[word]) {
          std::fprintf(stderr, "Bulk32 copy failed at slot %d word %d\n", slot,
                       word);
          return 1;
        }
  } else {
    std::puts("Bulk32 PTX test skipped: target or toolkit lacks 32B vectors");
  }

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12
  if (prop.major >= 9) {
    testBulkCopy<<<1, 32>>>(dst, bulkRan);
    if (!cudaOk(cudaDeviceSynchronize(), "testBulkCopy"))
      return 1;
    if (*bulkRan) {
      for (int i = 0; i < 64; ++i)
        if (dst[i] != uint32_t(i + 1))
          return 1;
    }
  }
  if (prop.major >= 10) {
    *bulkRan = 0;
    testAsyncStore<<<1, 32>>>(atomicWord, bulkRan);
    if (!cudaOk(cudaDeviceSynchronize(), "testAsyncStore"))
      return 1;
    if (*bulkRan && *atomicWord != 77u)
      return 1;
  }
#endif

  cudaFree(src);
  cudaFree(dst);
  cudaFree(dstWarp);
  cudaFree(dstThread);
  cudaFree(dstSingle);
  cudaFree(dstLegacy);
  cudaFree(results);
  cudaFree(atomicWord);
  cudaFree(elapsed);
  cudaFree(bulkRan);
  cudaFree(bytes8);
  cudaFree(halves16);
  cudaFree(wide64);
  cudaFree(floats32);
  cudaFree(blocks16);
  cudaFree(blocks32);
  cudaFree(formatRan);
  std::puts("NVIDIA default PlatformTraits passed");
  return 0;
}
