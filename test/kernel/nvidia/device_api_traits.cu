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
__global__ void compileDefaultNetProvider(DeviceAPI::Comm comm,
                                          DeviceAPI::Window window) {
  DeviceAPI::Net net(comm, 0);
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
  __device__ uint32_t operator()(uint32_t a, uint32_t b) const { return a + b; }
};

__global__ void testTraits(uint32_t *src, uint32_t *dst, uint32_t *dstWarp,
                           uint32_t *dstThread, uint32_t *dstSingle,
                           uint32_t *dstLegacy, uint32_t *results,
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

  Ops::copy(block, dst, src, 67u);
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

  __shared__ uint32_t scratch[32];
  scratch[lane] = uint32_t(lane);
  block.sync();
  using ScratchPtr = uint32_t *;
  const uint32_t local =
      Ops::load<decltype(thread), ScratchPtr, flagcxDeviceMemoryScratchpad,
                flagcxDeviceMemoryAccessKindNormal, flagcxDeviceScopeBlock,
                flagcxDeviceCacheL1, flagcxDeviceCacheHintNone>(
          thread, &scratch[lane], flagcxDeviceMemoryOrderRelaxed);
  Ops::store<decltype(thread), ScratchPtr, uint32_t,
             flagcxDeviceMemoryScratchpad, flagcxDeviceMemoryAccessKindNormal,
             flagcxDeviceScopeBlock, flagcxDeviceCacheL1,
             flagcxDeviceCacheHintNone>(thread, &scratch[lane], local + 1,
                                        flagcxDeviceMemoryOrderRelaxed);
  uint32_t privateValue = local;
  Ops::store<decltype(thread), ScratchPtr, uint32_t, flagcxDeviceMemoryPrivate,
             flagcxDeviceMemoryAccessKindNormal, flagcxDeviceScopeThread,
             flagcxDeviceCacheL1, flagcxDeviceCacheHintNone>(
      thread, &privateValue, local + 2, flagcxDeviceMemoryOrderRelaxed);
  const uint32_t privateRead =
      Ops::load<decltype(thread), ScratchPtr, flagcxDeviceMemoryPrivate,
                flagcxDeviceMemoryAccessKindNormal, flagcxDeviceScopeThread,
                flagcxDeviceCacheL1, flagcxDeviceCacheHintNone>(
          thread, &privateValue, flagcxDeviceMemoryOrderRelaxed);
  if (scratch[lane] != uint32_t(lane + 1) || privateRead != uint32_t(lane + 2))
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

    using Relaxed = std::integral_constant<flagcxDeviceMemoryOrder_t,
                                           flagcxDeviceMemoryOrderRelaxed>;
    const uint32_t immutable =
        Ops::load<decltype(thread), const uint32_t *,
                  flagcxDeviceLoadIntentImmutable>(thread, src, Relaxed{});
    const uint32_t noAllocate =
        Ops::load<decltype(thread), const uint32_t *, flagcxDeviceMemoryGlobal,
                  flagcxDeviceMemoryAccessKindNormal, flagcxDeviceScopeSystem,
                  flagcxDeviceCacheL1, flagcxDeviceCacheHintNoAllocate>(
            thread, src, Relaxed{});
    if (immutable != src[0] || noAllocate != src[0])
      results[0] |= 256u;

    Ops::store(thread, atomicWord, uint32_t{1}, flagcxDeviceMemoryOrderRelaxed);
    if (Ops::load(thread, atomicWord, flagcxDeviceMemoryOrderAcquire) != 1u)
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
    (void)DeviceAPI::Atomic::toNativeOrder(flagcxDeviceMemoryOrderAcquire);
    (void)DeviceAPI::Atomic::toNativeScope(flagcxDeviceScopeSystem);
    Ops::fence(thread, flagcxDeviceMemoryOrderRelease, flagcxDeviceScopeSystem);
  }
}

__global__ void testBulkCopy(uint32_t *dst, int *ran) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ alignas(16) uint32_t scratch[64];
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

__global__ void testAsyncStore(uint32_t *dst, int *ran) {
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
__global__ void compileMmio(uint32_t *mappedBar, bool enabled) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 &&                          \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2))
  if (enabled && threadIdx.x == 0) {
    using Coop = NvidiaTraits::CoopThread;
    Ops::store<Coop, uint32_t *, uint32_t, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryAccessKindMmio, flagcxDeviceScopeSystem,
               flagcxDeviceCacheL1, flagcxDeviceCacheHintNone>(
        Coop{}, mappedBar, uint32_t{1}, flagcxDeviceMemoryOrderRelease);
    (void)Ops::load<Coop, uint32_t *, flagcxDeviceMemoryGlobal,
                    flagcxDeviceMemoryAccessKindMmio, flagcxDeviceScopeSystem,
                    flagcxDeviceCacheL1, flagcxDeviceCacheHintNone>(
        Coop{}, mappedBar, flagcxDeviceMemoryOrderRelaxed);
  }
#else
  (void)mappedBar;
  (void)enabled;
#endif
}

__global__ void unsupportedBlockBroadcast(uint32_t *dst) {
  NvidiaTraits::CoopBlock block;
  dst[threadIdx.x] = Ops::broadcast(block, uint32_t{1}, 0);
}

__global__ void unsupportedBlockReduce(uint32_t *dst) {
  NvidiaTraits::CoopBlock block;
  dst[threadIdx.x] = Ops::reduce(block, uint32_t{1}, Sum{});
}

__global__ void unsupportedSpanBroadcast(uint32_t *dst) {
  NvidiaTraits::CoopTileSpan span(0, 1, 0);
  dst[threadIdx.x] = Ops::broadcast(span, uint32_t{1}, 0);
}

__global__ void unsupportedAnyReduce(uint32_t *dst) {
  NvidiaTraits::CoopBlock block;
  NvidiaTraits::CoopAny any(block);
  dst[threadIdx.x] = Ops::reduce(any, uint32_t{1}, Sum{});
}

__global__ void invalidRuntimeOrder(uint32_t *src,
                                    flagcxDeviceMemoryOrder_t order) {
  if (threadIdx.x == 0)
    (void)Ops::load(NvidiaTraits::CoopThread{}, src, order);
}

__global__ void testFetchMaxContention(uint32_t *word) {
  DeviceAPI::Atomic::fetchMax(word, uint32_t(threadIdx.x),
                              flagcxDeviceMemoryOrderAcqRel);
}

static bool cudaOk(cudaError_t result, const char *operation) {
  if (result == cudaSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
  return false;
}

int main(int argc, char **argv) {
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
          "bulk flag allocation"))
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

  if (argc > 1 && std::strcmp(argv[1], "--expect-block-trap") == 0) {
    unsupportedBlockBroadcast<<<1, 32>>>(results);
    return cudaDeviceSynchronize() == cudaSuccess ? 1 : 0;
  }
  if (argc > 1 && std::strcmp(argv[1], "--expect-reduce-trap") == 0) {
    unsupportedBlockReduce<<<1, 32>>>(results);
    return cudaDeviceSynchronize() == cudaSuccess ? 1 : 0;
  }
  if (argc > 1 && std::strcmp(argv[1], "--expect-span-trap") == 0) {
    unsupportedSpanBroadcast<<<1, 32>>>(results);
    return cudaDeviceSynchronize() == cudaSuccess ? 1 : 0;
  }
  if (argc > 1 && std::strcmp(argv[1], "--expect-any-trap") == 0) {
    unsupportedAnyReduce<<<1, 32>>>(results);
    return cudaDeviceSynchronize() == cudaSuccess ? 1 : 0;
  }
  if (argc > 1 && std::strcmp(argv[1], "--expect-order-trap") == 0) {
    invalidRuntimeOrder<<<1, 32>>>(src, flagcxDeviceMemoryOrderRelease);
    return cudaDeviceSynchronize() == cudaSuccess ? 1 : 0;
  }

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
  std::puts("NVIDIA default PlatformTraits passed");
  return 0;
}
