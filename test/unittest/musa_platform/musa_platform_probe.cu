/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Exercises the MUSA platform port independently of Host Device API and
 * communication backend integration. Run on the Linux MUSA server.
 ************************************************************************/

#include "device_api/platform_traits.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <musa_runtime.h>

#if !FLAGCX_CHECK_DEVICE_CC
#error "compile this probe with the MUSA device compiler"
#endif

using Traits = PlatformTraits<MusaPlatform>;
using Atomic = Traits::Atomic;
constexpr int kThreads = 64;

// Native SDK enums belong to the MUSA-compiled probe. The ordinary host
// probe independently checks the same public FlagCX layouts with g++.
static_assert(Atomic::toNativeOrder(flagcxDeviceMemoryOrderRelaxed) ==
                  musa::memory_order_relaxed,
              "relaxed order mapping changed");
static_assert(Atomic::toNativeOrder(flagcxDeviceMemoryOrderAcquire) ==
                  musa::memory_order_acquire,
              "acquire order mapping changed");
static_assert(Atomic::toNativeOrder(flagcxDeviceMemoryOrderRelease) ==
                  musa::memory_order_release,
              "release order mapping changed");
static_assert(Atomic::toNativeOrder(flagcxDeviceMemoryOrderAcqRel) ==
                  musa::memory_order_acq_rel,
              "acquire/release order mapping changed");
static_assert(Atomic::toNativeOrder(flagcxDeviceMemoryOrderSeqCst) ==
                  musa::memory_order_seq_cst,
              "sequentially consistent order mapping changed");
static_assert(Atomic::toNativeScope(flagcxDeviceScopeSystem) ==
                  musa::thread_scope_system,
              "system scope mapping changed");
static_assert(Atomic::toNativeScope(flagcxDeviceScopeDevice) ==
                  musa::thread_scope_device,
              "device scope mapping changed");
static_assert(Atomic::toNativeScope(flagcxDeviceScopeBlock) ==
                  musa::thread_scope_block,
              "block scope mapping changed");
static_assert(Atomic::toNativeScope(flagcxDeviceScopeThread) ==
                  musa::thread_scope_thread,
              "thread scope mapping changed");

struct Results {
  int simtWidth;
  int nativeWarpSize;
  size_t coopAnySize;
  size_t coopAnyAlignment;
  int lane[kThreads];
  flagcxLaneMask_t lowerMask[kThreads];
  flagcxLaneMask_t activeMask[kThreads];
  int tileRank[kThreads];
  flagcxLaneMask_t tileMask[kThreads];
  int lanesRank[kThreads];
  int spanRank[kThreads];
  int spanSize[kThreads];
  int spanSyncPassed[kThreads];
  int warpSpanSyncPassed[kThreads];
  int anyRank[kThreads];
  int anySize[kThreads];
  int anyGroupPassed[kThreads];
  uint32_t words32[4];
  uint64_t words64[4];
  int atomic32Passed[4];
  int atomic64Passed[4];
  uint32_t concurrent32;
  uint64_t concurrent64;
  uint32_t publicationFlag;
  uint32_t payload;
  int publicationPassed;
  uint32_t crossBlock32;
  uint64_t crossBlock64;
  uint32_t crossBlockPayload[kThreads];
  uint32_t crossBlockFlag;
  int crossBlockObserved;
  int crossBlockPassed;
};

template <typename T, flagcxDeviceScope_t Scope>
__device__ bool checkAtomics(T *word) {
  Atomic::store<T, Scope>(word, T{7}, flagcxDeviceMemoryOrderRelaxed);
  bool passed = Atomic::load<T, Scope>(word, flagcxDeviceMemoryOrderAcquire) ==
                T{7};
  passed &= Atomic::fetchAdd<T, Scope>(word, T{5},
                                     flagcxDeviceMemoryOrderRelease) == T{7};
  passed &= Atomic::fetchSub<T, Scope>(word, T{2},
                                     flagcxDeviceMemoryOrderAcqRel) == T{12};
  passed &= Atomic::fetchOr<T, Scope>(word, T{1},
                                    flagcxDeviceMemoryOrderRelaxed) == T{10};
  passed &= Atomic::fetchAnd<T, Scope>(word, T{7},
                                     flagcxDeviceMemoryOrderSeqCst) == T{11};
  passed &= Atomic::exchange<T, Scope>(word, T{9},
                                     flagcxDeviceMemoryOrderAcqRel) == T{3};

  T expected = T{8};
  passed &= !Atomic::compareExchange<T, Scope>(
      word, expected, T{13}, flagcxDeviceMemoryOrderRelaxed);
  passed &= expected == T{9};
  passed &= Atomic::compareExchange<T, Scope>(
      word, expected, T{14}, flagcxDeviceMemoryOrderRelease);
  passed &= Atomic::fetchMax<T, Scope>(word, T{16},
                                     flagcxDeviceMemoryOrderAcqRel) == T{14};
  passed &= Atomic::fetchMax<T, Scope>(word, T{15},
                                     flagcxDeviceMemoryOrderSeqCst) == T{16};
  Atomic::store<T, Scope>(word, T{21}, flagcxDeviceMemoryOrderRelease);
  passed &= Atomic::load<T, Scope>(word, flagcxDeviceMemoryOrderSeqCst) == T{21};
  return passed;
}

template <flagcxDeviceScope_t Scope>
__device__ void checkScope(Results *results, int index) {
  results->atomic32Passed[index] =
      checkAtomics<uint32_t, Scope>(&results->words32[index]);
  results->atomic64Passed[index] =
      checkAtomics<uint64_t, Scope>(&results->words64[index]);
}

// Check conversion, copy and synchronization for every supported group kind.
// These groups occupy contiguous threads; exchange with a peer in that group.
template <typename Group>
__device__ bool checkErasedGroup(Group group, unsigned *values) {
  Traits::CoopAny any(group);
  Traits::CoopAny copy(any);
  bool passed = any.threadRank() == group.threadRank() &&
                any.size() == group.size() &&
                copy.threadRank() == group.threadRank() &&
                copy.size() == group.size();
  const unsigned thread = threadIdx.x;
  const unsigned peer = thread - group.threadRank() +
                        (group.threadRank() + 1) % group.size();
  for (unsigned round = 0; round < 4; ++round) {
    values[thread] = round * kThreads + thread;
    any.sync();
    passed &= values[peer] == round * kThreads + peer;
    copy.sync();
  }
  return passed;
}

__global__ __launch_bounds__(kThreads) void anyGroupProbe(Results *results) {
  __shared__ unsigned values[kThreads];
  bool passed = checkErasedGroup(Traits::CoopBlock{}, values);
  passed &= checkErasedGroup(Traits::CoopWarp{}, values);
  passed &= checkErasedGroup(Traits::CoopTile<8>{}, values);
  const flagcxLaneMask_t tileMask = Traits::CoopTile<8>{}.laneMask();
  passed &= checkErasedGroup(Traits::CoopLanes{tileMask}, values);
  passed &= checkErasedGroup(Traits::CoopThread{}, values);
  Traits::CoopAny single;
  single.sync();
  passed &= single.threadRank() == 0 && single.size() == 1;
  results->anyGroupPassed[threadIdx.x] = passed;
}

// Shared by the full suite and the focused atomic/fence test. No span is
// constructed here; dynamic span/trap diagnostics remain in the full suite.
__device__ void exerciseAtomicChecks(Results *results) {
  const int thread = threadIdx.x;
  Traits::CoopBlock block;
  Atomic::fetchAdd<uint32_t, flagcxDeviceScopeDevice>(
      &results->concurrent32, uint32_t{1}, flagcxDeviceMemoryOrderRelaxed);
  Atomic::fetchAdd<uint64_t, flagcxDeviceScopeDevice>(
      &results->concurrent64, uint64_t{1}, flagcxDeviceMemoryOrderRelaxed);

  if (thread == 0) {
    results->simtWidth = Traits::Intrin::simtWidth;
    results->nativeWarpSize = warpSize;
    results->coopAnySize = sizeof(Traits::CoopAny);
    results->coopAnyAlignment = alignof(Traits::CoopAny);
    checkScope<flagcxDeviceScopeThread>(results, 0);
    checkScope<flagcxDeviceScopeBlock>(results, 1);
    checkScope<flagcxDeviceScopeDevice>(results, 2);
    checkScope<flagcxDeviceScopeSystem>(results, 3);
    Traits::Intrin::threadfenceDevice();
    Traits::Intrin::threadfenceSystem();
  }

  block.sync();
  // Two resident warps exchange payload without an intervening group
  // barrier. The loop is bounded so a visibility failure produces FAIL.
  if (thread == 0) {
    results->payload = 0x12345678u;
    Atomic::store<uint32_t, flagcxDeviceScopeDevice>(
        &results->publicationFlag, uint32_t{1},
        flagcxDeviceMemoryOrderRelease);
  }
  if (thread == 32) {
    for (int attempt = 0; attempt < 1000000; ++attempt) {
      if (Atomic::load<uint32_t, flagcxDeviceScopeDevice>(
              &results->publicationFlag, flagcxDeviceMemoryOrderAcquire)) {
        results->publicationPassed = results->payload == 0x12345678u;
        break;
      }
      Traits::Intrin::spinBackoff(0);
    }
  }
}

__global__ __launch_bounds__(kThreads) void atomicOnlyProbe(Results *results) {
  exerciseAtomicChecks(results);
}

// Two CTAs share device-scope counters. The producer CTA publishes the
// payload written by all its threads; a different CTA acquires and checks it.
// The producer never waits for the consumer, and the consumer wait is bounded.
__global__ __launch_bounds__(kThreads)
void crossBlockProbe(Results *results, unsigned round) {
  const unsigned thread = threadIdx.x;
  Atomic::fetchAdd<uint32_t, flagcxDeviceScopeDevice>(
      &results->crossBlock32, uint32_t{1}, flagcxDeviceMemoryOrderRelaxed);
  Atomic::fetchAdd<uint64_t, flagcxDeviceScopeDevice>(
      &results->crossBlock64, uint64_t{1}, flagcxDeviceMemoryOrderRelaxed);
  if (blockIdx.x == 0) {
    results->crossBlockPayload[thread] = 0x10000000u + round * kThreads + thread;
    Traits::CoopBlock{}.sync();
    if (thread == 0)
      Atomic::store<uint32_t, flagcxDeviceScopeDevice>(
          &results->crossBlockFlag, uint32_t{1}, flagcxDeviceMemoryOrderRelease);
  } else if (thread == 0) {
    for (int attempt = 0; attempt < 1000000; ++attempt) {
      if (Atomic::load<uint32_t, flagcxDeviceScopeDevice>(
              &results->crossBlockFlag, flagcxDeviceMemoryOrderAcquire)) {
        results->crossBlockObserved = 1;
        bool passed = true;
        for (unsigned i = 0; i < kThreads; ++i)
          passed &= results->crossBlockPayload[i] ==
                    0x10000000u + round * kThreads + i;
        results->crossBlockPassed = passed;
        break;
      }
      Traits::Intrin::spinBackoff(0);
    }
  }
}

__global__ __launch_bounds__(kThreads) void platformProbe(Results *results) {
  const int thread = threadIdx.x;
  Traits::CoopBlock block;
  Traits::CoopWarp warp;
  Traits::CoopTile<8> tile;
  Traits::CoopLanes lanes(Traits::Intrin::activemask());
  Traits::CoopTileSpan span(0, 2, 0);
  Traits::CoopAny any(warp);
  Traits::CoopAny anySpan(span);
  __shared__ unsigned spanValues[kThreads];

  results->lane[thread] = Traits::Intrin::lane();
  results->lowerMask[thread] = Traits::Intrin::lanemaskLt();
  results->activeMask[thread] = Traits::Intrin::activemask();
  results->tileRank[thread] = tile.threadRank();
  results->tileMask[thread] = tile.laneMask();
  results->lanesRank[thread] = lanes.threadRank();
  results->spanRank[thread] = span.threadRank();
  results->spanSize[thread] = span.size();
  results->anyRank[thread] = any.threadRank();
  results->anySize[thread] = any.size();

  tile.sync();
  lanes.sync();
  warp.sync();
  any.sync();
  span.sync();

  int spanSyncPassed = 1;
  for (unsigned round = 0; round < 32; ++round) {
    spanValues[thread] = round * kThreads + thread;
    span.sync();
    const unsigned peer = (thread + 32) % kThreads;
    spanSyncPassed &= spanValues[peer] == round * kThreads + peer;
    // The type-erased span must preserve the same synchronization scope.
    anySpan.sync();
  }
  results->spanSyncPassed[thread] = spanSyncPassed;

  exerciseAtomicChecks(results);
}

// Two independent warp spans complete different numbers of rounds. A
// whole-CTA barrier substituted here would couple the groups and can hang.
__global__ __launch_bounds__(kThreads) void warpSpanProbe(Results *results) {
  const unsigned thread = threadIdx.x;
  const unsigned warp = thread / 32;
  Traits::CoopTileSpan span(warp, 1, warp);
  Traits::CoopAny anySpan(span);
  __shared__ unsigned values[kThreads];
  int passed = 1;
  for (unsigned round = 0; round < (warp + 1) * 32; ++round) {
    values[thread] = round * kThreads + thread;
    span.sync();
    const unsigned peer = warp * 32 + (thread + 1) % 32;
    passed &= values[peer] == round * kThreads + peer;
    anySpan.sync();
  }
  results->warpSpanSyncPassed[thread] = passed;
}

static void checkValue(const char *name, int index, uint64_t actual,
                       uint64_t expected, int &failures) {
  if (actual != expected) {
    std::fprintf(stderr, "FAIL %s[%d]: got=%" PRIu64 " expected=%" PRIu64 "\n",
                 name, index, actual, expected);
    ++failures;
  }
}

#define CHECK_MUSA(call)                                                     \
  do {                                                                       \
    const musaError_t error = (call);                                         \
    if (error != musaSuccess) {                                               \
      std::fprintf(stderr, "%s failed: %s\n", #call, musaGetErrorString(error)); \
      return EXIT_FAILURE;                                                   \
    }                                                                        \
  } while (0)

static int reportKernelResources(const char *name, const void *kernel) {
  musaFuncAttributes attr{};
  CHECK_MUSA(musaFuncGetAttributes(&attr, kernel));
  std::fprintf(stderr,
               "%s: kernel=%p threads=%d maxThreadsPerBlock=%d numRegs=%d "
               "sharedBytes=%zu localBytes=%zu\n",
               name, const_cast<void *>(kernel), kThreads,
               attr.maxThreadsPerBlock, attr.numRegs,
               static_cast<size_t>(attr.sharedSizeBytes),
               static_cast<size_t>(attr.localSizeBytes));
  if (attr.maxThreadsPerBlock < kThreads) {
    std::fprintf(stderr, "%s cannot support the required %d-thread block\n",
                 name, kThreads);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

// Focused runs report only their selected checks. They do not imply that
// dynamic spans, cross-device visibility or communication passed.
static int runFocusedProbe(bool groups) {
  const char *name = groups ? "anyGroupProbe" : "atomicOnlyProbe";
  const void *kernel = groups ? reinterpret_cast<const void *>(anyGroupProbe)
                              : reinterpret_cast<const void *>(atomicOnlyProbe);
  if (reportKernelResources(name, kernel))
    return EXIT_FAILURE;
  Results *device = nullptr;
  Results host{};
  CHECK_MUSA(musaMalloc(reinterpret_cast<void **>(&device), sizeof(Results)));
  CHECK_MUSA(musaMemset(device, 0, sizeof(Results)));
  std::fprintf(stderr, "Launching %s (1 block, %d threads)\n", name, kThreads);
  if (groups) {
    anyGroupProbe<<<1, kThreads>>>(device);
  } else {
    atomicOnlyProbe<<<1, kThreads>>>(device);
  }
  CHECK_MUSA(musaGetLastError());
  CHECK_MUSA(musaDeviceSynchronize());
  CHECK_MUSA(musaMemcpy(&host, device, sizeof(Results), musaMemcpyDeviceToHost));
  CHECK_MUSA(musaFree(device));

  int failures = 0;
  if (groups) {
    for (int thread = 0; thread < kThreads; ++thread)
      checkValue("CoopAny group conversion and synchronization", thread,
                 host.anyGroupPassed[thread], 1, failures);
  } else {
    for (int scope = 0; scope < 4; ++scope) {
      checkValue("atomic32", scope, host.atomic32Passed[scope], 1, failures);
      checkValue("atomic64", scope, host.atomic64Passed[scope], 1, failures);
    }
    checkValue("contended atomic32", 0, host.concurrent32, kThreads, failures);
    checkValue("contended atomic64", 0, host.concurrent64, kThreads, failures);
    checkValue("release/acquire publication", 0, host.publicationPassed, 1,
               failures);
  }
  std::printf("MUSA %s-only probe %s (%d failures)\n",
              groups ? "groups" : "atomics",
              failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int runCrossBlockProbe() {
  if (reportKernelResources("crossBlockProbe",
                            reinterpret_cast<const void *>(crossBlockProbe)))
    return EXIT_FAILURE;
  Results *device = nullptr;
  Results host{};
  CHECK_MUSA(musaMalloc(reinterpret_cast<void **>(&device), sizeof(Results)));
  constexpr unsigned rounds = 8;
  int failures = 0;
  std::fprintf(stderr, "Launching crossBlockProbe (%u rounds, 2 blocks, %d threads/block)\n",
               rounds, kThreads);
  for (unsigned round = 0; round < rounds; ++round) {
    CHECK_MUSA(musaMemset(device, 0, sizeof(Results)));
    crossBlockProbe<<<2, kThreads>>>(device, round);
    CHECK_MUSA(musaGetLastError());
    CHECK_MUSA(musaDeviceSynchronize());
    CHECK_MUSA(musaMemcpy(&host, device, sizeof(Results), musaMemcpyDeviceToHost));
    checkValue("cross-block atomic32", round, host.crossBlock32, 2 * kThreads,
               failures);
    checkValue("cross-block atomic64", round, host.crossBlock64, 2 * kThreads,
               failures);
    checkValue("cross-block publication observed", round,
               host.crossBlockObserved, 1, failures);
    checkValue("cross-block payload visibility", round,
               host.crossBlockPassed, 1, failures);
  }
  CHECK_MUSA(musaFree(device));
  std::printf("MUSA cross-block probe %s (%u rounds, %d failures)\n",
              failures == 0 ? "PASS" : "FAIL", rounds, failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--groups-only") == 0)
    return runFocusedProbe(true);
  if (argc == 2 && std::strcmp(argv[1], "--atomics-only") == 0)
    return runFocusedProbe(false);
  if (argc == 2 && std::strcmp(argv[1], "--cross-block-only") == 0)
    return runCrossBlockProbe();
  if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--groups-only|--atomics-only|--cross-block-only]\n", argv[0]);
    return EXIT_FAILURE;
  }

  int resourceFailures = 0;
  resourceFailures += reportKernelResources(
      "platformProbe", reinterpret_cast<const void *>(platformProbe));
  resourceFailures += reportKernelResources(
      "warpSpanProbe", reinterpret_cast<const void *>(warpSpanProbe));
  resourceFailures += reportKernelResources(
      "anyGroupProbe", reinterpret_cast<const void *>(anyGroupProbe));
  if (resourceFailures)
    return EXIT_FAILURE;
  Results *device = nullptr;
  Results host{};
  CHECK_MUSA(musaMalloc(reinterpret_cast<void **>(&device), sizeof(Results)));
  CHECK_MUSA(musaMemset(device, 0, sizeof(Results)));
  std::fprintf(stderr, "Launching platformProbe (1 block, %d threads)\n", kThreads);
  platformProbe<<<1, kThreads>>>(device);
  CHECK_MUSA(musaGetLastError());
  CHECK_MUSA(musaDeviceSynchronize());
  std::fprintf(stderr, "Launching warpSpanProbe (1 block, %d threads)\n", kThreads);
  warpSpanProbe<<<1, kThreads>>>(device);
  CHECK_MUSA(musaGetLastError());
  CHECK_MUSA(musaDeviceSynchronize());
  std::fprintf(stderr, "Launching anyGroupProbe (1 block, %d threads)\n", kThreads);
  anyGroupProbe<<<1, kThreads>>>(device);
  CHECK_MUSA(musaGetLastError());
  CHECK_MUSA(musaDeviceSynchronize());
  CHECK_MUSA(musaMemcpy(&host, device, sizeof(Results), musaMemcpyDeviceToHost));
  CHECK_MUSA(musaFree(device));

  int failures = 0;
  checkValue("SIMT", 0, host.simtWidth, 32, failures);
  checkValue("warpSize", 0, host.nativeWarpSize, 32, failures);
  checkValue("CoopAny size", 0, host.coopAnySize, sizeof(Traits::CoopAny),
             failures);
  checkValue("CoopAny alignment", 0, host.coopAnyAlignment,
             alignof(Traits::CoopAny), failures);
  for (int thread = 0; thread < kThreads; ++thread) {
    const int lane = thread % 32;
    checkValue("lane", thread, host.lane[thread], lane, failures);
    checkValue("lanemaskLt", thread, host.lowerMask[thread],
               (uint64_t{1} << lane) - 1, failures);
    checkValue("activemask", thread, host.activeMask[thread], 0xffffffffull,
               failures);
    checkValue("tile rank", thread, host.tileRank[thread], lane % 8, failures);
    checkValue("tile mask", thread, host.tileMask[thread],
               uint64_t{0xff} << (lane & -8), failures);
    checkValue("lanes rank", thread, host.lanesRank[thread], lane, failures);
    checkValue("span rank", thread, host.spanRank[thread], thread, failures);
    checkValue("span size", thread, host.spanSize[thread], kThreads, failures);
    checkValue("full-CTA span synchronization", thread,
               host.spanSyncPassed[thread], 1, failures);
    checkValue("independent warp span synchronization", thread,
               host.warpSpanSyncPassed[thread], 1, failures);
    checkValue("CoopAny rank", thread, host.anyRank[thread], lane, failures);
    checkValue("CoopAny size", thread, host.anySize[thread], 32, failures);
    checkValue("CoopAny group conversion and synchronization", thread,
               host.anyGroupPassed[thread], 1, failures);
  }
  for (int scope = 0; scope < 4; ++scope) {
    checkValue("atomic32", scope, host.atomic32Passed[scope], 1, failures);
    checkValue("atomic64", scope, host.atomic64Passed[scope], 1, failures);
  }
  checkValue("contended atomic32", 0, host.concurrent32, kThreads, failures);
  checkValue("contended atomic64", 0, host.concurrent64, kThreads, failures);
  checkValue("release/acquire publication", 0, host.publicationPassed, 1,
             failures);
  std::printf("MUSA platform probe %s (%d failures)\n",
              failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
