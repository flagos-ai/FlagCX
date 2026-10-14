#ifndef FLAGCX_UNIRUNNER_GEMM_H_
#define FLAGCX_UNIRUNNER_GEMM_H_

#include "device_utils.h"

#include <cstdint>

constexpr uint64_t kUniRunnerGemmTileM = 64;
constexpr uint64_t kUniRunnerGemmTileN = 64;
constexpr uint64_t kUniRunnerGemmTileK = 16;
constexpr uint64_t kUniRunnerGemmThreads = 256;
constexpr uint64_t kUniRunnerGemmMicroM = 4;
constexpr uint64_t kUniRunnerGemmMicroN = 4;
constexpr uint64_t kUniRunnerGemmCounterStrideBytes = 128;

static_assert(kUniRunnerGemmTileM % kUniRunnerGemmMicroM == 0, "");
static_assert(kUniRunnerGemmTileN % kUniRunnerGemmMicroN == 0, "");
static_assert((kUniRunnerGemmTileM / kUniRunnerGemmMicroM) *
                      (kUniRunnerGemmTileN / kUniRunnerGemmMicroN) ==
                  kUniRunnerGemmThreads,
              "");
static_assert((kUniRunnerGemmTileM * kUniRunnerGemmTileK) %
                      kUniRunnerGemmThreads ==
                  0,
              "");
static_assert((kUniRunnerGemmTileK * kUniRunnerGemmTileN) %
                      kUniRunnerGemmThreads ==
                  0,
              "");
static_assert(kUniRunnerGemmCounterStrideBytes >= sizeof(uint32_t), "");
static_assert(kUniRunnerGemmCounterStrideBytes % alignof(uint32_t) == 0, "");

FLAGCX_HOST_DEVICE_INLINE uint64_t uniRunnerGemmCeilDivU64(uint64_t value,
                                                           uint64_t divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

FLAGCX_HOST_DEVICE_INLINE uint64_t uniRunnerGemmNumTileRows(uint64_t m) {
  return uniRunnerGemmCeilDivU64(m, kUniRunnerGemmTileM);
}

FLAGCX_HOST_DEVICE_INLINE uint64_t uniRunnerGemmNumTileCols(uint64_t n) {
  return uniRunnerGemmCeilDivU64(n, kUniRunnerGemmTileN);
}

FLAGCX_HOST_DEVICE_INLINE uint64_t uniRunnerGemmTotalTiles(uint64_t m,
                                                           uint64_t n) {
  return uniRunnerGemmNumTileRows(m) * uniRunnerGemmNumTileCols(n);
}

FLAGCX_HOST_DEVICE_INLINE uint64_t
uniRunnerGemmWorkerTileCount(uint64_t totalTiles, uint64_t workerCount,
                             uint64_t workerId) {
  uint64_t baseTiles = totalTiles / workerCount;
  uint64_t remainder = totalTiles % workerCount;
  return baseTiles + (workerId < remainder ? 1 : 0);
}

FLAGCX_HOST_DEVICE_INLINE uint64_t
uniRunnerGemmWorkerTileBegin(uint64_t totalTiles, uint64_t workerCount,
                             uint64_t workerId) {
  uint64_t baseTiles = totalTiles / workerCount;
  uint64_t remainder = totalTiles % workerCount;
  return workerId * baseTiles + (workerId < remainder ? workerId : remainder);
}

#endif // FLAGCX_UNIRUNNER_GEMM_H_
