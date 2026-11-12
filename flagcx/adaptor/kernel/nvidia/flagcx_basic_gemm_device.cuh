#ifndef FLAGCX_BASIC_GEMM_DEVICE_CUH_
#define FLAGCX_BASIC_GEMM_DEVICE_CUH_

#include "device_utils.h"
#include "uni_runner_gemm.h"

FLAGCX_DEVICE_INLINE_DECORATOR void flagcxBasicGemmDevice(
    uint64_t a, uint64_t b, uint64_t c, uint64_t m, uint64_t n, uint64_t k,
    uint64_t lda, uint64_t ldb, uint64_t ldc, uint64_t nthreads,
    uint64_t accumulate, uint64_t workerId, uint64_t workerCount,
    float *sharedA, float *sharedB) {
  const float *aPtr = reinterpret_cast<const float *>(a);
  const float *bPtr = reinterpret_cast<const float *>(b);
  float *cPtr = reinterpret_cast<float *>(c);
  (void)nthreads;
  uint64_t tid = FLAGCX_THREAD_IDX_X;
  uint64_t threadRow = tid / 16;
  uint64_t threadCol = tid % 16;
  uint64_t numTileCols = uniRunnerGemmNumTileCols(n);
  uint64_t totalTiles = uniRunnerGemmTotalTiles(m, n);
  uint64_t tileBegin =
      uniRunnerGemmWorkerTileBegin(totalTiles, workerCount, workerId);
  uint64_t tileCount =
      uniRunnerGemmWorkerTileCount(totalTiles, workerCount, workerId);

  for (uint64_t tileId = tileBegin; tileId < tileBegin + tileCount; ++tileId) {
    uint64_t tileRow = tileId / numTileCols;
    uint64_t tileCol = tileId % numTileCols;
    float accum[kUniRunnerGemmMicroM][kUniRunnerGemmMicroN] = {};

    for (uint64_t kBase = 0; kBase < k; kBase += kUniRunnerGemmTileK) {
      for (uint64_t loadIndex = tid;
           loadIndex < kUniRunnerGemmTileM * kUniRunnerGemmTileK;
           loadIndex += kUniRunnerGemmThreads) {
        uint64_t localM = loadIndex / kUniRunnerGemmTileK;
        uint64_t localK = loadIndex % kUniRunnerGemmTileK;
        uint64_t globalM = tileRow * kUniRunnerGemmTileM + localM;
        uint64_t globalK = kBase + localK;
        sharedA[loadIndex] = globalM < m && globalK < k
                                 ? aPtr[globalM * lda + globalK]
                                 : 0.0f;
      }
      for (uint64_t loadIndex = tid;
           loadIndex < kUniRunnerGemmTileK * kUniRunnerGemmTileN;
           loadIndex += kUniRunnerGemmThreads) {
        uint64_t localK = loadIndex / kUniRunnerGemmTileN;
        uint64_t localN = loadIndex % kUniRunnerGemmTileN;
        uint64_t globalK = kBase + localK;
        uint64_t globalN = tileCol * kUniRunnerGemmTileN + localN;
        sharedB[loadIndex] = globalK < k && globalN < n
                                 ? bPtr[globalK * ldb + globalN]
                                 : 0.0f;
      }
      FLAGCX_DEVICE_SYNC_THREADS();

#pragma unroll
      for (uint64_t kk = 0; kk < kUniRunnerGemmTileK; ++kk) {
#pragma unroll
        for (uint64_t mi = 0; mi < kUniRunnerGemmMicroM; ++mi) {
#pragma unroll
          for (uint64_t nj = 0; nj < kUniRunnerGemmMicroN; ++nj) {
            accum[mi][nj] +=
                sharedA[(threadRow * kUniRunnerGemmMicroM + mi) *
                            kUniRunnerGemmTileK +
                        kk] *
                sharedB[kk * kUniRunnerGemmTileN +
                        threadCol * kUniRunnerGemmMicroN + nj];
          }
        }
      }
      FLAGCX_DEVICE_SYNC_THREADS();
    }

#pragma unroll
    for (uint64_t mi = 0; mi < kUniRunnerGemmMicroM; ++mi) {
#pragma unroll
      for (uint64_t nj = 0; nj < kUniRunnerGemmMicroN; ++nj) {
        uint64_t globalM = tileRow * kUniRunnerGemmTileM +
                           threadRow * kUniRunnerGemmMicroM + mi;
        uint64_t globalN = tileCol * kUniRunnerGemmTileN +
                           threadCol * kUniRunnerGemmMicroN + nj;
        if (globalM < m && globalN < n) {
          uint64_t outputIdx = globalM * ldc + globalN;
          cPtr[outputIdx] = accumulate != 0
                                ? cPtr[outputIdx] + accum[mi][nj]
                                : accum[mi][nj];
        }
      }
    }
  }
}

#endif // FLAGCX_BASIC_GEMM_DEVICE_CUH_
