/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Unit tests for NVSHMEM CommTraits and host-side adaptor.
 * Build: make USE_NVIDIA=1 USE_SHMEM=1 SHMEM_HOME=/path/to/nvshmem
 * Run:   nvshmrun -np 2 build/bin/test_nvshmem_adaptor
 *
 * Kernels are compiled separately in
 *test/kernel/nvidia/nvshmem_adaptor_kernel.cu
 ************************************************************************/

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "nvshmem_adaptor_kernel.h"

// Pull in the traits for host-side struct layout (Comm, Net, Window)
#define FLAGCX_COMM_TRAITS_SHMEM
#define USE_NVIDIA_ADAPTOR
#include "device_api/comm_traits.h"
#include "shmem_adaptor.h"

using DC = CommTraits<NvshmemBackend>;

// ============================================================
// Test fixture: NVSHMEM init/finalize + device setup
// ============================================================
static int g_pe, g_npes;

static void setup() {
  nvshmem_init();
  g_pe = nvshmem_my_pe();
  g_npes = nvshmem_n_pes();
  int devCount = 1;
  cudaGetDeviceCount(&devCount);
  int dev = g_pe % devCount;
  cudaSetDevice(dev);
  if (g_pe == 0)
    printf("[nvshmem_test] %d PEs initialized, device=%d\n", g_npes, dev);
}

static void teardown() { nvshmem_finalize(); }

// ============================================================
// Test: put operation (PE 0 → PE 1)
// ============================================================
static void test_put() {
  if (g_npes < 2) {
    printf("[SKIP] test_put: requires >= 2 PEs\n");
    return;
  }
  const size_t N = 256;
  void *symBuf = nvshmem_malloc(N);
  assert(symBuf != nullptr);
  cudaMemset(symBuf, 0, N);

  // Source: local buffer with pattern
  void *srcBuf;
  cudaMalloc(&srcBuf, N);
  uint8_t pattern[256];
  memset(pattern, 0xAB, N);
  cudaMemcpy(srcBuf, pattern, N, cudaMemcpyHostToDevice);

  // Build Net on device
  DC::Comm dc = {};
  dc.rank = g_pe;
  dc.nRanks = g_npes;
  DC::Net hostNet(dc, 0);
  DC::Net *devNet;
  cudaMalloc(&devNet, sizeof(DC::Net));
  cudaMemcpy(devNet, &hostNet, sizeof(DC::Net), cudaMemcpyHostToDevice);

  if (g_pe == 0) {
    launchKernelTestPut(devNet, symBuf, N, symBuf, srcBuf, N, srcBuf, 1, N,
                        nullptr);
    cudaDeviceSynchronize();
    nvshmem_quiet();
  }
  nvshmem_barrier_all();

  // Verify on PE 1
  if (g_pe == 1) {
    uint8_t result[256];
    cudaMemcpy(result, symBuf, N, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < N; i++) {
      if (result[i] != 0xAB) {
        ok = false;
        break;
      }
    }
    printf("[%s] test_put\n", ok ? "PASS" : "FAIL");
  }

  cudaFree(devNet);
  cudaFree(srcBuf);
  nvshmem_free(symBuf);
}

// ============================================================
// Test: signal operation (PE 0 signals PE 1)
// ============================================================
static void test_signal_and_wait() {
  if (g_npes < 2) {
    printf("[SKIP] test_signal_and_wait: requires >= 2 PEs\n");
    return;
  }
  const int SIG_COUNT = 4;
  uint64_t *sigBuf = (uint64_t *)nvshmem_malloc(SIG_COUNT * sizeof(uint64_t));
  assert(sigBuf != nullptr);
  cudaMemset(sigBuf, 0, SIG_COUNT * sizeof(uint64_t));

  DC::Comm dc = {};
  dc.rank = g_pe;
  dc.nRanks = g_npes;
  dc.signalBuffer = sigBuf;
  dc.signalCount = SIG_COUNT;
  DC::Net hostNet(dc, 0);
  DC::Net *devNet;
  cudaMalloc(&devNet, sizeof(DC::Net));
  cudaMemcpy(devNet, &hostNet, sizeof(DC::Net), cudaMemcpyHostToDevice);

  nvshmem_barrier_all();

  if (g_pe == 0) {
    launchKernelTestSignal(devNet, 1, 0, nullptr);
    cudaDeviceSynchronize();
  }

  if (g_pe == 1) {
    launchKernelTestWaitSignal(devNet, 0, 1, nullptr);
    cudaError_t err = cudaDeviceSynchronize();
    printf("[%s] test_signal_and_wait\n",
           (err == cudaSuccess) ? "PASS" : "FAIL");
  }

  nvshmem_barrier_all();
  cudaFree(devNet);
  nvshmem_free(sigBuf);
}

// ============================================================
// Test: flush (quiet / fence)
// ============================================================
static void test_flush() {
  DC::Comm dc = {};
  dc.rank = g_pe;
  dc.nRanks = g_npes;
  DC::Net hostNet(dc, 0);
  DC::Net *devNet;
  cudaMalloc(&devNet, sizeof(DC::Net));
  cudaMemcpy(devNet, &hostNet, sizeof(DC::Net), cudaMemcpyHostToDevice);

  launchKernelTestFlushQuiet(devNet, nullptr);
  cudaError_t err1 = cudaDeviceSynchronize();
  launchKernelTestFlushFence(devNet, nullptr);
  cudaError_t err2 = cudaDeviceSynchronize();

  if (g_pe == 0)
    printf("[%s] test_flush\n",
           (err1 == cudaSuccess && err2 == cudaSuccess) ? "PASS" : "FAIL");

  cudaFree(devNet);
}

// ============================================================
// Test: counter wait (local spin)
// ============================================================
static void test_counter_wait() {
  const int CTR_COUNT = 2;
  uint64_t *ctrBuf;
  cudaMalloc(&ctrBuf, CTR_COUNT * sizeof(uint64_t));
  cudaMemset(ctrBuf, 0, CTR_COUNT * sizeof(uint64_t));

  DC::Comm dc = {};
  dc.rank = g_pe;
  dc.nRanks = g_npes;
  dc.counterBuffer = ctrBuf;
  dc.counterCount = CTR_COUNT;
  DC::Net hostNet(dc, 0);
  DC::Net *devNet;
  cudaMalloc(&devNet, sizeof(DC::Net));
  cudaMemcpy(devNet, &hostNet, sizeof(DC::Net), cudaMemcpyHostToDevice);

  // Increment counter[0] from host-launched kernel first
  launchKernelIncCounter(ctrBuf, 0, nullptr);
  cudaDeviceSynchronize();

  // Now wait should return immediately (counter already >= 1)
  launchKernelWaitCounter(devNet, 0, 1, nullptr);
  cudaError_t err = cudaDeviceSynchronize();

  if (g_pe == 0)
    printf("[%s] test_counter_wait\n", (err == cudaSuccess) ? "PASS" : "FAIL");

  cudaFree(devNet);
  cudaFree(ctrBuf);
}

// ============================================================
// Test: barrier (world-scope)
// ============================================================
static void test_barrier_world() {
  if (g_npes < 2) {
    printf("[SKIP] test_barrier_world: requires >= 2 PEs\n");
    return;
  }
  const int BARRIER_COUNT = 1;
  const int INTRA_BARRIER_COUNT = 1;

  uint64_t *intraSig = (uint64_t *)nvshmem_malloc(INTRA_BARRIER_COUNT * g_npes *
                                                  sizeof(uint64_t));
  assert(intraSig);
  cudaMemset(intraSig, 0, INTRA_BARRIER_COUNT * g_npes * sizeof(uint64_t));

  uint64_t *worldSig =
      (uint64_t *)nvshmem_malloc(BARRIER_COUNT * g_npes * sizeof(uint64_t));
  assert(worldSig);
  cudaMemset(worldSig, 0, BARRIER_COUNT * g_npes * sizeof(uint64_t));

  int totalBarriers = INTRA_BARRIER_COUNT + BARRIER_COUNT;
  uint64_t *barrierUsage;
  cudaMalloc(&barrierUsage, totalBarriers * sizeof(uint64_t));
  cudaMemset(barrierUsage, 0, totalBarriers * sizeof(uint64_t));

  DC::Comm dc = {};
  dc.rank = g_pe;
  dc.nRanks = g_npes;
  dc.intraRank = g_pe;
  dc.intraSize = g_npes;
  dc.intraBarrierSignals = intraSig;
  dc.intraBarrierCount = INTRA_BARRIER_COUNT;
  dc.interBarrierCount = 0;
  dc.worldBarrierSignals = worldSig;
  dc.worldBarrierCount = BARRIER_COUNT;
  dc.barrierUsage = barrierUsage;

  DC::Comm *devDc;
  cudaMalloc(&devDc, sizeof(DC::Comm));
  cudaMemcpy(devDc, &dc, sizeof(DC::Comm), cudaMemcpyHostToDevice);

  nvshmem_barrier_all();

  launchKernelTestBarrierWorld(devDc, 0, nullptr);
  cudaError_t err = cudaDeviceSynchronize();

  nvshmem_barrier_all();
  if (g_pe == 0)
    printf("[%s] test_barrier_world\n", (err == cudaSuccess) ? "PASS" : "FAIL");

  cudaFree(devDc);
  cudaFree(barrierUsage);
  nvshmem_free(worldSig);
  nvshmem_free(intraSig);
}

// ============================================================
// Test: host-side adaptor lifecycle
// ============================================================
static void test_host_adaptor_lifecycle() {
  flagcxResult_t r1 = shmemAdaptor->init(g_pe, g_npes);
  flagcxResult_t r2 = shmemAdaptor->init(g_pe, g_npes);
  bool initOk = (r1 == flagcxSuccess && r2 == flagcxSuccess);

  void *ptr = nullptr;
  flagcxResult_t r3 = shmemAdaptor->malloc(&ptr, 1024);
  bool mallocOk = (r3 == flagcxSuccess && ptr != nullptr);
  if (ptr)
    shmemAdaptor->free(ptr);

  shmemAdaptor->finalize();
  shmemAdaptor->finalize();

  if (g_pe == 0)
    printf("[%s] test_host_adaptor_lifecycle\n",
           (initOk && mallocOk) ? "PASS" : "FAIL");
}

// ============================================================
// Main
// ============================================================
int main() {
  setup();

  test_put();
  test_signal_and_wait();
  test_flush();
  test_counter_wait();
  test_barrier_world();
  test_host_adaptor_lifecycle();

  teardown();
  return 0;
}
