/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Unit tests for NVSHMEM CommTraits and host-side adaptor.
 * Build: nvcc -DFLAGCX_COMM_TRAITS_SHMEM -DUSE_NVIDIA_ADAPTOR ...
 *
 * Multi-PE tests require: nvshmrun -np 2 ./test_nvshmem_adaptor
 ************************************************************************/

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// Pull in the traits under test
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

static void teardown() {
  nvshmem_finalize();
}

// ============================================================
// Test: put operation (PE 0 → PE 1)
// ============================================================
__global__ void kernel_test_put(DC::Net *net, DC::Window dst, DC::Window src,
                                int peer, size_t bytes) {
  DC::Team team = {net->_dc.nRanks, net->_dc.rank, 1};
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  flagcxDevNet_None ra;
  flagcxDevNet_None la;
  flagcxDescriptorSmem desc;
  net->put(team, peer, dst, 0, src, 0, bytes, ra, la, coop, desc,
           flagcxDeviceScopeDevice, flagcxDeviceScopeDevice);
}

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
  uint8_t pattern[N];
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

  DC::Window src = {srcBuf, N, srcBuf};
  DC::Window dst = {symBuf, N, symBuf};

  if (g_pe == 0) {
    kernel_test_put<<<1, 32>>>(devNet, dst, src, 1, N);
    cudaDeviceSynchronize();
    nvshmem_quiet();
  }
  nvshmem_barrier_all();

  // Verify on PE 1
  if (g_pe == 1) {
    uint8_t result[N];
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
__global__ void kernel_test_signal(DC::Net *net, int peer,
                                   flagcxDevSignal_t sigId) {
  DC::Team team = {net->_dc.nRanks, net->_dc.rank, 1};
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  flagcxDevNet_SignalInc ra = {sigId};
  flagcxDescriptorSmem desc;
  net->signal(team, peer, ra, coop, desc, flagcxDeviceScopeDevice,
              flagcxDeviceScopeDevice);
}

__global__ void kernel_test_wait_signal(DC::Net *net,
                                        flagcxDevSignal_t sigId,
                                        uint64_t expected) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->waitSignal(coop, sigId, expected, 64, flagcxDeviceMemoryOrderAcquire);
}

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
    // Signal PE 1 at signal index 0
    kernel_test_signal<<<1, 32>>>(devNet, 1, 0);
    cudaDeviceSynchronize();
  }

  if (g_pe == 1) {
    // Wait for signal
    kernel_test_wait_signal<<<1, 32>>>(devNet, 0, 1);
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
__global__ void kernel_test_flush_quiet(DC::Net *net) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->flush(coop, flagcxDeviceMemoryOrderAcqRel);
}

__global__ void kernel_test_flush_fence(DC::Net *net) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->flush(coop, flagcxDeviceMemoryOrderRelease);
}

static void test_flush() {
  DC::Comm dc = {};
  dc.rank = g_pe;
  dc.nRanks = g_npes;
  DC::Net hostNet(dc, 0);
  DC::Net *devNet;
  cudaMalloc(&devNet, sizeof(DC::Net));
  cudaMemcpy(devNet, &hostNet, sizeof(DC::Net), cudaMemcpyHostToDevice);

  kernel_test_flush_quiet<<<1, 32>>>(devNet);
  cudaError_t err1 = cudaDeviceSynchronize();
  kernel_test_flush_fence<<<1, 32>>>(devNet);
  cudaError_t err2 = cudaDeviceSynchronize();

  if (g_pe == 0)
    printf("[%s] test_flush\n",
           (err1 == cudaSuccess && err2 == cudaSuccess) ? "PASS" : "FAIL");

  cudaFree(devNet);
}

// ============================================================
// Test: counter wait (local spin)
// ============================================================
__global__ void kernel_inc_counter(uint64_t *counterBuf, int idx) {
  atomicAdd((unsigned long long *)&counterBuf[idx], 1ULL);
}

__global__ void kernel_wait_counter(DC::Net *net,
                                    flagcxDevCounter_t counterId,
                                    uint64_t least) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->waitCounter(coop, counterId, least, 64,
                   flagcxDeviceMemoryOrderAcquire);
}

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
  kernel_inc_counter<<<1, 1>>>(ctrBuf, 0);
  cudaDeviceSynchronize();

  // Now wait should return immediately (counter already >= 1)
  kernel_wait_counter<<<1, 32>>>(devNet, 0, 1);
  cudaError_t err = cudaDeviceSynchronize();

  if (g_pe == 0)
    printf("[%s] test_counter_wait\n", (err == cudaSuccess) ? "PASS" : "FAIL");

  cudaFree(devNet);
  cudaFree(ctrBuf);
}

// ============================================================
// Test: barrier (world-scope)
// ============================================================
__global__ void kernel_test_barrier_world(DC::Comm *devDc, uint32_t ctaIdx) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  DC::Net net(*devDc, 0);
  Barrier<NvshmemBackend, flagcxTeamTagWorld,
          PlatformTraits<NvidiaPlatform>::CoopBlock>
      bar(coop, flagcxTeamTagWorld{}, net, *devDc, ctaIdx, false, 0);
  bar.sync();
}

static void test_barrier_world() {
  if (g_npes < 2) {
    printf("[SKIP] test_barrier_world: requires >= 2 PEs\n");
    return;
  }
  const int BARRIER_COUNT = 1;
  const int INTRA_BARRIER_COUNT = 1;

  // Intra barrier signals (world barrier composes intra internally)
  uint64_t *intraSig =
      (uint64_t *)nvshmem_malloc(INTRA_BARRIER_COUNT * g_npes * sizeof(uint64_t));
  assert(intraSig);
  cudaMemset(intraSig, 0, INTRA_BARRIER_COUNT * g_npes * sizeof(uint64_t));

  // World barrier signals
  uint64_t *worldSig =
      (uint64_t *)nvshmem_malloc(BARRIER_COUNT * g_npes * sizeof(uint64_t));
  assert(worldSig);
  cudaMemset(worldSig, 0, BARRIER_COUNT * g_npes * sizeof(uint64_t));

  // barrierUsage: [intra(0..INTRA-1), inter(none), world(INTRA..INTRA+WORLD-1)]
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

  kernel_test_barrier_world<<<1, 32>>>(devDc, 0);
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
  // Test init (idempotent/ref-counted)
  flagcxResult_t r1 = shmemAdaptor->init(g_pe, g_npes);
  flagcxResult_t r2 = shmemAdaptor->init(g_pe, g_npes);
  bool initOk = (r1 == flagcxSuccess && r2 == flagcxSuccess);

  // Test malloc/free
  void *ptr = nullptr;
  flagcxResult_t r3 = shmemAdaptor->malloc(&ptr, 1024);
  bool mallocOk = (r3 == flagcxSuccess && ptr != nullptr);
  if (ptr)
    shmemAdaptor->free(ptr);

  // Finalize (matches double init)
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
