/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * GPU wait cancellation and FlagCX/XPU NetIO primitive bridge; no NIC DMA.
 ************************************************************************/
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "device_api/flagcx_device.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <thread>
#include <type_traits>

using PT = PlatformTraits<NvidiaPlatform>;
using Bridge = flagcxXpuNetioNvidiaPlatform;
static_assert(std::is_same<xpu_netio_device_platform, Bridge>::value,
              "SDK must use the FlagCX primitive bridge");

__global__ void waitKernel(flagcxXpuNetioDeviceContext *state, uint64_t **peers,
                           uint64_t *epoch, uint32_t *control, int phase) {
  DeviceAPI::Comm comm{};
  comm.rank = comm.intraRank = 0;
  comm.nRanks = comm.intraSize = 2;
  comm.contextCount = 1;
  comm.signalCount = 1;
  comm.barrierSignalBase = 1;
  comm.signalBuffer = state->signalBuffer;
  comm.barrierPeers = peers;
  comm.epochBuffer = epoch;
  comm.nBarriers = 1;
  comm.netProvider = DefaultNetProviderGda;
  comm.netProviderId = 1;
  comm.gdaDeviceContext = state;
  PT::CoopBlock block;
  if (block.threadRank() == 0)
    PT::Atomic::store(control + 2, 1u, flagcxDeviceMemoryOrderRelease);
  if (phase == 0) {
    DeviceAPI::Net net(comm, 0);
    net.waitSignal(block, 0, 1, 64, flagcxDeviceMemoryOrderAcquire);
  } else {
    Barrier<DefaultBackend<NvidiaPlatform>, flagcxTeamTagIntra, PT::CoopBlock>
        barrier(block, comm, DeviceAPI::Team{2, 0, 1}, 0);
    barrier.wait(); // Peer 1 never arrives; all 32 threads must still exit.
  }
  if (block.threadRank() == 0)
    PT::Atomic::store(control + 3, 1u, flagcxDeviceMemoryOrderRelease);
}

__global__ void primitiveKernel(uint32_t *a, uint64_t *queue, uint8_t *copy,
                                uint32_t *results) {
  __shared__ uint8_t source[64];
  const unsigned lane = threadIdx.x;
  source[lane] = lane;
  source[lane + 32] = lane + 32;
  __syncthreads();
  constexpr uint32_t mask = 0x15; // Physical lanes 0, 2, 4.
  if (mask & (uint32_t{1} << lane)) {
    const auto broadcast = Bridge::warpBroadcast(lane, 4, mask);
    const auto maximum = Bridge::warpMax(lane, mask);
    results[lane] = broadcast == 4 && maximum == 4;
  }
  if (lane == 0) {
    *a = 1;
    assert(Bridge::casScoped<1>(a, 1u, 9u) == 1);
    assert(Bridge::casScoped<1>(a, 1u, 2u) == 9);
    assert(Bridge::maxScoped<1>(a, 12u) == 9);
    assert(Bridge::addScoped<1>(a, 2u) == 12);
    const uint64_t values[2] = {0x123456789abcdef0ULL, 0x9876543210abcdefULL};
    Bridge::storeQueue128(queue, values);
    Bridge::releaseSystem();
    assert(Bridge::loadDma64(queue) == values[0]);
    assert(Bridge::loadDma32(a) == 14);
    assert(Bridge::loadDma8(reinterpret_cast<uint8_t *>(a)) == 14);
    Bridge::copySharedToGlobal64(copy, source);
    Bridge::bulkCopyCommit();
    Bridge::bulkCopyWait();
    results[32] = Bridge::clockNs() != 0;
  }
}

int main() {
  uint32_t *host, *control;
  assert(cudaHostAlloc(&host, 4 * sizeof(uint32_t), cudaHostAllocMapped) ==
         cudaSuccess);
  assert(cudaHostGetDevicePointer(&control, host, 0) == cudaSuccess);
  flagcxXpuNetioDeviceContext *state;
  uint64_t **peers, *inboxes, *epoch, *signal;
#define ALLOC(p, n) assert(cudaMallocManaged(&(p), (n)) == cudaSuccess)
  ALLOC(state, sizeof(*state));
  ALLOC(peers, 2 * sizeof(*peers));
  ALLOC(inboxes, 4 * sizeof(*inboxes));
  ALLOC(epoch, sizeof(*epoch));
  ALLOC(signal, sizeof(*signal));
  std::memset(state, 0, sizeof(*state));
  state->contexts = reinterpret_cast<const xpu_netio_device_context_t *>(1);
  state->signalRemote = reinterpret_cast<const xpu_netio_remote_region_t *>(1);
  state->atomicResults.address = 1;
  state->errorFlag = control;
  state->abortFlag = control + 1;
  state->nranks = 2;
  state->signalCount = state->userSignalCount = 1;
  state->signalBuffer = signal;
  peers[0] = inboxes;
  peers[1] = inboxes + 2;
  cudaStream_t stream;
  cudaEvent_t done;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
         cudaSuccess);
  assert(cudaEventCreateWithFlags(&done, cudaEventDisableTiming) ==
         cudaSuccess);
  for (int phase = 0; phase < 3; ++phase) {
    std::memset(host, 0, 4 * sizeof(*host));
    std::memset(inboxes, 0, 4 * sizeof(*inboxes));
    *epoch = *signal = 0;
    state->timeoutNs = phase == 2 ? 5000000000ULL : 2000000ULL;
    waitKernel<<<1, 32, 0, stream>>>(state, peers, epoch, control, phase);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaEventRecord(done, stream) == cudaSuccess);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (__atomic_load_n(host + 2, __ATOMIC_ACQUIRE) == 0) {
      assert(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    if (phase == 2)
      __atomic_store_n(host + 1, 1u, __ATOMIC_RELEASE);
    cudaError_t status;
    while ((status = cudaEventQuery(done)) == cudaErrorNotReady) {
      assert(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    assert(status == cudaSuccess && host[3] == 1);
    assert(host[0] == (phase == 2 ? XPU_NETIO_ABORTED : XPU_NETIO_TIMEOUT));
    assert(*epoch == 0); // A failed IPC barrier must not commit its epoch.
  }
  uint32_t *atomic, *results;
  uint64_t *queue;
  uint8_t *copy;
  ALLOC(atomic, 4);
  ALLOC(results, 33 * sizeof(*results));
  ALLOC(queue, 16);
  ALLOC(copy, 64);
#undef ALLOC
  std::memset(results, 0, 33 * sizeof(*results));
  primitiveKernel<<<1, 32>>>(atomic, queue, copy, results);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(results[0] && results[2] && results[4] && results[32]);
  for (unsigned i = 0; i < 64; ++i)
    assert(copy[i] == i);
  for (void *p : {static_cast<void *>(state), static_cast<void *>(peers),
                  static_cast<void *>(inboxes), static_cast<void *>(epoch),
                  static_cast<void *>(signal), static_cast<void *>(atomic),
                  static_cast<void *>(results), static_cast<void *>(queue),
                  static_cast<void *>(copy)})
    assert(cudaFree(p) == cudaSuccess);
  assert(cudaFreeHost(host) == cudaSuccess);
  assert(cudaEventDestroy(done) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  std::puts(
      "PASS: GDA/IPC timeout, live Host abort, 32-thread exit, CAS old-value, "
      "masked warp, DMA loads, queue store, bulk copy; no NIC DMA");
}
