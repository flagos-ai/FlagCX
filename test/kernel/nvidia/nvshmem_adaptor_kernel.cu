/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM adaptor test kernels — device code + host-callable launchers.
 * Host test logic lives in test/unittest/shmem_adaptor/test_nvshmem_adaptor.cpp
 ************************************************************************/

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define FLAGCX_COMM_TRAITS_SHMEM
#define USE_NVIDIA_ADAPTOR
#include "device_api/comm_traits.h"

using DC = CommTraits<NvshmemBackend>;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

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

__global__ void kernel_test_signal(DC::Net *net, int peer,
                                   flagcxDevNetSignal_t sigId) {
  DC::Team team = {net->_dc.nRanks, net->_dc.rank, 1};
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  flagcxDevNet_SignalInc ra = {sigId};
  flagcxDescriptorSmem desc;
  net->signal(team, peer, ra, coop, desc, flagcxDeviceScopeDevice,
              flagcxDeviceScopeDevice);
}

__global__ void kernel_test_wait_signal(DC::Net *net,
                                        flagcxDevNetSignal_t sigId,
                                        uint64_t expected) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->waitSignal(coop, sigId, expected, 64, flagcxDeviceMemoryOrderAcqRel);
}

__global__ void kernel_test_flush_quiet(DC::Net *net) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->flush(coop, flagcxDeviceMemoryOrderAcqRel);
}

__global__ void kernel_test_flush_fence(DC::Net *net) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->flush(coop, flagcxDeviceMemoryOrderRelease);
}

__global__ void kernel_inc_counter(uint64_t *counterBuf, int idx) {
  atomicAdd((unsigned long long *)&counterBuf[idx], 1ULL);
}

__global__ void kernel_wait_counter(DC::Net *net,
                                    flagcxDevNetCounter_t counterId,
                                    uint64_t least) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  net->waitCounter(coop, counterId, least, 64, flagcxDeviceMemoryOrderAcqRel);
}

__global__ void kernel_test_barrier_world(DC::Comm *devDc, uint32_t ctaIdx) {
  typename PlatformTraits<NvidiaPlatform>::CoopBlock coop;
  DC::Net net(*devDc, 0);
  Barrier<NvshmemBackend, flagcxTeamTagWorld,
          PlatformTraits<NvidiaPlatform>::CoopBlock>
      bar(coop, flagcxTeamTagWorld{}, net, *devDc, ctaIdx, false, 0);
  bar.sync();
}

// ---------------------------------------------------------------------------
// Host-callable launchers (extern "C" for .cpp linkage)
// ---------------------------------------------------------------------------

extern "C" void launchKernelTestPut(void *devNet, void *dst, size_t dstSize,
                                     void *dstRaw, void *src, size_t srcSize,
                                     void *srcRaw, int peer, size_t bytes,
                                     void *stream) {
  DC::Window dstW = {dst, dstSize, dstRaw};
  DC::Window srcW = {src, srcSize, srcRaw};
  kernel_test_put<<<1, 32, 0, (cudaStream_t)stream>>>(
      (DC::Net *)devNet, dstW, srcW, peer, bytes);
}

extern "C" void launchKernelTestSignal(void *devNet, int peer, int sigId,
                                        void *stream) {
  kernel_test_signal<<<1, 32, 0, (cudaStream_t)stream>>>(
      (DC::Net *)devNet, peer, (flagcxDevNetSignal_t)sigId);
}

extern "C" void launchKernelTestWaitSignal(void *devNet, int sigId,
                                            uint64_t expected, void *stream) {
  kernel_test_wait_signal<<<1, 32, 0, (cudaStream_t)stream>>>(
      (DC::Net *)devNet, (flagcxDevNetSignal_t)sigId, expected);
}

extern "C" void launchKernelTestFlushQuiet(void *devNet, void *stream) {
  kernel_test_flush_quiet<<<1, 32, 0, (cudaStream_t)stream>>>(
      (DC::Net *)devNet);
}

extern "C" void launchKernelTestFlushFence(void *devNet, void *stream) {
  kernel_test_flush_fence<<<1, 32, 0, (cudaStream_t)stream>>>(
      (DC::Net *)devNet);
}

extern "C" void launchKernelIncCounter(uint64_t *counterBuf, int idx,
                                        void *stream) {
  kernel_inc_counter<<<1, 1, 0, (cudaStream_t)stream>>>(counterBuf, idx);
}

extern "C" void launchKernelWaitCounter(void *devNet, int counterId,
                                         uint64_t least, void *stream) {
  kernel_wait_counter<<<1, 32, 0, (cudaStream_t)stream>>>(
      (DC::Net *)devNet, (flagcxDevNetCounter_t)counterId, least);
}

extern "C" void launchKernelTestBarrierWorld(void *devDc, uint32_t ctaIdx,
                                              void *stream) {
  kernel_test_barrier_world<<<1, 32, 0, (cudaStream_t)stream>>>(
      (DC::Comm *)devDc, ctaIdx);
}
