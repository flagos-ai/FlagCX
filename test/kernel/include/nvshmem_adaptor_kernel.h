/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM adaptor test kernel launchers — host-callable declarations.
 * Implementations live in test/kernel/nvidia/nvshmem_adaptor_kernel.cu.
 ************************************************************************/

#ifndef TEST_KERNEL_NVSHMEM_ADAPTOR_KERNEL_H_
#define TEST_KERNEL_NVSHMEM_ADAPTOR_KERNEL_H_

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void launchKernelTestPut(void *devNet, void *dst, size_t dstSize, void *dstRaw,
                         void *src, size_t srcSize, void *srcRaw, int peer,
                         size_t bytes, void *stream);
void launchKernelTestSignal(void *devNet, int peer, int sigId, void *stream);
void launchKernelTestWaitSignal(void *devNet, int sigId, uint64_t expected,
                                void *stream);
void launchKernelTestFlushQuiet(void *devNet, void *stream);
void launchKernelTestFlushFence(void *devNet, void *stream);
void launchKernelIncCounter(uint64_t *counterBuf, int idx, void *stream);
void launchKernelWaitCounter(void *devNet, int counterId, uint64_t least,
                             void *stream);
void launchKernelTestBarrierWorld(void *devDc, uint32_t ctaIdx, void *stream);

#ifdef __cplusplus
}
#endif

#endif // TEST_KERNEL_NVSHMEM_ADAPTOR_KERNEL_H_
