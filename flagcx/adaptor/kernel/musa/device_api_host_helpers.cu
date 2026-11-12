/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device-compiled helpers for MUSA Device API host integration.
 ************************************************************************/

#include "device_api/flagcx_device_internal.h"
#include "device_api/flagcx_device_core.h"
#include <new>

extern "C" size_t flagcxDevNetSizeOf() { return sizeof(flagcxDevNet); }

static __global__ __launch_bounds__(256) void flagcxDevNetConstructKernel(flagcxDevNet *nets,
                                                   flagcxDevComm *comm,
                                                   int count) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < count)
    ::new (&nets[i]) flagcxDevNet(*comm, i);
}

extern "C" void flagcxDevNetLaunchConstruct(void *devNets, void *devComm,
                                            int count, void *stream) {
  if (count <= 0 || devNets == nullptr || devComm == nullptr)
    return;
  int threads = count < 256 ? count : 256;
  int blocks = (count + threads - 1) / threads;
  flagcxDevNetConstructKernel<<<blocks, threads, 0, (musaStream_t)stream>>>(
      static_cast<flagcxDevNet *>(devNets),
      static_cast<flagcxDevComm *>(devComm), count);
}
