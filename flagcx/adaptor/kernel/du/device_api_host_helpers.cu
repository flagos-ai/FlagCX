/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device-compiled helpers for device API host integration.
 * Compiled by nvcc, linked into libflagcx.so.
 * Called from flagcx_device.cc via extern "C".
 *
 * Other vendors provide equivalent implementations in their own
 * device-compiled files (.xpu, .hip, etc.).
 ************************************************************************/

#include "device_api/flagcx_device_internal.h"
#include "device_api/flagcx_device_core.h"
#include <new>

extern "C" size_t flagcxDevNetSizeOf() { return sizeof(flagcxDevNet); }

static __global__ void flagcxDevNetConstructKernel(flagcxDevNet *nets,
                                            flagcxDevComm *comm, int count) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < count) {
    ::new (&nets[i]) flagcxDevNet(*comm, i);
  }
}

// Launches the kernel to construct flagcxDevNet[] on device.
// The caller is responsible for patching comm->_netContexts and
// synchronizing the stream afterwards.
extern "C" void flagcxDevNetLaunchConstruct(void *devNets, void *devComm,
                                            int count, void *stream) {
  if (count <= 0 || devNets == nullptr || devComm == nullptr)
    return;
  int threads = (count < 256) ? count : 256;
  int blocks = (count + threads - 1) / threads;
  flagcxDevNetConstructKernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
      (flagcxDevNet *)devNets, (flagcxDevComm *)devComm, count);
}
