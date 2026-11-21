/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device-compiled helpers for device API host integration.
 * Compiled by hgcc, linked into libflagcx.so.
 * Called from default_dev_api_backend.cc via extern "C". PPU selects its own
 * DefaultBackend traits while using the CUDA-compatible launch syntax.
 *
 * Other vendors provide equivalent implementations in their own
 * device-compiled files (.xpu, .hip, etc.).
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include <new>

extern "C" size_t flagcxDevNetSizeOf() { return sizeof(flagcxDevNet); }

static __global__ void flagcxDevNetConstructKernel(flagcxDevNet *nets,
                                                   flagcxDevComm *comm,
                                                   int count) {
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
  flagcxDevNetConstructKernel<<<blocks, threads, 0, (hggcStream_t)stream>>>(
      (flagcxDevNet *)devNets, (flagcxDevComm *)devComm, count);
}
