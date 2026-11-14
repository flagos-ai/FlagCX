/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device-compiled helpers for MUSA Device API host integration.
 ************************************************************************/

#include "device_api/flagcx_device_internal.h"
#include "device_api/flagcx_device_core.h"
#include "debug.h"
#include <new>

extern "C" size_t flagcxDevNetSizeOf() { return sizeof(flagcxDevNet); }

static __global__ __launch_bounds__(256) void flagcxDevNetConstructKernel(flagcxDevNet *nets,
                                                   flagcxDevComm *comm,
                                                   int count) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < count)
    ::new (&nets[i]) flagcxDevNet(*comm, i);
}

extern "C" flagcxResult_t flagcxMusaDevNetLaunchConstructChecked(
    void *devNets, void *devComm, int count, void *stream) {
  if (count <= 0 || devNets == nullptr || devComm == nullptr)
    return flagcxInvalidArgument;
  musaError_t error = musaGetLastError();
  if (error != musaSuccess) {
    WARN("DevNet construct: prior MUSA error %d (%s)", (int)error,
         musaGetErrorString(error));
    return flagcxUnhandledDeviceError;
  }
  musaFuncAttributes attributes = {};
  error = musaFuncGetAttributes(
      &attributes, reinterpret_cast<const void *>(flagcxDevNetConstructKernel));
  if (error != musaSuccess) {
    WARN("DevNet construct: attribute query failed %d (%s)", (int)error,
         musaGetErrorString(error));
    return flagcxUnhandledDeviceError;
  }
  int threads = count < 256 ? count : 256;
  if (attributes.maxThreadsPerBlock < threads)
    return flagcxInvalidUsage;
  int blocks = 1 + (count - 1) / threads;
  flagcxDevNetConstructKernel<<<blocks, threads, 0, (musaStream_t)stream>>>(
      static_cast<flagcxDevNet *>(devNets),
      static_cast<flagcxDevComm *>(devComm), count);
  error = musaGetLastError();
  if (error != musaSuccess) {
    WARN("DevNet construct: launch failed %d (%s)", (int)error,
         musaGetErrorString(error));
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

// Retain the existing helper symbol for callers using the legacy interface.
extern "C" void flagcxDevNetLaunchConstruct(void *devNets, void *devComm,
                                            int count, void *stream) {
  (void)flagcxMusaDevNetLaunchConstructChecked(devNets, devComm, count, stream);
}
