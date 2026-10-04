/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "gdr_visibility_test.h"

#if defined(USE_NVIDIA_ADAPTOR)
#include "nvidia_adaptor.h"
#elif defined(USE_METAX_ADAPTOR)
#include "metax_adaptor.h"
#elif defined(USE_DU_ADAPTOR)
#include "du_adaptor.h"
#elif defined(USE_PPU_ADAPTOR)
#include "ppu_adaptor.h"
#else
#error "GDR visibility consumer requires a supported device adaptor"
#endif

namespace {

constexpr int kVisibilityConsumerThreads = 256;

__global__ void flagcxTestGdrVisibilityConsumerKernel(const uint8_t *data,
                                                      size_t size,
                                                      uint8_t expected,
                                                      int *status) {
  bool mismatch = false;
  for (size_t offset = threadIdx.x; offset < size; offset += blockDim.x) {
    if (data[offset] != expected) {
      mismatch = true;
      break;
    }
  }
  if (mismatch)
    atomicExch(status, 1);
  atomicAdd(status + 1, 1);
}

} // namespace

extern "C" flagcxResult_t
flagcxTestLaunchGdrVisibilityConsumer(const void *data, size_t size,
                                      uint8_t expected, int *status,
                                      flagcxStream_t stream) {
  if (data == nullptr || size == 0 || status == nullptr || stream == nullptr)
    return flagcxInvalidArgument;
  flagcxTestGdrVisibilityConsumerKernel<<<1, kVisibilityConsumerThreads, 0,
                                          stream->base>>>(
      static_cast<const uint8_t *>(data), size, expected, status);
  // Launch/runtime errors are reported by the following stream synchronize.
  return flagcxSuccess;
}
