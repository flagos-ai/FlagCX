// Reuse the platform-neutral CUDA-compatible native Device API tests. The
// build supplies CoreX's explicit device-pass marker to both compiler passes.
#include "../nvidia/device_api.cu"

// Compile every CoreX-supported RMW operation. Aligned uint64_t load/store is
// covered by the public tests; RMW intentionally uses the 32-bit domain.
__global__ void flagcxIluvatarAtomicContractKernel(uint32_t *value) {
  if (FLAGCX_THREAD_IDX_X != 0)
    return;
  uint32_t expected =
      DeviceAPI::Atomic::load(value, flagcxDeviceMemoryOrderAcquire);
  DeviceAPI::Atomic::store(value, expected, flagcxDeviceMemoryOrderRelease);
  DeviceAPI::Atomic::fetchAdd(value, uint32_t{1},
                              flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::fetchSub(value, uint32_t{1},
                              flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::fetchOr(value, uint32_t{1}, flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::fetchAnd(value, ~uint32_t{0},
                              flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::exchange(value, expected, flagcxDeviceMemoryOrderAcqRel);
  DeviceAPI::Atomic::compareExchange(value, expected, expected,
                                     flagcxDeviceMemoryOrderAcqRel);
}

// This kernel is intentionally not part of the normal success path. A CoreX
// runtime test launches it separately and expects a kernel error, proving that
// an unsupported partial mask fails rather than widening to a full-wave
// barrier and hanging.
__global__ void flagcxIluvatarUnsupportedCoopKernel() {
  flagcxCoopLanes partial(flagcxLaneMask_t{3});
  partial.sync();
}
