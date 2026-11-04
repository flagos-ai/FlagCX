// Compile this file once per FLAGCX_NEGATIVE_* macro. Every case must fail.
#include "device_api/platform_traits.h"
#include <type_traits>

using Traits = PlatformTraits<NvidiaPlatform>;
using Ops = Traits::CoopOps;

struct UnsupportedValue {
  uint32_t word[3];
};

__global__ void negative(uint32_t *ptr) {
  Traits::CoopThread coop;
#if defined(FLAGCX_NEGATIVE_MMIO_SCRATCHPAD)
  Ops::store<Traits::CoopThread, uint32_t *, uint32_t,
             flagcxDeviceMemoryScratchpad, flagcxDeviceMemoryAccessKindMmio,
             flagcxDeviceScopeSystem, flagcxDeviceCacheL1,
             flagcxDeviceCacheHintNone>(coop, ptr, uint32_t{1},
                                        flagcxDeviceMemoryOrderRelaxed);
#elif defined(FLAGCX_NEGATIVE_MMIO_CACHE)
  Ops::store<Traits::CoopThread, uint32_t *, uint32_t, flagcxDeviceMemoryGlobal,
             flagcxDeviceMemoryAccessKindMmio, flagcxDeviceScopeSystem,
             flagcxDeviceCacheL1, flagcxDeviceCacheHintNoAllocate>(
      coop, ptr, uint32_t{1}, flagcxDeviceMemoryOrderRelaxed);
#elif defined(FLAGCX_NEGATIVE_IMMUTABLE_ACQUIRE)
  (void)Ops::load<Traits::CoopThread, uint32_t *,
                  flagcxDeviceLoadIntentImmutable>(
      coop, ptr,
      std::integral_constant<flagcxDeviceMemoryOrder_t,
                             flagcxDeviceMemoryOrderAcquire>{});
#elif defined(FLAGCX_NEGATIVE_UNSUPPORTED_TYPE)
  (void)Ops::load(coop, reinterpret_cast<UnsupportedValue *>(ptr),
                  flagcxDeviceMemoryOrderRelaxed);
#else
#error "select a negative test case"
#endif
}
