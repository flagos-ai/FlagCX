#include "stream_consumer.h"

#include "kernel_operator.h"

#include <stdint.h>

constexpr uint32_t kPayloadBytes = 256;

extern "C" __global__ __vector__ void
StreamVisibilityProducer(__gm__ uint8_t *payload, uint8_t value) {
  if (AscendC::GetBlockIdx() != 0)
    return;
  for (uint32_t i = 0; i < kPayloadBytes; ++i)
    payload[i] = value;
#if __NPU_ARCH__ == 3510
  // Scalar GM stores on A5 can remain in the data cache at kernel exit.
  dcci(reinterpret_cast<__gm__ int64_t *>(payload),
       cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
#endif
}

extern "C" __global__ __vector__ void
StreamVisibilitySignal(__gm__ uint64_t *signal, uint64_t value) {
  if (AscendC::GetBlockIdx() != 0)
    return;
  *signal = value;
#if __NPU_ARCH__ == 3510
  dcci(reinterpret_cast<__gm__ int64_t *>(signal),
       cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
#endif
}

extern "C" __global__ __vector__ void
StreamVisibilitySignalReader(__gm__ const uint64_t *signal,
                             __gm__ uint32_t *status) {
  if (AscendC::GetBlockIdx() != 0)
    return;
  const uint64_t value = *signal;
  status[2] = static_cast<uint32_t>(value);
  status[3] = static_cast<uint32_t>(value >> 32);
#if __NPU_ARCH__ == 3510
  dcci(reinterpret_cast<__gm__ int64_t *>(status),
       cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
#endif
}

extern "C" __global__ __vector__ void
StreamVisibilityConsumer(__gm__ const uint8_t *payload, __gm__ uint32_t *status,
                         uint8_t expected) {
  if (AscendC::GetBlockIdx() != 0)
    return;
  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < kPayloadBytes; ++i)
    mismatches += payload[i] != expected;
  status[0] = mismatches;
  status[1] = kPayloadBytes;
#if __NPU_ARCH__ == 3510
  dcci(reinterpret_cast<__gm__ int64_t *>(status),
       cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
#endif
}

void launchStreamConsumer(void *payload, uint32_t *status, uint8_t expected,
                          void *stream) {
  StreamVisibilityConsumer<<<1, nullptr, stream>>>(
      static_cast<uint8_t *>(payload), status, expected);
}

void launchStreamProducer(void *payload, uint8_t value, void *stream) {
  StreamVisibilityProducer<<<1, nullptr, stream>>>(
      static_cast<uint8_t *>(payload), value);
}

void launchStreamSignal(void *signal, uint64_t value, void *stream) {
  StreamVisibilitySignal<<<1, nullptr, stream>>>(
      static_cast<uint64_t *>(signal), value);
}

void launchStreamSignalReader(void *signal, uint32_t *status, void *stream) {
  StreamVisibilitySignalReader<<<1, nullptr, stream>>>(
      static_cast<uint64_t *>(signal), status);
}
