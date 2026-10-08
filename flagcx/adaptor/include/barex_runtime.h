/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_BAREX_RUNTIME_H_
#define FLAGCX_BAREX_RUNTIME_H_

#include "flagcx.h"

#include <stddef.h>
#include <stdint.h>

// Internal controls for configuring the in-tree BAREX provider runtime from
// the generic P2P Engine. They do not extend the public net-adaptor or
// plugin-v1 ABI. Normal collective users do not enter this scope and therefore
// keep the historical single channel.
struct flagcxBarexRuntimeConnectionConfig {
  uint32_t channels;
};

// Keep this aligned with the public Engine's maximum QP/channel geometry.
constexpr uint32_t FLAGCX_BAREX_RUNTIME_MAX_CHANNELS = 8;

inline flagcxResult_t
flagcxBarexRuntimeValidateChannelCount(uint32_t channels) {
  return channels >= 1 && channels <= FLAGCX_BAREX_RUNTIME_MAX_CHANNELS
             ? flagcxSuccess
             : flagcxInvalidArgument;
}

// HELLO's former padding word carries the negotiated channel geometry.
// Zero remains the legacy encoding for one channel, lane zero.
inline flagcxResult_t flagcxBarexRuntimeEncodeHelloGeometry(uint32_t channels,
                                                            uint32_t lane,
                                                            uint32_t *wire) {
  if (wire == nullptr ||
      flagcxBarexRuntimeValidateChannelCount(channels) != flagcxSuccess ||
      lane >= channels || channels > UINT16_MAX || lane > UINT16_MAX)
    return flagcxInvalidArgument;
  *wire = channels == 1 && lane == 0 ? 0 : (lane << 16) | channels;
  return flagcxSuccess;
}

inline flagcxResult_t flagcxBarexRuntimeDecodeHelloGeometry(uint32_t wire,
                                                            uint32_t *channels,
                                                            uint32_t *lane) {
  if (channels == nullptr || lane == nullptr)
    return flagcxInvalidArgument;
  if (wire == 0) {
    *channels = 1;
    *lane = 0;
    return flagcxSuccess;
  }
  *channels = wire & UINT16_MAX;
  *lane = wire >> 16;
  if (flagcxBarexRuntimeValidateChannelCount(*channels) != flagcxSuccess ||
      *lane >= *channels)
    return flagcxInvalidArgument;
  return flagcxSuccess;
}

inline flagcxResult_t flagcxBarexRuntimeSelectLane(uint32_t channels,
                                                   uint64_t orderingKey,
                                                   uint32_t *lane) {
  if (lane == nullptr ||
      flagcxBarexRuntimeValidateChannelCount(channels) != flagcxSuccess)
    return flagcxInvalidArgument;
  *lane = static_cast<uint32_t>(orderingKey % channels);
  return flagcxSuccess;
}

inline flagcxResult_t flagcxBarexRuntimeEncodeListenGeometry(uint32_t device,
                                                             uint32_t channels,
                                                             uint32_t *wire) {
  if (wire == nullptr || device > UINT16_MAX ||
      flagcxBarexRuntimeValidateChannelCount(channels) != flagcxSuccess)
    return flagcxInvalidArgument;
  *wire = (channels << 16) | device;
  return flagcxSuccess;
}

inline flagcxResult_t
flagcxBarexRuntimeDecodeListenGeometry(uint32_t wire, uint32_t *device,
                                       uint32_t *channels) {
  if (device == nullptr || channels == nullptr)
    return flagcxInvalidArgument;
  *device = wire & UINT16_MAX;
  const uint32_t encodedChannels = wire >> 16;
  // Before the geometry field existed this word contained only the device.
  *channels = encodedChannels == 0 ? 1 : encodedChannels;
  return flagcxBarexRuntimeValidateChannelCount(*channels);
}

flagcxResult_t flagcxBarexRuntimeSetConnectionConfig(
    const struct flagcxBarexRuntimeConnectionConfig *config);
void flagcxBarexRuntimeClearConnectionConfig(void);
flagcxResult_t flagcxBarexRuntimeResetConnect(void *opaqueHandle);
// Transfer a failed deregistration handle from the generic Engine to BAREX.
// Direct adaptor callers keep ownership until they explicitly call this.
flagcxResult_t flagcxBarexRuntimeDeferMr(void *mhandle);
// Retry only handles whose ownership was explicitly transferred above.
flagcxResult_t flagcxBarexRuntimeDrainDeferredMrs(void);
size_t flagcxBarexRuntimeMrSegmentSize(int type);
flagcxResult_t flagcxBarexRuntimeGetCommChannels(void *comm,
                                                 uint32_t *channels);

#endif // FLAGCX_BAREX_RUNTIME_H_
