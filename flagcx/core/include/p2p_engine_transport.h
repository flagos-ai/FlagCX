/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_P2P_ENGINE_TRANSPORT_H_
#define FLAGCX_P2P_ENGINE_TRANSPORT_H_

#include "flagcx_net_adaptor.h"
#include "flagcx_p2p.h"
#include "net_transport.h"

#include <stddef.h>
#include <stdint.h>
#include <vector>

// A P2P registration may contain multiple provider MRs. BAREX uses this to
// keep GPU registrations below the provider limit; IBRC normally publishes a
// single segment. Every segment owns one opaque adaptor MR and its exported
// per-NIC keys.
struct flagcxP2pMrSegment {
  uintptr_t base = 0;
  size_t size = 0;
  void *adaptorMr = nullptr;
  struct flagcxNetMrInfo keys = {};
};

struct flagcxP2pMrRecord {
  FlagcxP2pMr id = 0;
  uintptr_t base = 0;
  size_t size = 0;
  std::vector<struct flagcxP2pMrSegment> segments;
};

struct flagcxP2pMrSlice {
  size_t segmentIndex = 0;
  uint64_t segmentOffset = 0;
  size_t size = 0;
};

// Encode/decode up to FLAGCX_NET_MAX_MR_KEYS rkeys in the existing 64-byte
// public descriptor. Key zero remains in the legacy rkey field; additional
// keys use the reserved padding. Other reserved fields are left untouched.
flagcxResult_t flagcxP2pDescSetKeys(struct FlagcxP2pRdmaDesc *desc,
                                    const uint32_t *rkeys, uint32_t nKeys);
flagcxResult_t flagcxP2pDescGetKey(const struct FlagcxP2pRdmaDesc *desc,
                                   uint32_t keyIndex, uint32_t *rkey);

flagcxResult_t
flagcxP2pMrRecordValidate(const struct flagcxP2pMrRecord *record);

// Split [address, address + size) at provider-MR boundaries. A zero-byte
// range is valid anywhere within the aggregate registration and yields no
// slices.
flagcxResult_t
flagcxP2pMrSplitRange(const struct flagcxP2pMrRecord *record, uintptr_t address,
                      size_t size,
                      std::vector<struct flagcxP2pMrSlice> *slices);

flagcxResult_t
flagcxP2pMrSegmentRegion(const struct flagcxP2pMrSegment *segment,
                         flagcxTransportRegion *region);

#endif // FLAGCX_P2P_ENGINE_TRANSPORT_H_
