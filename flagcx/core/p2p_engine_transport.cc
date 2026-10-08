/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "p2p_engine_transport.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

bool rangeFits(uintptr_t base, size_t capacity, uintptr_t address,
               size_t size) {
  if (address < base)
    return false;
  const uintptr_t offset = address - base;
  return offset <= capacity && size <= capacity - offset;
}

} // namespace

flagcxResult_t flagcxP2pDescSetKeys(struct FlagcxP2pRdmaDesc *desc,
                                    const uint32_t *rkeys, uint32_t nKeys) {
  if (desc == nullptr || rkeys == nullptr || nKeys == 0 ||
      nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return flagcxInvalidArgument;

  desc->rkey = rkeys[0];
  desc->nmsgs = nKeys;
  std::memset(desc->padding, 0, sizeof(desc->padding));
  for (uint32_t i = 1; i < nKeys; ++i) {
    std::memcpy(desc->padding + (i - 1) * sizeof(uint32_t), rkeys + i,
                sizeof(uint32_t));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pDescGetKey(const struct FlagcxP2pRdmaDesc *desc,
                                   uint32_t keyIndex, uint32_t *rkey) {
  if (desc == nullptr || rkey == nullptr)
    return flagcxInvalidArgument;

  // nmsgs == 0 is the legacy IBRC descriptor encoding. A single key belongs
  // to the connection-selected device, irrespective of its physical index.
  const uint32_t nKeys = desc->nmsgs == 0 ? 1 : desc->nmsgs;
  if (nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return flagcxInvalidArgument;
  if (nKeys == 1) {
    *rkey = desc->rkey;
    return flagcxSuccess;
  }
  if (keyIndex >= nKeys)
    return flagcxNotSupported;
  if (keyIndex == 0) {
    *rkey = desc->rkey;
    return flagcxSuccess;
  }
  std::memcpy(rkey, desc->padding + (keyIndex - 1) * sizeof(uint32_t),
              sizeof(uint32_t));
  return flagcxSuccess;
}

flagcxResult_t
flagcxP2pMrRecordValidate(const struct flagcxP2pMrRecord *record) {
  if (record == nullptr || record->base == 0 || record->size == 0 ||
      record->segments.empty())
    return flagcxInvalidArgument;
  if (record->base > std::numeric_limits<uintptr_t>::max() - record->size)
    return flagcxInvalidArgument;

  uintptr_t next = record->base;
  for (const flagcxP2pMrSegment &segment : record->segments) {
    if (segment.base != next || segment.size == 0 || segment.keys.nKeys == 0 ||
        segment.keys.nKeys > FLAGCX_NET_MAX_MR_KEYS)
      return flagcxInvalidArgument;
    if (segment.base > std::numeric_limits<uintptr_t>::max() - segment.size)
      return flagcxInvalidArgument;
    next = segment.base + segment.size;
  }
  return next == record->base + record->size ? flagcxSuccess
                                             : flagcxInvalidArgument;
}

flagcxResult_t
flagcxP2pMrSplitRange(const struct flagcxP2pMrRecord *record, uintptr_t address,
                      size_t size,
                      std::vector<struct flagcxP2pMrSlice> *slices) {
  if (slices == nullptr)
    return flagcxInvalidArgument;
  slices->clear();
  const flagcxResult_t validateResult = flagcxP2pMrRecordValidate(record);
  if (validateResult != flagcxSuccess)
    return validateResult;
  if (!rangeFits(record->base, record->size, address, size))
    return flagcxInvalidArgument;
  if (size == 0)
    return flagcxSuccess;

  uintptr_t cursor = address;
  size_t remaining = size;
  for (size_t i = 0; i < record->segments.size() && remaining > 0; ++i) {
    const flagcxP2pMrSegment &segment = record->segments[i];
    const uintptr_t segmentEnd = segment.base + segment.size;
    if (cursor >= segmentEnd)
      continue;
    if (cursor < segment.base)
      return flagcxInternalError;

    const size_t available = segmentEnd - cursor;
    const size_t sliceSize = std::min(remaining, available);
    flagcxP2pMrSlice slice;
    slice.segmentIndex = i;
    slice.segmentOffset = static_cast<uint64_t>(cursor - segment.base);
    slice.size = sliceSize;
    slices->push_back(slice);
    cursor += sliceSize;
    remaining -= sliceSize;
  }
  return remaining == 0 ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t
flagcxP2pMrSegmentRegion(const struct flagcxP2pMrSegment *segment,
                         flagcxTransportRegion *region) {
  if (segment == nullptr || region == nullptr || segment->base == 0 ||
      segment->size == 0 || segment->keys.nKeys == 0 ||
      segment->keys.nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return flagcxInvalidArgument;
  region->base = segment->base;
  region->size = segment->size;
  region->mrInfo = &segment->keys;
  region->localMrHandle = segment->adaptorMr;
  return flagcxSuccess;
}
