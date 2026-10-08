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

flagcxResult_t flagcxP2pMrSplitPair(
    const struct flagcxP2pMrRecord *local, uintptr_t localAddress,
    const struct flagcxP2pMrRecord *remote, uintptr_t remoteAddress,
    size_t size, size_t sliceSize, size_t fragmentSize,
    std::vector<struct flagcxP2pMrPairSlice> *slices) {
  if (slices == nullptr)
    return flagcxInvalidArgument;
  slices->clear();
  if (flagcxP2pMrRecordValidate(local) != flagcxSuccess ||
      flagcxP2pMrRecordValidate(remote) != flagcxSuccess ||
      !rangeFits(local->base, local->size, localAddress, size) ||
      !rangeFits(remote->base, remote->size, remoteAddress, size))
    return flagcxInvalidArgument;
  if (size == 0)
    return flagcxSuccess;

  size_t localIndex = 0;
  size_t remoteIndex = 0;
  size_t offset = 0;
  while (offset < size) {
    const uintptr_t localCursor = localAddress + offset;
    const uintptr_t remoteCursor = remoteAddress + offset;
    while (localIndex < local->segments.size() &&
           localCursor >= local->segments[localIndex].base +
                              local->segments[localIndex].size)
      ++localIndex;
    while (remoteIndex < remote->segments.size() &&
           remoteCursor >= remote->segments[remoteIndex].base +
                               remote->segments[remoteIndex].size)
      ++remoteIndex;
    if (localIndex >= local->segments.size() ||
        remoteIndex >= remote->segments.size())
      return flagcxInternalError;

    const flagcxP2pMrSegment &localSegment = local->segments[localIndex];
    const flagcxP2pMrSegment &remoteSegment = remote->segments[remoteIndex];
    if (localCursor < localSegment.base || remoteCursor < remoteSegment.base)
      return flagcxInternalError;
    size_t bytes = std::min(
        size - offset,
        std::min(static_cast<size_t>(localSegment.base + localSegment.size -
                                     localCursor),
                 static_cast<size_t>(remoteSegment.base + remoteSegment.size -
                                     remoteCursor)));
    if (sliceSize > 0 && bytes > sliceSize && bytes - sliceSize > fragmentSize)
      bytes = sliceSize;
    if (bytes == 0)
      return flagcxInternalError;
    slices->push_back({localIndex, remoteIndex, offset, bytes});
    offset += bytes;
  }
  return flagcxSuccess;
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

uint64_t flagcxP2pEngineOrderingKey(uint64_t transferId, uint64_t sliceIndex) {
  // SplitMix64 gives sequential transfer IDs and slice indices well-distributed
  // low bits. This matters because ordered-lane selection uses key % laneCount.
  uint64_t value = transferId + 0x9e3779b97f4a7c15ULL * (sliceIndex + 1);
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
