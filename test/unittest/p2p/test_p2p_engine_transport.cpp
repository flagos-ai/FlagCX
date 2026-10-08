/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "p2p_engine_transport.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

flagcxP2pMrSegment makeSegment(uintptr_t base, size_t size,
                               uint32_t nKeys = 1) {
  flagcxP2pMrSegment segment;
  segment.base = base;
  segment.size = size;
  segment.adaptorMr = reinterpret_cast<void *>(base + 1);
  segment.keys.nKeys = nKeys;
  for (uint32_t i = 0; i < nKeys; ++i) {
    segment.keys.lkeys[i] = 100 + i;
    segment.keys.rkeys[i] = 200 + i;
  }
  return segment;
}

flagcxP2pMrRecord makeThreeSegmentRecord() {
  flagcxP2pMrRecord record;
  record.id = 7;
  record.base = 0x1000;
  record.size = 0x3000;
  record.segments.push_back(makeSegment(0x1000, 0x1000, 2));
  record.segments.push_back(makeSegment(0x2000, 0x1000, 2));
  record.segments.push_back(makeSegment(0x3000, 0x1000, 2));
  return record;
}

} // namespace

TEST(P2pEngineTransportTest, DescriptorEncodesAllPerNicKeys) {
  FlagcxP2pRdmaDesc desc{};
  desc.rid = 19;
  desc.idx = 23;
  uint32_t keys[FLAGCX_NET_MAX_MR_KEYS];
  for (uint32_t i = 0; i < FLAGCX_NET_MAX_MR_KEYS; ++i)
    keys[i] = 0x1000 + i;

  ASSERT_EQ(flagcxP2pDescSetKeys(&desc, keys, FLAGCX_NET_MAX_MR_KEYS),
            flagcxSuccess);
  EXPECT_EQ(desc.nmsgs, FLAGCX_NET_MAX_MR_KEYS);
  EXPECT_EQ(desc.rid, 19u);
  EXPECT_EQ(desc.idx, 23u);
  for (uint32_t i = 0; i < FLAGCX_NET_MAX_MR_KEYS; ++i) {
    uint32_t key = 0;
    ASSERT_EQ(flagcxP2pDescGetKey(&desc, i, &key), flagcxSuccess);
    EXPECT_EQ(key, keys[i]);
  }
}

TEST(P2pEngineTransportTest, DescriptorRejectsMissingMultiNicKey) {
  FlagcxP2pRdmaDesc desc{};
  const uint32_t keys[] = {11, 22};
  ASSERT_EQ(flagcxP2pDescSetKeys(&desc, keys, 2), flagcxSuccess);

  uint32_t key = 0;
  EXPECT_EQ(flagcxP2pDescGetKey(&desc, 2, &key), flagcxNotSupported);
  EXPECT_EQ(
      flagcxP2pDescGetKey(&desc, std::numeric_limits<uint32_t>::max(), &key),
      flagcxNotSupported);
}

TEST(P2pEngineTransportTest, LegacyDescriptorUsesConnectionSelectedKey) {
  FlagcxP2pRdmaDesc desc{};
  desc.rkey = 99;
  desc.nmsgs = 0;

  uint32_t key = 0;
  ASSERT_EQ(flagcxP2pDescGetKey(&desc, 6, &key), flagcxSuccess);
  EXPECT_EQ(key, 99u);
  ASSERT_EQ(
      flagcxP2pDescGetKey(&desc, std::numeric_limits<uint32_t>::max(), &key),
      flagcxSuccess);
  EXPECT_EQ(key, 99u);
}

TEST(P2pEngineTransportTest, SplitsRangeAtProviderMrBoundaries) {
  flagcxP2pMrRecord record = makeThreeSegmentRecord();
  std::vector<flagcxP2pMrSlice> slices;

  ASSERT_EQ(flagcxP2pMrSplitRange(&record, 0x1800, 0x2000, &slices),
            flagcxSuccess);
  ASSERT_EQ(slices.size(), 3u);
  EXPECT_EQ(slices[0].segmentIndex, 0u);
  EXPECT_EQ(slices[0].segmentOffset, 0x800u);
  EXPECT_EQ(slices[0].size, 0x800u);
  EXPECT_EQ(slices[1].segmentIndex, 1u);
  EXPECT_EQ(slices[1].segmentOffset, 0u);
  EXPECT_EQ(slices[1].size, 0x1000u);
  EXPECT_EQ(slices[2].segmentIndex, 2u);
  EXPECT_EQ(slices[2].segmentOffset, 0u);
  EXPECT_EQ(slices[2].size, 0x800u);
}

TEST(P2pEngineTransportTest, ZeroLengthAtRegistrationEndIsValid) {
  flagcxP2pMrRecord record = makeThreeSegmentRecord();
  std::vector<flagcxP2pMrSlice> slices = {{9, 9, 9}};

  ASSERT_EQ(flagcxP2pMrSplitRange(&record, 0x4000, 0, &slices), flagcxSuccess);
  EXPECT_TRUE(slices.empty());
}

TEST(P2pEngineTransportTest, RejectsGapsAndOverflow) {
  flagcxP2pMrRecord record = makeThreeSegmentRecord();
  record.segments[1].base++;
  EXPECT_EQ(flagcxP2pMrRecordValidate(&record), flagcxInvalidArgument);

  record = makeThreeSegmentRecord();
  record.base = std::numeric_limits<uintptr_t>::max() - 8;
  record.size = 16;
  EXPECT_EQ(flagcxP2pMrRecordValidate(&record), flagcxInvalidArgument);
}

TEST(P2pEngineTransportTest, RegionViewResolvesBoundsAndPerNicKeys) {
  flagcxP2pMrSegment segment = makeSegment(0x8000, 0x1000, 2);
  flagcxTransportRegion region{};
  ASSERT_EQ(flagcxP2pMrSegmentRegion(&segment, &region), flagcxSuccess);

  uintptr_t address = 0;
  uint32_t key = 0;
  ASSERT_EQ(
      flagcxTransportResolveRegion(&region, 0x120, 0x200, 1, 1, &address, &key),
      flagcxSuccess);
  EXPECT_EQ(address, 0x8120u);
  EXPECT_EQ(key, 101u);

  ASSERT_EQ(
      flagcxTransportResolveRegion(&region, 0x120, 0x200, 1, 0, &address, &key),
      flagcxSuccess);
  EXPECT_EQ(key, 201u);
  EXPECT_EQ(
      flagcxTransportResolveRegion(&region, 0xf00, 0x200, 0, 0, &address, &key),
      flagcxInvalidArgument);
  EXPECT_EQ(flagcxTransportResolveRegion(&region, 0, 1, 2, 0, &address, &key),
            flagcxNotSupported);
}

TEST(P2pEngineTransportTest, SingleKeyRegionIgnoresPhysicalLaneIndex) {
  flagcxP2pMrSegment segment = makeSegment(0x9000, 0x1000);
  flagcxTransportRegion region{};
  ASSERT_EQ(flagcxP2pMrSegmentRegion(&segment, &region), flagcxSuccess);

  uintptr_t address = 0;
  uint32_t key = 0;
  ASSERT_EQ(flagcxTransportResolveRegion(&region, 0, 1, 7, 0, &address, &key),
            flagcxSuccess);
  EXPECT_EQ(key, 200u);
}
