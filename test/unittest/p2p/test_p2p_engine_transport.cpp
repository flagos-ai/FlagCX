/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "barex_runtime.h"
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

TEST(P2pEngineTransportTest, DescriptorPreservesSparsePhysicalNicKeys) {
  FlagcxP2pRdmaDesc desc{};
  const uint32_t keys[] = {0, 0x2200, 0, 0x4400};
  ASSERT_EQ(flagcxP2pDescSetKeys(&desc, keys, 4), flagcxSuccess);

  uint32_t key = 1;
  ASSERT_EQ(flagcxP2pDescGetKey(&desc, 0, &key), flagcxSuccess);
  EXPECT_EQ(key, 0u);
  ASSERT_EQ(flagcxP2pDescGetKey(&desc, 1, &key), flagcxSuccess);
  EXPECT_EQ(key, 0x2200u);
  ASSERT_EQ(flagcxP2pDescGetKey(&desc, 3, &key), flagcxSuccess);
  EXPECT_EQ(key, 0x4400u);
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

TEST(P2pEngineTransportTest, SplitsPairedRangesAtEitherProviderBoundary) {
  flagcxP2pMrRecord local = makeThreeSegmentRecord();
  flagcxP2pMrRecord remote;
  remote.id = 8;
  remote.base = 0x8000;
  remote.size = 0x3000;
  remote.segments.push_back(makeSegment(0x8000, 0x1800, 2));
  remote.segments.push_back(makeSegment(0x9800, 0x1800, 2));
  std::vector<flagcxP2pMrPairSlice> slices;

  ASSERT_EQ(flagcxP2pMrSplitPair(&local, 0x1800, &remote, 0x8800, 0x2000, 0, 0,
                                 &slices),
            flagcxSuccess);
  ASSERT_EQ(slices.size(), 4u);
  EXPECT_EQ(slices[0].localSegmentIndex, 0u);
  EXPECT_EQ(slices[0].remoteSegmentIndex, 0u);
  EXPECT_EQ(slices[0].offset, 0u);
  EXPECT_EQ(slices[0].size, 0x800u);
  EXPECT_EQ(slices[1].localSegmentIndex, 1u);
  EXPECT_EQ(slices[1].remoteSegmentIndex, 0u);
  EXPECT_EQ(slices[1].offset, 0x800u);
  EXPECT_EQ(slices[1].size, 0x800u);
  EXPECT_EQ(slices[2].localSegmentIndex, 1u);
  EXPECT_EQ(slices[2].remoteSegmentIndex, 1u);
  EXPECT_EQ(slices[2].offset, 0x1000u);
  EXPECT_EQ(slices[2].size, 0x800u);
  EXPECT_EQ(slices[3].localSegmentIndex, 2u);
  EXPECT_EQ(slices[3].remoteSegmentIndex, 1u);
  EXPECT_EQ(slices[3].offset, 0x1800u);
  EXPECT_EQ(slices[3].size, 0x800u);
}

TEST(P2pEngineTransportTest, PairSplitAppliesSliceLimitAfterMrBoundaries) {
  flagcxP2pMrRecord local = makeThreeSegmentRecord();
  flagcxP2pMrRecord remote = makeThreeSegmentRecord();
  remote.base = 0x8000;
  for (size_t i = 0; i < remote.segments.size(); ++i)
    remote.segments[i].base = remote.base + i * 0x1000;
  std::vector<flagcxP2pMrPairSlice> slices;

  ASSERT_EQ(flagcxP2pMrSplitPair(&local, 0x1000, &remote, 0x8000, 0x1000, 0x600,
                                 0x100, &slices),
            flagcxSuccess);
  ASSERT_EQ(slices.size(), 3u);
  EXPECT_EQ(slices[0].size, 0x600u);
  EXPECT_EQ(slices[1].offset, 0x600u);
  EXPECT_EQ(slices[1].size, 0x600u);
  EXPECT_EQ(slices[2].offset, 0xc00u);
  EXPECT_EQ(slices[2].size, 0x400u);
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

TEST(P2pEngineTransportTest, OrderingKeysDistributeSingleSliceTransfers) {
  uint64_t laneMask2 = 0;
  uint64_t laneMask4 = 0;
  for (uint64_t transferId = 1; transferId <= 64; ++transferId) {
    const uint64_t key = flagcxP2pEngineOrderingKey(transferId, 0);
    laneMask2 |= 1ULL << (key % 2);
    laneMask4 |= 1ULL << (key % 4);
  }
  EXPECT_EQ(laneMask2, 0x3u);
  EXPECT_EQ(laneMask4, 0xfu);
}

TEST(P2pEngineTransportTest, OrderingKeysDistinguishSlices) {
  const uint64_t transferId = 17;
  EXPECT_NE(flagcxP2pEngineOrderingKey(transferId, 0),
            flagcxP2pEngineOrderingKey(transferId, 1));
}

TEST(P2pEngineTransportTest, BarexHelloGeometryPreservesLegacyEncoding) {
  uint32_t wire = 1;
  ASSERT_EQ(flagcxBarexRuntimeEncodeHelloGeometry(1, 0, &wire), flagcxSuccess);
  EXPECT_EQ(wire, 0u);

  uint32_t channels = 0;
  uint32_t lane = 1;
  ASSERT_EQ(flagcxBarexRuntimeDecodeHelloGeometry(wire, &channels, &lane),
            flagcxSuccess);
  EXPECT_EQ(channels, 1u);
  EXPECT_EQ(lane, 0u);
}

TEST(P2pEngineTransportTest, BarexListenGeometryAcceptsLegacyDeviceOnlyWire) {
  uint32_t device = 0;
  uint32_t channels = 0;
  ASSERT_EQ(flagcxBarexRuntimeDecodeListenGeometry(3, &device, &channels),
            flagcxSuccess);
  EXPECT_EQ(device, 3u);
  EXPECT_EQ(channels, 1u);

  uint32_t wire = 0;
  ASSERT_EQ(flagcxBarexRuntimeEncodeListenGeometry(3, 4, &wire), flagcxSuccess);
  ASSERT_EQ(flagcxBarexRuntimeDecodeListenGeometry(wire, &device, &channels),
            flagcxSuccess);
  EXPECT_EQ(device, 3u);
  EXPECT_EQ(channels, 4u);
}

TEST(P2pEngineTransportTest, BarexHelloGeometryRejectsMismatchedLanes) {
  uint32_t wire = 0;
  ASSERT_EQ(flagcxBarexRuntimeEncodeHelloGeometry(4, 3, &wire), flagcxSuccess);
  uint32_t channels = 0;
  uint32_t lane = 0;
  ASSERT_EQ(flagcxBarexRuntimeDecodeHelloGeometry(wire, &channels, &lane),
            flagcxSuccess);
  EXPECT_EQ(channels, 4u);
  EXPECT_EQ(lane, 3u);

  EXPECT_EQ(flagcxBarexRuntimeEncodeHelloGeometry(4, 4, &wire),
            flagcxInvalidArgument);
  EXPECT_EQ(
      flagcxBarexRuntimeDecodeHelloGeometry((4u << 16) | 4u, &channels, &lane),
      flagcxInvalidArgument);
  EXPECT_EQ(flagcxBarexRuntimeEncodeHelloGeometry(
                FLAGCX_BAREX_RUNTIME_MAX_CHANNELS + 1, 0, &wire),
            flagcxInvalidArgument);
}

TEST(P2pEngineTransportTest, BarexOrderedDomainsUseStablePhysicalLanes) {
  uint64_t laneMask = 0;
  for (uint64_t key = 0; key < 32; ++key) {
    uint32_t first = 0;
    uint32_t second = 0;
    ASSERT_EQ(flagcxBarexRuntimeSelectLane(4, key, &first), flagcxSuccess);
    ASSERT_EQ(flagcxBarexRuntimeSelectLane(4, key, &second), flagcxSuccess);
    EXPECT_EQ(first, second);
    laneMask |= 1ULL << first;
  }
  EXPECT_EQ(laneMask, 0xfu);
}
