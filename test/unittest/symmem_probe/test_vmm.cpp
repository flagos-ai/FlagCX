#include "probe.hpp"
#include <dirent.h>
#include <unistd.h>

static size_t fdCount() {
  DIR *dir = opendir("/proc/self/fd");
  if (!dir)
    return 0;
  size_t count = 0;
  while (readdir(dir))
    ++count;
  closedir(dir);
  return count;
}

TEST_F(VmmProbe, AllocationAndRange) {
  allocate(1024 * 1024, false);
  ASSERT_FALSE(HasFatalFailure());
  void *base = nullptr;
  size_t bytes = 0;
  PROBE(deviceAdaptor->getAddressRange(static_cast<char *>(buffer) + 127, &base,
                                       &bytes));
  EXPECT_EQ(base, buffer);
  EXPECT_EQ(bytes, allocationBytes);
  std::vector<unsigned char> pattern(4096, 0xA7), actual(4096);
  PROBE(deviceAdaptor->deviceMemcpy(buffer, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr,
                                    nullptr));
  PROBE(deviceAdaptor->deviceMemcpy(actual.data(), buffer, actual.size(),
                                    flagcxMemcpyDeviceToHost, nullptr,
                                    nullptr));
  EXPECT_EQ(actual, pattern);
  int segmentsSeen = 0;
  PROBE(deviceAdaptor->symPhysSegmentCount(buffer, pattern.size(),
                                           &segmentsSeen));
  EXPECT_GE(segmentsSeen, 1);
}

TEST_F(VmmProbe, MultiSegmentSubrange) {
  allocate(4096, false);
  ASSERT_FALSE(HasFatalFailure());
  const size_t granularity = alignment;
  PROBE(deviceAdaptor->gdrMemFree(buffer, nullptr));
  buffer = nullptr;
  // Read per allocation; this does not alter the cached VMM_ENABLE switch.
  std::string value = std::to_string(granularity);
  ASSERT_EQ(setenv("FLAGCX_VMM_SEGMENT_BYTES", value.c_str(), 1), 0);
  allocate(3 * granularity, false);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(count, 3);
  const size_t start = granularity - 17;
  std::vector<unsigned char> pattern(64, 0x5C), actual(64);
  void *interior = static_cast<char *>(buffer) + start;
  int touched = 0;
  PROBE(deviceAdaptor->symPhysSegmentCount(interior, pattern.size(), &touched));
  EXPECT_EQ(touched, 2);
  PROBE(deviceAdaptor->deviceMemcpy(interior, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr,
                                    nullptr));
  PROBE(deviceAdaptor->deviceMemcpy(actual.data(), interior, actual.size(),
                                    flagcxMemcpyDeviceToHost, nullptr,
                                    nullptr));
  EXPECT_EQ(actual, pattern);
}

TEST_F(VmmProbe, AliasAndRetryableTeardown) {
  allocate();
  ASSERT_FALSE(HasFatalFailure());
  flagcxSymRankLayout layout{allocationBytes, offset, 1024 * 1024, 0, count};
  PROBE(deviceAdaptor->symFlatMapSegmentsBegin(1, 0, bundle, &layout, segments,
                                               count, allocationBytes,
                                               alignment, &flat));
  ASSERT_NE(flat, nullptr);
  // Map one segment, then reject the duplicate before completing the rest.
  PROBE(deviceAdaptor->symFlatMapSegmentOne(flat, 0, -1));
  EXPECT_EQ(deviceAdaptor->symFlatMapSegmentOne(flat, 0, -1),
            flagcxInvalidUsage);
  for (int i = 1; i < count; ++i) {
    PROBE(deviceAdaptor->symFlatMapSegmentOne(flat, i, -1));
  }
  std::vector<unsigned char> pattern(4096, 0x3D), actual(4096);
  PROBE(deviceAdaptor->deviceMemcpy(flat, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr,
                                    nullptr));
  PROBE(deviceAdaptor->deviceMemcpy(actual.data(), buffer, actual.size(),
                                    flagcxMemcpyDeviceToHost, nullptr,
                                    nullptr));
  EXPECT_EQ(actual, pattern);
  EXPECT_EQ(deviceAdaptor->symFlatVaFreeSegments(flat), flagcxInvalidUsage);
  PROBE(deviceAdaptor->symFlatMappingUnmapSegments(flat));
  PROBE(deviceAdaptor->symFlatMappingUnmapSegments(flat));
  PROBE(deviceAdaptor->symFlatVaFreeSegments(flat));
  flat = nullptr;
}

TEST_F(VmmProbe, LsaArenaOffsetsAndFdLifetime) {
  allocate();
  ASSERT_FALSE(HasFatalFailure());
  size_t capacity = 0, vaAlignment = 0;
  PROBE(deviceAdaptor->symLsaGetCapacity(&capacity, &vaAlignment));
  ASSERT_GT(capacity, 0u);
  ASSERT_EQ(vaAlignment, alignment);
  arenaBytes = 2 * allocationBytes;
  PROBE(deviceAdaptor->symLsaVaReserve(arenaBytes, alignment, &arena));
  const size_t before = fdCount();
  for (int repeat = 0; repeat < 4; ++repeat) {
    int fd = -1;
    PROBE(deviceAdaptor->symPhysExportSegment(bundle, 0, &fd));
    ASSERT_GE(fd, 0);
    EXPECT_EQ(close(fd), 0);
    flagcxSymRankLayout layout{allocationBytes, offset, 1024 * 1024, 0, count};
    void *slot = static_cast<char *>(arena) + (repeat % 2) * allocationBytes;
    PROBE(deviceAdaptor->symLsaMapBeginAt(slot, 1, 0, bundle, &layout, segments,
                                          count, allocationBytes, alignment,
                                          &flat));
    for (int i = 0; i < count; ++i) {
      PROBE(deviceAdaptor->symFlatMapSegmentOne(flat, i, -1));
    }
    PROBE(deviceAdaptor->symFlatMappingUnmapSegments(flat));
    PROBE(deviceAdaptor->symFlatVaFreeSegments(flat));
    flat = nullptr;
  }
  EXPECT_EQ(fdCount(), before);
}
