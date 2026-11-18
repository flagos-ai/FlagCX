#pragma once
#include "adaptor.h"
#include "flagcx.h"
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

// Every potentially blocking operation has an entry and exit marker.
template <typename F>
flagcxResult_t probeCall(const char *name, F operation) {
  std::fprintf(stderr, "SymMem probe begin: %s\n", name);
  std::fflush(stderr);
  auto result = operation();
  std::fprintf(stderr, "SymMem probe end: %s result=%d\n", name, (int)result);
  std::fflush(stderr);
  return result;
}
#define PROBE(call)                                                            \
  ASSERT_EQ(probeCall(#call, [&] { return (call); }), flagcxSuccess)

class VmmProbe : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_NE(deviceAdaptor, nullptr);
    ASSERT_STREQ(deviceAdaptor->name, "IXCUDA");
    PROBE(deviceAdaptor->setDevice(0));
    ASSERT_NE(deviceAdaptor->symPhysDescribe, nullptr);
    ASSERT_NE(deviceAdaptor->symFlatMapSegmentsBegin, nullptr);
    ASSERT_NE(deviceAdaptor->symLsaMapBeginAt, nullptr);
  }
  void TearDown() override {
    // Preserve backing resources if an earlier teardown phase fails.
    if (flat != nullptr) {
      if (deviceAdaptor->symFlatMappingUnmapSegments(flat) != flagcxSuccess) {
        ADD_FAILURE() << "mapping cleanup failed; backing resources retained";
        return;
      }
      if (deviceAdaptor->symFlatVaFreeSegments(flat) != flagcxSuccess) {
        ADD_FAILURE() << "mapping record/VA cleanup failed";
        return;
      }
    }
    if (arena != nullptr &&
        deviceAdaptor->symLsaVaFree(arena, arenaBytes) != flagcxSuccess) {
      ADD_FAILURE() << "arena cleanup failed";
      return;
    }
    if (bundle != nullptr) {
      EXPECT_EQ(deviceAdaptor->symPhysFreeSegments(bundle), flagcxSuccess);
    }
    if (buffer != nullptr) {
      EXPECT_EQ(deviceAdaptor->gdrMemFree(buffer, nullptr), flagcxSuccess);
    }
  }
  void allocate(size_t bytes = 1024 * 1024, bool describe = true) {
    PROBE(deviceAdaptor->gdrMemAlloc(&buffer, bytes, nullptr));
    ASSERT_NE(buffer, nullptr);
    PROBE(deviceAdaptor->getAddressRange(buffer, &allocationBase,
                                         &allocationBytes));
    ASSERT_EQ(buffer, allocationBase);
    ASSERT_GE(allocationBytes, bytes);
    if (!describe) {
      size_t capacity = 0;
      PROBE(deviceAdaptor->symLsaGetCapacity(&capacity, &alignment));
      PROBE(deviceAdaptor->symPhysSegmentCount(buffer, bytes, &count));
      ASSERT_GT(count, 0);
      return;
    }
    PROBE(deviceAdaptor->symPhysDescribe(buffer, bytes, &bundle, &segments,
                                         &count, &offset, &alignment));
    ASSERT_GT(count, 0);
    ASSERT_GT(alignment, 0u);
  }
  void netReadWrite(bool multi);
  void *buffer = nullptr, *allocationBase = nullptr, *bundle = nullptr;
  void *flat = nullptr, *arena = nullptr;
  const flagcxSymSegmentMeta *segments = nullptr;
  size_t allocationBytes = 0, offset = 0, alignment = 0, arenaBytes = 0;
  int count = 0;
};
