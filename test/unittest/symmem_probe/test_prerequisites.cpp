#include <gtest/gtest.h>

#include "probe.hpp"

#ifndef FLAGCX_ILUVATAR_HAS_VMM
#define FLAGCX_ILUVATAR_HAS_VMM 0
#endif

// Run once before the component/MPI probes. This checks the built provider,
// without calling the GPU driver or mistaking header support for hardware.
TEST(SymmemProbePrerequisites, VmmCallbacks) {
  std::fprintf(stderr, "SymMem prerequisite: compiled VMM support = %d\n",
               FLAGCX_ILUVATAR_HAS_VMM);
  EXPECT_EQ(FLAGCX_ILUVATAR_HAS_VMM, 1)
      << "See iluvatar-vmm-sdk logs for compiler errors and SDK prerequisites";
  ASSERT_NE(deviceAdaptor, nullptr);
  EXPECT_STREQ(deviceAdaptor->name, "IXCUDA");
#define CHECK_CALLBACK(name)                                                   \
  do {                                                                         \
    std::fprintf(stderr, "SymMem prerequisite: %s = %s\n", #name,              \
                 deviceAdaptor->name ? "available" : "unavailable");           \
    EXPECT_NE(deviceAdaptor->name, nullptr) << #name;                          \
  } while (0)
  CHECK_CALLBACK(gdrMemAlloc);
  CHECK_CALLBACK(gdrMemFree);
  CHECK_CALLBACK(getAddressRange);
  CHECK_CALLBACK(symPhysDescribe);
  CHECK_CALLBACK(symPhysExportSegment);
  CHECK_CALLBACK(symPhysFreeSegments);
  CHECK_CALLBACK(symPhysSegmentCount);
  CHECK_CALLBACK(symFlatMapSegmentsBegin);
  CHECK_CALLBACK(symFlatMapSegmentOne);
  CHECK_CALLBACK(symFlatMappingUnmapSegments);
  CHECK_CALLBACK(symFlatVaFreeSegments);
  CHECK_CALLBACK(symLsaGetCapacity);
  CHECK_CALLBACK(symLsaVaReserve);
  CHECK_CALLBACK(symLsaVaFree);
  CHECK_CALLBACK(symLsaMapBeginAt);
#undef CHECK_CALLBACK
  std::fflush(stderr);
}
