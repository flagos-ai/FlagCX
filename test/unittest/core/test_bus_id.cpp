// Unit tests for PCI Bus ID conversion utilities.
// Source: flagcx/service/include/utils.h (declared), flagcx/service/utils.cc
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "utils.h"

TEST(BusId, Int64ToBusIdAndBack) {
  // Known ID value — convert to string and back, verify round-trip
  int64_t origId = 0x00041001; // domain=0, bus=0x41, dev=0x00, func=1
  char busId[64] = {0};

  flagcxResult_t result = int64ToBusId(origId, busId);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_GT(strlen(busId), 0u);

  // Parse back to int64
  int64_t parsedId = 0;
  result = busIdToInt64(busId, &parsedId);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_EQ(parsedId, origId);
}

TEST(BusId, KnownFormat) {
  // int64ToBusId extracts: domain=id>>20, bus=(id&0xff000)>>12,
  //                         dev=(id&0xff0)>>4, func=id&0xf
  // For id=0x00041001: domain=0, bus=0x41, dev=0x00, func=1
  int64_t id = 0x00041001;
  char busId[64] = {0};

  flagcxResult_t result = int64ToBusId(id, busId);
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_STREQ(busId, "0000:41:00.1");
}

TEST(BusId, InvalidBusId) {
  // busIdToInt64 handles non-hex characters by stopping early.
  // "invalid" has no valid hex digits to parse, result should be 0.
  int64_t id = -1;
  flagcxResult_t result = busIdToInt64("invalid", &id);
  EXPECT_EQ(result, flagcxSuccess); // function always returns success
  // strtol("", NULL, 16) returns 0
  EXPECT_EQ(id, 0);
}
