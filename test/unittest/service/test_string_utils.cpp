// Unit tests for string conversion and host utility functions.
// Source: flagcx/service/include/utils.h (declared), flagcx/service/utils.cc
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "flagcx.h"
#include "utils.h"

// ---------- flagcxOpToString ----------

TEST(StringUtils, OpToString) {
  EXPECT_STREQ(flagcxOpToString(flagcxSum), "flagcxSum");
  EXPECT_STREQ(flagcxOpToString(flagcxMax), "flagcxMax");
  EXPECT_STREQ(flagcxOpToString(flagcxMin), "flagcxMin");
  EXPECT_STREQ(flagcxOpToString(flagcxProd), "flagcxProd");
  EXPECT_STREQ(flagcxOpToString(flagcxAvg), "flagcxAvg");
}

// ---------- flagcxDatatypeToString ----------

TEST(StringUtils, DatatypeToString) {
  EXPECT_STREQ(flagcxDatatypeToString(flagcxFloat32), "flagcxFloat32");
  EXPECT_STREQ(flagcxDatatypeToString(flagcxInt32), "flagcxInt32");
  EXPECT_STREQ(flagcxDatatypeToString(flagcxFloat64), "flagcxFloat64");
  EXPECT_STREQ(flagcxDatatypeToString(flagcxInt8), "flagcxInt8");
  EXPECT_STREQ(flagcxDatatypeToString(flagcxUint64), "flagcxUint64");
}

// ---------- getHostName ----------

TEST(StringUtils, GetHostName) {
  char buf[256] = {0};
  flagcxResult_t result = getHostName(buf, sizeof(buf), '.');
  EXPECT_EQ(result, flagcxSuccess);
  EXPECT_GT(strlen(buf), 0u);
}

// ---------- getHash ----------

TEST(StringUtils, GetHash) {
  // Same input produces same hash
  uint64_t h1 = getHash("hello", 5);
  uint64_t h2 = getHash("hello", 5);
  EXPECT_EQ(h1, h2);

  // Different inputs produce different hashes
  uint64_t h3 = getHash("world", 5);
  EXPECT_NE(h1, h3);

  // Empty string still produces a hash
  uint64_t h4 = getHash("", 0);
  EXPECT_NE(h4, 0u);
}
