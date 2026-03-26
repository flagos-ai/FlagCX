// Unit tests for c2c_ir.h XML tag parsing utilities.
// Source: flagcx/runner/include/c2c_ir.h (inline functions)
// Links against libflagcx (heavy header dependencies).

#include <cstring>
#include <gtest/gtest.h>

#include "c2c_ir.h"

// ---------- readTagValue ----------

TEST(C2cIR, ReadTagValueBasic) {
  char line[512];
  strcpy(line, "<count>42</count>");
  char buf[512] = {0};
  bool found = readTagValue(line, "count", buf);
  EXPECT_TRUE(found);
  EXPECT_STREQ(buf, "42");
}

TEST(C2cIR, ReadTagValueMissing) {
  char line[512];
  strcpy(line, "<other>42</other>");
  char buf[512] = {0};
  bool found = readTagValue(line, "count", buf);
  EXPECT_FALSE(found);
}

// ---------- readIntTag ----------

TEST(C2cIR, ReadIntTag) {
  char line[512];
  strcpy(line, "<rank>5</rank>");
  int val = readIntTag(line, "rank");
  EXPECT_EQ(val, 5);
}

// ---------- readSizeTag ----------

TEST(C2cIR, ReadSizeTag) {
  char line[512];
  strcpy(line, "<count>1048576</count>");
  size_t val = readSizeTag(line, "count");
  EXPECT_EQ(val, 1048576u);
}

// ---------- genC2cAlgoHash ----------

TEST(C2cIR, GenAlgoHash) {
  // Same inputs produce same hash
  size_t h1 = genC2cAlgoHash(100, 200, 0, flagcxCommOpAllReduce, flagcxSum);
  size_t h2 = genC2cAlgoHash(100, 200, 0, flagcxCommOpAllReduce, flagcxSum);
  EXPECT_EQ(h1, h2);

  // Different inputs produce different hashes
  size_t h3 = genC2cAlgoHash(100, 200, 1, flagcxCommOpAllReduce, flagcxSum);
  EXPECT_NE(h1, h3);

  size_t h4 = genC2cAlgoHash(100, 200, 0, flagcxCommOpBroadcast, flagcxSum);
  EXPECT_NE(h1, h4);
}
