// Unit tests for alignment math utilities.
// Pure templates/macros in align.h — no library link needed.

#include <gtest/gtest.h>

#include "align.h"

// ---------- divUp (template) ----------

TEST(Align, DivUpBasic) {
  EXPECT_EQ(divUp(10, 3), 4);
  EXPECT_EQ(divUp(9, 3), 3);
  EXPECT_EQ(divUp(0, 5), 0);
  EXPECT_EQ(divUp(1, 1), 1);
  EXPECT_EQ(divUp(7, 7), 1);
}

// ---------- DIVUP (macro) ----------

TEST(Align, DivUpMacro) {
  EXPECT_EQ(DIVUP(10, 3), 4);
  EXPECT_EQ(DIVUP(9, 3), 3);
  EXPECT_EQ(DIVUP(0, 5), 0);
  EXPECT_EQ(DIVUP(1, 1), 1);
}

// ---------- roundUp (template) ----------

TEST(Align, RoundUpBasic) {
  EXPECT_EQ(roundUp(10, 4), 12);
  EXPECT_EQ(roundUp(8, 4), 8);
  EXPECT_EQ(roundUp(1, 16), 16);
  EXPECT_EQ(roundUp(0, 8), 0);
  EXPECT_EQ(roundUp(15, 5), 15);
}

// ---------- ROUNDUP (macro) ----------

TEST(Align, RoundUpMacro) {
  EXPECT_EQ(ROUNDUP(10, 4), 12);
  EXPECT_EQ(ROUNDUP(8, 4), 8);
  EXPECT_EQ(ROUNDUP(1, 16), 16);
}

// ---------- alignUp (template, power-of-two) ----------

TEST(Align, AlignUpPowerOfTwo) {
  EXPECT_EQ(alignUp(13, 8), 16);
  EXPECT_EQ(alignUp(16, 8), 16);
  EXPECT_EQ(alignUp(1, 1024), 1024);
  EXPECT_EQ(alignUp(0, 64), 0);
  EXPECT_EQ(alignUp(4096, 4096), 4096);
  EXPECT_EQ(alignUp(4097, 4096), 8192);
}
