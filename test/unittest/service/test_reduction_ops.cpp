// Unit tests for host-side reduction operations.
// Source: flagcx/service/include/utils.h (inline templates sum<T>, min<T>,
// max<T>) Links against libflagcx.

#include <gtest/gtest.h>

#include "utils.h"

// ---------- sum<T> ----------

TEST(ReductionOps, SumFloat) {
  float a[] = {1.0f, 2.0f, 3.0f};
  float b[] = {4.0f, 5.0f, 6.0f};
  float c[3] = {0};

  sum<float>(c, a, b, 3);

  EXPECT_FLOAT_EQ(c[0], 5.0f);
  EXPECT_FLOAT_EQ(c[1], 7.0f);
  EXPECT_FLOAT_EQ(c[2], 9.0f);
}

TEST(ReductionOps, SumInt32) {
  int32_t a[] = {10, 20, 30};
  int32_t b[] = {1, 2, 3};
  int32_t c[3] = {0};

  sum<int32_t>(c, a, b, 3);

  EXPECT_EQ(c[0], 11);
  EXPECT_EQ(c[1], 22);
  EXPECT_EQ(c[2], 33);
}

// ---------- min<T> ----------

TEST(ReductionOps, MinFloat) {
  float a[] = {3.0f, 1.0f, 4.0f};
  float b[] = {1.0f, 5.0f, 2.0f};
  float c[3] = {0};

  // Note: using fully qualified call to avoid ambiguity with std::min
  ::min<float>(c, a, b, 3);

  EXPECT_FLOAT_EQ(c[0], 1.0f);
  EXPECT_FLOAT_EQ(c[1], 1.0f);
  EXPECT_FLOAT_EQ(c[2], 2.0f);
}

// ---------- max<T> ----------

TEST(ReductionOps, MaxFloat) {
  float a[] = {3.0f, 1.0f, 4.0f};
  float b[] = {1.0f, 5.0f, 2.0f};
  float c[3] = {0};

  // Note: using fully qualified call to avoid ambiguity with std::max
  ::max<float>(c, a, b, 3);

  EXPECT_FLOAT_EQ(c[0], 3.0f);
  EXPECT_FLOAT_EQ(c[1], 5.0f);
  EXPECT_FLOAT_EQ(c[2], 4.0f);
}

// ---------- Edge case: single element ----------

TEST(ReductionOps, SumSingleElement) {
  float a[] = {42.0f};
  float b[] = {8.0f};
  float c[1] = {0};

  sum<float>(c, a, b, 1);

  EXPECT_FLOAT_EQ(c[0], 50.0f);
}
