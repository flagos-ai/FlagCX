// Unit tests for memory allocation helpers.
// Source: flagcx/service/include/alloc.h
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "alloc.h"

// ---------- flagcxCalloc ----------

TEST(Alloc, CallocBasic) {
  int *ptr = nullptr;
  flagcxResult_t result = flagcxCalloc(&ptr, 100);
  EXPECT_EQ(result, flagcxSuccess);
  ASSERT_NE(ptr, nullptr);

  // Verify all bytes are zero
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(ptr[i], 0) << "Non-zero at index " << i;
  }

  free(ptr);
}

TEST(Alloc, CallocZeroElements) {
  // malloc(0) is implementation-defined; just ensure no crash
  int *ptr = nullptr;
  flagcxResult_t result = flagcxCalloc(&ptr, static_cast<size_t>(0));
  // Result may vary; we just verify no crash
  (void)result;
  // If a pointer was returned, free it
  if (ptr)
    free(ptr);
}

// ---------- flagcxRealloc ----------

TEST(Alloc, ReallocGrow) {
  int *ptr = nullptr;
  flagcxResult_t result = flagcxCalloc(&ptr, 10);
  ASSERT_EQ(result, flagcxSuccess);
  ASSERT_NE(ptr, nullptr);

  // Fill with known data
  for (int i = 0; i < 10; i++) {
    ptr[i] = i + 1;
  }

  result =
      flagcxRealloc(&ptr, static_cast<size_t>(10), static_cast<size_t>(20));
  EXPECT_EQ(result, flagcxSuccess);
  ASSERT_NE(ptr, nullptr);

  // Verify original data preserved
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(ptr[i], i + 1) << "Original data corrupted at index " << i;
  }

  // Verify new portion is zeroed
  for (int i = 10; i < 20; i++) {
    EXPECT_EQ(ptr[i], 0) << "New memory not zeroed at index " << i;
  }

  free(ptr);
}

TEST(Alloc, ReallocSameSize) {
  int *ptr = nullptr;
  flagcxResult_t result = flagcxCalloc(&ptr, 10);
  ASSERT_EQ(result, flagcxSuccess);

  int *origPtr = ptr;
  result =
      flagcxRealloc(&ptr, static_cast<size_t>(10), static_cast<size_t>(10));
  EXPECT_EQ(result, flagcxSuccess);
  // Pointer should be unchanged (no-op)
  EXPECT_EQ(ptr, origPtr);

  free(ptr);
}

TEST(Alloc, ReallocShrinkFails) {
  int *ptr = nullptr;
  flagcxResult_t result = flagcxCalloc(&ptr, 10);
  ASSERT_EQ(result, flagcxSuccess);

  result = flagcxRealloc(&ptr, static_cast<size_t>(10), static_cast<size_t>(5));
  EXPECT_EQ(result, flagcxInternalError);

  free(ptr);
}
