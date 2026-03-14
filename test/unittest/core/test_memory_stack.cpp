// Unit tests for flagcxMemoryStack — LIFO memory allocator.
// Source: flagcx/service/include/utils.h (inline implementation)
// Links against libflagcx.

#include <cstring>
#include <gtest/gtest.h>

#include "utils.h"

TEST(MemoryStack, ConstructAndDestruct) {
  flagcxMemoryStack stack;
  flagcxMemoryStackConstruct(&stack);
  // Should not crash on destruct
  flagcxMemoryStackDestruct(&stack);
}

TEST(MemoryStack, AllocInNilFrame) {
  flagcxMemoryStack stack;
  flagcxMemoryStackConstruct(&stack);

  int *p = flagcxMemoryStackAlloc<int>(&stack, 1);
  ASSERT_NE(p, nullptr);
  // Should be zero-initialized
  EXPECT_EQ(*p, 0);

  flagcxMemoryStackDestruct(&stack);
}

TEST(MemoryStack, PushPopFrame) {
  flagcxMemoryStack stack;
  flagcxMemoryStackConstruct(&stack);

  // Alloc in nil frame
  int *p1 = flagcxMemoryStackAlloc<int>(&stack, 1);
  ASSERT_NE(p1, nullptr);
  *p1 = 42;

  // Push a new frame, alloc within it
  flagcxMemoryStackPush(&stack);
  int *p2 = flagcxMemoryStackAlloc<int>(&stack, 1);
  ASSERT_NE(p2, nullptr);
  *p2 = 99;

  // Pop frame — p2 is invalidated, p1 should still be valid
  flagcxMemoryStackPop(&stack);
  EXPECT_EQ(*p1, 42);

  // Alloc again in nil frame — should succeed
  int *p3 = flagcxMemoryStackAlloc<int>(&stack, 1);
  ASSERT_NE(p3, nullptr);

  flagcxMemoryStackDestruct(&stack);
}

TEST(MemoryStack, MultipleAllocsInFrame) {
  flagcxMemoryStack stack;
  flagcxMemoryStackConstruct(&stack);

  flagcxMemoryStackPush(&stack);

  // Allocate 100 items in the same frame
  for (int i = 0; i < 100; i++) {
    int *p = flagcxMemoryStackAlloc<int>(&stack, 1);
    ASSERT_NE(p, nullptr) << "Alloc failed at iteration " << i;
    *p = i;
  }

  flagcxMemoryStackPop(&stack);
  flagcxMemoryStackDestruct(&stack);
}

TEST(MemoryStack, LargeAlloc) {
  flagcxMemoryStack stack;
  flagcxMemoryStackConstruct(&stack);

  // Allocate 1MB
  char *p = flagcxMemoryStackAlloc<char>(&stack, 1024 * 1024);
  ASSERT_NE(p, nullptr);
  // Verify it's zero-initialized by checking a few spots
  EXPECT_EQ(p[0], 0);
  EXPECT_EQ(p[512 * 1024], 0);
  EXPECT_EQ(p[1024 * 1024 - 1], 0);

  flagcxMemoryStackDestruct(&stack);
}
