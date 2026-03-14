// Unit tests for flagcxMemoryPool — free-list allocator.
// Source: flagcx/service/include/utils.h (inline implementation)
// Links against libflagcx.

#include <gtest/gtest.h>

#include "utils.h"

struct PoolItem {
  int value;
  char padding[60]; // ensure non-trivial size
};

class MemoryPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    flagcxMemoryStackConstruct(&stack);
    flagcxMemoryPoolConstruct(&pool);
  }
  void TearDown() override { flagcxMemoryStackDestruct(&stack); }

  flagcxMemoryStack stack;
  flagcxMemoryPool pool;
};

TEST_F(MemoryPoolTest, ConstructEmpty) {
  // A freshly constructed pool has null head
  EXPECT_EQ(pool.head, nullptr);
}

TEST_F(MemoryPoolTest, AllocAndFree) {
  PoolItem *item = flagcxMemoryPoolAlloc<PoolItem>(&pool, &stack);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->value, 0); // zero-initialized
  item->value = 42;

  // Free it back to pool
  flagcxMemoryPoolFree(&pool, item);
  EXPECT_NE(pool.head, nullptr); // pool is non-empty

  // Alloc again — should reuse from pool
  PoolItem *item2 = flagcxMemoryPoolAlloc<PoolItem>(&pool, &stack);
  ASSERT_NE(item2, nullptr);
  // item2 should have been zero-initialized by alloc
  EXPECT_EQ(item2->value, 0);
}

TEST_F(MemoryPoolTest, MultipleAllocFree) {
  constexpr int N = 10;
  PoolItem *items[N];

  // Allocate N objects
  for (int i = 0; i < N; i++) {
    items[i] = flagcxMemoryPoolAlloc<PoolItem>(&pool, &stack);
    ASSERT_NE(items[i], nullptr);
    items[i]->value = i;
  }

  // Free all back to pool
  for (int i = 0; i < N; i++) {
    flagcxMemoryPoolFree(&pool, items[i]);
  }

  // Alloc N again — all should be reused from pool
  for (int i = 0; i < N; i++) {
    PoolItem *item = flagcxMemoryPoolAlloc<PoolItem>(&pool, &stack);
    ASSERT_NE(item, nullptr);
  }
}

TEST_F(MemoryPoolTest, TakeAll) {
  // Alloc+free into pool A
  PoolItem *item1 = flagcxMemoryPoolAlloc<PoolItem>(&pool, &stack);
  PoolItem *item2 = flagcxMemoryPoolAlloc<PoolItem>(&pool, &stack);
  flagcxMemoryPoolFree(&pool, item1);
  flagcxMemoryPoolFree(&pool, item2);

  // pool B starts empty
  flagcxMemoryPool poolB;
  flagcxMemoryPoolConstruct(&poolB);
  EXPECT_EQ(poolB.head, nullptr);

  // Take all from pool into poolB
  flagcxMemoryPoolTakeAll(&poolB, &pool);
  EXPECT_NE(poolB.head, nullptr); // poolB has items
  EXPECT_EQ(pool.head, nullptr);  // pool is now empty
}
