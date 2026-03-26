// Unit tests for flagcxLRUCache<K, V> template.
// Pure C++ data structure — no MPI, no GPU, no FlagCX runtime needed.

#include <gtest/gtest.h>
#include <string>

// Include the LRU cache template directly from the runner headers.
// We only need the template definition — no FlagCX runtime linkage.
#include "c2c_algo.h"

class LRUCacheTest : public ::testing::Test {
protected:
  // Small capacity for easy eviction testing
  flagcxLRUCache<int, std::string> cache{3};
};

TEST_F(LRUCacheTest, BasicPutAndGet) {
  cache.put(1, "one");
  cache.put(2, "two");

  std::string val;
  EXPECT_TRUE(cache.get(1, val));
  EXPECT_EQ(val, "one");
  EXPECT_TRUE(cache.get(2, val));
  EXPECT_EQ(val, "two");
}

TEST_F(LRUCacheTest, GetMissReturnsFalse) {
  std::string val;
  EXPECT_FALSE(cache.get(42, val));

  cache.put(1, "one");
  EXPECT_FALSE(cache.get(99, val));
}

TEST_F(LRUCacheTest, CapacityEviction) {
  // Fill to capacity (3)
  cache.put(1, "one");
  cache.put(2, "two");
  cache.put(3, "three");

  // Insert a 4th — should evict key 1 (least recently used)
  cache.put(4, "four");

  std::string val;
  EXPECT_FALSE(cache.get(1, val)) << "key 1 should have been evicted";
  EXPECT_TRUE(cache.get(2, val));
  EXPECT_TRUE(cache.get(3, val));
  EXPECT_TRUE(cache.get(4, val));
}

TEST_F(LRUCacheTest, AccessOrderUpdate) {
  cache.put(1, "one");
  cache.put(2, "two");
  cache.put(3, "three");

  // Access key 1 — moves it to front (most recently used)
  std::string val;
  cache.get(1, val);

  // Insert key 4 — should evict key 2 (now the LRU), not key 1
  cache.put(4, "four");

  EXPECT_TRUE(cache.get(1, val)) << "key 1 was accessed, should not be evicted";
  EXPECT_FALSE(cache.get(2, val)) << "key 2 should have been evicted";
  EXPECT_TRUE(cache.get(3, val));
  EXPECT_TRUE(cache.get(4, val));
}

TEST_F(LRUCacheTest, UpdateExistingKey) {
  cache.put(1, "one");
  cache.put(2, "two");

  // Update key 1 with new value
  cache.put(1, "ONE");

  std::string val;
  EXPECT_TRUE(cache.get(1, val));
  EXPECT_EQ(val, "ONE");

  // The update should also move key 1 to front
  cache.put(3, "three");
  cache.put(4, "four");

  // key 2 should be evicted (was LRU after key 1 was updated)
  EXPECT_TRUE(cache.get(1, val)) << "key 1 was updated, should not be evicted";
  EXPECT_FALSE(cache.get(2, val)) << "key 2 should have been evicted";
}
