#include "../../../plugin/torch/flagcx/include/cleanup_flagcx.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

TEST(TorchPairCleanup, NumericOrderDoesNotDependOnInsertionOrDigitCount) {
  flagcxPairComms pairs;
  for (const auto &key : {std::make_pair(2, 10), std::make_pair(2, 2),
                          std::make_pair(0, 10), std::make_pair(0, 2)})
    pairs.emplace(key, nullptr);
  std::vector<std::pair<int, int>> order;
  for (const auto &item : pairs)
    order.push_back(item.first);
  EXPECT_EQ(order, (std::vector<std::pair<int, int>>{
                       {0, 2}, {0, 10}, {2, 2}, {2, 10}}));
}

TEST(TorchPairCleanup, ExplicitRetryRetainsCompleteOwnerUntilSuccess) {
  auto &registry = flagcxPendingBackendCleanup::instance();
  int attempts = 0;
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<int> retained = owner;
  registry.retain([owner, &attempts] {
    return ++attempts == 1 ? flagcxRemoteError : flagcxSuccess;
  });
  owner.reset();
  EXPECT_EQ(registry.retry(), flagcxRemoteError);
  EXPECT_FALSE(retained.expired());
  EXPECT_EQ(registry.retry(), flagcxSuccess);
  EXPECT_TRUE(retained.expired());
  EXPECT_EQ(attempts, 2);
  EXPECT_EQ(registry.retry(), flagcxSuccess);
  EXPECT_EQ(attempts, 2);
}

TEST(TorchPairCleanup, ExceptionKeepsTheRetryOwnerReachable) {
  auto &registry = flagcxPendingBackendCleanup::instance();
  int attempts = 0;
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<int> retained = owner;
  registry.retain([owner, &attempts]() -> flagcxResult_t {
    if (++attempts == 1)
      throw std::runtime_error("cleanup failed");
    return flagcxSuccess;
  });
  owner.reset();
  EXPECT_THROW(registry.retry(), std::runtime_error);
  EXPECT_FALSE(retained.expired());
  EXPECT_EQ(registry.retry(), flagcxSuccess);
  EXPECT_TRUE(retained.expired());
}
