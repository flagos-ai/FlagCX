#include "../../../plugin/torch/flagcx/include/flagcx_pair_comm.hpp"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <random>
#include <vector>

TEST(PairCommOrder, DifferentInsertionHistoriesCannotFormShutdownWaitCycle) {
  constexpr int ranks = 5;
  std::vector<std::pair<int, int>> pairs;
  for (int low = 0; low < ranks; low++)
    for (int high = low + 1; high < ranks; high++)
      pairs.emplace_back(low, high);
  for (unsigned seed = 0; seed < 40; seed++) {
    std::array<c10d::FlagcxPairCommMap, ranks> owners;
    for (int rank = 0; rank < ranks; rank++) {
      std::mt19937 generator(seed * ranks + rank);
      std::shuffle(pairs.begin(), pairs.end(), generator);
      for (const auto &pair : pairs)
        if (pair.first == rank || pair.second == rank)
          owners[rank].emplace(pair, nullptr);
    }
    // A blocking pair shutdown can advance only when both endpoints enter
    // that pair. This exercises the exact container used by the consumer.
    for (size_t remaining = pairs.size(); remaining != 0; remaining--) {
      bool advanced = false;
      for (int rank = 0; rank < ranks && !advanced; rank++) {
        if (owners[rank].empty())
          continue;
        auto pair = owners[rank].begin()->first;
        int peer = pair.first == rank ? pair.second : pair.first;
        if (!owners[peer].empty() && owners[peer].begin()->first == pair) {
          owners[rank].erase(pair);
          owners[peer].erase(pair);
          advanced = true;
        }
      }
      ASSERT_TRUE(advanced) << "shutdown wait cycle, seed=" << seed;
    }
    for (const auto &owner : owners)
      EXPECT_TRUE(owner.empty());
  }
}
