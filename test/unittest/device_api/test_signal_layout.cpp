#include "device_signal_test_layout.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <set>
#include <utility>
#include <vector>

int main() {
  // Compare the ownership formula against the actual grid-stride issue loop.
  for (int ctas : {1, 2, 4, 36})
    for (int threads : {1, 4, 128, 512})
      for (int ranks : {2, 8, 129, 513, 1031}) {
        std::vector<int> issued(ranks, -1);
        for (int b = 0; b < ctas; ++b) {
          int sent = 0;
          for (int t = 0; t < threads; ++t)
            for (int p = b * threads + t; p < ranks; p += ctas * threads) {
              assert(issued[p] == -1);
              issued[p] = b;
              if (p >= 2)
                ++sent;
            }
          assert(sent ==
                 countInterPeersAssignedTo(b, ctas, threads, ranks, 0, 2));
        }
        for (int p = 0; p < ranks; ++p)
          assert(issued[p] == signalOwnerCta(p, ctas, threads));
        std::set<int> slots;
        for (int kind = 0; kind < FlagcxTestSignalKinds; ++kind)
          for (int b = 0; b < ctas; ++b)
            assert(
                slots
                    .insert(legacySignalId(FlagcxTestSignalKind(kind), b, ctas))
                    .second);
        assert(slots.insert(dedicatedSignalId(ctas)).second);
        assert(int(slots.size()) == legacySignalCount(ctas));
      }

  // Enumerate all CTA arrival orders. Fast CTAs complete before slow CTAs
  // reset/read their slots, including with only one shared context.
  for (int contexts : {1, 2, 3, 4}) {
    std::vector<int> order{0, 1, 2, 3};
    do {
      std::set<std::pair<int, int>> owners;
      for (int b : order)
        for (int combo = 0; combo < 18; ++combo)
          assert(owners.emplace(b % contexts, unifiedCompletionId(b, combo))
                     .second);
      std::vector<int> signal(contexts * kFlagcxTestUnifiedCompletionCount, 0);
      for (int round = 0; round < 100; ++round)
        for (int b : order)
          for (int combo = 0; combo < 18; ++combo) {
            int id = (b % contexts) * kFlagcxTestUnifiedCompletionCount +
                     unifiedCompletionId(b, combo);
            signal[id] = 0;
            signal[id] += 3;
            assert(signal[id] == 3);
          }
      for (int b = 0; b < 4; ++b)
        for (int combo = 0; combo < 18; ++combo)
          assert(signal[(b % contexts) * kFlagcxTestUnifiedCompletionCount +
                        unifiedCompletionId(b, combo)] == 3);
    } while (std::next_permutation(order.begin(), order.end()));
  }
  std::puts(
      "signal ownership: PASS (grid stride, 1-4 contexts, 24 CTA orders)");
}
