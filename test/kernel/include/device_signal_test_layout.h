// Test-only completion ownership. Contexts may be shared; slots may not.
#ifndef FLAGCX_TEST_SIGNAL_LAYOUT_H_
#define FLAGCX_TEST_SIGNAL_LAYOUT_H_
#include <stdint.h>
#include <stdlib.h>
#ifdef FLAGCX_HOST_DEVICE_INLINE
#define FLAGCX_TEST_HD FLAGCX_HOST_DEVICE_INLINE
#else
#define FLAGCX_TEST_HD inline
#endif

enum FlagcxTestSignalKind {
  FlagcxTestPutInc,
  FlagcxTestPutAdd,
  FlagcxTestPipeline,
  FlagcxTestFlush,
  FlagcxTestPutValue,
  FlagcxTestSignal,
  FlagcxTestWaitFlush,
  FlagcxTestAlltoAll,
  FlagcxTestSignalKinds
};
constexpr int kFlagcxTestUnifiedCtas = 4;
constexpr int kFlagcxTestUnifiedSlots = 18;
constexpr int kFlagcxTestUnifiedCompletionCount =
    kFlagcxTestUnifiedCtas * kFlagcxTestUnifiedSlots;
constexpr int kFlagcxTestPipelineResultWords = 4;
constexpr int kFlagcxTestWaitResultWords = 4;

FLAGCX_TEST_HD int legacySignalId(FlagcxTestSignalKind kind, int cta,
                                  int ctas) {
  return int(kind) * ctas + cta;
}
FLAGCX_TEST_HD int dedicatedSignalId(int ctas) {
  return int(FlagcxTestSignalKinds) * ctas;
}
FLAGCX_TEST_HD int legacySignalCount(int ctas) {
  return dedicatedSignalId(ctas) + 1;
}
FLAGCX_TEST_HD int signalOwnerCta(int rank, int ctas, int threads) {
  return (rank % (ctas * threads)) / threads;
}
FLAGCX_TEST_HD int countInterPeersAssignedTo(int cta, int ctas, int threads,
                                             int ranks, int intraBase,
                                             int intraSize) {
  int count = 0;
  for (int peer = 0; peer < ranks; ++peer)
    if ((peer < intraBase || peer >= intraBase + intraSize) &&
        signalOwnerCta(peer, ctas, threads) == cta)
      ++count;
  return count;
}
FLAGCX_TEST_HD int unifiedCompletionId(int cta, int combo) {
  return cta * kFlagcxTestUnifiedSlots + combo;
}
// Launchers also serve benchmarks; their default is no injected skew.
inline unsigned flagcxTestHostSignalSkew() {
  const char *text = getenv("FLAGCX_TEST_SIGNAL_SKEW");
  if (!text)
    return 0;
  unsigned long value = strtoul(text, nullptr, 10);
  return value <= 1000000 ? (unsigned)value : 1000000;
}
FLAGCX_TEST_HD unsigned signalSkewMultiplier(int rank, int cta) {
  return (unsigned(rank + cta) % 4u) == 0 ? unsigned(cta + 1) : 0;
}
#ifdef FLAGCX_DEVICE_INLINE_DECORATOR
// Bounded, test-only launch skew. It creates no inter-CTA dependency.
FLAGCX_DEVICE_INLINE_DECORATOR void flagcxTestSignalDelay(int rank, int cta,
                                                          unsigned skew) {
  const unsigned loops = signalSkewMultiplier(rank, cta) * skew;
  for (volatile unsigned i = 0; i < loops; ++i) {
  }
}
#endif
#undef FLAGCX_TEST_HD
#endif
