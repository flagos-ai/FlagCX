// Test-only completion ownership. Contexts may be shared; slots may not.
#ifndef FLAGCX_TEST_SIGNAL_LAYOUT_H_
#define FLAGCX_TEST_SIGNAL_LAYOUT_H_
#include <stddef.h>
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
// Payload groups are independent of completion slots. Variants reuse these
// three coop kinds x two teams only after the host has checked the last round.
constexpr int kFlagcxTestPayloadGroups = 6;
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
FLAGCX_TEST_HD int unifiedPayloadCombo(int group, int variant, int variants) {
  return (group / 2 * variants + variant) * 2 + group % 2;
}
FLAGCX_TEST_HD int unifiedPayloadGroup(int combo, int variants) {
  return combo / (2 * variants) * 2 + combo % 2;
}
// Float payloads have a nonempty, element-aligned slice for every CTA. Tiny
// requests are raised to four elements; larger requests keep whole elements.
inline size_t unifiedPayloadBytes(size_t requested) {
  size_t elements = requested / sizeof(float);
  if (elements < size_t(kFlagcxTestUnifiedCtas))
    elements = kFlagcxTestUnifiedCtas;
  return elements * sizeof(float);
}
// The same registered buffers also serve S17's rank tags and S24's scalars.
// Return zero on overflow so callers can reject unsafe payload allocations.
inline size_t unifiedPayloadBufferBytes(size_t maximum, size_t resolutionTags) {
  size_t groupBytes = unifiedPayloadBytes(maximum);
  if (groupBytes > SIZE_MAX / kFlagcxTestPayloadGroups ||
      resolutionTags > SIZE_MAX / sizeof(float))
    return 0;
  size_t capacity = kFlagcxTestPayloadGroups * groupBytes;
  size_t tagBytes = resolutionTags * sizeof(float);
  size_t scalarBytes = size_t(kFlagcxTestUnifiedCtas) * 12 * sizeof(uint64_t);
  if (capacity < tagBytes)
    capacity = tagBytes;
  if (capacity < scalarBytes)
    capacity = scalarBytes;
  return capacity;
}
struct FlagcxTestPayloadSlice {
  size_t offset;
  size_t bytes;
};
FLAGCX_TEST_HD FlagcxTestPayloadSlice unifiedPayloadSlice(size_t bytes,
                                                          int cta) {
  size_t elements = bytes / sizeof(float);
  size_t q = elements / kFlagcxTestUnifiedCtas;
  size_t r = elements % kFlagcxTestUnifiedCtas;
  size_t before = size_t(cta) < r ? size_t(cta) : r;
  return {(size_t(cta) * q + before) * sizeof(float),
          (q + (size_t(cta) < r ? 1 : 0)) * sizeof(float)};
}
// Small exactly representable floats distinguish senders, scenarios and rounds
// without losing the low bits when large buffers are checked.
inline float unifiedPayloadValue(int rank, int scenario, int round,
                                 size_t index) {
  uint32_t v = uint32_t(index) * 2654435761u ^
               uint32_t(rank + 1) * 2246822519u ^
               uint32_t(scenario) * 3266489917u ^ uint32_t(round) * 668265263u;
  v ^= v >> 16;
  return float((v & 65535u) + 1u);
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
