#ifndef FLAGCX_TEST_SIGNAL_HOST_H_
#define FLAGCX_TEST_SIGNAL_HOST_H_
#include "device_signal_test_layout.h"
#include "flagcx_kernel.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

struct FlagcxSignalTestOptions {
  int contexts;
  int repeats;
  unsigned skew;
};

// Validate on every rank before creating any communication resources.
static FlagcxSignalTestOptions flagcxSignalTestOptions(MPI_Comm group) {
  const char *names[] = {"FLAGCX_TEST_CONTEXTS", "FLAGCX_TEST_SIGNAL_REPEATS",
                         "FLAGCX_TEST_SIGNAL_SKEW"};
  const long defaults[] = {4, 1, 0};
  const long maximum[] = {FLAGCX_DEVICE_CTA_COUNT, 10000, 1000000};
  long values[3];
  int valid = 1, allValid = 0;
  for (int i = 0; i < 3; ++i) {
    values[i] = defaults[i];
    if (const char *text = std::getenv(names[i])) {
      char *end = nullptr;
      errno = 0;
      values[i] = std::strtol(text, &end, 10);
      if (errno || end == text || *end || values[i] < (i == 2 ? 0 : 1) ||
          values[i] > maximum[i]) {
        std::fprintf(stderr, "Invalid %s=%s\n", names[i], text);
        valid = 0;
      }
    }
  }
  MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, group);
  if (!allValid)
    MPI_Abort(MPI_COMM_WORLD, 1);
  // A rank cannot choose fewer launches than its peer.
  for (int i = 0; i < 3; ++i) {
    int value = (int)values[i], low = 0, high = 0;
    MPI_Allreduce(&value, &low, 1, MPI_INT, MPI_MIN, group);
    MPI_Allreduce(&value, &high, 1, MPI_INT, MPI_MAX, group);
    if (low != high) {
      std::fprintf(stderr, "%s must agree on all test ranks\n", names[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  return {(int)values[0], (int)values[1], (unsigned)values[2]};
}

static void flagcxSignalTestCheckContexts(int actual, MPI_Comm group) {
  int low = 0, high = 0;
  MPI_Allreduce(&actual, &low, 1, MPI_INT, MPI_MIN, group);
  MPI_Allreduce(&actual, &high, 1, MPI_INT, MPI_MAX, group);
  if (low <= 0 || low != high) {
    std::fprintf(stderr,
                 "Signal tests need positive, matching context counts; "
                 "local=%d min=%d max=%d\n",
                 actual, low, high);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::printf("# signal tests: actual contexts=%d, CTA slots are independent\n",
              actual);
}

static bool flagcxWaitSignalResults(const uint64_t *r, int ctas,
                                    uint64_t incoming,
                                    bool *sawOverlap = nullptr) {
  bool ok = true;
  for (int b = 0; b < ctas; ++b) {
    const uint64_t *v = r + b * kFlagcxTestWaitResultWords;
    if (sawOverlap && v[3] != 0)
      *sawOverlap = true;
    if (v[0] != 0 || v[1] != incoming || v[2] != 1) {
      std::fprintf(stderr,
                   "CTA %d reset=%llu signal=%llu expected=%llu ran=%llu\n", b,
                   (unsigned long long)v[0], (unsigned long long)v[1],
                   (unsigned long long)incoming, (unsigned long long)v[2]);
      ok = false;
    }
  }
  return ok;
}

static bool flagcxSignalOverlapResult(bool observed, unsigned skew,
                                      MPI_Comm group) {
  int local = observed ? 1 : 0, any = 0;
  MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, group);
  if (skew && FLAGCX_DEVICE_CTA_COUNT > 1 && !any) {
    std::fprintf(stderr,
                 "Signal skew did not produce a completed CTA before "
                 "a late CTA initialized; increase FLAGCX_TEST_SIGNAL_SKEW.\n");
    return false;
  }
  return true;
}

static bool flagcxPipelineResults(const uint64_t *r, int ctas, int threads,
                                  int ranks, int myRank, int intraBase,
                                  int intraSize) {
  bool ok = true;
  for (int b = 0; b < ctas; ++b) {
    const uint64_t *v = r + b * kFlagcxTestPipelineResultWords;
    uint64_t sent = countInterPeersAssignedTo(b, ctas, threads, ranks,
                                              intraBase, intraSize);
    uint64_t received = b == signalOwnerCta(myRank, ctas, threads)
                            ? 2 * uint64_t(ranks - intraSize)
                            : 0;
    if (v[0] != 2 * sent || v[1] != sent + 1 || v[2] != received ||
        v[3] != received) {
      std::fprintf(stderr, "CTA %d pipeline completion mismatch\n", b);
      ok = false;
    }
  }
  return ok;
}

static bool flagcxUnifiedSignalResults(const int *r) {
  for (int b = 0; b < kFlagcxTestUnifiedCtas; ++b)
    if (r[b] != 1)
      return false;
  return true;
}
#endif
