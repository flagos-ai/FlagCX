#include "perf_common.h"
#include "uni_runner_algorithm.h"

static int64_t selectedUniRunnerAlgorithm() {
  const char *value = getenv("FLAGCX_UNIRUNNER_ALGO");
  return value == NULL ? -1 : atoll(value);
}

static void collFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxAllReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype,
                             ctx.redOp, ctx.comm, ctx.stream));
}

static double bwFactorFn(int totalProcs) {
  const int64_t algorithm = selectedUniRunnerAlgorithm();
  double factor = (double)(2 * (totalProcs - 1)) / (double)(totalProcs);
  if (algorithm == uniRunnerAlgorithmLocRed) {
    factor = 1;
  } else if (algorithm == uniRunnerAlgorithmRingAG) {
    factor = (double)(totalProcs - 1) / (double)(totalProcs);
  }
  return factor;
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  const bool useLocRed =
      selectedUniRunnerAlgorithm() == uniRunnerAlgorithmLocRed;
  perfInitReductionBuffers(ctx, size, count, useLocRed);
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  const int64_t algorithm = selectedUniRunnerAlgorithm();
  if (algorithm == uniRunnerAlgorithmLocRed) {
    // LocRed accumulates into recvbuff. Check one fresh invocation outside the
    // timed loop so the expected value does not depend on numIters.
    perfInitReductionBuffers(ctx, size, count, true);
    PERF_CHECK(ctx.devHandle->deviceSynchronize());
    collFn(ctx, count);
    PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));
  }
  perfPrintBuffer(ctx, count, "recvbuff");

  if (algorithm == uniRunnerAlgorithmLocRed) {
    perfCheckLocalReduction(ctx, count, "local-reduce");
  } else if (algorithm == uniRunnerAlgorithmRingAG) {
    perfCheckRingGather(ctx, count, "ring-gather");
  } else {
    perfCheckUniformReduction(ctx, count,
                              perfReduceAcrossRanks(ctx.redOp, ctx.totalProcs),
                              "all-reduce");
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
