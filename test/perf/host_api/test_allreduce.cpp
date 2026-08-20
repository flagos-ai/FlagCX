#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  flagcxAllReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype, ctx.redOp,
                  ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("FLAGCX_UNIRUNNER_USE_RINGAG");
  double factor = (double)(2 * (totalProcs - 1)) / (double)(totalProcs);
  if (envLocRed != NULL && atoi(envLocRed) == 1) {
    factor = 1;
  } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
    factor = (double)(totalProcs - 1) / (double)(totalProcs);
  }
  return factor;
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const bool useLocRed = envLocRed != NULL && atoi(envLocRed) == 1;
  perfInitReductionBuffers(ctx, size, count, useLocRed);
}

static void postIterFn(PerfContext &ctx, size_t, size_t count) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("FLAGCX_UNIRUNNER_USE_RINGAG");
  perfPrintBuffer(ctx, count, "recvbuff");

  if (envLocRed != NULL && atoi(envLocRed) == 1) {
    perfCheckLocalReduction(ctx, count, "local-reduce");
  } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
    perfCheckRingGather(ctx, count, "ring-gather");
  } else {
    perfCheckUniformReduction(ctx, count,
                              perfReduceAcrossRanks(ctx.redOp,
                                                    ctx.totalProcs),
                              "all-reduce");
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfEnableTypedReduction(ctx);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
