#include "perf_common.h"

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  sBuf = ctx.maxBytes;
  rBuf = ctx.maxBytes / ctx.totalProcs;
}

static void collFn(PerfContext &ctx, size_t count) {
  const size_t recvcount = count / ctx.totalProcs;
  flagcxReduceScatter(ctx.sendbuff, ctx.recvbuff, recvcount, ctx.datatype,
                      ctx.redOp, ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  perfInitReductionBuffers(ctx, size, count, false);
}

static void postIterFn(PerfContext &ctx, size_t, size_t count) {
  const size_t recvcount = count / ctx.totalProcs;
  perfPrintBuffer(ctx, recvcount, "recvbuff");
  perfCheckUniformReduction(ctx, recvcount,
                            perfReduceAcrossRanks(ctx.redOp,
                                                  ctx.totalProcs),
                            "reduce-scatter");
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);
  perfEnableTypedReduction(ctx);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
