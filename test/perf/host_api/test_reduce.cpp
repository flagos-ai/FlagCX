#include "perf_common.h"

static void warmupFn(PerfContext &ctx, size_t count) {
  flagcxReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype, ctx.redOp, 0,
               ctx.comm, ctx.stream);
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  flagcxReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype, ctx.redOp,
               root, ctx.comm, ctx.stream);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int) {
  perfInitReductionBuffers(ctx, size, count, false);
}

static void postIterFn(PerfContext &ctx, size_t, size_t count, int root) {
  if (ctx.proc != root)
    return;
  perfPrintBuffer(ctx, count, "recvbuff");
  perfCheckUniformReduction(ctx, count,
                            perfReduceAcrossRanks(ctx.redOp,
                                                  ctx.totalProcs),
                            "reduce");
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfEnableTypedReduction(ctx);
  perfWarmup(ctx, warmupFn);
  perfRootBenchmarkLoop(ctx, collFn, nullptr, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
