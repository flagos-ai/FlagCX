#include "perf_reduce_check.h"

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  sBuf = ctx.maxBytes;
  rBuf = ctx.maxBytes / ctx.totalProcs;
}

static void collFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxReduceScatter(ctx.sendbuff, ctx.recvbuff,
                                 count / ctx.totalProcs, ctx.datatype, ctx.op,
                                 ctx.comm, ctx.stream));
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t, size_t count) {
  const size_t recvCount = count / ctx.totalProcs;
  const size_t activeCount = recvCount * ctx.totalProcs;
  const size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);
  perfReduceFillInput(static_cast<unsigned char *>(ctx.hello), activeCount,
                      ctx.proc, ctx.totalProcs, ctx.datatype, ctx.op);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello,
                                         activeCount * typeSize,
                                         flagcxMemcpyHostToDevice, NULL));
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, recvCount * typeSize,
                                         flagcxMemDevice, NULL));
}

static void postIterFn(PerfContext &ctx, size_t, size_t count) {
  const size_t recvCount = count / ctx.totalProcs;
  const size_t bytes = recvCount * getFlagcxDataTypeSize(ctx.datatype);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, bytes,
                                         flagcxMemcpyDeviceToHost, NULL));
  perfReduceCheckOutput(static_cast<const unsigned char *>(ctx.hello),
                        recvCount, static_cast<size_t>(ctx.proc) * recvCount,
                        ctx.proc, ctx.totalProcs, ctx.datatype, ctx.op,
                        "ReduceScatter");
  if (ctx.proc == 0)
    printf("ReduceScatter data verified: %zu bytes per rank\n", bytes);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
