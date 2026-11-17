#include "perf_reduce_check.h"

static void warmupFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype,
                          ctx.redOp, 0, ctx.comm, ctx.stream));
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  PERF_CHECK(flagcxReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype,
                          ctx.redOp, root, ctx.comm, ctx.stream));
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int root) {
  const size_t bytes = count * getFlagcxDataTypeSize(ctx.datatype);
  perfReduceFillInput(static_cast<unsigned char *>(ctx.hello), count, ctx.proc,
                      ctx.totalProcs, ctx.datatype, ctx.redOp);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, bytes,
                                         flagcxMemcpyHostToDevice, NULL));
  if (ctx.proc == root)
    PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, size,
                                           flagcxMemDevice, NULL));
}

static void postIterFn(PerfContext &ctx, size_t, size_t count, int root) {
  if (ctx.proc != root)
    return;
  perfPrintBuffer(ctx, count, "recvbuff");
  const size_t bytes = count * getFlagcxDataTypeSize(ctx.datatype);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, bytes,
                                         flagcxMemcpyDeviceToHost, NULL));
  perfReduceCheckOutput(static_cast<const unsigned char *>(ctx.hello), count, 0,
                        ctx.proc, ctx.totalProcs, ctx.datatype, ctx.redOp,
                        "Reduce");
  printf("Reduce data verified: %zu bytes, root=%d\n", bytes, root);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, warmupFn);
  perfRootBenchmarkLoop(ctx, collFn, nullptr, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
