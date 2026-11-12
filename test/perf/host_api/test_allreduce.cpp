#include "perf_reduce_check.h"

static void collFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxAllReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype,
                             ctx.op, ctx.comm, ctx.stream));
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
  const size_t bytes = count * getFlagcxDataTypeSize(ctx.datatype);
  perfReduceFillInput(static_cast<unsigned char *>(ctx.hello), count, ctx.proc,
                      ctx.totalProcs, ctx.datatype, ctx.op);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, bytes,
                                         flagcxMemcpyHostToDevice, NULL));
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, size, flagcxMemDevice,
                                         NULL));
}

static void postIterFn(PerfContext &ctx, size_t, size_t count) {
  const size_t bytes = count * getFlagcxDataTypeSize(ctx.datatype);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, bytes,
                                         flagcxMemcpyDeviceToHost, NULL));
  perfReduceCheckOutput(static_cast<const unsigned char *>(ctx.hello), count, 0,
                        ctx.proc, ctx.totalProcs, ctx.datatype, ctx.op,
                        "AllReduce");
  if (ctx.proc == 0)
    printf("AllReduce data verified: %zu bytes per rank\n", bytes);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
