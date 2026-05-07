#include "perf_common.h"

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  sBuf = ctx.maxBytes;
  rBuf = ctx.maxBytes / ctx.totalProcs;
}

static void warmupFn(PerfContext &ctx, size_t count) {
  flagcxScatter(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs, flagcxFloat,
                0, ctx.comm, ctx.stream);
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  flagcxScatter(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs, flagcxFloat,
                root, ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int root) {
  for (int v = 0; v < ctx.totalProcs; v++) {
    for (size_t i = 0; i < count / ctx.totalProcs; i++) {
      ((float *)ctx.hello)[v * (count / ctx.totalProcs) + i] = v;
    }
  }
  if (ctx.proc == root) {
    ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                                flagcxMemcpyHostToDevice, NULL);
  }
  if (ctx.proc == root && ctx.color == 0 && ctx.printBuffer) {
    printf("root rank is %d\n", root);
    printf("sendbuff = ");
    for (size_t i = 0; i < 10; i++) {
      printf("%f ", ((float *)ctx.hello)[i]);
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count, int root) {
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size / ctx.totalProcs,
                              flagcxMemcpyDeviceToHost, NULL);
  if (ctx.color == 0 && ctx.printBuffer) {
    printf("rank %d recvbuff = %f\n", ctx.proc, ((float *)ctx.hello)[0]);
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);
  perfWarmup(ctx, warmupFn);
  perfRootBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
