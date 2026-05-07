#include "perf_common.h"

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  sBuf = ctx.maxBytes / ctx.totalProcs;
  rBuf = ctx.maxBytes;
}

static void warmupFn(PerfContext &ctx, size_t count) {
  flagcxGather(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs, flagcxFloat,
               0, ctx.comm, ctx.stream);
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  flagcxGather(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs, flagcxFloat,
               root, ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int root) {
  ((float *)ctx.hello)[0] = ctx.proc;
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size / ctx.totalProcs,
                              flagcxMemcpyHostToDevice, NULL);
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("root rank is %d\n", root);
    printf("sendbuff = ");
    printf("%f\n", ((float *)ctx.hello)[0]);
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count, int root) {
  if (ctx.proc == root) {
    memset(ctx.hello, 0, size);
    ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                flagcxMemcpyDeviceToHost, NULL);
    if (ctx.color == 0 && ctx.printBuffer) {
      printf("recvbuff = ");
      for (int i = 0; i < ctx.totalProcs; i++) {
        printf("%f ", ((float *)ctx.hello)[i * (count / ctx.totalProcs)]);
      }
      printf("\n");
    }
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
