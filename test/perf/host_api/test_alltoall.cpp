#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs,
                 flagcxFloat, ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  for (int i = 0; i < ctx.totalProcs; i++) {
    ((float *)ctx.hello)[i * (count / ctx.totalProcs)] = 10 * ctx.proc + i;
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("sendbuff = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%f ", ((float *)ctx.hello)[i * (count / ctx.totalProcs)]);
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("recvbuff = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%f ", ((float *)ctx.hello)[i * (count / ctx.totalProcs)]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
