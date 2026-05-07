#include "perf_common.h"

static void warmupFn(PerfContext &ctx, size_t count) {
  flagcxBroadcast(ctx.sendbuff, ctx.recvbuff, count, flagcxFloat, 0, ctx.comm,
                  ctx.stream);
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  flagcxBroadcast(ctx.sendbuff, ctx.recvbuff, count, flagcxFloat, root,
                  ctx.comm, ctx.stream);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int root) {
  for (size_t i = 0; i < count; i++) {
    ((float *)ctx.hello)[i] = ctx.proc;
  }
  if (ctx.proc == root) {
    ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                                flagcxMemcpyHostToDevice, NULL);
  }
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
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
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("recvbuff = ");
    for (size_t i = 0; i < 10; i++) {
      printf("%f ", ((float *)ctx.hello)[i]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, warmupFn);
  perfRootBenchmarkLoop(ctx, collFn, nullptr, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
