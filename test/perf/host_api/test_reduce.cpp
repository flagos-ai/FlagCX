#include "perf_common.h"

static void warmupFn(PerfContext &ctx, size_t count) {
  flagcxReduce(ctx.sendbuff, ctx.recvbuff, count, flagcxFloat, flagcxSum, 0,
               ctx.comm, ctx.stream);
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  flagcxReduce(ctx.sendbuff, ctx.recvbuff, count, flagcxFloat, flagcxSum, root,
               ctx.comm, ctx.stream);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int root) {
  for (size_t i = 0; i < count; i++) {
    ((float *)ctx.hello)[i] = i % 10 * (1 << ctx.proc);
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
  if (ctx.proc == root && ctx.color == 0 && ctx.printBuffer) {
    printf("proc %d (root rank) sendbuff = ", root);
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
  if (ctx.proc == root && ctx.color == 0 && ctx.printBuffer) {
    printf("proc %d (root rank) recvbuff = ", root);
    for (size_t i = 0; i < 10; i++) {
      printf("%f ", ((float *)ctx.hello)[i]);
    }
    printf("\n");
    int correct = 1;
    for (size_t i = 0; i < count; i++) {
      if ((i % 10 == 0 && ((float *)ctx.hello)[i] != 0) ||
          ((float *)ctx.hello)[i] /
                  (float)(i % 10 * ((1 << ctx.totalProcs) - 1)) >
              1 + 1e-5 ||
          ((float *)ctx.hello)[i] /
                  (float)(i % 10 * ((1 << ctx.totalProcs) - 1)) <
              1 - 1e-5) {
        printf("wrong output at offset %lu, expected %f, got %f\n", i,
               (float)(i % 10 * ((1 << ctx.totalProcs) - 1)),
               ((float *)ctx.hello)[i]);
        correct = 0;
        break;
      }
    }
    printf("proc %d (root rank) correctness = %d\n", ctx.proc, correct);
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
