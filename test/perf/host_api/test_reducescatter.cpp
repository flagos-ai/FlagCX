#include "perf_common.h"

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  sBuf = ctx.maxBytes;
  rBuf = ctx.maxBytes / ctx.totalProcs;
}

static void collFn(PerfContext &ctx, size_t count) {
  flagcxReduceScatter(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs,
                      flagcxFloat, flagcxSum, ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  size_t recvcount = count / ctx.totalProcs;
  size_t index = 0;
  float value = 0.0;
  for (size_t i = 0; i < count; i++) {
    ((float *)ctx.hello)[i] = value;
    if (index == recvcount - 1) {
      index = 0;
      value += 1.0;
    } else {
      index++;
    }
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, ctx.stream);
  ctx.devHandle->streamSynchronize(ctx.stream);
  if (ctx.color == 0 && ctx.printBuffer) {
    printf("proc %d sendbuff = ", ctx.proc);
    for (size_t i = ctx.proc * recvcount; i < ctx.proc * recvcount + 10; i++) {
      printf("%f ", ((float *)ctx.hello)[i]);
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  size_t recvcount = count / ctx.totalProcs;
  size_t recvsize = size / ctx.totalProcs;
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, recvsize,
                              flagcxMemcpyDeviceToHost, ctx.stream);
  ctx.devHandle->streamSynchronize(ctx.stream);
  if (ctx.color == 0 && ctx.printBuffer) {
    printf("proc %d recvbuff = ", ctx.proc);
    for (size_t i = 0; i < 10; i++) {
      printf("%f ", ((float *)ctx.hello)[i]);
    }
    printf("\n");
    int correct = 1;
    for (size_t i = 0; i < recvcount; i++) {
      if (((float *)ctx.hello)[i] != (float)(ctx.proc) * ctx.totalProcs) {
        correct = 0;
        printf("rank %d offset %lu wrong output %f\n", ctx.proc, i,
               ((float *)ctx.hello)[i]);
        break;
      }
    }
    printf("rank %d correctness = %d\n", ctx.proc, correct);
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
