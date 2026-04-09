#include "perf_common.h"

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  sBuf = ctx.maxBytes / ctx.totalProcs;
  rBuf = ctx.maxBytes;
}

static void collFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxAllGather(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs,
                             ctx.datatype, ctx.comm, ctx.stream));
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)totalProcs;
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  if (ctx.datatype != flagcxFloat32)
    return;
  size_t count_per_rank = count / ctx.totalProcs;
  for (size_t i = 0; i < count_per_rank; i++) {
    ((float *)ctx.hello)[i] = (1 << ctx.proc) * (i % 10);
  }

  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size / ctx.totalProcs,
                              flagcxMemcpyHostToDevice, NULL);
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("sendbuff =");
    for (size_t i = 0; i < count_per_rank && i < 10; i++) {
      printf(" %f", ((float *)ctx.hello)[i]);
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  if (ctx.datatype != flagcxFloat32)
    return;
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);

  int correct = 1;
  size_t count_per_rank = count / ctx.totalProcs;
  for (size_t src_rank = 0; src_rank < (size_t)ctx.totalProcs; src_rank++) {
    for (size_t offset = 0; offset < count_per_rank; offset++) {
      size_t recv_offset = src_rank * count_per_rank + offset;
      float expected = (float)((1 << src_rank) * (offset % 10));
      float actual = ((float *)ctx.hello)[recv_offset];
      if (actual != expected) {
        printf("rank %d wrong output at offset %zu (src rank %zu, local "
               "offset %zu), expected %f, got %f\n",
               ctx.proc, recv_offset, src_rank, offset, expected, actual);
        correct = 0;
        break;
      }
    }
    if (!correct) {
      break;
    }
  }

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
  perfSetup(ctx, argc, argv, bufSizeFn);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn, false);
  perfTeardown(ctx);
  return 0;
}
