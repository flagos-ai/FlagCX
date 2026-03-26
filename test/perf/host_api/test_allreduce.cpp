#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  flagcxAllReduce(ctx.sendbuff, ctx.recvbuff, count, flagcxFloat, flagcxSum,
                  ctx.comm, ctx.stream);
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
  for (size_t i = 0; i < count; i++) {
    ((float *)ctx.hello)[i] = i % 10 * (1 << ctx.proc);
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
  ctx.devHandle->deviceMemcpy(ctx.recvbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
  if (ctx.color == 0 && ctx.printBuffer) {
    printf("rank %d sendbuff = ", ctx.proc);
    for (size_t i = 0; i < 10; i++) {
      printf("%f ", ((float *)ctx.hello)[i]);
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("FLAGCX_UNIRUNNER_USE_RINGAG");
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
  if (ctx.color == 0 && ctx.printBuffer) {
    printf("rank %d recvbuff = ", ctx.proc);
    for (size_t i = 0; i < 10; i++) {
      printf("%f ", ((float *)ctx.hello)[i]);
    }
    printf("\n");

    if (envLocRed != NULL && atoi(envLocRed) == 1) {
      /* red correctness check */
      int redCorrect = 1;
      for (size_t i = 0; i < count; i++) {
        if (i * ctx.totalProcs / count == (size_t)ctx.proc)
          continue;
        if (((float *)ctx.hello)[i] != (float)(i % 10 * (1 << ctx.proc))) {
          printf("rank %d wrong output at offset %lu, expected %f, got %f\n ",
                 ctx.proc, i, (float)(i % 10 * (1 << ctx.proc)),
                 ((float *)ctx.hello)[i]);
          redCorrect = 0;
          break;
        }
      }
      for (size_t i = ctx.proc * count / ctx.totalProcs;
           i < (ctx.proc + 1) * count / ctx.totalProcs; i++) {
        if (((float *)ctx.hello)[i] !=
            (float)(i % 10 * (1 << (ctx.proc + 1)))) {
          printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                 ctx.proc, i, (float)(i % 10 * (1 << (ctx.proc + 1))),
                 ((float *)ctx.hello)[i]);
          redCorrect = 0;
          break;
        }
      }
      printf("rank %d reduce correctness = %d\n", ctx.proc, redCorrect);
    } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
      /* p2p correctness check */
      int p2pCorrect = 1;
      for (size_t i = 0; i < count; i++) {
        if (((float *)ctx.hello)[i] !=
            (float)(i % 10 * (1 << (i * ctx.totalProcs / count)))) {
          printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                 ctx.proc, i,
                 (float)(i % 10 * (1 << (i * ctx.totalProcs / count))),
                 ((float *)ctx.hello)[i]);
          p2pCorrect = 0;
          break;
        }
      }
      printf("rank %d p2p correctness = %d\n", ctx.proc, p2pCorrect);
    } else {
      /* all-reduce correctness check */
      int arCorrect = 1;
      for (size_t i = 0; i < count; i++) {
        if ((i % 10 == 0 && ((float *)ctx.hello)[i] != 0) ||
            ((float *)ctx.hello)[i] /
                    (float)(i % 10 * ((1 << ctx.totalProcs) - 1)) >
                1 + 1e-5 ||
            ((float *)ctx.hello)[i] /
                    (float)(i % 10 * ((1 << ctx.totalProcs) - 1)) <
                1 - 1e-5) {
          printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                 ctx.proc, i, (float)(i % 10 * ((1 << ctx.totalProcs) - 1)),
                 ((float *)ctx.hello)[i]);
          arCorrect = 0;
          break;
        }
      }
      printf("rank %d all-reduce correctness = %d\n", ctx.proc, arCorrect);
    }
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
