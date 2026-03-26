#include "perf_common.h"

struct AlltoallvData {
  size_t *hSendcounts;
  size_t *hRecvcounts;
  size_t *hSdispls;
  size_t *hRdispls;
};

static void computeCounts(PerfContext &ctx, size_t perPeerCount) {
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  size_t sdis = 0, rdis = 0;
  for (int i = 0; i < ctx.totalProcs; i++) {
    if (ctx.proc % 2 == 0) {
      if (i % 2 == 0) {
        d->hSendcounts[i] = 2 * perPeerCount;
        d->hRecvcounts[i] = 2 * perPeerCount;
        d->hSdispls[i] = sdis;
        d->hRdispls[i] = rdis;
        if (i == ctx.proc) {
          d->hSendcounts[i] = 0;
          d->hRecvcounts[i] = 0;
        }
        sdis += 2 * perPeerCount;
        rdis += 2 * perPeerCount;
      } else {
        d->hSendcounts[i] = 0;
        d->hRecvcounts[i] = 0;
        d->hSdispls[i] = sdis;
        d->hRdispls[i] = rdis;
      }
    } else {
      if (i % 2 == 1) {
        d->hSendcounts[i] = 2 * perPeerCount;
        d->hRecvcounts[i] = 2 * perPeerCount;
        d->hSdispls[i] = sdis;
        d->hRdispls[i] = rdis;
        if (i == ctx.proc) {
          d->hSendcounts[i] = 0;
          d->hRecvcounts[i] = 0;
        }
        sdis += 2 * perPeerCount;
        rdis += 2 * perPeerCount;
      } else {
        d->hSendcounts[i] = 0;
        d->hRecvcounts[i] = 0;
        d->hSdispls[i] = sdis;
        d->hRdispls[i] = rdis;
      }
    }
  }
}

static void warmupFn(PerfContext &ctx, size_t count) {
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  computeCounts(ctx, count / ctx.totalProcs);
  flagcxAlltoAllv(ctx.sendbuff, d->hSendcounts, d->hSdispls, ctx.recvbuff,
                  d->hRecvcounts, d->hRdispls, flagcxFloat, ctx.comm,
                  ctx.stream);
}

static void collFn(PerfContext &ctx, size_t count) {
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  flagcxAlltoAllv(ctx.sendbuff, d->hSendcounts, d->hSdispls, ctx.recvbuff,
                  d->hRecvcounts, d->hRdispls, flagcxFloat, ctx.comm,
                  ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  size_t perPeer = count / ctx.totalProcs;

  for (int i = 0; i < ctx.totalProcs; i++) {
    ((float *)ctx.hello)[i * perPeer] = 10 * ctx.proc + i;
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);

  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("sendbuff = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%f ", ((float *)ctx.hello)[i * perPeer]);
    }
    printf("\n");
  }

  computeCounts(ctx, perPeer);

  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("hSendcounts = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%ld ", d->hSendcounts[i]);
    }
    printf("\n");
    printf("hRecvcounts = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%ld ", d->hRecvcounts[i]);
    }
    printf("\n");
    printf("hSdispls = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%ld ", d->hSdispls[i]);
    }
    printf("\n");
    printf("hRdispls = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%ld ", d->hRdispls[i]);
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  size_t perPeer = count / ctx.totalProcs;
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("recvbuff = ");
    for (int i = 0; i < ctx.totalProcs; i++) {
      printf("%f ", ((float *)ctx.hello)[i * perPeer]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);

  AlltoallvData data;
  data.hSendcounts = (size_t *)malloc(ctx.totalProcs * sizeof(size_t));
  data.hRecvcounts = (size_t *)malloc(ctx.totalProcs * sizeof(size_t));
  data.hSdispls = (size_t *)malloc(ctx.totalProcs * sizeof(size_t));
  data.hRdispls = (size_t *)malloc(ctx.totalProcs * sizeof(size_t));
  ctx.userData = &data;

  perfWarmup(ctx, warmupFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);

  free(data.hSendcounts);
  free(data.hRecvcounts);
  free(data.hSdispls);
  free(data.hRdispls);
  perfTeardown(ctx);
  return 0;
}
