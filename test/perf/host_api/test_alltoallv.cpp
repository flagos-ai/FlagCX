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
                  d->hRecvcounts, d->hRdispls, ctx.datatype, ctx.comm,
                  ctx.stream);
}

static void collFn(PerfContext &ctx, size_t count) {
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  flagcxAlltoAllv(ctx.sendbuff, d->hSendcounts, d->hSdispls, ctx.recvbuff,
                  d->hRecvcounts, d->hRdispls, ctx.datatype, ctx.comm,
                  ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);
  size_t perPeer = count / ctx.totalProcs;

  memset(ctx.hello, 0, size);
  for (int i = 0; i < ctx.totalProcs; i++) {
    float val = (float)(10 * ctx.proc + i);
    size_t offset = (size_t)i * perPeer * typeSize;
    memcpy((char *)ctx.hello + offset, &val,
           sizeof(float) < typeSize ? sizeof(float) : typeSize);
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);

  computeCounts(ctx, perPeer);
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
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
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn, false);

  free(data.hSendcounts);
  free(data.hRecvcounts);
  free(data.hSdispls);
  free(data.hRdispls);
  perfTeardown(ctx);
  return 0;
}
