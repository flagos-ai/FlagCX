#include "perf_alltoall_check.h"
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
  PERF_CHECK(flagcxAlltoAllv(ctx.sendbuff, d->hSendcounts, d->hSdispls,
                             ctx.recvbuff, d->hRecvcounts, d->hRdispls,
                             ctx.datatype, ctx.comm, ctx.stream));
}

static void collFn(PerfContext &ctx, size_t count) {
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  PERF_CHECK(flagcxAlltoAllv(ctx.sendbuff, d->hSendcounts, d->hSdispls,
                             ctx.recvbuff, d->hRecvcounts, d->hRdispls,
                             ctx.datatype, ctx.comm, ctx.stream));
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);
  size_t perPeer = count / ctx.totalProcs;
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  computeCounts(ctx, perPeer);

  memset(ctx.hello, 0, size);
  for (int i = 0; i < ctx.totalProcs; i++) {
    if (d->hSendcounts[i] == 0)
      continue;
    const size_t offset = d->hSdispls[i] * typeSize;
    const size_t bytes = d->hSendcounts[i] * typeSize;
    if (offset > size || bytes > size - offset)
      perfAbort("AllToAllv send displacement is out of bounds", __FILE__,
                __LINE__);
    perfAlltoallFill((unsigned char *)ctx.hello + offset, bytes, ctx.proc, i,
                     size);
  }
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                                         flagcxMemcpyHostToDevice, NULL));
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, size, flagcxMemDevice,
                                         NULL));
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  (void)count;
  AlltoallvData *d = (AlltoallvData *)ctx.userData;
  const size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                         flagcxMemcpyDeviceToHost, NULL));
  const unsigned char *host = (const unsigned char *)ctx.hello;
  size_t checked = 0;
  for (int peer = 0; peer < ctx.totalProcs; ++peer) {
    const size_t offset = d->hRdispls[peer] * typeSize;
    const size_t bytes = d->hRecvcounts[peer] * typeSize;
    if (offset > size || bytes > size - offset ||
        (bytes != 0 && offset < checked))
      perfAbort("AllToAllv receive displacement is out of bounds", __FILE__,
                __LINE__);
    if (offset > checked)
      perfAlltoallCheckEmpty(host + checked, offset - checked, ctx.proc,
                             checked, "AllToAllv");
    if (bytes != 0) {
      perfAlltoallCheck(host + offset, bytes, ctx.proc, peer, offset, size,
                        "AllToAllv");
      checked = offset + bytes;
    }
  }
  perfAlltoallCheckEmpty(host + checked, size - checked, ctx.proc, checked,
                         "AllToAllv");
  if (ctx.proc == 0)
    printf("AllToAllv data verified: %zu bytes per rank\n", size);
  fflush(stdout);
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
