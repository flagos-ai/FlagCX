#include "perf_alltoall_check.h"
#include "perf_common.h"

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  sBuf = ctx.maxBytes;
  rBuf = ctx.maxBytes / ctx.totalProcs;
}

static void warmupFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxScatter(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs,
                           ctx.datatype, 0, ctx.comm, ctx.stream));
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  PERF_CHECK(flagcxScatter(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs,
                           ctx.datatype, root, ctx.comm, ctx.stream));
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int root) {
  const size_t peerBytes =
      (count / ctx.totalProcs) * getFlagcxDataTypeSize(ctx.datatype);
  if (ctx.proc == root) {
    unsigned char *host = static_cast<unsigned char *>(ctx.hello);
    memset(host, 0, size);
    for (int peer = 0; peer < ctx.totalProcs; ++peer)
      perfDataFill(host + peer * peerBytes, peerBytes, root, peer, size);
    PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                                           flagcxMemcpyHostToDevice, NULL));
  }
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, peerBytes,
                                         flagcxMemDevice, NULL));
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count, int root) {
  const size_t peerBytes =
      (count / ctx.totalProcs) * getFlagcxDataTypeSize(ctx.datatype);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, peerBytes,
                                         flagcxMemcpyDeviceToHost, NULL));
  perfDataCheck(static_cast<const unsigned char *>(ctx.hello), peerBytes,
                ctx.proc, root, ctx.proc, 0, size, "Scatter");
  if (ctx.proc == 0)
    printf("Scatter data verified: %zu bytes, root=%d\n", size, root);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);
  perfWarmup(ctx, warmupFn);
  perfRootBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn, false);
  perfTeardown(ctx);
  return 0;
}
