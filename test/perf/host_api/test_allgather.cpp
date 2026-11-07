#include "perf_alltoall_check.h"
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
  const size_t peerBytes =
      (count / ctx.totalProcs) * getFlagcxDataTypeSize(ctx.datatype);
  perfDataFill(static_cast<unsigned char *>(ctx.hello), peerBytes, ctx.proc, 0,
               size);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, peerBytes,
                                         flagcxMemcpyHostToDevice, NULL));
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, size, flagcxMemDevice,
                                         NULL));
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  const size_t peerBytes =
      (count / ctx.totalProcs) * getFlagcxDataTypeSize(ctx.datatype);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                         flagcxMemcpyDeviceToHost, NULL));
  const unsigned char *host = static_cast<const unsigned char *>(ctx.hello);
  for (int peer = 0; peer < ctx.totalProcs; ++peer)
    perfDataCheck(host + peer * peerBytes, peerBytes, ctx.proc, peer, 0,
                  peer * peerBytes, size, "AllGather");
  const size_t transferredBytes = peerBytes * ctx.totalProcs;
  perfAlltoallCheckEmpty(host + transferredBytes, size - transferredBytes,
                         ctx.proc, transferredBytes, "AllGather");
  if (ctx.proc == 0)
    printf("AllGather data verified: %zu bytes per rank\n", size);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn, false);
  perfTeardown(ctx);
  return 0;
}
