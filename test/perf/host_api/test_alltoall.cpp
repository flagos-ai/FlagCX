#include "perf_alltoall_check.h"
#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs,
                            ctx.datatype, ctx.comm, ctx.stream));
}

static double bwFactorFn(int totalProcs) {
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  const size_t peerBytes =
      (count / ctx.totalProcs) * getFlagcxDataTypeSize(ctx.datatype);
  unsigned char *host = static_cast<unsigned char *>(ctx.hello);
  memset(host, 0, size);
  for (int peer = 0; peer < ctx.totalProcs; ++peer)
    perfAlltoallFill(host + peer * peerBytes, peerBytes, ctx.proc, peer, size);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
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
    perfAlltoallCheck(host + peer * peerBytes, peerBytes, ctx.proc, peer,
                      peer * peerBytes, size, "AllToAll");
  perfAlltoallCheckEmpty(host + peerBytes * ctx.totalProcs,
                         size - peerBytes * ctx.totalProcs, ctx.proc,
                         peerBytes * ctx.totalProcs, "AllToAll");
  if (ctx.proc == 0)
    printf("AllToAll data verified: %zu bytes per rank\n", size);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn, false);
  perfTeardown(ctx);
  return 0;
}
