#include "perf_alltoall_check.h"
#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  int recvPeer = (ctx.proc - 1 + ctx.totalProcs) % ctx.totalProcs;
  int sendPeer = (ctx.proc + 1) % ctx.totalProcs;
  PERF_CHECK(flagcxGroupStart(ctx.comm));
  PERF_CHECK(flagcxSend(ctx.sendbuff, count, ctx.datatype, sendPeer, ctx.comm,
                        ctx.stream));
  PERF_CHECK(flagcxRecv(ctx.recvbuff, count, ctx.datatype, recvPeer, ctx.comm,
                        ctx.stream));
  PERF_CHECK(flagcxGroupEnd(ctx.comm));
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  (void)count;
  const int sendPeer = (ctx.proc + 1) % ctx.totalProcs;
  perfDataFill(static_cast<unsigned char *>(ctx.hello), size, ctx.proc,
               sendPeer, size);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                                         flagcxMemcpyHostToDevice, NULL));
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, size, flagcxMemDevice,
                                         NULL));
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  (void)count;
  const int recvPeer = (ctx.proc - 1 + ctx.totalProcs) % ctx.totalProcs;
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                         flagcxMemcpyDeviceToHost, NULL));
  perfDataCheck(static_cast<const unsigned char *>(ctx.hello), size, ctx.proc,
                recvPeer, ctx.proc, 0, size, "SendRecv");
  if (ctx.proc == 0)
    printf("SendRecv data verified: %zu bytes per rank\n", size);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  printf("SendRecv rank=%d send_peer=%d recv_peer=%d max_bytes=%zu\n", ctx.proc,
         (ctx.proc + 1) % ctx.totalProcs,
         (ctx.proc - 1 + ctx.totalProcs) % ctx.totalProcs, ctx.maxBytes);
  fflush(stdout);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, nullptr, dataInitFn, postIterFn, false);
  perfTeardown(ctx);
  return 0;
}
