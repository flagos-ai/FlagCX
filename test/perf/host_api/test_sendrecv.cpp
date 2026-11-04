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

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  printf("SendRecv rank=%d send_peer=%d recv_peer=%d max_bytes=%zu\n", ctx.proc,
         (ctx.proc + 1) % ctx.totalProcs,
         (ctx.proc - 1 + ctx.totalProcs) % ctx.totalProcs, ctx.maxBytes);
  fflush(stdout);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, nullptr, nullptr, nullptr, false);
  perfTeardown(ctx);
  return 0;
}
