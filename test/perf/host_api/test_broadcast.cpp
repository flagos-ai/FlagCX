#include "perf_alltoall_check.h"
#include "perf_common.h"

static void warmupFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxBroadcast(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype, 0,
                             ctx.comm, ctx.stream));
}

static void collFn(PerfContext &ctx, size_t count, int root) {
  PERF_CHECK(flagcxBroadcast(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype,
                             root, ctx.comm, ctx.stream));
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count, int root) {
  (void)count;
  if (ctx.proc == root) {
    perfDataFill(static_cast<unsigned char *>(ctx.hello), size, root, root,
                 size);
    PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                                           flagcxMemcpyHostToDevice, NULL));
  }
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, size, flagcxMemDevice,
                                         NULL));
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count, int root) {
  (void)count;
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                         flagcxMemcpyDeviceToHost, NULL));
  perfDataCheck(static_cast<const unsigned char *>(ctx.hello), size, ctx.proc,
                root, root, 0, size, "Broadcast");
  if (ctx.proc == 0)
    printf("Broadcast data verified: %zu bytes per rank, root=%d\n", size,
           root);
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, warmupFn);
  perfRootBenchmarkLoop(ctx, collFn, nullptr, dataInitFn, postIterFn, false);
  perfTeardown(ctx);
  return 0;
}
