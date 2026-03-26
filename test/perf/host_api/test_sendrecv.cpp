#include "perf_common.h"

static void collFn(PerfContext &ctx, size_t count) {
  int recvPeer = (ctx.proc - 1 + ctx.totalProcs) % ctx.totalProcs;
  int sendPeer = (ctx.proc + 1) % ctx.totalProcs;
  flagcxGroupStart(ctx.comm);
  flagcxSend(ctx.sendbuff, count, flagcxFloat, sendPeer, ctx.comm, ctx.stream);
  flagcxRecv(ctx.recvbuff, count, flagcxFloat, recvPeer, ctx.comm, ctx.stream);
  flagcxGroupEnd(ctx.comm);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  strcpy((char *)ctx.hello, "_0x1234");
  strcpy((char *)ctx.hello + size / 3, "_0x5678");
  strcpy((char *)ctx.hello + size / 3 * 2, "_0x9abc");
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
  if (ctx.proc == 0 && ctx.color == 0 && ctx.printBuffer) {
    printf("sendbuff = ");
    printf("%s", (const char *)((char *)ctx.hello));
    printf("%s", (const char *)((char *)ctx.hello + size / 3));
    printf("%s\n", (const char *)((char *)ctx.hello + size / 3 * 2));
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
  if (ctx.proc == 0 && ctx.color == 0 && ctx.printBuffer) {
    printf("recvbuff = ");
    printf("%s", (const char *)((char *)ctx.hello));
    printf("%s", (const char *)((char *)ctx.hello + size / 3));
    printf("%s\n", (const char *)((char *)ctx.hello + size / 3 * 2));
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, nullptr, dataInitFn, postIterFn);
  perfTeardown(ctx);
  return 0;
}
