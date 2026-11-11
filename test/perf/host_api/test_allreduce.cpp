#include "perf_reduce_check.h"
#include "uni_runner_algorithm.h"

static int64_t selectedUniRunnerAlgorithm() {
  const char *value = getenv("FLAGCX_UNIRUNNER_ALGO");
  return value == NULL ? -1 : atoll(value);
}

static void collFn(PerfContext &ctx, size_t count) {
  PERF_CHECK(flagcxAllReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype,
                             ctx.redOp, ctx.comm, ctx.stream));
}

static double bwFactorFn(int totalProcs) {
  const int64_t algorithm = selectedUniRunnerAlgorithm();
  double factor = (double)(2 * (totalProcs - 1)) / (double)(totalProcs);
  if (algorithm == uniRunnerAlgorithmLocRed) {
    factor = 1;
  } else if (algorithm == uniRunnerAlgorithmRingAG) {
    factor = (double)(totalProcs - 1) / (double)(totalProcs);
  }
  return factor;
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  const int64_t algorithm = selectedUniRunnerAlgorithm();
  if (algorithm == uniRunnerAlgorithmLocRed ||
      algorithm == uniRunnerAlgorithmRingAG) {
    perfInitReductionBuffers(ctx, size, count,
                             algorithm == uniRunnerAlgorithmLocRed);
    return;
  }
  const size_t bytes = count * ctx.typeSize;
  perfReduceFillInput(static_cast<unsigned char *>(ctx.hello), count, ctx.proc,
                      ctx.totalProcs, ctx.datatype, ctx.redOp);
  PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, bytes,
                                         flagcxMemcpyHostToDevice, NULL));
  PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, size, flagcxMemDevice,
                                         NULL));
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  const int64_t algorithm = selectedUniRunnerAlgorithm();
  if (algorithm == uniRunnerAlgorithmLocRed) {
    // LocRed accumulates into recvbuff. Check one fresh invocation outside the
    // timed loop so the expected value does not depend on numIters.
    perfInitReductionBuffers(ctx, size, count, true);
    PERF_CHECK(ctx.devHandle->deviceSynchronize());
    collFn(ctx, count);
    PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));
  }
  perfPrintBuffer(ctx, count, "recvbuff");

  if (algorithm == uniRunnerAlgorithmLocRed) {
    perfCheckLocalReduction(ctx, count, "local-reduce");
  } else if (algorithm == uniRunnerAlgorithmRingAG) {
    perfCheckRingGather(ctx, count, "ring-gather");
  } else {
    const size_t bytes = count * ctx.typeSize;
    PERF_CHECK(ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, bytes,
                                           flagcxMemcpyDeviceToHost, NULL));
    perfReduceCheckOutput(static_cast<const unsigned char *>(ctx.hello), count,
                          0, ctx.proc, ctx.totalProcs, ctx.datatype, ctx.redOp,
                          "AllReduce");
    if (ctx.proc == 0)
      printf("AllReduce data verified: %zu bytes per rank\n", bytes);
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
