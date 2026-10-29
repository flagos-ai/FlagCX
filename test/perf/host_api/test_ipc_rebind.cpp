#include "perf_common.h"
#include "uni_runner_algorithm.h"

// Rotate buffers on only two ranks. All ranks must still take the same IPC
// descriptor-exchange or reuse path on each AllReduce invocation.
int main(int argc, char **argv) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  constexpr size_t count = 1024;
  constexpr size_t bytes = count * sizeof(float);
  const char *configuredAlgorithm = getenv("FLAGCX_UNIRUNNER_ALGO");
  if (ctx.totalProcs != 4 || ctx.datatype != flagcxFloat32 ||
      ctx.redOp != flagcxSum || ctx.maxBytes < bytes || ctx.localRegister ||
      configuredAlgorithm == nullptr ||
      atoll(configuredAlgorithm) != uniRunnerAlgorithmIpcRingAR)
    perfAbort("ipc_rebind requires four ranks, IPC Ring AR, float32 sum and "
              "4K buffers",
              __FILE__, __LINE__);

  void *alternateSend = nullptr;
  void *alternateRecv = nullptr;
  if (ctx.proc == 1)
    PERF_CHECK(ctx.devHandle->deviceMalloc(&alternateSend, bytes,
                                           flagcxMemDevice, NULL));
  if (ctx.proc == 0)
    PERF_CHECK(ctx.devHandle->deviceMalloc(&alternateRecv, bytes,
                                           flagcxMemDevice, NULL));

  for (int round = 0; round < 4; ++round) {
    const bool rotated = round == 1 || round == 2;
    void *send = rotated && ctx.proc == 1 ? alternateSend : ctx.sendbuff;
    void *recv = rotated && ctx.proc == 0 ? alternateRecv : ctx.recvbuff;
    for (size_t i = 0; i < count; ++i)
      static_cast<float *>(ctx.hello)[i] = static_cast<float>(ctx.proc + 1);
    PERF_CHECK(ctx.devHandle->deviceMemcpy(
        send, ctx.hello, bytes, flagcxMemcpyHostToDevice, ctx.stream));
    PERF_CHECK(ctx.devHandle->deviceMemset(recv, 0, bytes, flagcxMemDevice,
                                           ctx.stream));
    MPI_Barrier(ctx.splitComm);
    PERF_CHECK(flagcxAllReduce(send, recv, count, flagcxFloat32, flagcxSum,
                               ctx.comm, ctx.stream));
    PERF_CHECK(ctx.devHandle->deviceMemcpy(
        ctx.hello, recv, bytes, flagcxMemcpyDeviceToHost, ctx.stream));
    PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));
    for (size_t i = 0; i < count; ++i) {
      if (static_cast<float *>(ctx.hello)[i] != 10.0f) {
        fprintf(stderr, "rank %d round %d element %zu: expected 10, got %g\n",
                ctx.proc, round, i, static_cast<float *>(ctx.hello)[i]);
        ctx.accuracyFailed = true;
        break;
      }
    }
    MPI_Barrier(ctx.splitComm);
  }

  // Imported CUDA IPC mappings remain valid until communicator destruction.
  PERF_CHECK(ctx.devHandle->streamDestroy(ctx.stream));
  PERF_CHECK(flagcxCommDestroy(ctx.comm));
  PERF_CHECK(ctx.devHandle->deviceFree(ctx.sendbuff, flagcxMemDevice, NULL));
  PERF_CHECK(ctx.devHandle->deviceFree(ctx.recvbuff, flagcxMemDevice, NULL));
  if (alternateSend != nullptr)
    PERF_CHECK(ctx.devHandle->deviceFree(alternateSend, flagcxMemDevice, NULL));
  if (alternateRecv != nullptr)
    PERF_CHECK(ctx.devHandle->deviceFree(alternateRecv, flagcxMemDevice, NULL));
  free(ctx.hello);
  PERF_CHECK(flagcxDeviceHandleFree(ctx.devHandle));
  delete ctx.args;
  const int result = ctx.accuracyFailed ? 1 : 0;
  MPI_Finalize();
  return result;
}
