#include "perf_common.h"
#include <cstdio>

void perfSetup(PerfContext &ctx, int argc, char **argv,
               size_t sendBufSize, size_t recvBufSize) {
  // Parse arguments
  ctx.args = new parser(argc, argv);
  ctx.minBytes = ctx.args->getMinBytes();
  ctx.maxBytes = ctx.args->getMaxBytes();
  ctx.stepFactor = ctx.args->getStepFactor();
  ctx.numWarmupIters = ctx.args->getWarmupIters();
  ctx.numIters = ctx.args->getTestIters();
  ctx.printBuffer = ctx.args->isPrintBuffer();
  ctx.root = ctx.args->getRootRank();
  ctx.splitMask = ctx.args->getSplitMask();
  ctx.localRegister = ctx.args->getLocalRegister();

  // Initialize FlagCX handles
  flagcxHandleInit(&ctx.handler);
  ctx.comm = ctx.handler->comm;
  ctx.devHandle = ctx.handler->devHandle;

  // Initialize MPI environment
  ctx.color = 0;
  ctx.worldSize = 1;
  ctx.worldRank = 0;
  ctx.totalProcs = 1;
  ctx.proc = 0;
  initMpiEnv(argc, argv, ctx.worldRank, ctx.worldSize, ctx.proc,
             ctx.totalProcs, ctx.color, ctx.splitComm, ctx.splitMask);

  // Adjust root for totalProcs
  if (ctx.root >= 0)
    ctx.root = ctx.root % ctx.totalProcs;

  // GPU setup
  int nGpu;
  ctx.devHandle->getDeviceCount(&nGpu);
  ctx.devHandle->setDevice(ctx.worldRank % nGpu);

  // Create and broadcast uniqueId
  flagcxUniqueId_t &uniqueId = ctx.handler->uniqueId;
  if (ctx.proc == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            ctx.splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  // Initialize communicator
  flagcxCommInitRank(&ctx.handler->comm, ctx.totalProcs, uniqueId, ctx.proc);
  ctx.comm = ctx.handler->comm;

  // Create stream
  ctx.devHandle->streamCreate(&ctx.stream);

  // Buffer sizes (default to maxBytes)
  size_t sBufSize = sendBufSize > 0 ? sendBufSize : ctx.maxBytes;
  size_t rBufSize = recvBufSize > 0 ? recvBufSize : ctx.maxBytes;
  size_t hBufSize = ctx.maxBytes; // host buffer always maxBytes

  // Allocate buffers
  ctx.sendbuff = nullptr;
  ctx.recvbuff = nullptr;
  ctx.sendHandle = nullptr;
  ctx.recvHandle = nullptr;

  if (ctx.localRegister) {
    flagcxMemAlloc(&ctx.sendbuff, sBufSize);
    flagcxMemAlloc(&ctx.recvbuff, rBufSize);
    flagcxCommRegister(ctx.comm, ctx.sendbuff, sBufSize, &ctx.sendHandle);
    flagcxCommRegister(ctx.comm, ctx.recvbuff, rBufSize, &ctx.recvHandle);
  } else {
    ctx.devHandle->deviceMalloc(&ctx.sendbuff, sBufSize, flagcxMemDevice, NULL);
    ctx.devHandle->deviceMalloc(&ctx.recvbuff, rBufSize, flagcxMemDevice, NULL);
  }
  ctx.hello = malloc(hBufSize);
  memset(ctx.hello, 0, hBufSize);
}

void perfTeardown(PerfContext &ctx) {
  if (ctx.localRegister) {
    flagcxCommDeregister(ctx.comm, ctx.sendHandle);
    flagcxCommDeregister(ctx.comm, ctx.recvHandle);
    flagcxMemFree(ctx.sendbuff);
    flagcxMemFree(ctx.recvbuff);
  } else {
    ctx.devHandle->deviceFree(ctx.sendbuff, flagcxMemDevice, NULL);
    ctx.devHandle->deviceFree(ctx.recvbuff, flagcxMemDevice, NULL);
  }
  free(ctx.hello);
  ctx.devHandle->streamDestroy(ctx.stream);
  flagcxCommDestroy(ctx.comm);
  flagcxHandleFree(ctx.handler);
  delete ctx.args;

  MPI_Finalize();
}

void perfWarmup(PerfContext &ctx, PerfCollFn fn) {
  // Warmup for large size
  size_t largeCount = ctx.maxBytes / sizeof(float);
  for (int i = 0; i < ctx.numWarmupIters; i++) {
    fn(ctx, largeCount);
  }
  ctx.devHandle->streamSynchronize(ctx.stream);

  // Warmup for small size
  size_t smallCount = ctx.minBytes / sizeof(float);
  for (int i = 0; i < ctx.numWarmupIters; i++) {
    fn(ctx, smallCount);
  }
  ctx.devHandle->streamSynchronize(ctx.stream);
}

void perfBenchmarkLoop(PerfContext &ctx, PerfCollFn collFn,
                       PerfBwFactorFn bwFactorFn,
                       PerfDataInitFn dataInitFn) {
  for (size_t size = ctx.minBytes; size <= ctx.maxBytes;
       size *= ctx.stepFactor) {
    size_t count = size / sizeof(float);

    // Optional data initialization
    if (dataInitFn) {
      dataInitFn(ctx, size, count);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Timed loop
    ctx.tim.reset();
    for (int i = 0; i < ctx.numIters; i++) {
      collFn(ctx, count);
    }
    ctx.devHandle->streamSynchronize(ctx.stream);

    // Compute average elapsed time across all ranks
    double elapsed_time = ctx.tim.elapsed() / ctx.numIters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed_time /= ctx.worldSize;

    // Bandwidth calculation
    double baseBw = (double)(size) / 1.0E9 / elapsed_time;
    double algBw = baseBw;
    double factor = bwFactorFn ? bwFactorFn(ctx.totalProcs) : 1.0;
    double busBw = baseBw * factor;

    if (ctx.proc == 0 && ctx.color == 0) {
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: "
             "%lf GB/s; Bus bandwidth: %lf GB/s\n",
             size, elapsed_time, algBw, busBw);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}
