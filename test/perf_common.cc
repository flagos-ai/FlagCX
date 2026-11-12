#include "perf_common.h"
#include <cstdio>

[[noreturn]] void perfAbort(const char *message, const char *file, int line) {
  fprintf(stderr, "Perf test failure at %s:%d: %s\n", file, line, message);
  fflush(stderr);

  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  if (mpiInitialized) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  abort();
}

void perfCheck(flagcxResult_t result, const char *expression, const char *file,
               int line) {
  if (result == flagcxSuccess) {
    return;
  }

  char message[1024];
  snprintf(message, sizeof(message), "%s returned %d (%s)", expression,
           static_cast<int>(result), flagcxGetErrorString(result));
  perfAbort(message, file, line);
}

void perfSetup(PerfContext &ctx, int argc, char **argv,
               PerfBufSizeFn bufSizeFn) {
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
  ctx.verifyData = ctx.args->isVerificationEnabled();

  // Datatype/op from CLI (may be -1 for "all")
  // For warmup, default to float/sum when "all" is selected
  int parsedDt = ctx.args->getDataType();
  int parsedOp = ctx.args->getOp();
  // Table indices match enum values (test_types[i] == i), so direct cast is
  // valid
  ctx.datatype = (flagcxDataType_t)(parsedDt >= 0 ? parsedDt : flagcxFloat);
  ctx.op = (flagcxRedOp_t)(parsedOp >= 0 ? parsedOp : flagcxSum);

  // Initialize FlagCX device handle
  PERF_CHECK(flagcxDeviceHandleInit(&ctx.devHandle));

  // Initialize MPI environment
  ctx.color = 0;
  ctx.worldSize = 1;
  ctx.worldRank = 0;
  ctx.totalProcs = 1;
  ctx.proc = 0;
  initMpiEnv(argc, argv, ctx.worldRank, ctx.worldSize, ctx.proc, ctx.totalProcs,
             ctx.color, ctx.splitComm, ctx.splitMask);

  // Adjust root for totalProcs
  if (ctx.root >= 0)
    ctx.root = ctx.root % ctx.totalProcs;

  // GPU setup
  int nGpu;
  PERF_CHECK(ctx.devHandle->getDeviceCount(&nGpu));
  if (nGpu <= 0) {
    perfAbort("no accelerator devices are visible", __FILE__, __LINE__);
  }
  PERF_CHECK(ctx.devHandle->setDevice(ctx.worldRank % nGpu));

  // Create and broadcast uniqueId
  flagcxUniqueId uniqueId;
  if (ctx.proc == 0)
    PERF_CHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            ctx.splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  // Initialize communicator
  PERF_CHECK(
      flagcxCommInitRank(&ctx.comm, ctx.totalProcs, &uniqueId, ctx.proc));

  // Create stream
  PERF_CHECK(ctx.devHandle->streamCreate(&ctx.stream));

  // Buffer sizes: call bufSizeFn if provided (totalProcs is now known)
  size_t sBufSize = ctx.maxBytes;
  size_t rBufSize = ctx.maxBytes;
  if (bufSizeFn) {
    bufSizeFn(ctx, sBufSize, rBufSize);
  }
  size_t hBufSize = ctx.maxBytes; // host buffer always maxBytes

  // Allocate buffers
  ctx.sendbuff = nullptr;
  ctx.recvbuff = nullptr;
  ctx.sendHandle = nullptr;
  ctx.recvHandle = nullptr;

  if (ctx.localRegister) {
    PERF_CHECK(flagcxMemAlloc(&ctx.sendbuff, sBufSize));
    PERF_CHECK(flagcxMemAlloc(&ctx.recvbuff, rBufSize));
    PERF_CHECK(
        flagcxCommRegister(ctx.comm, ctx.sendbuff, sBufSize, &ctx.sendHandle));
    PERF_CHECK(
        flagcxCommRegister(ctx.comm, ctx.recvbuff, rBufSize, &ctx.recvHandle));
  } else {
    PERF_CHECK(ctx.devHandle->deviceMalloc(&ctx.sendbuff, sBufSize,
                                           flagcxMemDevice, NULL));
    PERF_CHECK(ctx.devHandle->deviceMalloc(&ctx.recvbuff, rBufSize,
                                           flagcxMemDevice, NULL));
  }
  ctx.hello = malloc(hBufSize);
  if (ctx.hello == nullptr) {
    perfAbort("host buffer allocation failed", __FILE__, __LINE__);
  }
  memset(ctx.hello, 0, hBufSize);

  // Zero-init device buffers to avoid UB in tests without custom dataInitFn
  if (ctx.sendbuff && ctx.recvbuff) {
    PERF_CHECK(ctx.devHandle->deviceMemset(ctx.sendbuff, 0, sBufSize,
                                           flagcxMemDevice, ctx.stream));
    PERF_CHECK(ctx.devHandle->deviceMemset(ctx.recvbuff, 0, rBufSize,
                                           flagcxMemDevice, ctx.stream));
    PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));
  }

  ctx.userData = nullptr;
}

void perfTeardown(PerfContext &ctx) {
  if (ctx.localRegister) {
    PERF_CHECK(flagcxCommDeregister(ctx.comm, ctx.sendHandle));
    PERF_CHECK(flagcxCommDeregister(ctx.comm, ctx.recvHandle));
    PERF_CHECK(flagcxMemFree(ctx.sendbuff));
    PERF_CHECK(flagcxMemFree(ctx.recvbuff));
  } else {
    PERF_CHECK(ctx.devHandle->deviceFree(ctx.sendbuff, flagcxMemDevice, NULL));
    PERF_CHECK(ctx.devHandle->deviceFree(ctx.recvbuff, flagcxMemDevice, NULL));
  }
  free(ctx.hello);
  PERF_CHECK(ctx.devHandle->streamDestroy(ctx.stream));
  PERF_CHECK(flagcxCommDestroy(ctx.comm));
  PERF_CHECK(flagcxDeviceHandleFree(ctx.devHandle));
  delete ctx.args;

  MPI_Finalize();
}

void perfWarmup(PerfContext &ctx, PerfCollFn fn) {
  size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);
  if (typeSize == 0) {
    perfAbort("unknown datatype (size=0)", __FILE__, __LINE__);
  }
  // Warmup for large size
  size_t largeCount = ctx.maxBytes / typeSize;
  for (int i = 0; i < ctx.numWarmupIters; i++) {
    fn(ctx, largeCount);
  }
  PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));

  // Warmup for small size
  size_t smallCount = ctx.minBytes / typeSize;
  for (int i = 0; i < ctx.numWarmupIters; i++) {
    fn(ctx, smallCount);
  }
  PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));
}

void perfBenchmarkLoop(PerfContext &ctx, PerfCollFn collFn,
                       PerfBwFactorFn bwFactorFn, PerfDataInitFn dataInitFn,
                       PerfPostIterFn postIterFn, bool iterateOps) {
  if (ctx.stepFactor <= 1) {
    perfAbort("stepFactor must be greater than 1", __FILE__, __LINE__);
  }

  // Determine which types and ops to run
  int parsedType = ctx.args->getDataType();
  int parsedOp = ctx.args->getOp();

  int typeCount, opCount;
  const flagcxDataType_t *runTypes;
  const flagcxRedOp_t *runOps;
  const char **runTypeNames;
  const char **runOpNames;

  flagcxDataType_t singleType = ctx.datatype;
  flagcxRedOp_t singleOp = ctx.op;
  const char *singleTypeName = test_typenames[parsedType >= 0 ? parsedType : 0];
  const char *singleOpName = test_opnames[parsedOp >= 0 ? parsedOp : 0];

  if (parsedType != -1) {
    typeCount = 1;
    runTypes = &singleType;
    runTypeNames = &singleTypeName;
  } else {
    typeCount = test_typenum;
    runTypes = test_types;
    runTypeNames = test_typenames;
  }

  if (parsedOp != -1 || !iterateOps) {
    opCount = 1;
    runOps = &singleOp;
    runOpNames = &singleOpName;
  } else {
    opCount = test_opnum;
    runOps = test_ops;
    runOpNames = test_opnames;
  }

  for (int ti = 0; ti < typeCount; ti++) {
    for (int oi = 0; oi < opCount; oi++) {
      ctx.datatype = runTypes[ti];
      ctx.op = runOps[oi];
      size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);

      if (ctx.proc == 0 && ctx.color == 0) {
        printf("#\n# datatype: %s, op: %s\n#\n", runTypeNames[ti],
               runOpNames[oi]);
      }

      for (size_t size = ctx.minBytes; size <= ctx.maxBytes;
           size *= ctx.stepFactor) {
        size_t count = size / typeSize;

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
        PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));

        // Compute average elapsed time across all ranks
        double elapsedTime = ctx.tim.elapsed() / ctx.numIters;
        MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        elapsedTime /= ctx.worldSize;

        // Bandwidth calculation
        double baseBw = (double)(size) / 1.0E9 / elapsedTime;
        double algBw = baseBw;
        double factor = bwFactorFn ? bwFactorFn(ctx.totalProcs) : 1.0;
        double busBw = baseBw * factor;

        if (ctx.proc == 0 && ctx.color == 0) {
          printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: "
                 "%lf GB/s; Bus bandwidth: %lf GB/s\n",
                 size, elapsedTime, algBw, busBw);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Optional post-iteration callback
        if (ctx.verifyData && postIterFn) {
          postIterFn(ctx, size, count);
        }
      }
    }
  }
}

void perfRootBenchmarkLoop(PerfContext &ctx, PerfRootCollFn collFn,
                           PerfBwFactorFn bwFactorFn,
                           PerfRootDataInitFn dataInitFn,
                           PerfRootPostIterFn postIterFn, bool iterateOps) {
  if (ctx.stepFactor <= 1) {
    perfAbort("stepFactor must be greater than 1", __FILE__, __LINE__);
  }

  // Determine which types and ops to run
  int parsedType = ctx.args->getDataType();
  int parsedOp = ctx.args->getOp();

  int typeCount, opCount;
  const flagcxDataType_t *runTypes;
  const flagcxRedOp_t *runOps;
  const char **runTypeNames;
  const char **runOpNames;

  flagcxDataType_t singleType = ctx.datatype;
  flagcxRedOp_t singleOp = ctx.op;
  const char *singleTypeName = test_typenames[parsedType >= 0 ? parsedType : 0];
  const char *singleOpName = test_opnames[parsedOp >= 0 ? parsedOp : 0];

  if (parsedType != -1) {
    typeCount = 1;
    runTypes = &singleType;
    runTypeNames = &singleTypeName;
  } else {
    typeCount = test_typenum;
    runTypes = test_types;
    runTypeNames = test_typenames;
  }

  if (parsedOp != -1 || !iterateOps) {
    opCount = 1;
    runOps = &singleOp;
    runOpNames = &singleOpName;
  } else {
    opCount = test_opnum;
    runOps = test_ops;
    runOpNames = test_opnames;
  }

  for (int ti = 0; ti < typeCount; ti++) {
    for (int oi = 0; oi < opCount; oi++) {
      ctx.datatype = runTypes[ti];
      ctx.op = runOps[oi];
      size_t typeSize = getFlagcxDataTypeSize(ctx.datatype);

      if (ctx.proc == 0 && ctx.color == 0) {
        printf("#\n# datatype: %s, op: %s\n#\n", runTypeNames[ti],
               runOpNames[oi]);
      }

      for (size_t size = ctx.minBytes; size <= ctx.maxBytes;
           size *= ctx.stepFactor) {
        int beginRoot, endRoot;
        double sumAlgBw = 0;
        double sumBusBw = 0;
        double sumTime = 0;
        int testCount = 0;

        if (ctx.root != -1) {
          beginRoot = endRoot = ctx.root;
        } else {
          beginRoot = 0;
          endRoot = ctx.totalProcs - 1;
        }

        for (int r = beginRoot; r <= endRoot; r++) {
          size_t count = size / typeSize;

          if (dataInitFn) {
            dataInitFn(ctx, size, count, r);
          }

          MPI_Barrier(MPI_COMM_WORLD);

          ctx.tim.reset();
          for (int i = 0; i < ctx.numIters; i++) {
            collFn(ctx, count, r);
          }
          PERF_CHECK(ctx.devHandle->streamSynchronize(ctx.stream));

          MPI_Barrier(MPI_COMM_WORLD);

          double elapsedTime = ctx.tim.elapsed() / ctx.numIters;
          MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE,
                        MPI_SUM, MPI_COMM_WORLD);
          elapsedTime /= ctx.worldSize;

          double baseBw = (double)(size) / 1.0E9 / elapsedTime;
          double algBw = baseBw;
          double factor = bwFactorFn ? bwFactorFn(ctx.totalProcs) : 1.0;
          double busBw = baseBw * factor;
          sumAlgBw += algBw;
          sumBusBw += busBw;
          sumTime += elapsedTime;
          testCount++;

          if (ctx.verifyData && postIterFn) {
            postIterFn(ctx, size, count, r);
          }
        }

        if (ctx.proc == 0 && ctx.color == 0) {
          double algBw = sumAlgBw / testCount;
          double busBw = sumBusBw / testCount;
          double elapsedTime = sumTime / testCount;
          printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: "
                 "%lf GB/s; Bus bandwidth: %lf GB/s\n",
                 size, elapsedTime, algBw, busBw);
        }
      }
    }
  }
}
