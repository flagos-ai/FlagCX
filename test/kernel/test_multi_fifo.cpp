/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * test_multi_fifo: benchmark one-sided AlltoAll with varying interContextCount
 * (number of FIFO contexts). Runs with 1 context first, then 2, 4, 8, 16, 32,
 * printing latency and bandwidth for each.
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <algorithm>
#include <cstring>
#include <iostream>

#define DATATYPE flagcxFloat

static bool runAlltoAll(flagcxComm_t comm, flagcxDeviceHandle_t devHandle,
                        flagcxStream_t stream, void *sendBuff, void *recvBuff,
                        flagcxWindow_t sendWin, flagcxWindow_t recvWin,
                        void *hello, size_t minBytes, size_t maxBytes,
                        int stepFactor, int numWarmupIters, int numIters,
                        int totalProcs, int proc, int worldSize,
                        int contextCount) {
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 1;
  reqs.interContextCount = contextCount;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  FLAGCXCHECK(flagcxDevMemCreate(comm, recvBuff, maxBytes, recvWin, &recvMem));

  // Warm-up
  for (int i = 0; i < numWarmupIters; i++) {
    FLAGCXCHECK(flagcxInterOneSidedAlltoAll(
        sendMem, recvMem,
        std::max((size_t)1, maxBytes / sizeof(float) / totalProcs), DATATYPE,
        devComm, stream));
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  bool allPass = true;

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    size_t count = size / sizeof(float) / totalProcs;
    if (count == 0)
      count = 1;

    float *helloFloat = (float *)hello;
    for (int r = 0; r < totalProcs; r++)
      for (size_t i = 0; i < count; i++)
        helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
    FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hello, size,
                                        flagcxMemcpyHostToDevice, NULL));
    memset(hello, 0, size);
    FLAGCXCHECK(devHandle->deviceMemcpy(recvBuff, hello, size,
                                        flagcxMemcpyHostToDevice, NULL));

    MPI_Barrier(MPI_COMM_WORLD);

    timer tim;
    for (int i = 0; i < numIters; i++) {
      FLAGCXCHECK(flagcxInterOneSidedAlltoAll(sendMem, recvMem, count, DATATYPE,
                                              devComm, stream));
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    if (numIters == 0) {
      MPI_Barrier(MPI_COMM_WORLD);
      continue;
    }
    double elapsedTime = tim.elapsed() / numIters;

    // Verify
    memset(hello, 0, size);
    FLAGCXCHECK(devHandle->deviceMemcpy(hello, recvBuff, size,
                                        flagcxMemcpyDeviceToHost, NULL));
    helloFloat = (float *)hello;
    bool correct = true;
    for (int src = 0; src < totalProcs && correct; src++)
      for (size_t i = 0; i < count && correct; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (helloFloat[src * count + i] != expected) {
          correct = false;
          if (proc == 0)
            printf("  ctx=%d MISMATCH at recvBuff[%d*%zu+%zu]: got %.0f "
                   "expected %.0f\n",
                   contextCount, src, count, i, helloFloat[src * count + i],
                   expected);
        }
      }
    if (!correct)
      allPass = false;

    MPI_Allreduce(MPI_IN_PLACE, &elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;
    double bw = (double)size / 1.0E9 / elapsedTime;

    if (proc == 0)
      printf("  ctx=%2d  %8zu bytes  %8.3f us  %7.3f GB/s  %s\n", contextCount,
             size, elapsedTime * 1e6, bw, correct ? "PASS" : "FAIL");

    MPI_Barrier(MPI_COMM_WORLD);
  }

  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));

  return allPass;
}

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();
  uint64_t splitMask = args.getSplitMask();
  int localRegister = args.getLocalRegister();

  if (stepFactor <= 1) {
    fprintf(stderr,
            "Error: stepFactor must be > 1 to avoid infinite loop in "
            "size-sweep (got %d)\n",
            stepFactor);
    return 1;
  }

  flagcxHandlerGroup_t handler;
  FLAGCXCHECK(flagcxHandleInit(&handler));
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  if (localRegister == 0) {
    if (proc == 0)
      printf("One-sided ops require -R 1 or -R 2. Skipping.\n");
    FLAGCXCHECK(flagcxHandleFree(handler));
    MPI_Finalize();
    return 0;
  }

  // Ensure enough data for all CTAs to get work
  size_t minElemBytes = (size_t)FLAGCX_DEVICE_CTA_COUNT * 32 * sizeof(float);
  if (minBytes < minElemBytes)
    minBytes = minElemBytes;
  if (maxBytes < minBytes)
    maxBytes = minBytes;

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, uniqueId, proc));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  void *sendBuff = nullptr, *recvBuff = nullptr;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;

  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, maxBytes));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, maxBytes));

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC));
    FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, maxBytes, &recvWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC));
  } else {
    FLAGCXCHECK(flagcxCommRegister(comm, sendBuff, maxBytes, &sendHandle));
    FLAGCXCHECK(flagcxCommRegister(comm, recvBuff, maxBytes, &recvHandle));
  }

  void *hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);

  // Deduplicate context counts and cap at FLAGCX_DEVICE_CTA_COUNT
  int rawCounts[] = {1, 2, 4, 8, 16, 32};
  int ctxCounts[8];
  int nConfigs = 0;
  for (int i = 0; i < (int)(sizeof(rawCounts) / sizeof(rawCounts[0])); i++) {
    int c = rawCounts[i];
    if (c > FLAGCX_DEVICE_CTA_COUNT)
      c = FLAGCX_DEVICE_CTA_COUNT;
    bool dup = false;
    for (int j = 0; j < nConfigs; j++)
      if (ctxCounts[j] == c) {
        dup = true;
        break;
      }
    if (!dup)
      ctxCounts[nConfigs++] = c;
  }

  if (proc == 0) {
    printf("\n# Multi-FIFO benchmark: one-sided AlltoAll, %d ranks\n",
           totalProcs);
    printf("# regMode: %s\n", localRegister == 2 ? "window" : "ipc");
    printf("# Columns: contextCount, size, latency, bandwidth, correctness\n");
    printf("# FLAGCX_DEVICE_CTA_COUNT = %d\n\n", FLAGCX_DEVICE_CTA_COUNT);
  }

  bool anyFail = false;
  for (int ci = 0; ci < nConfigs; ci++) {
    if (proc == 0)
      printf("--- interContextCount = %d ---\n", ctxCounts[ci]);
    bool pass = runAlltoAll(comm, devHandle, stream, sendBuff, recvBuff,
                            sendWin, recvWin, hello, minBytes, maxBytes,
                            stepFactor, numWarmupIters, numIters, totalProcs,
                            proc, worldSize, ctxCounts[ci]);
    if (!pass)
      anyFail = true;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (proc == 0)
    printf("\nOverall: %s\n", anyFail ? "FAIL" : "PASS");

  FLAGCXCHECK(devHandle->streamDestroy(stream));

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin));
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin));
  } else {
    FLAGCXCHECK(flagcxCommDeregister(comm, sendHandle));
    FLAGCXCHECK(flagcxCommDeregister(comm, recvHandle));
  }

  FLAGCXCHECK(flagcxMemFree(sendBuff));
  FLAGCXCHECK(flagcxMemFree(recvBuff));
  free(hello);
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return anyFail ? 1 : 0;
}
