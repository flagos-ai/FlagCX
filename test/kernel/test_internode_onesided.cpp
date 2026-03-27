/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Benchmark for FlagCX inter-node one-sided AlltoAll using Device API.
 *
 * Tests one-sided put + waitSignal + flush (NCCL GIN AlltoAll pattern):
 *   All ranks: put data to all peers, signal each peer, waitSignal, flush.
 *
 * Works with N MPI ranks (N >= 2).
 *
 * Usage: mpirun -np N ./test_kernel_internode_onesided [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>     -p <printbuffer 0/1>
 *   -R <regMode>   0=raw(deviceMalloc), 1=IPC(flagcxMemAlloc+CommRegister),
 *                  2=window(flagcxMemAlloc+CommWindowRegister)
 *   One-sided ops require -R 1 or -R 2.
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <algorithm>
#include <cstring>
#include <unistd.h>

#define DATATYPE flagcxFloat

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();
  int printBuffer = args.isPrintBuffer();
  uint64_t splitMask = args.getSplitMask();
  int localRegister = args.getLocalRegister();

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

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, uniqueId, proc));

  if (totalProcs < 2) {
    if (proc == 0)
      printf("test_kernel_internode_onesided requires at least 2 ranks.\n");
    FLAGCXCHECK(flagcxCommDestroy(comm));
    FLAGCXCHECK(flagcxHandleFree(handler));
    MPI_Finalize();
    return 0;
  }

  if (localRegister == 0) {
    if (proc == 0)
      printf("One-sided ops require -R 1 or -R 2. Skipping.\n");
    FLAGCXCHECK(flagcxCommDestroy(comm));
    FLAGCXCHECK(flagcxHandleFree(handler));
    MPI_Finalize();
    return 0;
  }

  // AlltoAll buffer layout: [rank0_data][rank1_data]...[rankN_data]
  // Each chunk has countPerPeer elements (= maxBytes / nRanks / sizeof(float))
  // Total buffer size = maxBytes (contains data for all peers)

  void *sendBuff = nullptr, *recvBuff = nullptr;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;

  if (localRegister == 0) {
    FLAGCXCHECK(
        devHandle->deviceMalloc(&sendBuff, maxBytes, flagcxMemDevice, NULL));
    FLAGCXCHECK(
        devHandle->deviceMalloc(&recvBuff, maxBytes, flagcxMemDevice, NULL));
  } else {
    FLAGCXCHECK(flagcxMemAlloc(&sendBuff, maxBytes));
    FLAGCXCHECK(flagcxMemAlloc(&recvBuff, maxBytes));
  }

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC));
    FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, maxBytes, &recvWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxCommRegister(comm, sendBuff, maxBytes, &sendHandle));
    FLAGCXCHECK(flagcxCommRegister(comm, recvBuff, maxBytes, &recvHandle));
  }

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Host buffer for data preparation and verification
  void *hostBuff = malloc(maxBytes);
  memset(hostBuff, 0, maxBytes);

  // Create device communicator with inter-node barrier + signal
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 1;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create device memory handles
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  FLAGCXCHECK(flagcxDevMemCreate(comm, recvBuff, maxBytes, recvWin, &recvMem));

  if (proc == 0 && color == 0) {
    printf("# FlagCX Device API Inter-node One-sided AlltoAll Benchmark\n");
    printf("# nRanks: %d, regMode: %s\n", totalProcs,
           localRegister == 2   ? "window"
           : localRegister == 1 ? "ipc"
                                : "raw (no registration)");
    printf("# %-12s %-14s %-14s %-8s\n", "Size(B)", "Time(us)", "BW(GB/s)",
           "Result");
  }

  // Warm-up
  for (int i = 0; i < numWarmupIters; i++) {
    size_t countPerPeer =
        std::max((size_t)1, maxBytes / sizeof(float) / totalProcs);
    FLAGCXCHECK(flagcxInterOneSidedAlltoAll(sendMem, recvMem, countPerPeer,
                                            DATATYPE, devComm, stream));
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  // Benchmark loop
  timer tim;
  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    size_t countPerPeer = size / sizeof(float) / totalProcs;
    if (countPerPeer == 0)
      countPerPeer = 1;

    // Initialize sendbuff: sendbuff[r * countPerPeer + i] =
    //   proc * 1000 + r * 100 + i
    // After alltoall: recvbuff[src * countPerPeer + i] =
    //   src * 1000 + proc * 100 + i
    float *hostFloat = (float *)hostBuff;
    for (int r = 0; r < totalProcs; r++) {
      for (size_t i = 0; i < countPerPeer; i++) {
        hostFloat[r * countPerPeer + i] =
            (float)(proc * 1000 + r * 100 + (int)i);
      }
    }
    FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostBuff, size,
                                        flagcxMemcpyHostToDevice, NULL));
    // Clear recvbuff
    FLAGCXCHECK(
        devHandle->deviceMemset(recvBuff, 0, size, flagcxMemDevice, NULL));

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      FLAGCXCHECK(flagcxInterOneSidedAlltoAll(sendMem, recvMem, countPerPeer,
                                              DATATYPE, devComm, stream));
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    double elapsedTime = tim.elapsed() / numIters;

    // Verify correctness
    memset(hostBuff, 0, size);
    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, size,
                                        flagcxMemcpyDeviceToHost, NULL));
    hostFloat = (float *)hostBuff;
    bool correct = true;
    for (int src = 0; src < totalProcs && correct; src++) {
      for (size_t i = 0; i < countPerPeer && correct; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (hostFloat[src * countPerPeer + i] != expected) {
          correct = false;
          if (proc == 0) {
            printf("  MISMATCH rank%d recvbuff[%d*%zu+%zu]: got %.0f expected "
                   "%.0f\n",
                   proc, src, countPerPeer, i,
                   hostFloat[src * countPerPeer + i], expected);
          }
        }
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;
    double bandwidth = (double)size / 1.0e9 / elapsedTime;

    if (proc == 0 && color == 0) {
      printf("  %-12zu %-14.3lf %-14.3lf %-8s\n", size, elapsedTime * 1e6,
             bandwidth, correct ? "PASS" : "FAIL");
    }

    if (printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d sendbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        float *sendFloat = (float *)hostBuff;
        // Re-read sendbuff for display
        devHandle->deviceMemcpy(hostBuff, sendBuff, size,
                                flagcxMemcpyDeviceToHost, NULL);
        sendFloat = (float *)hostBuff;
        printf(" %.0f", sendFloat[p * countPerPeer]);
      }
      printf("\n");
      // Re-read recvbuff for display
      devHandle->deviceMemcpy(hostBuff, recvBuff, size,
                              flagcxMemcpyDeviceToHost, NULL);
      hostFloat = (float *)hostBuff;
      printf("rank%d recvbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", hostFloat[p * countPerPeer]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Destroy stream first (sync any pending work)
  FLAGCXCHECK(devHandle->streamDestroy(stream));

  // Destroy device memory handles
  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));

  // Destroy device communicator (before comm destroy)
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));

  // Deregister buffer (before comm destroy)
  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin));
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxCommDeregister(comm, sendHandle));
    FLAGCXCHECK(flagcxCommDeregister(comm, recvHandle));
  }

  // Destroy comm to stop kernel proxy thread BEFORE freeing device memory
  FLAGCXCHECK(flagcxCommDestroy(comm));

  // Free buffer
  if (localRegister >= 1) {
    FLAGCXCHECK(flagcxMemFree(sendBuff));
    FLAGCXCHECK(flagcxMemFree(recvBuff));
  } else if (localRegister == 0) {
    FLAGCXCHECK(devHandle->deviceFree(sendBuff, flagcxMemDevice, NULL));
    FLAGCXCHECK(devHandle->deviceFree(recvBuff, flagcxMemDevice, NULL));
  }
  free(hostBuff);
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}
