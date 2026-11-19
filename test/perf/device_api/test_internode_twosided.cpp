/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "device_api.h"
#include "device_api_window.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include <algorithm>
#include <cstring>
#include <iostream>

#if defined(FLAGCX_TEST_ALLOCATOR_SHMEM) && defined(USE_NVIDIA_ADAPTOR)
extern "C" void flagcxNvshmemSyncDeviceState();
extern "C" void flagcxNvshmemFinalizeDeviceState();
#endif

#define DATATYPE flagcxFloat

int main(int argc, char *argv[]) {
  parser args(argc, argv, 2);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();
  int printBuffer = args.isPrintBuffer();
  uint64_t splitMask = args.getSplitMask();
  int winFlags;
  if (!deviceApiWindowMode(args, &winFlags))
    return 1;

  flagcxDeviceHandle_t devHandle;
  flagcxComm_t comm;
  FLAGCXCHECK(flagcxDeviceHandleInit(&devHandle));
  flagcxUniqueId uniqueId;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  if (proc == 0)
    printf("Two-sided Device API operations are unavailable on SHMEM backends. "
           "Skipping.\n");
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));
  MPI_Finalize();
  return 0;
#endif

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, &uniqueId, proc));
  if (proc == 0 && color == 0)
    FLAGCXCHECK(deviceApiReportWindowMode(comm, winFlags));

  const flagcxMemAllocator_t memAllocator = flagcxMemCCL;

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Device communicator creation initializes the selected SHMEM runtime
  // before any symmetric allocation is attempted.
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));
#if defined(FLAGCX_TEST_ALLOCATOR_SHMEM) && defined(USE_NVIDIA_ADAPTOR)
  flagcxNvshmemSyncDeviceState();
#endif

  void *sendBuff = nullptr, *recvBuff = nullptr, *hello;
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;
  size_t count;
  timer tim;

  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, maxBytes, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, maxBytes, memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                       winFlags, memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, maxBytes, &recvWin,
                                       winFlags, memAllocator));

  hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);

  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  FLAGCXCHECK(flagcxDevMemCreate(comm, recvBuff, maxBytes, recvWin, &recvMem));

  if (proc == 0 && color == 0) {
    printf("\n# FIFO AlltoAll test (two-sided send/recv, -R %d)\n", 2);
  }

  // Warm-up
  for (int i = 0; i < numWarmupIters; i++) {
    FLAGCXCHECK(launchKernelNetTwoSidedAlltoAll(
        sendMem, recvMem,
        std::max((size_t)1, maxBytes / sizeof(float) / totalProcs), DATATYPE,
        devComm, stream));
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  for (int i = 0; i < numWarmupIters; i++) {
    FLAGCXCHECK(launchKernelNetTwoSidedAlltoAll(
        sendMem, recvMem,
        std::max((size_t)1, minBytes / sizeof(float) / totalProcs), DATATYPE,
        devComm, stream));
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    count = size / sizeof(float) / totalProcs;
    if (count == 0)
      count = 1;

    // Initialize sendBuff: sendBuff[r * count + i] = proc * 1000 + r * 100 + i
    // After alltoall: recvBuff[src * count + i] = src * 1000 + proc * 100 + i
    float *helloFloat = (float *)hello;
    for (int r = 0; r < totalProcs; r++) {
      for (size_t i = 0; i < count; i++) {
        helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
      }
    }
    FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hello, size,
                                        flagcxMemcpyHostToDevice, NULL));
    memset(hello, 0, size);
    FLAGCXCHECK(devHandle->deviceMemcpy(recvBuff, hello, size,
                                        flagcxMemcpyHostToDevice, NULL));

    if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d sendBuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      FLAGCXCHECK(launchKernelNetTwoSidedAlltoAll(sendMem, recvMem, count,
                                                  DATATYPE, devComm, stream));
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    double elapsedTime = tim.elapsed() / numIters;

    // Verify correctness
    memset(hello, 0, size);
    FLAGCXCHECK(devHandle->deviceMemcpy(hello, recvBuff, size,
                                        flagcxMemcpyDeviceToHost, NULL));
    helloFloat = (float *)hello;
    bool correct = true;
    for (int src = 0; src < totalProcs && correct; src++) {
      for (size_t i = 0; i < count && correct; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (helloFloat[src * count + i] != expected) {
          correct = false;
          if (proc == 0) {
            printf("  MISMATCH at recvBuff[%d*%zu+%zu]: got %.0f expected "
                   "%.0f\n",
                   src, count, i, helloFloat[src * count + i], expected);
          }
        }
      }
    }

    if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d recvBuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }

    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;
    double bw = (double)(size) / 1.0E9 / elapsedTime;

    if (proc == 0 && color == 0) {
      printf("FIFO AlltoAll %zu bytes; %.3lf us; %.3lf GB/s; %s\n", size,
             elapsedTime * 1e6, bw, correct ? "PASS" : "FAIL");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ==========================================================================
  // Window AlltoAll test (requires -R 2 for window registration)
  // Falls back to FIFO AlltoAll on Default path when window not available
  // ==========================================================================
  {
    flagcxDevComm_t a2aDevComm = nullptr;
    flagcxDevMem_t a2aSendMem = nullptr, a2aRecvMem = nullptr;

    if (proc == 0 && color == 0) {
      printf("\n# Window AlltoAll test (two-sided send/recv, -R 2)\n");
    }

    flagcxDevCommRequirements a2aReqs =
        FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
    a2aReqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
    FLAGCXCHECK(flagcxDevCommCreate(comm, &a2aReqs, &a2aDevComm));

    FLAGCXCHECK(
        flagcxDevMemCreate(comm, sendBuff, maxBytes, sendWin, &a2aSendMem));
    FLAGCXCHECK(
        flagcxDevMemCreate(comm, recvBuff, maxBytes, recvWin, &a2aRecvMem));

    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      count = size / sizeof(float) / totalProcs;
      if (count == 0)
        count = 1;

      // Initialize sendBuff: sendBuff[r * count + i] = proc * 1000 + r * 100 +
      // i After alltoall: recvBuff[src * count + i] = src * 1000 + proc * 100 +
      // i
      float *helloFloat = (float *)hello;
      for (int r = 0; r < totalProcs; r++) {
        for (size_t i = 0; i < count; i++) {
          helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
        }
      }
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hello, size,
                                          flagcxMemcpyHostToDevice, NULL));
      memset(hello, 0, size);
      FLAGCXCHECK(devHandle->deviceMemcpy(recvBuff, hello, size,
                                          flagcxMemcpyHostToDevice, NULL));

      if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
        printf("rank%d sendBuff:", proc);
        for (int p = 0; p < totalProcs; p++) {
          printf(" %.0f", helloFloat[p * count]);
        }
        printf("\n");
      }

      MPI_Barrier(MPI_COMM_WORLD);

      tim.reset();
      for (int i = 0; i < numIters; i++) {
        FLAGCXCHECK(launchKernelNetTwoSidedAlltoAll(
            a2aSendMem, a2aRecvMem, count, DATATYPE, a2aDevComm, stream));
      }
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      double elapsedTime = tim.elapsed() / numIters;

      // Verify correctness
      memset(hello, 0, size);
      FLAGCXCHECK(devHandle->deviceMemcpy(hello, recvBuff, size,
                                          flagcxMemcpyDeviceToHost, NULL));
      helloFloat = (float *)hello;
      bool correct = true;
      for (int src = 0; src < totalProcs && correct; src++) {
        for (size_t i = 0; i < count && correct; i++) {
          float expected = (float)(src * 1000 + proc * 100 + (int)i);
          if (helloFloat[src * count + i] != expected) {
            correct = false;
            if (proc == 0) {
              printf("  MISMATCH at recvBuff[%d*%zu+%zu]: got %.0f expected "
                     "%.0f\n",
                     src, count, i, helloFloat[src * count + i], expected);
            }
          }
        }
      }

      if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
        printf("rank%d recvBuff:", proc);
        for (int p = 0; p < totalProcs; p++) {
          printf(" %.0f", helloFloat[p * count]);
        }
        printf("\n");
      }

      MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsedTime /= worldSize;
      double bw = (double)(size) / 1.0E9 / elapsedTime;

      if (proc == 0 && color == 0) {
        printf("Window AlltoAll %zu bytes; %.3lf us; %.3lf GB/s; %s\n", size,
               elapsedTime * 1e6, bw, correct ? "PASS" : "FAIL");
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Cleanup
    FLAGCXCHECK(flagcxDevMemDestroy(comm, a2aSendMem));
    FLAGCXCHECK(flagcxDevMemDestroy(comm, a2aRecvMem));
    FLAGCXCHECK(flagcxDevCommDestroy(comm, a2aDevComm));
  }

  FLAGCXCHECK(deviceApiCloseWindows(comm, splitComm, devHandle, stream,
                                    {&sendMem, &recvMem}, {&sendWin, &recvWin},
                                    memAllocator));
  FLAGCXCHECK(flagcxMemFree(sendBuff, memAllocator));
  FLAGCXCHECK(flagcxMemFree(recvBuff, memAllocator));
  FLAGCXCHECK(devHandle->streamDestroy(stream));

  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
#if defined(FLAGCX_TEST_ALLOCATOR_SHMEM) && defined(USE_NVIDIA_ADAPTOR)
  flagcxNvshmemFinalizeDeviceState();
#endif
  FLAGCXCHECK(flagcxCommDestroy(comm));
  free(hello);
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return 0;
}
