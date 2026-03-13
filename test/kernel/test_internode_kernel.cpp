#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <algorithm>
#include <cstring>
#include <iostream>

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

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  void *sendbuff = nullptr, *recvbuff = nullptr, *hello;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  size_t count;
  timer tim;

  if (localRegister == 2) {
    // Window mode: VMM alloc with comm (for flagcxCommWindowRegister later)
    FLAGCXCHECK(flagcxMemAlloc(&sendbuff, maxBytes, comm));
    FLAGCXCHECK(flagcxMemAlloc(&recvbuff, maxBytes, comm));
  } else if (localRegister == 1) {
    // Zero-copy: alloc + register for NIC RDMA access
    FLAGCXCHECK(flagcxMemAlloc(&sendbuff, maxBytes, comm));
    FLAGCXCHECK(flagcxMemAlloc(&recvbuff, maxBytes, comm));
    FLAGCXCHECK(flagcxCommRegister(comm, sendbuff, maxBytes, &sendHandle));
    FLAGCXCHECK(flagcxCommRegister(comm, recvbuff, maxBytes, &recvHandle));
  } else {
    // Unregistered
    FLAGCXCHECK(
        devHandle->deviceMalloc(&sendbuff, maxBytes, flagcxMemDevice, NULL));
    FLAGCXCHECK(
        devHandle->deviceMalloc(&recvbuff, maxBytes, flagcxMemDevice, NULL));
  }
  hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);

  // Create device communicator for AlltoAll demo
  // Request IPC barriers — bar.sync() uses signal-based intra barrier
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.fields[0] = FLAGCX_DEVICE_CTA_COUNT;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Create raw device memory handles for send/recv buffers
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(NULL, sendbuff, maxBytes, NULL, &sendMem));
  FLAGCXCHECK(flagcxDevMemCreate(NULL, recvbuff, maxBytes, NULL, &recvMem));

  if (proc == 0 && color == 0) {
    printf("\n# FIFO AlltoAll test (two-sided send/recv, -R %d)\n",
           localRegister);
  }

  // Warm-up
  for (int i = 0; i < numWarmupIters; i++) {
    FLAGCXCHECK(flagcxInterAlltoAllDemo(
        sendMem, recvMem,
        std::max((size_t)1, maxBytes / sizeof(float) / totalProcs), DATATYPE,
        devComm, stream));
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  for (int i = 0; i < numWarmupIters; i++) {
    FLAGCXCHECK(flagcxInterAlltoAllDemo(
        sendMem, recvMem,
        std::max((size_t)1, minBytes / sizeof(float) / totalProcs), DATATYPE,
        devComm, stream));
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    count = size / sizeof(float) / totalProcs;
    if (count == 0)
      count = 1;

    // Initialize sendbuff: sendbuff[r * count + i] = proc * 1000 + r * 100 + i
    // After alltoall: recvbuff[src * count + i] = src * 1000 + proc * 100 + i
    float *helloFloat = (float *)hello;
    for (int r = 0; r < totalProcs; r++) {
      for (size_t i = 0; i < count; i++) {
        helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
      }
    }
    FLAGCXCHECK(devHandle->deviceMemcpy(sendbuff, hello, size,
                                        flagcxMemcpyHostToDevice, NULL));
    memset(hello, 0, size);
    FLAGCXCHECK(devHandle->deviceMemcpy(recvbuff, hello, size,
                                        flagcxMemcpyHostToDevice, NULL));

    if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d sendbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      FLAGCXCHECK(flagcxInterAlltoAllDemo(sendMem, recvMem, count, DATATYPE,
                                          devComm, stream));
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    double elapsedTime = tim.elapsed() / numIters;

    // Verify correctness
    memset(hello, 0, size);
    FLAGCXCHECK(devHandle->deviceMemcpy(hello, recvbuff, size,
                                        flagcxMemcpyDeviceToHost, NULL));
    helloFloat = (float *)hello;
    bool correct = true;
    for (int src = 0; src < totalProcs && correct; src++) {
      for (size_t i = 0; i < count && correct; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (helloFloat[src * count + i] != expected) {
          correct = false;
          if (proc == 0) {
            printf("  MISMATCH at recvbuff[%d*%zu+%zu]: got %.0f expected "
                   "%.0f\n",
                   src, count, i, helloFloat[src * count + i], expected);
          }
        }
      }
    }

    if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d recvbuff:", proc);
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
  // Falls back to FIFO AlltoAll on Tier 2/3 when window not available
  // ==========================================================================
  if (localRegister == 2) {
    // Register windows (returns nullptr on Tier 2/3)
    flagcxWindow_t sendWin = nullptr, recvWin = nullptr;
    FLAGCXCHECK(flagcxCommWindowRegister(comm, sendbuff, maxBytes, &sendWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC));
    FLAGCXCHECK(flagcxCommWindowRegister(comm, recvbuff, maxBytes, &recvWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC));

    // Shared pointers for whichever path we take
    flagcxDevComm_t a2aDevComm = nullptr;
    flagcxDevMem_t a2aSendMem = nullptr, a2aRecvMem = nullptr;
    bool useWindow = (sendWin != nullptr && recvWin != nullptr);

    if (useWindow) {
      // Tier 1: Window AlltoAll path (one-sided put + signal)
      if (proc == 0 && color == 0) {
        printf("\n# Window AlltoAll test (one-sided devNet)\n");
      }

      // Create device communicator with barrier + signal requirements
      flagcxDevCommRequirements interReqs =
          FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
      interReqs.fields[2] = FLAGCX_DEVICE_CTA_COUNT; // barrierCount
      interReqs.fields[3] = 1;                       // signalCount
      FLAGCXCHECK(flagcxDevCommCreate(comm, &interReqs, &a2aDevComm));

      // Create window-mode device memory handles
      FLAGCXCHECK(
          flagcxDevMemCreate(comm, sendbuff, maxBytes, sendWin, &a2aSendMem));
      FLAGCXCHECK(
          flagcxDevMemCreate(comm, recvbuff, maxBytes, recvWin, &a2aRecvMem));
    } else {
      // Tier 2/3: fallback to FIFO AlltoAll (two-sided send/recv)
      if (proc == 0 && color == 0) {
        printf("\n# FIFO AlltoAll test (window not available, two-sided "
               "fallback)\n");
      }

      flagcxDevCommRequirements fallbackReqs =
          FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
      fallbackReqs.fields[0] = FLAGCX_DEVICE_CTA_COUNT;
      FLAGCXCHECK(flagcxDevCommCreate(comm, &fallbackReqs, &a2aDevComm));

      FLAGCXCHECK(
          flagcxDevMemCreate(comm, sendbuff, maxBytes, NULL, &a2aSendMem));
      FLAGCXCHECK(
          flagcxDevMemCreate(comm, recvbuff, maxBytes, NULL, &a2aRecvMem));
    }

    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      count = size / sizeof(float) / totalProcs;
      if (count == 0)
        count = 1;

      // Initialize sendbuff: sendbuff[r * count + i] = proc * 1000 + r * 100 +
      // i After alltoall: recvbuff[src * count + i] = src * 1000 + proc * 100 +
      // i
      float *helloFloat = (float *)hello;
      for (int r = 0; r < totalProcs; r++) {
        for (size_t i = 0; i < count; i++) {
          helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
        }
      }
      FLAGCXCHECK(devHandle->deviceMemcpy(sendbuff, hello, size,
                                          flagcxMemcpyHostToDevice, NULL));
      memset(hello, 0, size);
      FLAGCXCHECK(devHandle->deviceMemcpy(recvbuff, hello, size,
                                          flagcxMemcpyHostToDevice, NULL));

      if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
        printf("rank%d sendbuff:", proc);
        for (int p = 0; p < totalProcs; p++) {
          printf(" %.0f", helloFloat[p * count]);
        }
        printf("\n");
      }

      MPI_Barrier(MPI_COMM_WORLD);

      tim.reset();
      for (int i = 0; i < numIters; i++) {
        FLAGCXCHECK(flagcxInterAlltoAllDemo(a2aSendMem, a2aRecvMem, count,
                                            DATATYPE, a2aDevComm, stream));
      }
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      double elapsedTime = tim.elapsed() / numIters;

      // Verify correctness
      memset(hello, 0, size);
      FLAGCXCHECK(devHandle->deviceMemcpy(hello, recvbuff, size,
                                          flagcxMemcpyDeviceToHost, NULL));
      helloFloat = (float *)hello;
      bool correct = true;
      for (int src = 0; src < totalProcs && correct; src++) {
        for (size_t i = 0; i < count && correct; i++) {
          float expected = (float)(src * 1000 + proc * 100 + (int)i);
          if (helloFloat[src * count + i] != expected) {
            correct = false;
            if (proc == 0) {
              printf("  MISMATCH at recvbuff[%d*%zu+%zu]: got %.0f expected "
                     "%.0f\n",
                     src, count, i, helloFloat[src * count + i], expected);
            }
          }
        }
      }

      if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
        printf("rank%d recvbuff:", proc);
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
        printf("%s AlltoAll %zu bytes; %.3lf us; %.3lf GB/s; %s\n",
               useWindow ? "Window" : "FIFO", size, elapsedTime * 1e6, bw,
               correct ? "PASS" : "FAIL");
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Cleanup
    FLAGCXCHECK(flagcxDevMemDestroy(comm, a2aSendMem));
    FLAGCXCHECK(flagcxDevMemDestroy(comm, a2aRecvMem));
    FLAGCXCHECK(flagcxDevCommDestroy(comm, a2aDevComm));
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin));
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin));
  }

  // Destroy stream first (sync any pending work)
  FLAGCXCHECK(devHandle->streamDestroy(stream));

  // Destroy raw device memory handles
  FLAGCXCHECK(flagcxDevMemDestroy(NULL, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(NULL, recvMem));

  // Destroy device communicator before comm destroy
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));

  if (localRegister == 1) {
    // deregister buffer (must be done before comm destroy)
    FLAGCXCHECK(flagcxCommDeregister(comm, sendHandle));
    FLAGCXCHECK(flagcxCommDeregister(comm, recvHandle));
  }

  // Free -R 1/-R 2 buffers before comm destroy (flagcxMemFree needs comm alive)
  if (localRegister >= 1) {
    FLAGCXCHECK(flagcxMemFree(sendbuff, comm));
    FLAGCXCHECK(flagcxMemFree(recvbuff, comm));
  }

  // Destroy comm to stop kernel proxy thread BEFORE freeing device memory
  FLAGCXCHECK(flagcxCommDestroy(comm));

  if (localRegister == 0) {
    FLAGCXCHECK(devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL));
    FLAGCXCHECK(devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL));
  }
  free(hello);
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}
