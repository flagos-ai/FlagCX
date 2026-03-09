/*************************************************************************
 * Intra-node kernel tests using FlagCX Device API.
 *
 * Currently tests AllReduce: each rank fills its buffer with (rank+1), then
 * AllReduce(sum) produces nRanks*(nRanks+1)/2 on every element.
 *
 * Tests performance: warmup + timed iterations over multiple message sizes.
 *
 * Usage: mpirun -np <nGPUs> ./test_intranode_kernel [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>     -p <printbuffer 0/1>
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <cassert>
#include <cmath>
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

  assert(stepFactor > 1 && "Step factor must be > 1 to avoid infinite loop "
                           "when increasing message size");

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

  // Create device communicator for custom kernel usage
  flagcxDevComm_t devComm = nullptr;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.fields[0] = FLAGCX_DEVICE_CTA_COUNT; // lsaBarrierCount
  reqs.fields[1] = 0;                       // lsaMultimem
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Allocate device buffers
  void *sendbuff, *recvbuff;
  FLAGCXCHECK(
      devHandle->deviceMalloc(&sendbuff, maxBytes, flagcxMemDevice, NULL));
  FLAGCXCHECK(
      devHandle->deviceMalloc(&recvbuff, maxBytes, flagcxMemDevice, NULL));

  // Allocate registered buffer + device memory handle
  // -R 0: Raw (cudaMalloc + implicit IPC via flagcxDevMemCreate)
  // -R 1: IPC (cudaMalloc + flagcxCommRegister + flagcxDevMemCreate)
  // -R 2: Window (VMM + flagcxCommWindowRegister + flagcxDevMemCreate)
  void *regBuff = nullptr;
  void *regHandle = nullptr;
  flagcxWindow_t win = nullptr;
  flagcxDevMem_t devMem = nullptr;
  // -R 0 uses cudaMalloc (IPC-compatible).
  // -R 1/-R 2 use flagcxMemAlloc with comm.
  if (localRegister == 0) {
    FLAGCXCHECK(
        devHandle->deviceMalloc(&regBuff, maxBytes, flagcxMemDevice, NULL));
  } else {
    FLAGCXCHECK(flagcxMemAlloc(&regBuff, maxBytes, comm));
  }
  if (localRegister == 2) {
    // Window mode (NCCL > 2.28 only; graceful fallback on Tier 2/3)
    FLAGCXCHECK(flagcxCommWindowRegister(comm, regBuff, maxBytes, &win,
                                         FLAGCX_WIN_DEFAULT));
    FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, maxBytes, win, &devMem));
  } else if (localRegister == 1) {
    // IPC mode: explicit NIC registration + implicit IPC peer exchange
    FLAGCXCHECK(flagcxCommRegister(comm, regBuff, maxBytes, &regHandle));
    FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, maxBytes, nullptr, &devMem));
  } else {
    // Raw mode: no explicit registration, implicit IPC via flagcxDevMemCreate
    FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, maxBytes, nullptr, &devMem));
  }

  // Host buffer for initialization and verification
  void *hostbuff = malloc(maxBytes);

  if (proc == 0 && color == 0) {
    printf("# FlagCX Device API Intra-node AllReduce Benchmark\n");
    printf("# nRanks: %d, regMode: %s\n", totalProcs,
           localRegister == 2   ? "window"
           : localRegister == 1 ? "ipc"
                                : "raw (implicit IPC)");
    printf("# %-12s %-14s %-14s %-14s %-8s\n", "Size(B)", "Time(us)",
           "AlgBW(GB/s)", "BusBW(GB/s)", "Correct");
  }

  // Warmup with max size
  {
    size_t count = maxBytes / sizeof(float);
    for (int i = 0; i < numWarmupIters; i++) {
      FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, sendbuff,
                                          count * sizeof(float),
                                          flagcxMemcpyDeviceToDevice, stream));
      FLAGCXCHECK(
          flagcxIntraAllReduceDemo(devMem, count, DATATYPE, devComm, stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(recvbuff, regBuff,
                                          count * sizeof(float),
                                          flagcxMemcpyDeviceToDevice, stream));
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
  }

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;
    size_t bytes = count * sizeof(float);

    // Initialize: each rank fills sendbuff with (rank + 1)
    float *hbuf = (float *)hostbuff;
    for (size_t i = 0; i < count; i++) {
      hbuf[i] = (float)(proc + 1);
    }
    FLAGCXCHECK(devHandle->deviceMemcpy(sendbuff, hostbuff, bytes,
                                        flagcxMemcpyHostToDevice, NULL));

    MPI_Barrier(MPI_COMM_WORLD);

    // Timed iterations
    timer tim;
    for (int i = 0; i < numIters; i++) {
      FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, sendbuff, bytes,
                                          flagcxMemcpyDeviceToDevice, stream));
      FLAGCXCHECK(
          flagcxIntraAllReduceDemo(devMem, count, DATATYPE, devComm, stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(recvbuff, regBuff, bytes,
                                          flagcxMemcpyDeviceToDevice, stream));
    }
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    double elapsed = tim.elapsed() / numIters;

    // Reduce elapsed time across ranks for consistent reporting
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed /= worldSize;

    // Bandwidth calculation
    // AllReduce: 2*(N-1)/N * size for bus bandwidth
    double algBW = (double)size / 1.0e9 / elapsed;
    double busFW = algBW * 2.0 * (totalProcs - 1) / (double)totalProcs;

    // Verify correctness: expected value = sum(1..nRanks) = nRanks*(nRanks+1)/2
    memset(hostbuff, 0, bytes);
    FLAGCXCHECK(devHandle->deviceMemcpy(hostbuff, recvbuff, bytes,
                                        flagcxMemcpyDeviceToHost, NULL));

    float expected = (float)(totalProcs * (totalProcs + 1)) / 2.0f;
    int correct = 1;
    for (size_t i = 0; i < count && correct; i++) {
      if (fabsf(hbuf[i] - expected) > 1e-3f) {
        correct = 0;
        if (printBuffer) {
          printf("rank%d: MISMATCH at [%zu]: got %.4f, expected %.4f\n", proc,
                 i, hbuf[i], expected);
        }
      }
    }
    // Global correctness check
    MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (proc == 0 && color == 0) {
      printf("  %-12zu %-14.2f %-14.4f %-14.4f %-8s\n", size, elapsed * 1e6,
             algBW, busFW, correct ? "PASS" : "FAIL");
    }

    if (printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d result[0..3]:", proc);
      for (size_t i = 0; i < 4 && i < count; i++) {
        printf(" %.2f", hbuf[i]);
      }
      printf(" (expected: %.2f)\n", expected);
    }
  }

  // Cleanup
  FLAGCXCHECK(flagcxDevMemDestroy(comm, devMem));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, win));
  } else if (localRegister == 1) {
    FLAGCXCHECK(flagcxCommDeregister(comm, regHandle));
  }
  if (localRegister == 0) {
    FLAGCXCHECK(devHandle->deviceFree(regBuff, flagcxMemDevice, NULL));
  } else {
    FLAGCXCHECK(flagcxMemFree(regBuff, comm));
  }
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL));
  free(hostbuff);
  FLAGCXCHECK(flagcxHandleFree(handler));

  MPI_Finalize();
  return 0;
}
