/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * API correctness test for FlagCX inter-node Device API (Net path).
 *
 * Aligned 1:1 with device_ir_inter S1–S15:
 *   K1:  DevNetGetFromComm
 *   K2:  Signal/Counter Reset (resetSignal, resetCounter, readSignal, shadow)
 *   K3:  Put + SigInc (putSignalInc)
 *   K4:  Put + SigAdd (putSignalAddDecoupled)
 *   K5:  Put + SigInc + CtrInc (CounterPipeline)
 *   K6:  Put(None) + Flush + Signal (FlushDecouple)
 *   K7:  PutValue
 *   K8:  Get
 *   K9:  Signal (SigInc + SigAdd, standalone)
 *   K10: Shadow (commented — FollowShadow + MeetShadow)
 *   K11: WaitSignal + Flush (standalone)
 *   K12: Inter Barrier
 *   K13: World Barrier
 *   K14: AlltoAll (one-sided composite)
 *   K15: AlltoAll (two-sided, commented)
 *
 * Requirements:
 *   - Multi-node, OR single-node with FLAGCX_P2P_DISABLE=1
 *   - FLAGCX_USE_HETERO_COMM=1 (for DevComm setup)
 *   - -R 1 (IPC) or -R 2 (window) for symmetric memory registration
 *
 * Usage: mpirun -np N ./test_device_api_inter [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>
 *   -R <regMode>   1=IPC(flagcxMemAlloc+CommRegister)
 *                  2=window(flagcxMemAlloc+CommWindowRegister)
 *
 * Signal/counter slot assignments:
 *   slot 0: K3(SignalInc), K5(CounterPipeline), K6(FlushDecouple),
 *K14(OneSided) slot 1: K7(PutValue), K9(Signal) slot 2: K10(Shadow) counter 0:
 *K5(CounterInc)
 *
 * DevCommRequirements: interSignalCount=3, interCounterCount=1
 ************************************************************************/

#include "device_api.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define DATATYPE flagcxFloat

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

static void initSendBuff(void *sendBuff, size_t countPerPeer, int nRanks,
                         int myRank, flagcxDeviceHandle_t devHandle,
                         flagcxStream_t stream, void *hostScratch) {
  float *h = (float *)hostScratch;
  for (int r = 0; r < nRanks; r++)
    for (size_t i = 0; i < countPerPeer; i++)
      h[(size_t)r * countPerPeer + i] =
          (float)(myRank * 1000 + r * 100 + (int)i);
  devHandle->deviceMemcpy(sendBuff, hostScratch,
                          (size_t)nRanks * countPerPeer * sizeof(float),
                          flagcxMemcpyHostToDevice, stream);
  devHandle->streamSynchronize(stream);
}

static bool verifyAlltoAll(const float *buf, size_t countPerPeer, int nRanks,
                           int myRank) {
  for (int src = 0; src < nRanks; src++)
    for (size_t i = 0; i < countPerPeer; i++) {
      float expected = (float)(src * 1000 + myRank * 100 + (int)i);
      if (buf[(size_t)src * countPerPeer + i] != expected)
        return false;
    }
  return true;
}

static bool verifyCounterPipeline(const uint64_t *hResult, const float *buf,
                                  size_t countPerPeer, int nRanks) {
  uint64_t nInterRanks = hResult[1];
  if (hResult[0] != 2 * nInterRanks)
    return false;
  for (int src = 0; src < nRanks; src++)
    if (buf[(size_t)src * countPerPeer] != 999.0f)
      return false;
  return true;
}

static bool verifyPutValue(const void *buf, size_t putValBase, int nRanks,
                           int myRank) {
  const uint64_t *vals = (const uint64_t *)((const char *)buf + putValBase);
  for (int src = 0; src < nRanks; src++) {
    uint64_t expected = (uint64_t)src * 1000u + (uint64_t)myRank;
    if (vals[src] != expected)
      return false;
  }
  return true;
}

static bool verifyReset(const uint64_t *r) {
  return r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0;
}

static void printResult(const char *name, bool ok, int rank) {
  printf("[rank %d] %-30s %s\n", rank, name, ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int localRegister = args.getLocalRegister();
  uint64_t splitMask = args.getSplitMask();

  if (stepFactor <= 1) {
    printf("Error: stepFactor must be > 1, got %d\n", stepFactor);
    MPI_Finalize();
    return 1;
  }

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

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, &uniqueId, proc));

  int intraSize = comm->localRanks;
  int nNodes = 0;
  if (intraSize > 0 && totalProcs % intraSize == 0)
    nNodes = totalProcs / intraSize;
  if (nNodes < 2) {
    if (proc == 0) {
      printf("ERROR: inter test requires at least two logical nodes; "
             "totalRanks=%d intraSize=%d\n",
             totalProcs, intraSize);
    }
    FLAGCXCHECK(flagcxCommDestroy(comm));
    FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));
    MPI_Finalize();
    return 1;
  }

  if (localRegister == 0) {
    if (proc == 0)
      printf("ERROR: one-sided ops require -R 1 or -R 2.\n");
    FLAGCXCHECK(flagcxCommDestroy(comm));
    FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));
    MPI_Finalize();
    return 1;
  }

  size_t recvBuffSize = maxBytes + (size_t)totalProcs * sizeof(uint64_t);
  const size_t putValBase = maxBytes;

  void *sendBuff = nullptr, *recvBuff = nullptr;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  void *hostBuff = malloc(recvBuffSize);
  memset(hostBuff, 0, recvBuffSize);

  uint64_t *dResultBuf = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc(
      (void **)&dResultBuf, 4 * sizeof(uint64_t), flagcxMemDevice, NULL));
  uint64_t hResultBuf[4] = {};

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 3;
  reqs.interCounterCount = 1;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

#ifdef FLAGCX_COMM_TRAITS_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, maxBytes, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, recvBuffSize, memAllocator));

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC,
                                         memAllocator));
    FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, recvBuffSize, &recvWin,
                                         FLAGCX_WIN_COLL_SYMMETRIC,
                                         memAllocator));
  } else {
    FLAGCXCHECK(flagcxCommRegister(comm, sendBuff, maxBytes, &sendHandle,
                                   memAllocator));
    FLAGCXCHECK(flagcxCommRegister(comm, recvBuff, recvBuffSize, &recvHandle,
                                   memAllocator));
  }

  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  FLAGCXCHECK(
      flagcxDevMemCreate(comm, recvBuff, recvBuffSize, recvWin, &recvMem));

  if (proc == 0 && color == 0) {
    printf("# FlagCX Device API Inter-Node Test\n");
    printf("# nRanks: %d, regMode: %s\n", totalProcs,
           localRegister == 2 ? "window" : "ipc");
    printf("# K1=DevNetGetFromComm  K2=Reset  K3=PutSigInc  K4=PutSigAdd\n");
    printf("# K5=CounterPipeline  K6=FlushDecouple  K7=PutValue  K8=Get\n");
    printf("# K9=Signal  K10=Shadow(commented)  K11=WaitSignal+Flush\n");
    printf("# K12=InterBarrier  K13=WorldBarrier  K14=OneSidedAlltoAll\n");
    printf("# K15=TwoSidedAlltoAll(commented)\n#\n");
  }

  // --- K1: DevNetGetFromComm ---
  bool allPass = true;
  FLAGCXCHECK(devHandle->deviceMemset(dResultBuf, 0, 4 * sizeof(uint64_t),
                                      flagcxMemDevice, stream));
  FLAGCXCHECK(launchKernelNetGetFromComm(devComm, (int *)dResultBuf, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf, 2 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, stream));
  {
    int *k1Res = (int *)hResultBuf;
    bool k1Ok = (k1Res[0] == 1);
    allPass &= k1Ok;
    if (proc == 0 && color == 0)
      printf("  %-30s %s (intraSize=%d)\n", "K1 DevNetGetFromComm",
             k1Ok ? "PASS" : "FAIL", k1Res[1]);
  }

  // Warm-up: K3 only
  for (int i = 0; i < numWarmupIters; i++) {
    size_t cp =
        std::max((size_t)1, maxBytes / sizeof(float) / (size_t)totalProcs);
    FLAGCXCHECK(launchKernelNetPutSignalInc(sendMem, recvMem, cp, DATATYPE,
                                            devComm, stream));
  }
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  // Initial K2 reset — establishes clean signal/counter/shadow state
  FLAGCXCHECK(launchKernelNetReset(devComm, stream, dResultBuf));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Main test loop
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t countPerPeer = size / sizeof(float) / (size_t)totalProcs;
    if (countPerPeer == 0)
      countPerPeer = 1;
    size_t floatSize = (size_t)totalProcs * countPerPeer * sizeof(float);

    if (proc == 0 && color == 0)
      printf("# Size = %zu bytes, countPerPeer = %zu\n", size, countPerPeer);

    MPI_Barrier(MPI_COMM_WORLD);

    // --- K2: Reset ---
    FLAGCXCHECK(launchKernelNetReset(devComm, stream, dResultBuf));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                        4 * sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, stream));
    bool k2Ok = verifyReset(hResultBuf);
    printResult("K2 Reset", k2Ok, proc);
    allPass &= k2Ok;
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K3: Put + SigInc ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice,
                                        stream));
    FLAGCXCHECK(launchKernelNetPutSignalInc(sendMem, recvMem, countPerPeer,
                                            DATATYPE, devComm, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        flagcxMemcpyDeviceToHost, stream));
    bool k3Ok =
        verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs, proc);
    printResult("K3 PutSigInc", k3Ok, proc);
    allPass &= k3Ok;
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K4: Put + SigAdd ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice,
                                        stream));
    FLAGCXCHECK(launchKernelNetPutSignalAdd(sendMem, recvMem, countPerPeer,
                                            DATATYPE, devComm, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        flagcxMemcpyDeviceToHost, stream));
    bool k4Ok =
        verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs, proc);
    printResult("K4 PutSigAdd", k4Ok, proc);
    allPass &= k4Ok;
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K5: Put + SigInc + CtrInc (CounterPipeline) ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice,
                                        stream));
    FLAGCXCHECK(launchKernelNetCounterPipeline(
        sendMem, recvMem, countPerPeer, DATATYPE, devComm, stream, dResultBuf));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                        4 * sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        flagcxMemcpyDeviceToHost, stream));
    bool k5Ok = verifyCounterPipeline(hResultBuf, (const float *)hostBuff,
                                      countPerPeer, totalProcs);
    printResult("K5 CounterPipeline", k5Ok, proc);
    allPass &= k5Ok;
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K6: Put(None) + Flush + Signal (FlushDecouple) ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice,
                                        stream));
    FLAGCXCHECK(launchKernelNetFlushDecouple(sendMem, recvMem, countPerPeer,
                                             DATATYPE, devComm, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        flagcxMemcpyDeviceToHost, stream));
    bool k6Ok =
        verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs, proc);
    printResult("K6 FlushDecouple", k6Ok, proc);
    allPass &= k6Ok;
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K7: PutValue ---
    FLAGCXCHECK(devHandle->deviceMemset((char *)recvBuff + putValBase, 0,
                                        (size_t)totalProcs * sizeof(uint64_t),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelNetPutValue(recvMem, devComm, stream, putValBase));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy((char *)hostBuff + putValBase,
                                        (char *)recvBuff + putValBase,
                                        (size_t)totalProcs * sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, stream));
    bool k7Ok = verifyPutValue(hostBuff, putValBase, totalProcs, proc);
    printResult("K7 PutValue", k7Ok, proc);
    allPass &= k7Ok;
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K8: Get --- SKIPPED (get unsupported on vendor path)
    // initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
    //              hostBuff);
    // FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
    // flagcxMemDevice,
    //                                     stream));
    // FLAGCXCHECK(launchKernelNetGet(sendMem, recvMem, countPerPeer, DATATYPE,
    //                                devComm, stream));
    // FLAGCXCHECK(devHandle->streamSynchronize(stream));
    // FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
    //                                     flagcxMemcpyDeviceToHost, stream));
    // bool k8Ok =
    //     verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs,
    //     proc);
    // printResult("K8 Get", k8Ok, proc);
    printResult("K8 Get (SKIP)", true, proc);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K9: Signal (standalone SigInc + SigAdd) ---
    FLAGCXCHECK(launchKernelNetSignal(devComm, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    printResult("K9 Signal", true, proc); // hang-free = PASS
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K10: Shadow (commented — FollowShadow + MeetShadow) ---
    // FLAGCXCHECK(launchKernelNetFollowShadow(devComm, stream));
    // FLAGCXCHECK(devHandle->streamSynchronize(stream));
    // printResult("K10a FollowShadow", true, proc);
    // MPI_Barrier(MPI_COMM_WORLD);
    //
    // FLAGCXCHECK(launchKernelNetMeetShadow(devComm, stream));
    // FLAGCXCHECK(devHandle->streamSynchronize(stream));
    // printResult("K10b MeetShadow", true, proc);
    // MPI_Barrier(MPI_COMM_WORLD);

    // --- K11: WaitSignal + Flush (standalone) ---
    FLAGCXCHECK(launchKernelNetWaitSignalFlush(devComm, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    printResult("K11 WaitSignal+Flush", true, proc); // hang-free = PASS
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K12: Inter Barrier ---
    FLAGCXCHECK(devHandle->deviceMemset(dResultBuf, 0, 4 * sizeof(uint64_t),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(
        launchKernelInterBarrier(devComm, (int *)dResultBuf, 3, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                        sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, stream));
    {
      int barResult = (int)hResultBuf[0];
      bool k12Ok = (barResult == 1);
      printResult("K12 InterBarrier", k12Ok, proc);
      allPass &= k12Ok;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K13: World Barrier ---
    FLAGCXCHECK(devHandle->deviceMemset(dResultBuf, 0, 4 * sizeof(uint64_t),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelWorldBarrier(devComm, (int *)dResultBuf, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                        sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, stream));
    {
      bool k13Ok = ((int)hResultBuf[0] == 1);
      printResult("K13 WorldBarrier", k13Ok, proc);
      allPass &= k13Ok;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K14: AlltoAll (one-sided composite) ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice,
                                        stream));
    FLAGCXCHECK(launchKernelNetOneSidedAlltoAll(sendMem, recvMem, countPerPeer,
                                                DATATYPE, devComm, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        flagcxMemcpyDeviceToHost, stream));
    bool k14Ok =
        verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs, proc);
    printResult("K14 OneSidedAlltoAll", k14Ok, proc);
    allPass &= k14Ok;
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K15: AlltoAll (two-sided, commented) ---
    // initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
    //              hostBuff);
    // FLAGCXCHECK(
    //     devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice,
    //     NULL));
    // FLAGCXCHECK(launchKernelNetTwoSidedAlltoAll(sendMem, recvMem,
    // countPerPeer,
    //                                         DATATYPE, devComm, stream));
    // FLAGCXCHECK(devHandle->streamSynchronize(stream));
    // FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
    //                                     flagcxMemcpyDeviceToHost, NULL));
    // bool k15Ok =
    //     verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs,
    //     proc);
    // printResult("K15 TwoSidedAlltoAll", k15Ok, proc);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (proc == 0 && color == 0)
      printf("#\n");

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Summary
  MPI_Barrier(MPI_COMM_WORLD);
  int pass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&pass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  printf("[rank %d] === Overall: %s ===\n", proc, globalPass ? "PASS" : "FAIL");

  // Cleanup
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(devHandle->deviceFree(dResultBuf, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));

  if (localRegister == 2) {
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin, memAllocator));
    FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin, memAllocator));
  } else {
    FLAGCXCHECK(flagcxCommDeregister(comm, sendHandle, memAllocator));
    FLAGCXCHECK(flagcxCommDeregister(comm, recvHandle, memAllocator));
  }

  FLAGCXCHECK(flagcxMemFree(sendBuff, memAllocator));
  FLAGCXCHECK(flagcxMemFree(recvBuff, memAllocator));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  free(hostBuff);
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
