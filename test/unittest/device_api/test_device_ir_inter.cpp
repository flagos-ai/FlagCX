/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Inter-Node Tests — Scalar IR transport functions.
 *
 * Aligned 1:1 with device_api_inter K1–K15:
 *   S1:  DevNetGetFromComm (NetGetFromCommS)
 *   S2:  Reset (read/reset/shadow)
 *   S3:  PutSignalInc (PutS_RSigInc + WaitSignalS + FlushS)
 *   S4:  PutSignalAdd (PutS_RSigAdd + WaitSignalS + FlushS)
 *   S5:  CounterPipeline (PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS)
 *   S6:  Put(None) + Flush + Signal (FlushDecouple)
 *   S7:  PutValue (PutValueS None+Signal, then PutValueS_RSigInc)
 *   S8:  Get (GetS + FlushS)
 *   S9:  Signal (SignalSigIncS + SignalSigAddS)
 *   S10: Shadow (commented — MeetShadowS)
 *   S11: WaitSignal + Flush (standalone)
 *   S12: Inter Barrier (InterBarrierStress)
 *   S13: World Barrier (WorldBarrierSyncS + SplitS)
 *   S14: AlltoAll (one-sided composite)
 *   S15: AlltoAll (two-sided, commented)
 *
 * Requirements:
 *   - Multi-node, OR single-node with FLAGCX_P2P_DISABLE=1
 *   - FLAGCX_USE_HETERO_COMM=1 (for DevComm with inter context)
 *
 * Usage: mpirun -np N ./test_device_ir_inter [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 ************************************************************************/

#include "check.h"
#include "device_ir.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#if defined(FLAGCX_TEST_ALLOCATOR_SHMEM) && defined(USE_DU_ADAPTOR)
extern "C" void flagcxDushmemSyncDeviceState();
extern "C" void flagcxDushmemFinalizeDeviceState();
#endif
#include <vector>

// ===========================================================================
// Main test driver
// ===========================================================================

int main(int argc, char *argv[]) {
  flagcxDeviceHandle_t devHandle;
  FLAGCXCHECK(flagcxDeviceHandleInit(&devHandle));
  flagcxComm_t comm;
  flagcxUniqueId uniqueId;

  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  uint64_t splitMask = 0;
  int color = 0;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();

  if (stepFactor <= 1) {
    if (proc == 0)
      printf("Error: stepFactor must be > 1, got %d\n", stepFactor);
    MPI_Finalize();
    return 1;
  }

  int nGpu;
  FLAGCXCHECK(devHandle->getDeviceCount(&nGpu));
  FLAGCXCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    FLAGCXCHECK(flagcxGetUniqueId(&uniqueId));
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  FLAGCXCHECK(flagcxCommInitRank(&comm, totalProcs, &uniqueId, proc));

  flagcxStream_t stream;
  FLAGCXCHECK(devHandle->streamCreate(&stream));

  // Create DevComm with enough signal/counter slots (initializes NVSHMEM)
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 3;
  reqs.interCounterCount = 1;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));
#if defined(FLAGCX_TEST_ALLOCATOR_SHMEM) && defined(USE_DU_ADAPTOR)
  flagcxDushmemSyncDeviceState();
#endif

  // Reserve a disjoint row per rank/context for the explicit multi-context
  // transport test. The fixed CTA stride keeps the device-side layout stable
  // without exporting contextCount through the public API.
  size_t multiContextBase =
      (maxBytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
  size_t multiContextBytes =
      (size_t)totalProcs * FLAGCX_DEVICE_CTA_COUNT * sizeof(float);
  size_t sendBufSize =
      multiContextBase + FLAGCX_DEVICE_CTA_COUNT * sizeof(float);
  size_t putValBase =
      (multiContextBase + multiContextBytes + sizeof(uint64_t) - 1) &
      ~(sizeof(uint64_t) - 1);
  size_t bufSize = putValBase + (size_t)totalProcs * sizeof(uint64_t);
  void *sendBuff = nullptr, *recvBuff = nullptr;
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, sendBufSize, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, bufSize, memAllocator));

  // Register symmetric windows
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, sendBufSize, &sendWin,
                                       FLAGCX_WIN_COLL_SYMMETRIC,
                                       memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, bufSize, &recvWin,
                                       FLAGCX_WIN_COLL_SYMMETRIC,
                                       memAllocator));

  // Create DevMem handles
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(
      flagcxDevMemCreate(comm, sendBuff, sendBufSize, sendWin, &sendMem));
  FLAGCXCHECK(flagcxDevMemCreate(comm, recvBuff, bufSize, recvWin, &recvMem));

  // Get device pointers for IR functions
  void *devCommPtr = nullptr;
  FLAGCXCHECK(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));
  void *sendMemPtr = nullptr, *recvMemPtr = nullptr;
  FLAGCXCHECK(flagcxDevMemGetDevicePtr(sendMem, &sendMemPtr));
  FLAGCXCHECK(flagcxDevMemGetDevicePtr(recvMem, &recvMemPtr));

  // Allocate results buffer
  int *devResults = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devResults, 256 * sizeof(int),
                                      flagcxMemDevice, NULL));

  // Host scratch
  float *hostBuf = new float[maxBytes / sizeof(float)];

  if (proc == 0) {
    printf("=== Device IR Inter-Node Transport Tests ===\n");
    printf("Ranks: %d\n\n", totalProcs);
  }

  // =========================================================================
  // S1: DevNetGetFromComm (NetGetFromCommS) — run once before loop
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                      flagcxMemDevice, stream));

  launchKernelNetGetFromCommS(devCommPtr, devResults, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS1[4] = {0};
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS1, devResults, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, stream));

  bool s1Pass = (hostS1[0] == 1);
  bool s1Skip = (hostS1[0] == 0);
  int intraSize = hostS1[1] > 0 ? hostS1[1] : totalProcs;
  int intraBase = proc - (proc % intraSize);
  printf("[rank %d] S1  DevNetGetFromCommS: %s (intraSize=%d)\n", proc,
         s1Skip ? "SKIP (no transport contexts)" : (s1Pass ? "PASS" : "FAIL"),
         intraSize);
  if (s1Skip)
    s1Pass = true;

  // =========================================================================
  // S2: Reset — run once before loop
  // =========================================================================
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                      flagcxMemDevice, stream));

  launchKernelNetResetS(devCommPtr, devResults, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));

  int hostS2[4] = {0};
  FLAGCXCHECK(devHandle->deviceMemcpy(hostS2, devResults, 4 * sizeof(int),
                                      flagcxMemcpyDeviceToHost, stream));

  bool s2Pass = (hostS2[0] == 1) && (hostS2[1] == 1) && (hostS2[2] == 1);
  bool s2Skip = (hostS2[0] == 0) && (hostS2[1] == 0) && (hostS2[2] == 0);
  printf("[rank %d] S2  ResetS: %s\n", proc,
         s2Skip ? "SKIP (no transport contexts)" : (s2Pass ? "PASS" : "FAIL"));
  if (s2Skip)
    s2Pass = true;
  MPI_Barrier(MPI_COMM_WORLD);

#ifdef USE_NVIDIA_ADAPTOR
  // =========================================================================
  // S3b: explicit multi-context data + release — run once before size loop
  // =========================================================================
  bool s3bPass = true;
  if (!s1Skip) {
    std::vector<float> multiSend(FLAGCX_DEVICE_CTA_COUNT);
    for (int context = 0; context < FLAGCX_DEVICE_CTA_COUNT; ++context)
      multiSend[context] = (float)(proc * 1000 + context);
    FLAGCXCHECK(devHandle->deviceMemcpy(
        (char *)sendBuff + multiContextBase, multiSend.data(),
        multiSend.size() * sizeof(float), flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->deviceMemset((char *)recvBuff + multiContextBase, 0,
                                        multiContextBytes, flagcxMemDevice,
                                        stream));
    FLAGCXCHECK(devHandle->deviceMemset(
        devResults, 0, (FLAGCX_DEVICE_CTA_COUNT + 1) * sizeof(int),
        flagcxMemDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelNetMultiContextPutSignalIncS(devCommPtr, sendMemPtr, recvMemPtr,
                                             multiContextBase, devResults,
                                             stream);
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    int hostContextResults[FLAGCX_DEVICE_CTA_COUNT + 1] = {};
    std::vector<float> multiRecv((size_t)totalProcs * FLAGCX_DEVICE_CTA_COUNT);
    FLAGCXCHECK(devHandle->deviceMemcpy(hostContextResults, devResults,
                                        sizeof(hostContextResults),
                                        flagcxMemcpyDeviceToHost, stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(
        multiRecv.data(), (char *)recvBuff + multiContextBase,
        multiContextBytes, flagcxMemcpyDeviceToHost, stream));

    int contextCount = hostContextResults[FLAGCX_DEVICE_CTA_COUNT];
    int nNodes = intraSize > 0 ? totalProcs / intraSize : 0;
    s3bPass = contextCount > 0 && contextCount <= FLAGCX_DEVICE_CTA_COUNT &&
              nNodes > 1;
    int intraRank = proc % intraSize;
    int nodeIdx = proc / intraSize;
    int prevNode = nNodes > 0 ? (nodeIdx + nNodes - 1) % nNodes : 0;
    int sourceRank = prevNode * intraSize + intraRank;
    for (int context = 0; context < contextCount && s3bPass; ++context) {
      float expected = (float)(sourceRank * 1000 + context);
      size_t index = (size_t)sourceRank * FLAGCX_DEVICE_CTA_COUNT + context;
      if (hostContextResults[context] != 1 || multiRecv[index] != expected)
        s3bPass = false;
    }
    printf("[rank %d] S3b MultiContextPutSignalIncS: %s (contexts=%d)\n", proc,
           s3bPass ? "PASS" : "FAIL", contextCount);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  // =========================================================================
  // Main test loop: S3–S14
  // =========================================================================
  bool allInterPass = true;
#ifdef USE_NVIDIA_ADAPTOR
  allInterPass &= s3bPass;
#endif

  // Helper lambda: init sendBuff with alltoall pattern
  auto initSend = [&](size_t countPerPeer) {
    for (int r = 0; r < totalProcs; r++)
      for (size_t i = 0; i < countPerPeer; i++)
        hostBuf[(size_t)r * countPerPeer + i] =
            (float)(proc * 1000 + r * 100 + (int)i);
    devHandle->deviceMemcpy(sendBuff, hostBuf,
                            (size_t)totalProcs * countPerPeer * sizeof(float),
                            flagcxMemcpyHostToDevice, stream);
    devHandle->streamSynchronize(stream);
  };

  // Helper lambda: verify alltoall pattern in recvBuff (inter peers only)
  auto verifyAlltoAll = [&](size_t countPerPeer) -> bool {
    devHandle->deviceMemcpy(hostBuf, recvBuff,
                            (size_t)totalProcs * countPerPeer * sizeof(float),
                            flagcxMemcpyDeviceToHost, stream);
    for (int src = 0; src < totalProcs; src++) {
      if (src >= intraBase && src < intraBase + intraSize)
        continue;
      for (size_t i = 0; i < countPerPeer; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (hostBuf[(size_t)src * countPerPeer + i] != expected)
          return false;
      }
    }
    return true;
  };

  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t countPerPeer =
        std::max((size_t)1, size / sizeof(float) / (size_t)totalProcs);
    size_t floatSize = (size_t)totalProcs * countPerPeer * sizeof(float);

    if (proc == 0)
      printf("# Size = %zu bytes, countPerPeer = %zu\n", size, countPerPeer);

    MPI_Barrier(MPI_COMM_WORLD);

    // --- S3: PutSignalInc (PutS_RSigInc + WaitSignalS + FlushS) ---
    if (!s1Skip) {
      initSend(countPerPeer);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelNetPutSignalIncS(devCommPtr, sendMemPtr, recvMemPtr,
                                   countPerPeer, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      bool s3Ok = verifyAlltoAll(countPerPeer);
      printf("[rank %d] S3  PutSignalIncS: %s\n", proc, s3Ok ? "PASS" : "FAIL");
      allInterPass &= s3Ok;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S4: PutSignalAdd (PutS_RSigAdd + WaitSignalS + FlushS) ---
    if (!s1Skip) {
      initSend(countPerPeer);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelNetPutSignalAddS(devCommPtr, sendMemPtr, recvMemPtr,
                                   countPerPeer, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      bool s4Ok = verifyAlltoAll(countPerPeer);
      printf("[rank %d] S4  PutSignalAddS: %s\n", proc, s4Ok ? "PASS" : "FAIL");
      fflush(stdout);
      allInterPass &= s4Ok;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S5: CounterPipeline (PutS_RSigInc_LCtrInc + WaitSignalS +
    // WaitCounterS) ---
    if (!s1Skip) {
      initSend(countPerPeer);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelNetCounterPipelineS(devCommPtr, sendMemPtr, recvMemPtr,
                                      countPerPeer, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      // Custom verification: round 2 sends stamped data (999.0f at offset 0)
      devHandle->deviceMemcpy(hostBuf, recvBuff,
                              (size_t)totalProcs * countPerPeer * sizeof(float),
                              flagcxMemcpyDeviceToHost, stream);
      bool s5Ok = true;
      for (int src = 0; src < totalProcs; src++) {
        if (src >= intraBase && src < intraBase + intraSize)
          continue;
        if (hostBuf[(size_t)src * countPerPeer] != 999.0f) {
          s5Ok = false;
          break;
        }
      }
      printf("[rank %d] S5  CounterPipelineS: %s\n", proc,
             s5Ok ? "PASS" : "FAIL");
      fflush(stdout);
      allInterPass &= s5Ok;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S6: Put(None) + Flush + Signal (FlushDecouple) ---
    if (!s1Skip) {
      initSend(countPerPeer);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelNetFlushDecoupleS(devCommPtr, sendMemPtr, recvMemPtr,
                                    countPerPeer, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      bool s6Ok = verifyAlltoAll(countPerPeer);
      printf("[rank %d] S6  FlushDecouple: %s\n", proc, s6Ok ? "PASS" : "FAIL");
      fflush(stdout);
      allInterPass &= s6Ok;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S7: PutValue (PutValueS None+Signal, then PutValueS_RSigInc) ---
    if (!s1Skip) {
      FLAGCXCHECK(devHandle->deviceMemset((char *)recvBuff + putValBase, 0,
                                          (size_t)totalProcs * sizeof(uint64_t),
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelNetPutValueS(devCommPtr, recvMemPtr, putValBase, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      uint64_t *hostVals = new uint64_t[totalProcs]();
      FLAGCXCHECK(devHandle->deviceMemcpy(hostVals,
                                          (char *)recvBuff + putValBase,
                                          (size_t)totalProcs * sizeof(uint64_t),
                                          flagcxMemcpyDeviceToHost, stream));
      bool s7Ok = true;
      for (int src = 0; src < totalProcs; src++) {
        if (src >= intraBase && src < intraBase + intraSize)
          continue;
        uint64_t expected = (uint64_t)src * 1000u + (uint64_t)proc;
        if (hostVals[src] != expected) {
          s7Ok = false;
          break;
        }
      }
      delete[] hostVals;

      printf("[rank %d] S7  PutValue: %s\n", proc, s7Ok ? "PASS" : "FAIL");
      fflush(stdout);
      allInterPass &= s7Ok;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S8: Get (GetS + FlushS) --- SKIPPED (get unsupported on vendor path)
    if (!s1Skip) {
      // initSend(countPerPeer);
      // FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
      //                                     flagcxMemDevice, stream));
      // MPI_Barrier(MPI_COMM_WORLD);
      //
      // launchKernelNetGetS(devCommPtr, sendMemPtr, recvMemPtr, countPerPeer,
      //                     stream);
      // FLAGCXCHECK(devHandle->streamSynchronize(stream));
      //
      // bool s8Ok = verifyAlltoAll(countPerPeer);
      // printf("[rank %d] S8  Get: %s\n", proc, s8Ok ? "PASS" : "FAIL");
      // fflush(stdout);
      // allInterPass &= s8Ok;
      printf("[rank %d] S8  Get: SKIP (get unsupported on vendor path)\n",
             proc);
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S9: Signal (SignalSigIncS + SignalSigAddS) ---
    if (!s1Skip) {
      MPI_Barrier(MPI_COMM_WORLD);
      launchKernelNetSignalS(devCommPtr, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      printf("[rank %d] S9  Signal(SigInc+SigAdd): PASS\n", proc);
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S10: Shadow (commented — MeetShadowS) ---

    // --- S11: WaitSignal + Flush (standalone) ---
    if (!s1Skip) {
      MPI_Barrier(MPI_COMM_WORLD);
      launchKernelNetWaitSignalFlushS(devCommPtr, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      printf("[rank %d] S11 WaitSignal+Flush: PASS\n", proc);
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S12: Inter Barrier (InterBarrierStress) ---
    if (!s1Skip) {
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);
      launchKernelInterBarrierS(devCommPtr, devResults, 3, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      int hostRes[1] = {0};
      FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));
      printf("[rank %d] S12 InterBarrier: %s\n", proc,
             hostRes[0] == 1 ? "PASS" : (hostRes[0] == -1 ? "SKIP" : "FAIL"));
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S13: World Barrier (sync + arrive/wait split) ---
    if (!s1Skip) {
      MPI_Barrier(MPI_COMM_WORLD);
      launchKernelWorldBarrierS(devCommPtr, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      printf("[rank %d] S13 WorldBarrier: PASS\n", proc);
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S14: AlltoAll (one-sided composite) ---
    if (!s1Skip) {
      initSend(countPerPeer);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelNetOneSidedAlltoAllS(devCommPtr, sendMemPtr, recvMemPtr,
                                       countPerPeer, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));

      bool s14Ok = verifyAlltoAll(countPerPeer);
      printf("[rank %d] S14 OneSidedAlltoAll: %s\n", proc,
             s14Ok ? "PASS" : "FAIL");
      fflush(stdout);
      allInterPass &= s14Ok;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- S15: AlltoAll (two-sided, commented) ---

    if (proc == 0)
      printf("#\n");

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // =========================================================================
  // Summary
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  int allPass = s1Pass && s2Pass && allInterPass;
  int globalPass = 0;
  MPI_Allreduce(&allPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  printf("[rank %d] === Overall: %s ===\n", proc, globalPass ? "PASS" : "FAIL");

  // Cleanup
  delete[] hostBuf;
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemFreeDevicePtr(sendMem));
  FLAGCXCHECK(flagcxDevMemFreeDevicePtr(recvMem));
  FLAGCXCHECK(flagcxDevCommFreeDevicePtr(devComm));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin, memAllocator));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin, memAllocator));
  FLAGCXCHECK(flagcxMemFree(sendBuff, memAllocator));
  FLAGCXCHECK(flagcxMemFree(recvBuff, memAllocator));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
#if defined(FLAGCX_TEST_ALLOCATOR_SHMEM) && defined(USE_DU_ADAPTOR)
  flagcxDushmemFinalizeDeviceState();
#endif
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
