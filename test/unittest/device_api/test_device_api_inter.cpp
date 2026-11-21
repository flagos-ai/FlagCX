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
 *   K4b: Multi-context Put + standalone release
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
 *   - -R 2 (Window registration prepares IPC or symmetric mappings)
 *
 * Usage: mpirun -np N ./test_device_api_inter [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>
 *   -R 2          flagcxMemAlloc + CommWindowRegister
 *
 * Signal/counter slot assignments:
 *   slot 0: K3(SignalInc), K5(CounterPipeline), K6(FlushDecouple),
 *K14(OneSided) slot 1: K7(PutValue), K9(Signal) slot 2: K4b(MultiContext),
 *K10(Shadow) counter 0: K5(CounterInc)
 *
 * DevCommRequirements: interSignalCount=3, interCounterCount=1
 ************************************************************************/

#include "check.h"
#include "device_api.h"
#include "device_api_window.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "global_comm.h"
#if defined(FLAGCX_USE_XPU_NETIO)
#include "device_api/comm_traits.h"
#include "device_api/devnet/xpu_netio_gda_host.h"
#include "net_gda_test.h"
#include "utils.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unistd.h>

#define DATATYPE flagcxFloat

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

static void initSendBuff(void *sendBuff, size_t countPerPeer, int nRanks,
                         int myRank, flagcxDeviceHandle_t devHandle,
                         flagcxStream_t stream, void *hostScratch,
                         int round = 0) {
  float *h = (float *)hostScratch;
  for (int r = 0; r < nRanks; r++)
    for (size_t i = 0; i < countPerPeer; i++)
      h[(size_t)r * countPerPeer + i] =
          (float)(myRank * 1000 + r * 100 + (int)i + round * 17);
  devHandle->deviceMemcpy(sendBuff, hostScratch,
                          (size_t)nRanks * countPerPeer * sizeof(float),
                          flagcxMemcpyHostToDevice, stream);
  devHandle->streamSynchronize(stream);
}

static bool verifyAlltoAll(const float *buf, size_t countPerPeer, int nRanks,
                           int myRank, int round = 0) {
  for (int src = 0; src < nRanks; src++)
    for (size_t i = 0; i < countPerPeer; i++) {
      float expected = (float)(src * 1000 + myRank * 100 + (int)i + round * 17);
      if (buf[(size_t)src * countPerPeer + i] != expected)
        return false;
    }
  return true;
}

#ifdef USE_NVIDIA_ADAPTOR
static bool verifyInterAlltoAll(const float *buf, size_t countPerPeer,
                                int nRanks, int myRank, int intraSize) {
  int myNode = myRank / intraSize;
  for (int src = 0; src < nRanks; src++) {
    if (src / intraSize == myNode)
      continue;
    for (size_t i = 0; i < countPerPeer; i++) {
      float expected = (float)(src * 1000 + myRank * 100 + (int)i);
      if (buf[(size_t)src * countPerPeer + i] != expected)
        return false;
    }
  }
  return true;
}
#endif

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

static void printSkip(const char *name, const char *reason, int rank) {
  printf("[rank %d] %-30s SKIP (%s)\n", rank, name, reason);
}

static flagcxResult_t synchronizeInterTest(flagcxDeviceHandle_t device,
                                           flagcxStream_t stream,
                                           flagcxDevComm_t comm) {
  FLAGCXCHECK(device->streamSynchronize(stream));
#if defined(FLAGCX_USE_XPU_NETIO)
  if (comm->gdaProvider != 0)
    return XpuNetioGda::pollError(comm);
#else
  (void)comm;
#endif
  return flagcxSuccess;
}

static void printResult(const char *name, bool ok, int rank) {
  printf("[rank %d] %-30s %s\n", rank, name, ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  int winFlags;
  if (!deviceApiWindowMode(args, &winFlags))
    return 1;
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numTestIters = std::max(1, args.getTestIters());
  uint64_t splitMask = args.getSplitMask();

  if (minBytes == 0 || maxBytes < minBytes) {
    printf("Error: invalid message-size range\n");
    return 1;
  }
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
    FLAGCXCHECK(flagcxCommFinalize(comm));
    FLAGCXCHECK(flagcxCommFinalize(comm));
    FLAGCXCHECK(flagcxCommDestroy(comm));
    FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));
    MPI_Finalize();
    return 1;
  }

  if (maxBytes < static_cast<size_t>(totalProcs) * sizeof(float) ||
      maxBytes > std::numeric_limits<size_t>::max() -
                     static_cast<size_t>(totalProcs) * sizeof(uint64_t)) {
    fprintf(stderr, "Message buffers must hold at least one float per rank\n");
    MPI_Abort(splitComm, 1);
  }
  size_t recvBuffSize = maxBytes + (size_t)totalProcs * sizeof(uint64_t);
  const size_t putValBase = maxBytes;

  void *sendBuff = nullptr, *recvBuff = nullptr;
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
#if defined(FLAGCX_USE_XPU_NETIO)
  int netEnabled = 1, netType = -1;
  // The backend performs the collective configuration validation. Do not
  // return locally on malformed input before peers enter that validation.
  const bool requestGda =
      flagcxReadDevNetConfig(&netEnabled, &netType) == flagcxSuccess &&
      netEnabled != 0 && netType == 1;
  if (requestGda) {
    // The MLX5 adapter supports indexed 64-bit signals on one context.
    // Counters and multiple contexts remain explicit SKIPs.
    reqs.interContextCount = 1;
    reqs.interSignalCount = 3;
    reqs.interCounterCount = 0;
    reqs.interForceEnable = true;
  }
#endif
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  bool useGda = false;
  bool supportsSignal = true, supportsCounter = true, supportsPutValue = true;
  bool supportsMultiContext = true;
  unsigned requiredAlignment = 1;
#if defined(FLAGCX_USE_XPU_NETIO)
  size_t gdaChunkBytes = 0;
  useGda = devComm->gdaProvider != 0;
  if (requestGda && !useGda) {
    fprintf(stderr, "Explicit GDA test selected a different Net provider\n");
    MPI_Abort(splitComm, 1);
  }
  if (useGda) {
    unsigned capabilities = 0;
    uint32_t *errorFlag = nullptr;
    uint64_t maxChunkBytes = 0;
    FLAGCXCHECK(XpuNetioGda::queryComm(devComm, &capabilities, &errorFlag,
                                       &maxChunkBytes, &requiredAlignment));
    if (errorFlag == nullptr || requiredAlignment == 0 ||
        requiredAlignment > (1u << 20) ||
        (requiredAlignment & (requiredAlignment - 1)) != 0 ||
        maxChunkBytes < requiredAlignment) {
      fprintf(stderr, "Invalid GDA capability profile\n");
      MPI_Abort(splitComm, 1);
    }
    unsigned commonCapabilities = 0, commonAlignment = 0;
    MPI_Allreduce(&capabilities, &commonCapabilities, 1, MPI_UNSIGNED, MPI_BAND,
                  splitComm);
    MPI_Allreduce(&requiredAlignment, &commonAlignment, 1, MPI_UNSIGNED,
                  MPI_MAX, splitComm);
    requiredAlignment = commonAlignment;
    uint64_t commonMaxChunk = 0;
    MPI_Allreduce(&maxChunkBytes, &commonMaxChunk, 1, MPI_UINT64_T, MPI_MIN,
                  splitComm);
    supportsSignal =
        (commonCapabilities & DefaultNetGdaSignal) != 0 &&
        (commonCapabilities & DefaultNetGdaSignalCompletion) != 0 &&
        reqs.interSignalCount >= 3;
    supportsCounter =
        (commonCapabilities & DefaultNetGdaCounterCompletion) != 0 &&
        reqs.interCounterCount > 0;
    supportsPutValue = (commonCapabilities & DefaultNetGdaPutValue) != 0;
    gdaChunkBytes = static_cast<size_t>(
        std::min<uint64_t>(commonMaxChunk,
                           std::max<uint64_t>(requiredAlignment, 3072)) /
        requiredAlignment * requiredAlignment);
    supportsMultiContext = devComm->contextCount > 1;
    FLAGCXCHECK(XpuNetioGda::clearError(devComm));
  }
#endif

#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, maxBytes, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, recvBuffSize, memAllocator));

  FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                       winFlags, memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, recvBuffSize, &recvWin,
                                       winFlags, memAllocator));

  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  FLAGCXCHECK(
      flagcxDevMemCreate(comm, recvBuff, recvBuffSize, recvWin, &recvMem));

  if (proc == 0 && color == 0) {
    printf("# FlagCX Device API Inter-Node Test\n");
    printf("# nRanks: %d, regMode: window, Net provider: %s\n", totalProcs,
           useGda ? "GDA" : "backend default");
    if (useGda) {
      printf("# GDA: unsupported public operations remain explicit TODO/SKIP "
             "cases\n");
      printf("# GDA user signals=%d, counters=%d, multiple contexts=%d\n",
             supportsSignal, supportsCounter, supportsMultiContext);
    }
    FLAGCXCHECK(deviceApiReportWindowMode(comm, winFlags));
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
  FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
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

  // Warm up on a supported data path; all ranks use the same size/profile.
  const size_t warmupCount = maxBytes / sizeof(float) / (size_t)totalProcs;
  const bool warmupAligned =
      (warmupCount * sizeof(float)) % requiredAlignment == 0;
  if (supportsSignal || (useGda && warmupAligned)) {
    initSendBuff(sendBuff, warmupCount, totalProcs, proc, devHandle, stream,
                 hostBuff);
    FLAGCXCHECK(devHandle->deviceMemset(
        recvBuff, 0, (size_t)totalProcs * warmupCount * sizeof(float),
        flagcxMemDevice, stream));
    for (int i = 0; i < numWarmupIters; i++) {
      if (supportsSignal) {
        FLAGCXCHECK(launchKernelNetPutSignalInc(sendMem, recvMem, warmupCount,
                                                DATATYPE, devComm, stream));
      } else {
        FLAGCXCHECK(launchKernelNetOneSidedAlltoAll(
            sendMem, recvMem, warmupCount, DATATYPE, devComm, stream));
      }
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
    }
  }

  // Reset only the user completion plane; Barrier channels keep their rounds.
  if (supportsSignal && supportsCounter) {
    FLAGCXCHECK(launchKernelNetReset(devComm, stream, dResultBuf));
    FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
  }
  MPI_Barrier(splitComm);
  unsigned alltoallRuns = 0;

  // Main test loop
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t countPerPeer = size / sizeof(float) / (size_t)totalProcs;
    if (countPerPeer == 0)
      countPerPeer = 1;
    size_t floatSize = (size_t)totalProcs * countPerPeer * sizeof(float);

    if (proc == 0 && color == 0)
      printf("# Size = %zu bytes, countPerPeer = %zu\n", size, countPerPeer);

    MPI_Barrier(MPI_COMM_WORLD);

    if (supportsSignal && supportsCounter) {
      // --- K2: Reset ---
      FLAGCXCHECK(launchKernelNetReset(devComm, stream, dResultBuf));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                          4 * sizeof(uint64_t),
                                          flagcxMemcpyDeviceToHost, stream));
      bool k2Ok = verifyReset(hResultBuf);
      printResult("K2 Reset", k2Ok, proc);
      allPass &= k2Ok;
    } else {
      printSkip("K2 Reset", "user signal/counter operations unavailable", proc);
    }
    MPI_Barrier(splitComm);

    if (supportsSignal) {
      // --- K3: Put + SigInc ---
      initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                   hostBuff);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(launchKernelNetPutSignalInc(sendMem, recvMem, countPerPeer,
                                              DATATYPE, devComm, stream));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                          flagcxMemcpyDeviceToHost, stream));
      bool k3Ok = verifyAlltoAll((const float *)hostBuff, countPerPeer,
                                 totalProcs, proc);
      printResult("K3 PutSigInc", k3Ok, proc);
      allPass &= k3Ok;
    } else {
      printSkip("K3 PutSigInc", "public SignalInc unavailable", proc);
    }
    MPI_Barrier(splitComm);

    if (supportsSignal) {
      // --- K4: Put + SigAdd ---
      initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                   hostBuff);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(launchKernelNetPutSignalAdd(sendMem, recvMem, countPerPeer,
                                              DATATYPE, devComm, stream));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                          flagcxMemcpyDeviceToHost, stream));
      bool k4Ok = verifyAlltoAll((const float *)hostBuff, countPerPeer,
                                 totalProcs, proc);
      printResult("K4 PutSigAdd", k4Ok, proc);
      allPass &= k4Ok;
    } else {
      printSkip("K4 PutSigAdd", "public SignalAdd unavailable", proc);
    }
    MPI_Barrier(splitComm);

#ifdef USE_NVIDIA_ADAPTOR
    if (supportsSignal && supportsMultiContext) {
      // --- K4b: multi-context data + standalone release ---
      initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                   hostBuff);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      // Make launch skew deterministic for the first size. Without the kernel's
      // pre-communication rendezvous, peers can signal rank 0 before it
      // captures its baseline and rank 0 waits forever for one extra signal.
      if (size == minBytes && proc == 0)
        usleep(100000);
      FLAGCXCHECK(launchKernelNetMultiContextPutRelease(
          sendMem, recvMem, countPerPeer, DATATYPE, devComm, stream));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                          flagcxMemcpyDeviceToHost, stream));
      bool k4bOk = verifyInterAlltoAll((const float *)hostBuff, countPerPeer,
                                       totalProcs, proc, intraSize);
      printResult("K4b MultiContextPutRelease", k4bOk, proc);
      allPass &= k4bOk;
    } else {
      printSkip("K4b MultiContextPutRelease",
                "multiple contexts or public signals unavailable", proc);
    }
    MPI_Barrier(splitComm);
#endif

    if (supportsSignal && supportsCounter) {
      // --- K5: Put + SigInc + CtrInc (CounterPipeline) ---
      initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                   hostBuff);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(launchKernelNetCounterPipeline(sendMem, recvMem, countPerPeer,
                                                 DATATYPE, devComm, stream,
                                                 dResultBuf));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                          4 * sizeof(uint64_t),
                                          flagcxMemcpyDeviceToHost, stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                          flagcxMemcpyDeviceToHost, stream));
      bool k5Ok = verifyCounterPipeline(hResultBuf, (const float *)hostBuff,
                                        countPerPeer, totalProcs);
      printResult("K5 CounterPipeline", k5Ok, proc);
      allPass &= k5Ok;
    } else {
      printSkip("K5 CounterPipeline", "public signals/counters unavailable",
                proc);
    }
    MPI_Barrier(splitComm);

    if (supportsSignal) {
      // --- K6: Put(None) + Flush + Signal (FlushDecouple) ---
      initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                   hostBuff);
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(launchKernelNetFlushDecouple(sendMem, recvMem, countPerPeer,
                                               DATATYPE, devComm, stream));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                          flagcxMemcpyDeviceToHost, stream));
      bool k6Ok = verifyAlltoAll((const float *)hostBuff, countPerPeer,
                                 totalProcs, proc);
      printResult("K6 FlushDecouple", k6Ok, proc);
      allPass &= k6Ok;
    } else {
      printSkip("K6 FlushDecouple", "public standalone signal unavailable",
                proc);
    }
    MPI_Barrier(splitComm);

    if (supportsPutValue && supportsSignal) {
      // --- K7: PutValue ---
      FLAGCXCHECK(devHandle->deviceMemset((char *)recvBuff + putValBase, 0,
                                          (size_t)totalProcs * sizeof(uint64_t),
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(
          launchKernelNetPutValue(recvMem, devComm, stream, putValBase));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      FLAGCXCHECK(devHandle->deviceMemcpy((char *)hostBuff + putValBase,
                                          (char *)recvBuff + putValBase,
                                          (size_t)totalProcs * sizeof(uint64_t),
                                          flagcxMemcpyDeviceToHost, stream));
      bool k7Ok = verifyPutValue(hostBuff, putValBase, totalProcs, proc);
      printResult("K7 PutValue", k7Ok, proc);
      allPass &= k7Ok;
    } else {
      printSkip("K7 PutValue",
                "this case requires putValue plus public signals", proc);
    }
    MPI_Barrier(splitComm);

    // --- K8: Get --- SKIPPED (get unsupported on vendor path)
    // initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
    //              hostBuff);
    // FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
    // flagcxMemDevice,
    //                                     stream));
    // FLAGCXCHECK(launchKernelNetGet(sendMem, recvMem, countPerPeer, DATATYPE,
    //                                devComm, stream));
    // FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
    // FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
    //                                     flagcxMemcpyDeviceToHost, stream));
    // bool k8Ok =
    //     verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs,
    //     proc);
    // printResult("K8 Get", k8Ok, proc);
    printSkip("K8 Get", "remote read test not enabled", proc);
    MPI_Barrier(MPI_COMM_WORLD);

    if (supportsSignal) {
      // --- K9: Signal (standalone SigInc + SigAdd) ---
      FLAGCXCHECK(launchKernelNetSignal(devComm, stream));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      printResult("K9 Signal", true, proc); // hang-free = PASS
    } else {
      printSkip("K9 Signal", "public SignalInc/Add unavailable", proc);
    }
    MPI_Barrier(splitComm);

    // --- K10: Shadow (commented — FollowShadow + MeetShadow) ---
    // FLAGCXCHECK(launchKernelNetFollowShadow(devComm, stream));
    // FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
    // printResult("K10a FollowShadow", true, proc);
    // MPI_Barrier(MPI_COMM_WORLD);
    //
    // FLAGCXCHECK(launchKernelNetMeetShadow(devComm, stream));
    // FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
    // printResult("K10b MeetShadow", true, proc);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (supportsSignal) {
      // --- K11: WaitSignal + Flush (standalone) ---
      FLAGCXCHECK(launchKernelNetWaitSignalFlush(devComm, stream));
      FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
      printResult("K11 WaitSignal+Flush", true, proc); // hang-free = PASS
    } else {
      printSkip("K11 WaitSignal+Flush", "public signal wait unavailable", proc);
    }
    MPI_Barrier(splitComm);

    // --- K12: Inter Barrier ---
    FLAGCXCHECK(devHandle->deviceMemset(dResultBuf, 0, 4 * sizeof(uint64_t),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(
        launchKernelInterBarrier(devComm, (int *)dResultBuf, 3, stream));
    FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
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
    FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
    FLAGCXCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                        sizeof(uint64_t),
                                        flagcxMemcpyDeviceToHost, stream));
    {
      bool k13Ok = ((int)hResultBuf[0] == 1);
      printResult("K13 WorldBarrier", k13Ok, proc);
      allPass &= k13Ok;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K14: AlltoAll (same launcher selects the actual Net provider) ---
    if ((countPerPeer * sizeof(float)) % requiredAlignment == 0) {
      for (int iteration = 0; iteration < numTestIters; ++iteration) {
        // Skew rank 0 before receive initialization. Early peers must wait
        // for its kernel to reach the entry Barrier before posting puts.
        if (useGda && size == minBytes && iteration == 0 && proc == 0)
          usleep(100000);
        const int round = useGda ? iteration + 1 : 0;
        initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle,
                     stream, hostBuff, round);
        FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, floatSize,
                                            flagcxMemDevice, stream));
#if defined(FLAGCX_USE_XPU_NETIO)
        if (useGda) {
          static_assert(sizeof(flagcxNetGdaVisibilityResult) ==
                            sizeof(hResultBuf),
                        "diagnostic buffer size");
          FLAGCXCHECK(devHandle->deviceMemset(dResultBuf, 0, sizeof(hResultBuf),
                                              flagcxMemDevice, stream));
          FLAGCXCHECK(devHandle->deviceMemset(
              dResultBuf + 3, 0xff, sizeof(uint64_t), flagcxMemDevice, stream));
          // Small aligned chunks exercise multiple notifications and tails.
          FLAGCXCHECK(launchKernelNetOneSidedAlltoAll(
              sendMem, recvMem, countPerPeer, DATATYPE, devComm, stream, round,
              std::min(gdaChunkBytes, countPerPeer * sizeof(float)),
              reinterpret_cast<flagcxNetGdaVisibilityResult *>(dResultBuf)));
        } else
#endif
          FLAGCXCHECK(launchKernelNetOneSidedAlltoAll(
              sendMem, recvMem, countPerPeer, DATATYPE, devComm, stream));
        FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
        FLAGCXCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
        bool k14Ok = verifyAlltoAll((const float *)hostBuff, countPerPeer,
                                    totalProcs, proc, round);
#if defined(FLAGCX_USE_XPU_NETIO)
        if (useGda) {
          flagcxNetGdaVisibilityResult result{};
          FLAGCXCHECK(
              devHandle->deviceMemcpy(&result, dResultBuf, sizeof(result),
                                      flagcxMemcpyDeviceToHost, stream));
          FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
          const bool visible =
              result.checkedBytes == floatSize && result.mismatchCount == 0;
          printResult("K14 GPU payload after signal", visible, proc);
          if (!visible)
            fprintf(stderr,
                    "[rank %d] GDA immediate payload check: checked=%llu "
                    "mismatches=%llu first=%llu\n",
                    proc, (unsigned long long)result.checkedBytes,
                    (unsigned long long)result.mismatchCount,
                    (unsigned long long)result.firstMismatchOffset);
          k14Ok &= visible;
        }
#endif
        printResult("K14 OneSidedAlltoAll", k14Ok, proc);
        allPass &= k14Ok;
        ++alltoallRuns;
      }
    } else {
      printSkip("K14 OneSidedAlltoAll", "message violates provider alignment",
                proc);
    }
    MPI_Barrier(splitComm);

    // --- K15: AlltoAll (two-sided, commented) ---
    // initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
    //              hostBuff);
    // FLAGCXCHECK(
    //     devHandle->deviceMemset(recvBuff, 0, floatSize, flagcxMemDevice,
    //     NULL));
    // FLAGCXCHECK(launchKernelNetTwoSidedAlltoAll(sendMem, recvMem,
    // countPerPeer,
    //                                         DATATYPE, devComm, stream));
    // FLAGCXCHECK(synchronizeInterTest(devHandle, stream, devComm));
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
    if (size > maxBytes / static_cast<size_t>(stepFactor))
      break;
  }

  if (useGda && alltoallRuns == 0) {
    fprintf(stderr, "GDA suite did not execute an aligned AlltoAll case\n");
    allPass = false;
  }

  // Summary
  MPI_Barrier(MPI_COMM_WORLD);
  int pass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&pass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  printf("[rank %d] === Overall: %s ===\n", proc, globalPass ? "PASS" : "FAIL");

  // Cleanup
  FLAGCXCHECK(deviceApiCloseWindows(comm, splitComm, devHandle, stream,
                                    {&sendMem, &recvMem}, {&sendWin, &recvWin},
                                    memAllocator));
  FLAGCXCHECK(devHandle->deviceFree(dResultBuf, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->streamDestroy(stream));

  FLAGCXCHECK(flagcxMemFree(sendBuff, memAllocator));
  FLAGCXCHECK(flagcxMemFree(recvBuff, memAllocator));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(flagcxCommFinalize(comm));
  FLAGCXCHECK(flagcxCommFinalize(comm));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  free(hostBuff);
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
