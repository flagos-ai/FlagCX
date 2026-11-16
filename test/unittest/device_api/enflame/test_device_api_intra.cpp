/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Enflame-specific Intra-node Device API test
 * Uses dual-buffer pattern: separate data buffer + helper buffer
 * instead of single buffer with (count+i) offset.
 ************************************************************************/

#include "check.h"
#include "device_api.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define DATATYPE flagcxFloat

static void printResult(const char *name, bool ok, int rank) {
  printf("[rank %d] %-35s %s\n", rank, name, ok ? "PASS" : "FAIL");
}

// Print result with SKIP support: -1 means the sub-test is not supported.
// allExpected is the number of sub-tests to check.
static void printResultWithSkip(const char *name, const int *results,
                                int count, int rank) {
  bool allPass = true;
  bool anySkip = false;
  for (int i = 0; i < count; i++) {
    if (results[i] == -1) {
      anySkip = true;
    } else if (results[i] != 1) {
      allPass = false;
    }
  }
  if (!allPass) {
    printf("[rank %d] %-35s %s\n", rank, name, "FAIL");
  } else if (anySkip) {
    printf("[rank %d] %-35s %s\n", rank, name, "SKIP (partial)");
  } else {
    printf("[rank %d] %-35s %s\n", rank, name, "PASS");
  }
}

int main(int argc, char *argv[]) {
  flagcxDeviceHandle_t devHandle;
  flagcxComm_t comm;
  FLAGCXCHECK(flagcxDeviceHandleInit(&devHandle));
  flagcxUniqueId uniqueId;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  uint64_t splitMask = 0;
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

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = 0;
  reqs.interSignalCount = 0;
  reqs.interCounterCount = 0;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Enflame dual-buffer: allocate TWO separate buffers, each maxBytes
  size_t bufSize = maxBytes;
  void *regBuff = nullptr;
  void *helperBuff = nullptr;
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&helperBuff, bufSize, memAllocator));

  flagcxWindow_t win = nullptr;
  flagcxWindow_t helperWin = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(
      comm, regBuff, bufSize, &win, FLAGCX_WIN_COLL_SYMMETRIC, memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(
      comm, helperBuff, bufSize, &helperWin, FLAGCX_WIN_COLL_SYMMETRIC, memAllocator));

  flagcxDevMem_t devMem = nullptr;
  flagcxDevMem_t helperMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));
  FLAGCXCHECK(flagcxDevMemCreate(comm, helperBuff, bufSize, helperWin, &helperMem));

  int *devResults = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devResults, 64 * sizeof(int),
                                      flagcxMemDevice, NULL));
  int hostResults[64] = {};

  float *devOutput = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devOutput, bufSize,
                                      flagcxMemDevice, NULL));

  size_t maxCount = bufSize / sizeof(float);
  float *hostBuf = new float[maxCount];

  if (proc == 0) {
    printf("# FlagCX Device API Intra-Node Test (Enflame)\n");
    printf("# nRanks: %d\n#\n", totalProcs);
  }

  int peer = (proc + 1) % totalProcs;
  bool allPass = true;

  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;

    if (proc == 0)
      printf("# Size = %zu bytes, count = %zu\n", size, count);

    MPI_Barrier(MPI_COMM_WORLD);

    // K1: Comm Queries
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 8 * sizeof(int),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelCommQueries(devMem, devComm, devResults, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                        6 * sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool k1Ok = (hostResults[0] == 1) && (hostResults[1] == proc) &&
                (hostResults[2] == totalProcs) && (hostResults[3] == proc) &&
                (hostResults[4] == totalProcs);
    printResult("K1 CommQueries", k1Ok, proc);
    allPass &= k1Ok;

    // K2: Coop Groups
    // Expected: [1, -1, 1, -1, -1] for block, warp, thread, Tile<8>, lanes
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 16 * sizeof(int),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelCoopGroups(devResults, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                        16 * sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    // Check each sub-test with detailed output
    const char *k2Names[5] = {"block", "warp", "thread", "Tile<8>", "lanes"};
    const int expectedK2[5] = {1, -1, 1, -1, -1};
    bool k2Ok = true;
    for (int i = 0; i < 5; i++) {
      const char *status;
      if (expectedK2[i] == 1) {
        // Supported feature: expect 1
        status = (hostResults[i] == 1) ? "pass" : "fail";
        if (hostResults[i] != 1) k2Ok = false;
      } else {
        // Unsupported feature: expect -1
        status = (hostResults[i] == -1) ? "skip (unsupported)" : "fail";
        if (hostResults[i] != -1) k2Ok = false;
      }
      printf("[rank %d] K2  CoopGroups %-12s: %s\n", proc, k2Names[i], status);
    }
    printResult("K2 CoopGroups", k2Ok, proc);
    allPass &= k2Ok;

    // K3: Team
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 8 * sizeof(int),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelTeam(devComm, devResults, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                        8 * sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool k3Ok = (hostResults[0] == 1) && (hostResults[1] == 1) &&
                (hostResults[2] == 1) && (hostResults[3] == 1);
    printResult("K3 Team", k3Ok, proc);
    allPass &= k3Ok;

    // K4: Local Pointer
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                        flagcxMemDevice, stream));
    {
      float magic = 42.0f;
      FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, &magic, sizeof(float),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
    }
    FLAGCXCHECK(launchKernelLocalPointer(devMem, regBuff, devResults, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                        4 * sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool k4Ok = (hostResults[0] == 1);
    printResult("K4 LocalPointer", k4Ok, proc);
    allPass &= k4Ok;

    // K5: Intra Pointer (Enflame: pass both devMem and helperMem)
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)proc;
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelIntraPointer(devMem, helperMem, devComm, devOutput,
                                         count, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, devOutput,
                                        count * sizeof(float),
                                        flagcxMemcpyDeviceToHost, stream));
    bool k5Ok = true;
    for (size_t i = 0; i < count; i++) {
      if (fabsf(hostBuf[i] - (float)peer) > 1e-3f) {
        k5Ok = false;
        break;
      }
    }
    printResult("K5 IntraPointer", k5Ok, proc);
    allPass &= k5Ok;

    // K6: Peer Pointer (Enflame: pass both devMem and helperMem)
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelPeerPointer(devMem, helperMem, devComm, devOutput,
                                        count, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, devOutput,
                                        count * sizeof(float),
                                        flagcxMemcpyDeviceToHost, stream));
    bool k6Ok = true;
    for (size_t i = 0; i < count; i++) {
      if (fabsf(hostBuf[i] - (float)peer) > 1e-3f) {
        k6Ok = false;
        break;
      }
    }
    printResult("K6 PeerPointer(team)", k6Ok, proc);
    allPass &= k6Ok;

    // K8: Intra Barrier Sync (Enflame: pass both devMem and helperMem)
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 1);
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelIntraBarrierSync(devMem, helperMem, devComm,
                                             devOutput, count, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, devOutput,
                                        count * sizeof(float),
                                        flagcxMemcpyDeviceToHost, stream));
    bool k8Ok = true;
    float k8Expected = (float)(peer + 1);
    for (size_t i = 0; i < count; i++) {
      if (fabsf(hostBuf[i] - k8Expected) > 1e-3f) {
        k8Ok = false;
        break;
      }
    }
    printResult("K8 IntraBarrierSync", k8Ok, proc);
    allPass &= k8Ok;

    // K9: Intra Barrier Arrive/Wait (Enflame: pass both devMem and helperMem)
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 100);
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));
    FLAGCXCHECK(launchKernelIntraBarrierArriveWait(devMem, helperMem, devComm,
                                                   devOutput, count, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, devOutput,
                                        count * sizeof(float),
                                        flagcxMemcpyDeviceToHost, stream));
    bool k9Ok = true;
    float k9Expected = (float)(peer + 100);
    for (size_t i = 0; i < count; i++) {
      if (fabsf(hostBuf[i] - k9Expected) > 1e-3f) {
        k9Ok = false;
        break;
      }
    }
    printResult("K9 IntraBarrierArriveWait", k9Ok, proc);
    allPass &= k9Ok;

    // K10: IntraAllReduce (Enflame: pass both devMem and helperMem)
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 1);
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(launchKernelIntraAllReduce(devMem, helperMem, count, flagcxFloat,
                                           devComm, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, regBuff, count * sizeof(float),
                                        flagcxMemcpyDeviceToHost, stream));
    float k10Expected = (float)(totalProcs * (totalProcs + 1) / 2);
    bool k10Ok = true;
    for (size_t i = 0; i < count; i++) {
      if (fabsf(hostBuf[i] - k10Expected) > 1e-1f) {
        k10Ok = false;
        break;
      }
    }
    printResult("K10 IntraAllReduce(composite)", k10Ok, proc);
    allPass &= k10Ok;

    if (proc == 0)
      printf("#\n");

    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int pass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&pass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  printf("[rank %d] === Overall: %s ===\n", proc, globalPass ? "PASS" : "FAIL");

  delete[] hostBuf;
  FLAGCXCHECK(devHandle->deviceFree(devOutput, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, devMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, helperMem));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, win, memAllocator));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, helperWin, memAllocator));
  FLAGCXCHECK(flagcxMemFree(regBuff, memAllocator));
  FLAGCXCHECK(flagcxMemFree(helperBuff, memAllocator));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
