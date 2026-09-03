/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Intra-node Device API test — exercises struct-based Device API functions
 * that only require single-node (intra-node) setup.
 *
 * Tests (K1–K10, aligned with device_ir_intra S1–S10):
 *   K1:  Comm Queries (hasWindow, getIntraRank/Size, getRank/Size)
 *   K2:  Coop Groups (Block/Tile/Warp/Lanes/TileSpan threadRank/size/sync)
 *   K3:  Team (flagcxTeamIntra, RankToWorld, RankToIntra, RankIsMember)
 *   K4:  Local Pointer (flagcxGetLocalPointer, getRawPtr)
 *   K5:  Intra Pointer (flagcxGetIntraPointer — IPC/window peer read)
 *   K6:  Peer Pointer  (flagcxGetPeerPointer with team)
 *   K7:  Multicast Pointer (flagcxGetMulticastPointer — commented, NVLS)
 *   K8:  Intra Barrier Sync (flagcxDevBarrier<Intra> sync)
 *   K9:  Intra Barrier Arrive/Wait (flagcxDevBarrier<Intra> arrive+wait)
 *   K10: IntraAllReduce (composite — peer pointer + barrier end-to-end)
 *
 * Usage: mpirun -np N ./test_device_api_intra [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>  -R <regMode>
 *   Runs on any single node, no network or HETERO_COMM required.
 *   -R 2 recommended (window registration for peer pointer access).
 ************************************************************************/

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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

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

  // Create DevComm (initializes the selected SHMEM runtime before allocation)
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = 0;
  reqs.interSignalCount = 0;
  reqs.interCounterCount = 0;
  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Buffer sized to maxBytes, symmetric window registered
  size_t bufSize = maxBytes;
  void *regBuff = nullptr;
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&regBuff, bufSize, memAllocator));

  flagcxWindow_t win = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(
      comm, regBuff, bufSize, &win, FLAGCX_WIN_COLL_SYMMETRIC, memAllocator));

  // Create DevMem
  flagcxDevMem_t devMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  // Results buffer
  int *devResults = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devResults, 64 * sizeof(int),
                                      flagcxMemDevice, NULL));
  int hostResults[64] = {};

  // Output buffer for pointer/barrier tests (sized to maxBytes)
  float *devOutput = nullptr;
  FLAGCXCHECK(devHandle->deviceMalloc((void **)&devOutput, bufSize,
                                      flagcxMemDevice, NULL));

  // Host scratch buffer (reused across iterations)
  size_t maxCount = bufSize / sizeof(float);
  float *hostBuf = new float[maxCount];

  if (proc == 0) {
    printf("# FlagCX Device API Intra-Node Test\n");
    printf("# nRanks: %d\n#\n", totalProcs);
  }

  int peer = (proc + 1) % totalProcs;
  bool allPass = true;

  // =========================================================================
  // Main test loop
  // =========================================================================
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;

    if (proc == 0)
      printf("# Size = %zu bytes, count = %zu\n", size, count);

    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // K1: Comm Queries
    // -----------------------------------------------------------------------
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 8 * sizeof(int),
                                        flagcxMemDevice, stream));

    FLAGCXCHECK(launchKernelCommQueries(devMem, devComm, devResults, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                        6 * sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool k1Ok = (hostResults[0] == 1) &&          // hasWindow
                (hostResults[1] == proc) &&       // intraRank
                (hostResults[2] == totalProcs) && // intraSize
                (hostResults[3] == proc) &&       // rank
                (hostResults[4] == totalProcs);   // size
    printResult("K1 CommQueries", k1Ok, proc);
    allPass &= k1Ok;

    // -----------------------------------------------------------------------
    // K2: Coop Groups
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 16 * sizeof(int),
                                        flagcxMemDevice, stream));

    FLAGCXCHECK(launchKernelCoopGroups(devResults, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                        16 * sizeof(int),
                                        flagcxMemcpyDeviceToHost, stream));

    bool k2Ok = (hostResults[0] == 1) && (hostResults[1] == 1) &&
                (hostResults[2] == 1) && (hostResults[3] == 1) &&
                (hostResults[4] == 1);
    printResult("K2 CoopGroups", k2Ok, proc);
    allPass &= k2Ok;

    // -----------------------------------------------------------------------
    // K3: Team
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // K4: Local Pointer
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                        flagcxMemDevice, stream));

    // Write a known value to regBuff so kernel can verify localPtr reads it
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

    // -----------------------------------------------------------------------
    // K5: Intra Pointer
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    // Write known pattern: each rank fills regBuff with its rank value
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)proc;
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));

    FLAGCXCHECK(
        launchKernelIntraPointer(devMem, devComm, devOutput, count, stream));
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

    // -----------------------------------------------------------------------
    // K6: Peer Pointer (team-based)
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));

    FLAGCXCHECK(
        launchKernelPeerPointer(devMem, devComm, devOutput, count, stream));
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

    // -----------------------------------------------------------------------
    // K7: Multicast Pointer (commented — NVLS-dependent)
    // -----------------------------------------------------------------------
    // MPI_Barrier(MPI_COMM_WORLD);
    // ...

    // -----------------------------------------------------------------------
    // K8: Intra Barrier Sync
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    // Write rank+1 to local buffer
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 1);
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));

    FLAGCXCHECK(launchKernelIntraBarrierSync(devMem, devComm, devOutput, count,
                                             stream));
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

    // -----------------------------------------------------------------------
    // K9: Intra Barrier Arrive/Wait
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    // Write rank+100 to local buffer
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 100);
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                        flagcxMemDevice, stream));

    FLAGCXCHECK(launchKernelIntraBarrierArriveWait(devMem, devComm, devOutput,
                                                   count, stream));
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

    // -----------------------------------------------------------------------
    // K10: IntraAllReduce (composite)
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 1); // each rank contributes rank+1
    FLAGCXCHECK(devHandle->deviceMemcpy(regBuff, hostBuf, count * sizeof(float),
                                        flagcxMemcpyHostToDevice, stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    FLAGCXCHECK(launchKernelIntraAllReduce(devMem, count, flagcxFloat, devComm,
                                           stream));
    FLAGCXCHECK(devHandle->streamSynchronize(stream));

    FLAGCXCHECK(devHandle->deviceMemcpy(hostBuf, regBuff, count * sizeof(float),
                                        flagcxMemcpyDeviceToHost, stream));
    // AllReduce sum: expected = sum(1..N) = N*(N+1)/2
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

  // =========================================================================
  // Summary
  // =========================================================================
  MPI_Barrier(MPI_COMM_WORLD);

  int pass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&pass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  printf("[rank %d] === Overall: %s ===\n", proc, globalPass ? "PASS" : "FAIL");

  // Cleanup
  delete[] hostBuf;
  FLAGCXCHECK(devHandle->deviceFree(devOutput, flagcxMemDevice, NULL));
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, devMem));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, win, memAllocator));
  FLAGCXCHECK(flagcxMemFree(regBuff, memAllocator));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));
  FLAGCXCHECK(flagcxDeviceHandleFree(devHandle));

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
