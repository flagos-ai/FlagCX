/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Intra-Node Tests — host driver exercising FlagCX Device API
 * IR wrapper functions that only require intra-node (single-node) setup.
 *
 * Tests S-suffixed (scalar) IR functions (aligned with device_api_intra
 * K1–K10):
 *   S1:  Comm Queries (DevCommGetRank, DevCommGetSize,
 *DevCommGetIntraRank/Size) S2:  Coop Groups (CoopThreadRankS, CoopSizeS,
 *CoopSyncS, TileSpan, Lanes) S3:  Team Queries (TeamRankToWorldS) S4:  Local
 *Pointer (GetLocalPointerS) S5:  Intra Pointer (GetIntraPointerS) S6:  Peer
 *Pointer (GetPeerPointerS) S7:  Multicast Pointer (GetMulticastPointerS) —
 *commented, NVLS-dependent S8:  Intra Barrier Sync (IntraBarrierSyncS) S9:
 *Intra Barrier Sync Two Epochs (SyncS Release + read + SyncS Acquire) S10:
 *Intra AllReduce (composite using barriers + pointers)
 *
 * Usage: mpirun -np N ./test_device_ir_intra [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 ************************************************************************/

#include "check.h"
#include "device_api_report.h"
#include "device_api_test_session.hpp"
#include "device_ir.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

// ===========================================================================
// Main test driver
// ===========================================================================

static int runDeviceApiTest(int argc, char *argv[],
                            DeviceApiTestSession &resources) {
  auto &devHandle = resources.devHandle;
  auto &comm = resources.comm;
  flagcxUniqueId uniqueId;

  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
// Print with rank prefix from all ranks, flushing immediately
#define RPRINTF(...)                                                           \
  do {                                                                         \
    printf("[rank %d] ", proc);                                                \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  } while (0)
  auto &splitComm = resources.group;
  uint64_t splitMask = 0;
  int color = 0;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);
  resources.check(flagcxDeviceHandleInit(&devHandle));

  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();

  if (stepFactor <= 1) {
    if (proc == 0)
      printf("Error: stepFactor must be > 1, got %d\n", stepFactor);
    return 1;
  }

  int nGpu;
  resources.check(devHandle->getDeviceCount(&nGpu));
  resources.check(devHandle->setDevice(worldRank % nGpu));

  flagcxResult_t uniqueStatus = flagcxSuccess;
  if (proc == 0)
    uniqueStatus = flagcxGetUniqueId(&uniqueId);
  resources.check(uniqueStatus);
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  resources.checkInit(flagcxCommInitRank(&comm, totalProcs, &uniqueId, proc));

  auto &stream = resources.stream;
  resources.check(devHandle->streamCreate(&stream));

  // Create DevComm (initializes the selected SHMEM runtime before allocation)
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 4;
  reqs.interBarrierCount = 4;
  reqs.interSignalCount = 2;
  reqs.interCounterCount = 1;
  auto &devComm = resources.devComm;
  resources.check(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Allocate test buffer sized to maxBytes
  size_t bufSize = maxBytes;
  auto &regBuff = resources.regBuff;
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  resources.memAllocator = flagcxMemSHMEM;
  const auto &memAllocator = resources.memAllocator;
#else
  resources.memAllocator = flagcxMemCCL;
  const auto &memAllocator = resources.memAllocator;
#endif
  resources.check(flagcxMemAlloc(&regBuff, bufSize, memAllocator));

  // Register symmetric window
  auto &win = resources.win;
  resources.check(flagcxCommWindowRegister(
      comm, regBuff, bufSize, &win, flagcxTestWindowFlags(), memAllocator));

  // Create DevMem
  auto &devMem = resources.devMem;
  resources.check(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  // Get device pointers
  void *devCommPtr = nullptr;
  resources.check(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));
  void *devMemPtr = nullptr;
  resources.check(flagcxDevMemGetDevicePtr(devMem, &devMemPtr));

  int peer = (proc + 1) % totalProcs;

  // Allocate reusable device buffers
  auto &devResults = resources.devResults;
  resources.check(devHandle->deviceMalloc(
      (void **)&devResults, 16 * sizeof(int), flagcxMemDevice, NULL));

  auto &devOutput = resources.devOutput;
  resources.check(devHandle->deviceMalloc((void **)&devOutput, bufSize,
                                          flagcxMemDevice, NULL));

  // Host scratch buffer
  size_t maxCount = bufSize / sizeof(float);
  auto &hostBuf = resources.hostBuf;
  hostBuf = new float[maxCount];

  if (proc == 0) {
    printf("# FlagCX Device IR Intra-Node Test\n");
    printf("# nRanks: %d\n#\n", totalProcs);
  }

  bool allPass = true;

  // =========================================================================
  // Main test loop
  // =========================================================================
  const DeviceApiReporter reporter("test_device_ir_intra", proc);

  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;

    if (proc == 0)
      printf("# Size = %zu bytes, count = %zu\n", size, count);

    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // S1: Comm Queries (Scalar)
    // -----------------------------------------------------------------------
    resources.check(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                            flagcxMemDevice, stream));

    launchKernelCommQueriesS(devCommPtr, devResults, stream);
    resources.check(devHandle->streamSynchronize(stream));

    int hostS1[4];
    resources.check(devHandle->deviceMemcpy(hostS1, devResults, 4 * sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    bool s1Pass = (hostS1[0] == proc) && (hostS1[1] == totalProcs) &&
                  (hostS1[2] == proc) && (hostS1[3] == totalProcs);
    RPRINTF("S1  CommQueries(Scalar): %s\n", s1Pass ? "PASS" : "FAIL");
    allPass &= s1Pass;

    // -----------------------------------------------------------------------
    // S2: Coop Groups (Scalar)
    // -----------------------------------------------------------------------
    resources.check(devHandle->deviceMemset(devResults, 0, 3 * sizeof(int),
                                            flagcxMemDevice, stream));

    launchKernelCoopGroupsS(devCommPtr, devResults, stream);
    resources.check(devHandle->streamSynchronize(stream));

    int hostS2[3];
    resources.check(devHandle->deviceMemcpy(hostS2, devResults, 3 * sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    bool s2Pass = (hostS2[0] == 1) && (hostS2[1] == 1) && (hostS2[2] == 1);
    RPRINTF("S2  CoopGroups(Scalar): %s\n", s2Pass ? "PASS" : "FAIL");
    allPass &= s2Pass;

    // -----------------------------------------------------------------------
    // S3: Team Queries (Scalar)
    // -----------------------------------------------------------------------
    resources.check(devHandle->deviceMemset(devResults, 0, 2 * sizeof(int),
                                            flagcxMemDevice, stream));

    launchKernelTeamQueriesS(devCommPtr, devResults, stream);
    resources.check(devHandle->streamSynchronize(stream));

    int hostS3[2];
    resources.check(devHandle->deviceMemcpy(hostS3, devResults, 2 * sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    bool s3Pass = (hostS3[1] == proc);
    RPRINTF("S3  TeamQueries(Scalar): %s\n", s3Pass ? "PASS" : "FAIL");
    allPass &= s3Pass;

    // -----------------------------------------------------------------------
    // S4: Local Pointer (Scalar)
    // -----------------------------------------------------------------------
    resources.check(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                            flagcxMemDevice, stream));

    // Write a known value to regBuff so kernel can verify localPtr reads it
    {
      float magic = 42.0f;
      resources.check(devHandle->deviceMemcpy(
          regBuff, &magic, sizeof(float), flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
    }

    launchKernelLocalPointerS(devMemPtr, regBuff, devResults, stream);
    resources.check(devHandle->streamSynchronize(stream));

    int hostS4 = 0;
    resources.check(devHandle->deviceMemcpy(&hostS4, devResults, sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    bool s4Pass = (hostS4 == 1);
    RPRINTF("S4  LocalPointer(Scalar): %s\n", s4Pass ? "PASS" : "FAIL");
    allPass &= s4Pass;

    // -----------------------------------------------------------------------
    // S5: Intra Pointer (Scalar)
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc * 1000 + (int)i);
    resources.check(devHandle->deviceMemcpy(regBuff, hostBuf,
                                            count * sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
    resources.check(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    launchKernelIntraPointerS(devCommPtr, devMemPtr, devOutput, count, stream);
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
                                            count * sizeof(float),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    bool s5Pass = true;
    for (size_t i = 0; i < count; i++) {
      float expected = (float)(peer * 1000 + (int)i);
      if (fabsf(hostBuf[i] - expected) > 1e-3f) {
        s5Pass = false;
        break;
      }
    }
    RPRINTF("S5  IntraPointer(Scalar): %s\n", s5Pass ? "PASS" : "FAIL");
    allPass &= s5Pass;

    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // S6: Peer Pointer (Scalar) — team-based
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc * 1000 + (int)i);
    resources.check(devHandle->deviceMemcpy(regBuff, hostBuf,
                                            count * sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
    resources.check(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    launchKernelPeerPointerS(devCommPtr, devMemPtr, devOutput, count, stream);
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
                                            count * sizeof(float),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    bool s6Pass = true;
    for (size_t i = 0; i < count; i++) {
      float expected = (float)(peer * 1000 + (int)i);
      if (fabsf(hostBuf[i] - expected) > 1e-3f) {
        s6Pass = false;
        break;
      }
    }
    RPRINTF("S6  PeerPointer(Scalar): %s\n", s6Pass ? "PASS" : "FAIL");
    allPass &= s6Pass;

    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // S7: Multicast Pointer (Scalar) — commented, NVLS-dependent
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // S8: Intra Barrier Sync (Scalar)
    // -----------------------------------------------------------------------
    resources.check(devHandle->deviceMemset(regBuff, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));
    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    launchKernelIntraBarrierSyncS(devCommPtr, devMemPtr, (float *)regBuff,
                                  devOutput, count, stream);
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
                                            count * sizeof(float),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    float expectedS8 = (float)(peer + 1);
    bool s8Pass = true;
    for (size_t i = 0; i < count; i++) {
      if (!(fabsf(hostBuf[i] - expectedS8) <= 1e-3f)) {
        s8Pass = false;
        break;
      }
    }
    RPRINTF("S8  IntraBarrierSync(Scalar): %s\n", s8Pass ? "PASS" : "FAIL");
    allPass &= s8Pass;

    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // S9: two full scalar barriers (Release + read + Acquire)
    // -----------------------------------------------------------------------
    resources.check(devHandle->deviceMemset(regBuff, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));
    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    launchKernelIntraBarrierArriveWaitS(devCommPtr, devMemPtr, (float *)regBuff,
                                        devOutput, count, stream);
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
                                            count * sizeof(float),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    float expectedS9 = (float)(peer + 500);
    bool s9Pass = true;
    for (size_t i = 0; i < count; i++) {
      if (!(fabsf(hostBuf[i] - expectedS9) <= 1e-3f)) {
        s9Pass = false;
        break;
      }
    }
    RPRINTF("S9  IntraBarrierTwoEpochs(Scalar): %s\n",
            s9Pass ? "PASS" : "FAIL");
    allPass &= s9Pass;

    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // S10: Intra AllReduce (Scalar) — composite
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 1); // each rank contributes rank+1
    resources.check(devHandle->deviceMemcpy(regBuff, hostBuf,
                                            count * sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
    resources.check(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    launchKernelIntraAllReduceS(devCommPtr, devMemPtr, (float *)regBuff, count,
                                stream);
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, regBuff,
                                            count * sizeof(float),
                                            flagcxMemcpyDeviceToHost, stream));
    resources.check(devHandle->streamSynchronize(stream));

    // AllReduce sum: expected = sum(1..N) = N*(N+1)/2
    float expectedS10 = (float)(totalProcs * (totalProcs + 1) / 2);
    bool s10Pass = true;
    for (size_t i = 0; i < count; i++) {
      if (fabsf(hostBuf[i] - expectedS10) > 1e-1f) {
        s10Pass = false;
        break;
      }
    }
    RPRINTF("S10 IntraAllReduce(Scalar): %s\n", s10Pass ? "PASS" : "FAIL");
    reporter.report("S10", size, s10Pass ? "PASS" : "FAIL");
    allPass &= s10Pass;

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

  // One checked cleanup path serves successful and failed runs.
  resources.check(resources.finish());
  return globalPass ? 0 : 1;
}

int main(int argc, char *argv[]) {
  return deviceApiTestMain(argc, argv, runDeviceApiTest);
}
