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
 *
 * Capability switches (set by test/unittest/device_api/Makefile): a platform
 * that cannot satisfy a case compiles it out instead of shipping a launcher
 * that would report it green.  FLAGCX_TEST_NO_COOP_GROUPS drops K2,
 * FLAGCX_TEST_NO_PEER_POINTER drops K5/K6, FLAGCX_TEST_NO_INTRA_BARRIER drops
 * K8/K9/K10; each site carries the reason.  K1/K3/K4 are unconditional and are
 * what a reduced platform such as Ascend 910C still has to pass.
 ************************************************************************/

#include "check.h"
#include "device_api.h"
#include "device_api_report.h"
#include "device_api_test_session.hpp"
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

static int runDeviceApiTest(int argc, char *argv[],
                            DeviceApiTestSession &resources) {
  auto &devHandle = resources.devHandle;
  auto &comm = resources.comm;
  flagcxUniqueId uniqueId;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  auto &splitComm = resources.group;
  uint64_t splitMask = 0;
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
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = 0;
  reqs.interSignalCount = 0;
  reqs.interCounterCount = 0;
  auto &devComm = resources.devComm;
  resources.check(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Buffer sized to maxBytes, symmetric window registered
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

  auto &win = resources.win;
  resources.check(flagcxCommWindowRegister(
      comm, regBuff, bufSize, &win, flagcxTestWindowFlags(), memAllocator));

  // Create DevMem
  auto &devMem = resources.devMem;
  resources.check(flagcxDevMemCreate(comm, regBuff, bufSize, win, &devMem));

  // Results buffer
  auto &devResults = resources.devResults;
  resources.check(devHandle->deviceMalloc(
      (void **)&devResults, 64 * sizeof(int), flagcxMemDevice, NULL));
  int hostResults[64] = {};

  // Output buffer for pointer/barrier tests (sized to maxBytes)
  auto &devOutput = resources.devOutput;
  resources.check(devHandle->deviceMalloc((void **)&devOutput, bufSize,
                                          flagcxMemDevice, NULL));

  // Host scratch buffer (reused across iterations)
  size_t maxCount = bufSize / sizeof(float);
  auto &hostBuf = resources.hostBuf;
  hostBuf = new float[maxCount];

  if (proc == 0) {
    printf("# FlagCX Device API Intra-Node Test\n");
    printf("# nRanks: %d\n#\n", totalProcs);
  }

  int peer = (proc + 1) % totalProcs;
  (void)peer; // unused when the peer-access/barrier cases are compiled out
  bool allPass = true;

  // =========================================================================
  // Main test loop
  // =========================================================================
  const DeviceApiReporter reporter("test_device_api_intra", proc);

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
    resources.check(devHandle->deviceMemset(devResults, 0, 8 * sizeof(int),
                                            flagcxMemDevice, stream));

    resources.check(
        launchKernelCommQueries(devMem, devComm, devResults, stream));
    resources.check(devHandle->streamSynchronize(stream));
    resources.check(devHandle->deviceMemcpy(hostResults, devResults,
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
#ifndef FLAGCX_TEST_NO_COOP_GROUPS
    MPI_Barrier(MPI_COMM_WORLD);
    resources.check(devHandle->deviceMemset(devResults, 0, 16 * sizeof(int),
                                            flagcxMemDevice, stream));

    resources.check(launchKernelCoopGroups(devResults, stream));
    resources.check(devHandle->streamSynchronize(stream));
    resources.check(devHandle->deviceMemcpy(hostResults, devResults,
                                            16 * sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));

    bool k2Ok = (hostResults[0] == 1) && (hostResults[1] == 1) &&
                (hostResults[2] == 1) && (hostResults[3] == 1) &&
                (hostResults[4] == 1);
    printResult("K2 CoopGroups", k2Ok, proc);
    allPass &= k2Ok;
#else
    // Ascend 910C gives FlagCX one logical thread per AI core: CoopBlock has
    // size()==1/threadRank()==0 and CoopTile<N> static_asserts N==1
    // (ascend_platform_traits.h:598-615), so this case has no rank/size pair
    // to report and no working sync.  Compiled out on that platform by
    // test/unittest/device_api/Makefile rather than answered with a stub.
#endif

    // -----------------------------------------------------------------------
    // K3: Team
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    resources.check(devHandle->deviceMemset(devResults, 0, 8 * sizeof(int),
                                            flagcxMemDevice, stream));

    resources.check(launchKernelTeam(devComm, devResults, stream));
    resources.check(devHandle->streamSynchronize(stream));
    resources.check(devHandle->deviceMemcpy(hostResults, devResults,
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
    resources.check(devHandle->deviceMemset(devResults, 0, 4 * sizeof(int),
                                            flagcxMemDevice, stream));

    // Write a known value to regBuff so kernel can verify localPtr reads it
    {
      float magic = 42.0f;
      resources.check(devHandle->deviceMemcpy(
          regBuff, &magic, sizeof(float), flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
    }

    resources.check(
        launchKernelLocalPointer(devMem, regBuff, devResults, stream));
    resources.check(devHandle->streamSynchronize(stream));
    resources.check(devHandle->deviceMemcpy(hostResults, devResults,
                                            4 * sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));

    bool k4Ok = (hostResults[0] == 1);
    printResult("K4 LocalPointer", k4Ok, proc);
    allPass &= k4Ok;

    // -----------------------------------------------------------------------
    // K5: Intra Pointer
    // -----------------------------------------------------------------------
#ifndef FLAGCX_TEST_NO_PEER_POINTER
    MPI_Barrier(MPI_COMM_WORLD);

    // Write known pattern: each rank fills regBuff with its rank value
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)proc;
    resources.check(devHandle->deviceMemcpy(regBuff, hostBuf,
                                            count * sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
    resources.check(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    resources.check(
        launchKernelIntraPointer(devMem, devComm, devOutput, count, stream));
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
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
    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    resources.check(
        launchKernelPeerPointer(devMem, devComm, devOutput, count, stream));
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
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
#else
    // Both cases dereference a pointer into another rank's memory
    // (flagcxGetIntraPointer / flagcxGetPeerPointer).  Whether the symmetric
    // window registered by flagcxCommWindowRegister really maps peer device
    // memory on Ascend 910C is unmeasured, and no launcher in
    // test/kernel/ascend/device_api.asc claims it: compiling the cases out
    // keeps the suite's PASS honest until that mapping is demonstrated.
#endif

    // -----------------------------------------------------------------------
    // K7: Multicast Pointer (commented — NVLS-dependent)
    // -----------------------------------------------------------------------
    // MPI_Barrier(MPI_COMM_WORLD);
    // ...

    // -----------------------------------------------------------------------
    // K8: Intra Barrier Sync
    // -----------------------------------------------------------------------
#ifndef FLAGCX_TEST_NO_INTRA_BARRIER
    MPI_Barrier(MPI_COMM_WORLD);

    // Write rank+1 to local buffer
    for (size_t i = 0; i < count; i++)
      hostBuf[i] = (float)(proc + 1);
    resources.check(devHandle->deviceMemcpy(regBuff, hostBuf,
                                            count * sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
    resources.check(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    resources.check(launchKernelIntraBarrierSync(devMem, devComm, devOutput,
                                                 count, stream));
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
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
    resources.check(devHandle->deviceMemcpy(regBuff, hostBuf,
                                            count * sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
    resources.check(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    resources.check(devHandle->deviceMemset(devOutput, 0, count * sizeof(float),
                                            flagcxMemDevice, stream));

    resources.check(launchKernelIntraBarrierArriveWait(
        devMem, devComm, devOutput, count, stream));
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, devOutput,
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
    resources.check(devHandle->deviceMemcpy(regBuff, hostBuf,
                                            count * sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
    resources.check(devHandle->streamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    resources.check(launchKernelIntraAllReduce(devMem, count, flagcxFloat,
                                               devComm, stream));
    resources.check(devHandle->streamSynchronize(stream));

    resources.check(devHandle->deviceMemcpy(hostBuf, regBuff,
                                            count * sizeof(float),
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
    reporter.report("K10", size, k10Ok ? "PASS" : "FAIL");
    allPass &= k10Ok;
#else
    reporter.report("K10", size, "SKIP");
    // All three wait on ranks before reading what they wrote: K8/K9 go
    // through flagcxDevBarrier<Intra>, and K10 is the same composite the
    // barrier feeds.  The IPC pass has neither the peer mapping nor a
    // barrier the atomic layer can support (struct CceAtomic,
    // ascend_platform_traits.h:201, rejecting add/sub/bitOr/bitAnd/exch/cas
    // at :439-488), so its build compiles these cases out; the cann-shmem
    // build backs all three and grades them (test/kernel/ascend/
    // device_api.asc).
#endif

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
#if defined(USE_NVIDIA_ADAPTOR) && defined(FLAGCX_COMM_TRAITS_DEFAULT)
  // A singleton Inter team still reserves one signal slot per requested
  // barrier channel. Exercise the Barrier-only creation path without user
  // signal or counter resources.
  flagcxDevCommRequirements interOnlyReqs =
      FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  interOnlyReqs.interBarrierCount = 1;
  auto &interOnlyComm = resources.interOnlyComm;
  resources.check(flagcxDevCommCreate(comm, &interOnlyReqs, &interOnlyComm));
  resources.check(flagcxDevCommDestroy(comm, interOnlyComm));
  interOnlyComm = nullptr;
#endif
  resources.check(resources.finish());
  return globalPass ? 0 : 1;
}

int main(int argc, char *argv[]) {
  return deviceApiTestMain(argc, argv, runDeviceApiTest);
}
