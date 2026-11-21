/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Unified Inter Suite Tests — INTER + WORLD teams
 * Tests one-sided operations with inter-node communication.
 *
 * Tests S16–S25 (INTER + WORLD teams across cooperative variants):
 *   S16: DevBarrier — INTER + WORLD (merged)
 *   S17: DevTeamResolution — INTER + WORLD
 *   S18: DevPut — INTER + WORLD
 *   S19: DevGet — INTER + WORLD
 *   S20: DevSignalStandalone — INTER + WORLD
 *   S21: DevPutSignalWait — INTER + WORLD
 *   S22: DevPut_RSig — INTER + WORLD
 *   S23: DevPutCounter — INTER + WORLD
 *   S24: DevPutValue_RSig — INTER + WORLD
 *   S25: DevSignalShadowFlush — INTER + WORLD
 *
 * Requirements:
 *   - Multi-node with 2+ nodes (requires INTER team)
 *   - FLAGCX_USE_HETERO_COMM=1 (for DevComm)
 *
 * Usage: mpirun -np N ./test_device_ir_unified_inter [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 ************************************************************************/

#include "check.h"
#include "device_ir.h"
#include "device_signal_test.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "global_comm.h"
#include "tools.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

constexpr int kUnifiedIrSignalComboCount = 12;
constexpr int kUnifiedIrInterTeamResolutionComboCount = 9;

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
#define RPRINTF(...)                                                           \
  do {                                                                         \
    printf("[rank %d] ", proc);                                                \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  } while (0)
  MPI_Comm splitComm;
  uint64_t splitMask = 0;
  int color = 0;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);
  const auto signalOptions = flagcxSignalTestOptions(splitComm);

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

  // Create DevComm with signal/counter/barrier slots
  // Per CTA: 12 signal variants; S23 uses 18 signal/counter slots.
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interContextCount = signalOptions.contexts;
  reqs.interSignalCount = kFlagcxTestUnifiedCompletionCount;
  reqs.interCounterCount = kFlagcxTestUnifiedCompletionCount;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Six payload groups (coop x team), partitioned across CTAs. Completion
  // slots stay independent; variants reuse payload only after host validation.
  size_t bufSize =
      unifiedPayloadBufferBytes(maxBytes, size_t(9) * size_t(totalProcs));
  if (bufSize == 0) {
    std::fprintf(stderr, "Unified payload allocation size overflow\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (proc == 0)
    printf(
        "# payload: %d groups, %zu bytes per buffer; CTAs share each group\n",
        kFlagcxTestPayloadGroups, bufSize);
  void *sendBuff = nullptr, *recvBuff = nullptr;
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, bufSize, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, bufSize, memAllocator));

  // Register symmetric windows
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr;
  FLAGCXCHECK(flagcxCommWindowRegister(comm, sendBuff, bufSize, &sendWin,
                                       flagcxTestWindowFlags(), memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, bufSize, &recvWin,
                                       flagcxTestWindowFlags(), memAllocator));

  // Create DevMem handles
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  FLAGCXCHECK(flagcxDevMemCreate(comm, sendBuff, bufSize, sendWin, &sendMem));
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

#ifndef FLAGCX_TEST_NO_ATOMIC_COMPLETION
  FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                      flagcxMemDevice, stream));
  launchKernelSignalTestContextCountS(devCommPtr, devResults, stream);
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  int actualContexts = 0;
  FLAGCXCHECK(devHandle->deviceMemcpy(&actualContexts, devResults, sizeof(int),
                                      flagcxMemcpyDeviceToHost, stream));
  FLAGCXCHECK(devHandle->streamSynchronize(stream));
  flagcxSignalTestCheckContexts(actualContexts, splitComm);
#endif

  // Host scratch mirrors the already combination-scaled device buffers.
  float *hostSend = new float[bufSize / sizeof(float)];
  float *hostRecv = new float[bufSize / sizeof(float)];

  // Team geometry
  int intraSize = comm->localRanks;
  int intraRank = comm->localRank;
  int nNodes = totalProcs / intraSize;
  int nodeIdx = proc / intraSize;

  if (proc == 0) {
    printf("=== Device IR Unified Inter Suite (INTER + WORLD teams) ===\n");
    printf("Ranks: %d, Nodes: %d, IntraSize: %d\n", totalProcs, nNodes,
           intraSize);
    if (FLAGCX_TEST_UNIFIED_INTER_PUT_COOP_EXPECTED_MASK ==
        FLAGCX_TEST_UNIFIED_PUT_COOP_ALL_MASK) {
      printf("S21-S22 expected cases: THREAD/WARP/BLOCK x 2 variants x "
             "INTER/WORLD\n");
      printf("S23 expected cases: THREAD/WARP/BLOCK x 3 variants x "
             "INTER/WORLD\n\n");
    } else {
      printf("S21-S22 expected cases: WARP/BLOCK x 2 variants x "
             "INTER/WORLD\n");
      printf("S23 expected cases: WARP/BLOCK x 3 variants x INTER/WORLD\n");
      printf("S21-S23 unsupported coop cases: THREAD x INTER/WORLD\n\n");
    }
  }

  if (nNodes < 2) {
    if (proc == 0) {
      printf("ERROR: Inter suite requires multi-node setup (nNodes >= 2)\n");
      printf("Current configuration: nNodes = %d\n", nNodes);
    }
    MPI_Finalize();
    return 1;
  }

  bool allPass = true;
  const uint32_t expectedPutCoopMask =
      FLAGCX_TEST_UNIFIED_INTER_PUT_COOP_EXPECTED_MASK;
  const uint32_t expectedPutCounterMask =
      FLAGCX_TEST_UNIFIED_INTER_PUT_COUNTER_EXPECTED_MASK;
  // Each completion scenario must publish a verdict from every CTA.
  int zeroResults[kFlagcxTestUnifiedCtas] = {};

  // =========================================================================
  // Main test loop
  // =========================================================================
  for (int iteration = 0; iteration < signalOptions.repeats; ++iteration) {
    for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
      size_t bytes = unifiedPayloadBytes(size);
      size_t count = bytes / sizeof(float);

      if (proc == 0)
        printf("# Size = %zu bytes/group (requested %zu), count = %zu\n", bytes,
               size, count);

      MPI_Barrier(MPI_COMM_WORLD);

      // =======================================================================
      // S16: DevBarrier — 3 coop kinds × 2 teams × 2 forms = 12 cases
      // =======================================================================
      {
        int hostResults[4];
        bool s16Pass = true;

        // Sub-block A: BarrierSync
        FLAGCXCHECK(devHandle->deviceMemset(
            devResults, 0, FLAGCX_DEVICE_CTA_COUNT * sizeof(int),
            flagcxMemDevice, stream));
        launchKernelDevBarrierInterWorldS(devCommPtr, devResults, stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                            4 * sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        for (int i = 0; i < 4; i++) {
          if (hostResults[i] != 1) {
            s16Pass = false;
            break;
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Sub-block B: BarrierArrive + BarrierWait (split)
        FLAGCXCHECK(devHandle->deviceMemset(
            devResults, 0, FLAGCX_DEVICE_CTA_COUNT * sizeof(int),
            flagcxMemDevice, stream));
        launchKernelDevBarrierArriveWaitInterWorldS(devCommPtr, devResults,
                                                    stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                            4 * sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        for (int i = 0; i < 4; i++) {
          if (hostResults[i] != 1) {
            s16Pass = false;
            break;
          }
        }

        RPRINTF("S16 DevBarrier(INTER+WORLD): %s\n", s16Pass ? "PASS" : "FAIL");
        allPass &= s16Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // =======================================================================
      // S17: DevTeamResolution — 3 coop × INTER/WORLD/INTRA = 9 cases
      // =======================================================================
      {
        int maxRanks = totalProcs;
        if (nNodes > maxRanks)
          maxRanks = nNodes;
        size_t s17Size =
            kUnifiedIrInterTeamResolutionComboCount * maxRanks * sizeof(float);

        // Keep rank 0 distinguishable from the zero-initialized receive buffer.
        float myTag = (float)(proc + 1);
        FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, &myTag, sizeof(float),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, s17Size,
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevTeamResolutionInterWorldS(
            devCommPtr, recvMemPtr, sendMemPtr, devResults, stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes = 0;
        FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        float *s17Recv =
            new float[kUnifiedIrInterTeamResolutionComboCount * maxRanks];
        FLAGCXCHECK(devHandle->deviceMemcpy(s17Recv, recvBuff, s17Size,
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s17Pass = (hostRes == 1);
        int prevNode = (nodeIdx + nNodes - 1) % nNodes;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;

        for (int combo = 0; combo < 6 && s17Pass; combo++) {
          int teamIdx = combo % 2; // 0=INTER, 1=WORLD
          size_t baseOff = combo * maxRanks;

          if (teamIdx == 0) { // INTER
            float expected = (float)(prevNode * intraSize + intraRank + 1);
            if (s17Recv[baseOff + prevNode] != expected) {
              s17Pass = false;
            }
          } else { // WORLD
            float expected = (float)(prevWorld + 1);
            if (s17Recv[baseOff + prevWorld] != expected) {
              s17Pass = false;
            }
          }
        }

        // Combinations 6-8 are THREAD/WARP/BLOCK INTRA rings on the same
        // virtual-node topology.  They specifically cover ranks on node 1,
        // where worldRank != intraRank.
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        for (int combo = 6;
             combo < kUnifiedIrInterTeamResolutionComboCount && s17Pass;
             combo++) {
          float expected = (float)(nodeIdx * intraSize + prevIntra + 1);
          if (s17Recv[combo * maxRanks + prevIntra] != expected) {
            s17Pass = false;
          }
        }

        RPRINTF("S17 DevTeamResolution(INTER+WORLD+INTRA-locality): %s\n",
                s17Pass ? "PASS" : "FAIL");
        allPass &= s17Pass;
        delete[] s17Recv;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // =======================================================================
      // S18: DevPut + DevPutValue — INTER + WORLD
      // =======================================================================
      {
        // --- Sub-block A: DevPut ---
        for (size_t i = 0; i < 6 * count; i++)
          hostSend[i] = (float)(proc * 1000 + i);
        FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 6 * bytes,
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevPutInterWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                      devResults, bytes, stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes = 0;
        FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s18Pass = (hostRes == 1);
        int prevNode = (nodeIdx + nNodes - 1) % nNodes;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;
        int prevNodeBase = prevNode * intraSize + intraRank;

        for (int combo = 0; combo < 6 && s18Pass; combo++) {
          int teamIdx = combo % 2;
          size_t off = combo * count;
          int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;
          for (size_t i = 0; i < count && s18Pass; i++) {
            float expected = (float)(senderRank * 1000 + off + i);
            if (hostRecv[off + i] != expected)
              s18Pass = false;
          }
        }

        // --- Sub-block B: DevPutValue (6 combos, scalar uint64 per slot) ---
        FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * sizeof(uint64_t),
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevPutValueInterWorldS(devCommPtr, recvMemPtr, devResults,
                                           bytes, stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        uint64_t hostRecvV[6];
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRecvV, recvBuff,
                                            6 * sizeof(uint64_t),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        if (hostRes != 1)
          s18Pass = false;
        for (int combo = 0; combo < 6 && s18Pass; combo++) {
          int teamIdx = combo % 2;
          int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;
          uint64_t expected = (uint64_t)(senderRank * 100 + combo);
          if (hostRecvV[combo] != expected)
            s18Pass = false;
        }

        RPRINTF("S18 DevPut+PutValue(INTER+WORLD): %s\n",
                s18Pass ? "PASS" : "FAIL");
        allPass &= s18Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // =======================================================================
      // S19: DevGet — INTER + WORLD
      // =======================================================================
#ifdef FLAGCX_TEST_NO_REMOTE_READ
      RPRINTF(
          "S19 DevGet(INTER+WORLD): SKIP (no remote read on this device)\n");
      MPI_Barrier(MPI_COMM_WORLD);
#else
      {
        for (size_t i = 0; i < 6 * count; i++)
          hostSend[i] = (float)(proc * 2000 + i);
        FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 6 * bytes,
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevGetInterWorldS(devCommPtr, sendMemPtr, recvMemPtr,
                                      devResults, bytes, stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes = 0;
        FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s19Pass = (hostRes == 1);
        int nextNode = (nodeIdx + 1) % nNodes;
        int nextWorld = (proc + 1) % totalProcs;
        int nextNodeBase = nextNode * intraSize + intraRank;

        for (int combo = 0; combo < 6 && s19Pass; combo++) {
          int teamIdx = combo % 2;
          size_t off = combo * count;
          int sourceRank = (teamIdx == 0) ? nextNodeBase : nextWorld;

          for (size_t i = 0; i < count && s19Pass; i++) {
            float expected = (float)(sourceRank * 2000 + off + i);
            if (hostRecv[off + i] != expected) {
              s19Pass = false;
            }
          }
        }

        RPRINTF("S19 DevGet(INTER+WORLD): %s\n", s19Pass ? "PASS" : "FAIL");
        allPass &= s19Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

      // =======================================================================
      // S20: DevSignalStandalone — 3 coop × 2 teams × 2 actions = 12 cases
      //      — INTER + WORLD
      // =======================================================================
      {
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevSignalStandaloneInterWorldS(devCommPtr, devResults,
                                                   stream, signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s20Pass = flagcxUnifiedSignalResults(hostRes);
        RPRINTF("S20 DevSignalStandalone(INTER+WORLD): %s\n",
                s20Pass ? "PASS" : "FAIL");
        allPass &= s20Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // S21/S22: two signal variants. S23: three completion variants.
      // Each launch runs six coop x team groups, split across all CTAs.
      // Check every variant before reusing these groups for the next round.
      for (int scenario = 21; scenario <= 23; ++scenario) {
        const int variants = scenario == 23 ? 3 : 2;
        const char *names[] = {"DevPutSignalWait", "DevPut_RSig",
                               "DevPutCounter"};
        const uint32_t expectedMask =
            scenario == 23 ? expectedPutCounterMask : expectedPutCoopMask;
        bool scenarioPass = true;
        int prevNode = (nodeIdx + nNodes - 1) % nNodes;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;
        int prevNodeBase = prevNode * intraSize + intraRank;
        for (int variant = 0; variant < variants; ++variant) {
          const int roundTag = iteration * 3 + variant;
          const size_t payloadBytes = kFlagcxTestPayloadGroups * bytes;
          for (size_t i = 0; i < kFlagcxTestPayloadGroups * count; ++i)
            hostSend[i] = unifiedPayloadValue(proc, scenario, roundTag, i);
          FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, payloadBytes,
                                              flagcxMemcpyHostToDevice,
                                              stream));
          // NaN sentinel: a missing write cannot match a previous round's data.
          FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0xff, payloadBytes,
                                              flagcxMemDevice, stream));
          FLAGCXCHECK(devHandle->deviceMemcpy(
              devResults, zeroResults, sizeof(zeroResults),
              flagcxMemcpyHostToDevice, stream));
          FLAGCXCHECK(devHandle->streamSynchronize(stream));
          MPI_Barrier(MPI_COMM_WORLD);

          if (scenario == 21)
            launchKernelDevPutSignalWaitInterWorldS(
                devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes, stream,
                signalOptions.skew, variant);
          else if (scenario == 22)
            launchKernelDevPutRSigInterWorldS(
                devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes, stream,
                signalOptions.skew, variant);
          else
            launchKernelDevPutCounterInterWorldS(
                devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes, stream,
                signalOptions.skew, variant);
          FLAGCXCHECK(devHandle->streamSynchronize(stream));
          MPI_Barrier(MPI_COMM_WORLD);

          int hostRes[kFlagcxTestUnifiedCtas] = {};
          FLAGCXCHECK(
              devHandle->deviceMemcpy(hostRes, devResults, sizeof(hostRes),
                                      flagcxMemcpyDeviceToHost, stream));
          FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, payloadBytes,
                                              flagcxMemcpyDeviceToHost,
                                              stream));
          FLAGCXCHECK(devHandle->streamSynchronize(stream));
          bool roundPass = flagcxUnifiedSignalResults(hostRes);
          for (int group = 0; group < kFlagcxTestPayloadGroups; ++group) {
            int combo = unifiedPayloadCombo(group, variant, variants);
            if ((expectedMask & (uint32_t{1} << combo)) == 0)
              continue;
            int senderRank = group % 2 == 0 ? prevNodeBase : prevWorld;
            size_t off = size_t(group) * count;
            for (size_t i = 0; i < count; ++i) {
              float expected =
                  unifiedPayloadValue(senderRank, scenario, roundTag, off + i);
              if (hostRecv[off + i] != expected) {
                std::fprintf(
                    stderr,
                    "S%d variant=%d group=%d index=%zu payload mismatch\n",
                    scenario, variant, group, i);
                roundPass = false;
                break;
              }
            }
          }
          scenarioPass &= roundPass;
          // All ranks finish consuming this round before any buffer is reused.
          MPI_Barrier(MPI_COMM_WORLD);
        }
        RPRINTF("S%d %s(INTER+WORLD): %s\n", scenario, names[scenario - 21],
                scenarioPass ? "PASS" : "FAIL");
        allPass &= scenarioPass;
      }

      // =======================================================================
      // S24: DevPutValue_RSigInc/Add — 3 coop × 2 teams × 2 variants = 12
      // ResetSignal → assert ReadSignal==0 → PutValue_RSigInc/RSigAdd →
      // WaitSignal → assert ReadSignal==expected
      // Each combo writes 1 uint64_t scalar value.
      // =======================================================================
      {
        FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0,
                                            kFlagcxTestUnifiedCtas *
                                                kUnifiedIrSignalComboCount *
                                                sizeof(uint64_t),
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevPutValueRSigInterWorldS(devCommPtr, recvMemPtr,
                                               devResults, bytes, stream,
                                               signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        uint64_t hostRecvV[kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount];
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRecvV, recvBuff,
                                            kFlagcxTestUnifiedCtas *
                                                kUnifiedIrSignalComboCount *
                                                sizeof(uint64_t),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s24Pass = flagcxUnifiedSignalResults(hostRes);
        int prevNode = (nodeIdx + nNodes - 1) % nNodes;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;
        int prevNodeBase = prevNode * intraSize + intraRank;

        for (int payload = 0;
             payload < kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount &&
             s24Pass;
             ++payload) {
          int combo = payload % kUnifiedIrSignalComboCount;
          int teamIdx = combo % 2;
          int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;
          uint64_t expected = (uint64_t)(senderRank * 100 + payload);
          if (hostRecvV[payload] != expected) {
            s24Pass = false;
          }
        }

        RPRINTF("S24 DevPutValue_RSig(INTER+WORLD): %s\n",
                s24Pass ? "PASS" : "FAIL");
        allPass &= s24Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // =======================================================================
      // S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush
      //      — INTER + WORLD
      // ResetSignal → assert ReadSignal==0 → IncreaseSignalShadow(5) →
      // SignalInc × 5 → WaitSignalMeetShadow → assert ReadSignal==5 → Flush
      // =======================================================================
      {
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevSignalShadowFlushInterWorldS(devCommPtr, devResults,
                                                    stream, signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s25Pass = flagcxUnifiedSignalResults(hostRes);
        RPRINTF("S25 DevSignalShadowFlush(INTER+WORLD): %s\n",
                s25Pass ? "PASS" : "FAIL");
        allPass &= s25Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      if (proc == 0)
        printf("\n");
    }
  } // repeated suite

  int localPass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&localPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  // Cleanup
  delete[] hostSend;
  delete[] hostRecv;
  FLAGCXCHECK(devHandle->deviceFree(devResults, flagcxMemDevice, NULL));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, sendMem));
  FLAGCXCHECK(flagcxDevMemDestroy(comm, recvMem));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, sendWin, memAllocator));
  FLAGCXCHECK(flagcxCommWindowDeregister(comm, recvWin, memAllocator));
  FLAGCXCHECK(flagcxMemFree(sendBuff, memAllocator));
  FLAGCXCHECK(flagcxMemFree(recvBuff, memAllocator));
  FLAGCXCHECK(flagcxDevCommDestroy(comm, devComm));
  FLAGCXCHECK(devHandle->streamDestroy(stream));
  FLAGCXCHECK(flagcxCommDestroy(comm));

  if (proc == 0) {
    printf("=== Final Result: %s ===\n",
           globalPass ? "ALL PASS" : "SOME FAILED");
  }

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
