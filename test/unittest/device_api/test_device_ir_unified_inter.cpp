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
  // Inter suite uses 6 combinations (INTER + WORLD) for most tests
  // S18 uses 8 signal slots (includes extra BLOCK single-leader patterns)
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount =
      8; // 8 slots for S18 (6 standard + 2 BLOCK single-leader)
  reqs.interCounterCount = 6; // 6 slots for S23 per-combo counter tracking

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Allocate send/recv buffers (8x for S18's 8-combination regions)
  size_t bufSize = maxBytes * 8;
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
                                       FLAGCX_WIN_COLL_SYMMETRIC,
                                       memAllocator));
  FLAGCXCHECK(flagcxCommWindowRegister(comm, recvBuff, bufSize, &recvWin,
                                       FLAGCX_WIN_COLL_SYMMETRIC,
                                       memAllocator));

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
    printf("Ranks: %d, Nodes: %d, IntraSize: %d\n\n", totalProcs, nNodes,
           intraSize);
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
  // S21-S25 atomically clear this value if any device context fails.
  int passResult = 1;

  // =========================================================================
  // Main test loop
  // =========================================================================
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t count = size / sizeof(float);
    if (count == 0)
      count = 1;
    size_t bytes = count * sizeof(float);

    if (proc == 0)
      printf("# Size = %zu bytes, count = %zu\n", bytes, count);

    MPI_Barrier(MPI_COMM_WORLD);

    // =======================================================================
    // S16: DevBarrier — INTER + WORLD (BarrierSync + ArriveWait)
    // =======================================================================
    {
      int hostResults[4];
      bool s16Pass = true;

      // Sub-block A: BarrierSync
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0,
                                          FLAGCX_DEVICE_CTA_COUNT * sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevBarrierInterWorldS(devCommPtr, devResults, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                          4 * sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));
      for (int i = 0; i < 4; i++) {
        if (hostResults[i] != 1) {
          s16Pass = false;
          break;
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // Sub-block B: BarrierArrive + BarrierWait (split)
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0,
                                          FLAGCX_DEVICE_CTA_COUNT * sizeof(int),
                                          flagcxMemDevice, stream));
      launchKernelDevBarrierArriveWaitInterWorldS(devCommPtr, devResults,
                                                  stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(hostResults, devResults,
                                          4 * sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));
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
    // S17: DevTeamResolution — INTER + WORLD, plus nonzero-node INTRA
    // =======================================================================
    {
      int maxRanks = totalProcs;
      if (nNodes > maxRanks)
        maxRanks = nNodes;
      size_t s17Size = 7 * maxRanks * sizeof(float);

      float myTag = (float)proc;
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, &myTag, sizeof(float),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, s17Size, flagcxMemDevice,
                                          stream));
      FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevTeamResolutionInterWorldS(devCommPtr, recvMemPtr,
                                               sendMemPtr, devResults, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      float *s17Recv = new float[7 * maxRanks];
      FLAGCXCHECK(devHandle->deviceMemcpy(s17Recv, recvBuff, s17Size,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s17Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;

      for (int combo = 0; combo < 6 && s17Pass; combo++) {
        int teamIdx = combo % 2; // 0=INTER, 1=WORLD
        size_t baseOff = combo * maxRanks;

        if (teamIdx == 0) { // INTER
          float expected = (float)(prevNode * intraSize + intraRank);
          if (s17Recv[baseOff + prevNode] != expected) {
            s17Pass = false;
          }
        } else { // WORLD
          float expected = (float)prevWorld;
          if (s17Recv[baseOff + prevWorld] != expected) {
            s17Pass = false;
          }
        }
      }

      // Combination 6 is an INTRA ring on the same virtual-node topology.
      // It specifically covers ranks on node 1, where worldRank != intraRank.
      if (s17Pass) {
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        float expected = (float)(nodeIdx * intraSize + prevIntra);
        if (s17Recv[6 * maxRanks + prevIntra] != expected) {
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
      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

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
      uint64_t hostRecvV[6];
      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecvV, recvBuff,
                                          6 * sizeof(uint64_t),
                                          flagcxMemcpyDeviceToHost, stream));
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

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

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

    // =======================================================================
    // S20: DevSignalStandalone (signal-only: Inc+Add+Wait+Read+Reset)
    //      — INTER + WORLD
    // =======================================================================
    {
      FLAGCXCHECK(devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevSignalStandaloneInterWorldS(devCommPtr, devResults,
                                                 stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      bool s20Pass = (hostRes == 1);
      RPRINTF("S20 DevSignalStandalone(INTER+WORLD): %s\n",
              s20Pass ? "PASS" : "FAIL");
      allPass &= s20Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S21: DevPutSignalWait — INTER + WORLD (6 combinations, split put+signal)
    // ResetSignal → Put → SignalInc/SignalAdd → WaitSignal → ReadSignal verify
    // =======================================================================
    {
      for (size_t i = 0; i < 6 * count; i++)
        hostSend[i] = (float)(proc * 3000 + i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 6 * bytes,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutSignalWaitInterWorldS(
          devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s21Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;
      int prevNodeBase = prevNode * intraSize + intraRank;

      for (int combo = 0; combo < 6 && s21Pass; combo++) {
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;

        for (size_t i = 0; i < count && s21Pass; i++) {
          float expected = (float)(senderRank * 3000 + off + i);
          if (hostRecv[off + i] != expected) {
            s21Pass = false;
          }
        }
      }

      RPRINTF("S21 DevPutSignalWait(INTER+WORLD): %s\n",
              s21Pass ? "PASS" : "FAIL");
      allPass &= s21Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S22: DevPut_RSigInc + DevPut_RSigAdd — INTER + WORLD
    // ResetSignal → assert ReadSignal==0 → Put_RSigInc/RSigAdd →
    // WaitSignal → assert ReadSignal==expected
    // =======================================================================
    {
      for (size_t i = 0; i < 6 * count; i++)
        hostSend[i] = (float)(proc * 4000 + i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 6 * bytes,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutRSigInterWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                        devResults, bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s22Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;
      int prevNodeBase = prevNode * intraSize + intraRank;

      for (int combo = 0; combo < 6 && s22Pass; combo++) {
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;

        for (size_t i = 0; i < count && s22Pass; i++) {
          float expected = (float)(senderRank * 4000 + off + i);
          if (hostRecv[off + i] != expected) {
            s22Pass = false;
          }
        }
      }

      RPRINTF("S22 DevPut_RSig(INTER+WORLD): %s\n", s22Pass ? "PASS" : "FAIL");
      allPass &= s22Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc
    //      — INTER + WORLD (counter mega-scenario)
    // ResetCounter/Signal → assert Read==0 → Put_*LCtrInc variants →
    // WaitCounter → assert ReadCounter==1 → WaitSignal (if applicable) →
    // assert ReadSignal==expected
    // =======================================================================
    {
      for (size_t i = 0; i < 6 * count; i++)
        hostSend[i] = (float)(proc * 5000 + i);
      FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend, 6 * bytes,
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutCounterInterWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                           devResults, bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff, 6 * bytes,
                                          flagcxMemcpyDeviceToHost, stream));

      bool s23Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;
      int prevNodeBase = prevNode * intraSize + intraRank;

      for (int combo = 0; combo < 6 && s23Pass; combo++) {
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;

        for (size_t i = 0; i < count && s23Pass; i++) {
          float expected = (float)(senderRank * 5000 + off + i);
          if (hostRecv[off + i] != expected) {
            s23Pass = false;
          }
        }
      }

      RPRINTF("S23 DevPutCounter(INTER+WORLD): %s\n",
              s23Pass ? "PASS" : "FAIL");
      allPass &= s23Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // =======================================================================
    // S24: DevPutValue_RSigInc + DevPutValue_RSigAdd — INTER + WORLD
    // ResetSignal → assert ReadSignal==0 → PutValue_RSigInc/RSigAdd →
    // WaitSignal → assert ReadSignal==expected
    // Each combo writes 1 uint64_t scalar value.
    // =======================================================================
    {
      FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0, 6 * sizeof(uint64_t),
                                          flagcxMemDevice, stream));
      FLAGCXCHECK(devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutValueRSigInterWorldS(devCommPtr, recvMemPtr, devResults,
                                             bytes, stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      uint64_t hostRecvV[6];
      FLAGCXCHECK(devHandle->deviceMemcpy(hostRecvV, recvBuff,
                                          6 * sizeof(uint64_t),
                                          flagcxMemcpyDeviceToHost, stream));

      bool s24Pass = (hostRes == 1);
      int prevNode = (nodeIdx + nNodes - 1) % nNodes;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;
      int prevNodeBase = prevNode * intraSize + intraRank;

      for (int combo = 0; combo < 6 && s24Pass; combo++) {
        int teamIdx = combo % 2;
        int senderRank = (teamIdx == 0) ? prevNodeBase : prevWorld;
        uint64_t expected = (uint64_t)(senderRank * 100 + combo);
        if (hostRecvV[combo] != expected) {
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
      FLAGCXCHECK(devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                          flagcxMemcpyHostToDevice, stream));
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevSignalShadowFlushInterWorldS(devCommPtr, devResults,
                                                  stream);
      FLAGCXCHECK(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                          flagcxMemcpyDeviceToHost, stream));

      bool s25Pass = (hostRes == 1);
      RPRINTF("S25 DevSignalShadowFlush(INTER+WORLD): %s\n",
              s25Pass ? "PASS" : "FAIL");
      allPass &= s25Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (proc == 0)
      printf("\n");
  }

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
    printf("=== Final Result: %s ===\n", allPass ? "ALL PASS" : "SOME FAILED");
  }

  MPI_Finalize();
  return allPass ? 0 : 1;
}
