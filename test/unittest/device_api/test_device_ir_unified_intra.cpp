/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR Unified Intra Suite Tests — INTRA + WORLD teams
 * Tests one-sided operations with intra-node communication.
 *
 * Tests S16–S25 (INTRA + WORLD teams across cooperative variants):
 *   S16: DevBarrier — INTRA + WORLD (merged)
 *   S17: DevTeamResolution — INTRA + WORLD
 *   S18: DevPut — INTRA + WORLD
 *   S19: DevGet — INTRA + WORLD
 *   S20: DevSignalStandalone — INTRA + WORLD
 *   S21: DevPutSignalWait — INTRA + WORLD
 *   S22: DevPut_RSig — INTRA + WORLD
 *   S23: DevPutCounter — INTRA + WORLD
 *   S24: DevPutValue_RSig — INTRA + WORLD
 *   S25: DevSignalShadowFlush — INTRA + WORLD
 *
 * Requirements:
 *   - Single-node with 2+ GPUs (P2P path)
 *   - FLAGCX_USE_HETERO_COMM=1 (for DevComm)
 *
 * Usage: mpirun -np N ./test_device_ir_unified_intra [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 ************************************************************************/

#include "check.h"
#include "device_ir.h"
#include "device_signal_test.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

constexpr int kUnifiedIrSignalComboCount = 12;
constexpr int kUnifiedIrS23ComboCount = 18;

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
  // Signal variants use 12 slots; S23 uses 18 signal/counter slots.
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interContextCount = signalOptions.contexts;
  reqs.interSignalCount = kFlagcxTestUnifiedCompletionCount;
  reqs.interCounterCount = kFlagcxTestUnifiedCompletionCount;

  flagcxDevComm_t devComm = nullptr;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Allocate send/recv buffers for S23, the largest combination matrix.
  if (maxBytes > SIZE_MAX / kFlagcxTestUnifiedCompletionCount)
    MPI_Abort(MPI_COMM_WORLD, 1);
  size_t bufSize = std::max(maxBytes, size_t(sizeof(uint64_t))) *
                   kFlagcxTestUnifiedCompletionCount;
  void *sendBuff = nullptr, *recvBuff = nullptr;
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  flagcxMemAllocator_t memAllocator = flagcxMemSHMEM;
#else
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
#endif
  FLAGCXCHECK(flagcxMemAlloc(&recvBuff, bufSize, memAllocator));
  FLAGCXCHECK(flagcxMemAlloc(&sendBuff, bufSize, memAllocator));

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

  // Team geometry (single-node: intraSize == totalProcs, nNodes == 1)
  int intraSize = totalProcs;
  int intraRank = proc;

  if (proc == 0) {
    printf("=== Device IR Unified Intra Suite (INTRA + WORLD teams) ===\n");
    printf("Ranks: %d, IntraSize: %d\n", totalProcs, intraSize);
    printf("S21-S22 expected cases: THREAD/WARP/BLOCK x 2 variants x "
           "INTRA/WORLD\n");
    printf("S23 expected cases: THREAD/WARP/BLOCK x 3 variants x "
           "INTRA/WORLD\n\n");
#ifdef FLAGCX_TEST_NO_CROSS_RANK_COMM
    // Say it on the banner (MC952 review P1).  A skipped case never
    // reaches the report as a pass, and what remains runnable here must
    // never be read as cross-rank certification.
    printf("NOTE: S16-S18 capability-gated SKIP (peer path not operable).\n"
           "This run grades launch/init machinery only: a world-size-1\n"
           "pass would certify LOCAL SMOKE at best, never cross-rank\n"
           "rendezvous.  The four-rank certification target stays open\n"
           "pending a working peer transport.\n\n");
#endif
  }

  bool allPass = true;
  // Verdicts actually executed (a capability SKIP does not increment).
  // If a build gates every case, the final line says NO CASES GRADED
  // rather than showing an empty "ALL PASS".
  int gradedCases = 0;
  const uint32_t expectedPutCoopMask =
      FLAGCX_TEST_UNIFIED_INTRA_PUT_COOP_EXPECTED_MASK;
  const uint32_t expectedPutCounterMask =
      FLAGCX_TEST_UNIFIED_INTRA_PUT_COUNTER_EXPECTED_MASK;
  // Each completion scenario must publish a verdict from every CTA.
#ifndef FLAGCX_TEST_NO_ATOMIC_COMPLETION
  int zeroResults[kFlagcxTestUnifiedCtas] = {};
#endif

  for (int iteration = 0; iteration < signalOptions.repeats; ++iteration) {
    for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
      size_t count = size / sizeof(float);
      if (count == 0)
        count = 1;
      size_t bytes = count * sizeof(float);

      if (proc == 0)
        printf("# Size = %zu bytes, count = %zu\n", bytes, count);

      MPI_Barrier(MPI_COMM_WORLD);

      // S16: DevBarrier — 3 coop kinds × 2 teams × 2 forms = 12 cases
#ifdef FLAGCX_TEST_NO_CROSS_RANK_COMM
      // No operable peer store/poll path on this backend: scalar stores into
      // IPC-imported peer GM fault-freeze the AI core even with
      // ENABLE_PEER_ACCESS (measured; see ascend.sh).  At world size 1 the
      // barrier's arrive/wait peer loop iterates over zero peers, so a PASS
      // here would certify nothing cross-rank - report SKIP instead.
      RPRINTF("S16 DevBarrier(INTRA+WORLD): SKIP (peer path not operable)\n");
      MPI_Barrier(MPI_COMM_WORLD);
#else
      const char *skipS16 = std::getenv("FLAGCX_CI_UNIFIED_IR_SKIP_S16");
      if (skipS16 != nullptr && skipS16[0] != '\0' && skipS16[0] != '0') {
        if (proc == 0) {
          const char *reason =
              std::getenv("FLAGCX_CI_UNIFIED_IR_SKIP_S16_REASON");
          printf("S16 DevBarrier(INTRA+WORLD): SKIP (CI ignore_test: %s)\n",
                 reason != nullptr ? reason : "temporarily blacklisted");
          fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      } else {
        int hostResults[4];
        bool s16Pass = true;

        // Sub-block A: BarrierSync
        FLAGCXCHECK(devHandle->deviceMemset(
            devResults, 0, FLAGCX_DEVICE_CTA_COUNT * sizeof(int),
            flagcxMemDevice, stream));
        launchKernelDevBarrierIntraWorldS(devCommPtr, devResults, stream);
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
        launchKernelDevBarrierArriveWaitIntraWorldS(devCommPtr, devResults,
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

        RPRINTF("S16 DevBarrier(INTRA+WORLD): %s\n", s16Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s16Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

      // S17: DevTeamResolution — INTRA + WORLD (6 combinations)
#ifdef FLAGCX_TEST_NO_CROSS_RANK_COMM
      // Each combo's expected value is the PREVIOUS peer rank's tag; at world
      // size 1 the resolution degenerates to a self read.  Same capability
      // gate as S16 (no operable peer path on this backend).
      RPRINTF("S17 DevTeamResolution(INTRA+WORLD): SKIP (peer path not\n"
              "operable)\n");
      MPI_Barrier(MPI_COMM_WORLD);
#else
      {
        int maxRanks = intraSize;
        if (totalProcs > maxRanks)
          maxRanks = totalProcs;
        size_t s17Size = 6 * maxRanks * sizeof(float);

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

        launchKernelDevTeamResolutionIntraWorldS(
            devCommPtr, recvMemPtr, sendMemPtr, devResults, stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes = 0;
        FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        float *s17Recv = new float[6 * maxRanks];
        FLAGCXCHECK(devHandle->deviceMemcpy(s17Recv, recvBuff, s17Size,
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s17Pass = (hostRes == 1);
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;

        for (int combo = 0; combo < 6 && s17Pass; combo++) {
          int teamIdx = combo % 2; // 0=INTRA, 1=WORLD
          size_t baseOff = combo * maxRanks;

          if (teamIdx == 0) { // INTRA
            float expected = (float)(prevIntra + 1);
            if (s17Recv[baseOff + prevIntra] != expected) {
              s17Pass = false;
            }
          } else { // WORLD
            float expected = (float)(prevWorld + 1);
            if (s17Recv[baseOff + prevWorld] != expected) {
              s17Pass = false;
            }
          }
        }

        RPRINTF("S17 DevTeamResolution(INTRA+WORLD): %s\n",
                s17Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s17Pass;
        delete[] s17Recv;
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

      // S18: DevPut + DevPutValue — INTRA + WORLD
#ifdef FLAGCX_TEST_NO_CROSS_RANK_COMM
      // put/putValue to the previous rank; at world size 1 the destination
      // resolves to self, so the run is a copy through loopback, not the
      // cross-rank one-sided path the case certifies (MC952 review P1).
      RPRINTF("S18 DevPut+PutValue(INTRA+WORLD): SKIP (peer path not\n"
              "operable)\n");
      MPI_Barrier(MPI_COMM_WORLD);
#else
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

        launchKernelDevPutIntraWorldS(devCommPtr, recvMemPtr, sendMemPtr,
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
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;

        for (int combo = 0; combo < 6 && s18Pass; combo++) {
          int teamIdx = combo % 2;
          size_t off = combo * count;
          int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;
          for (size_t i = 0; i < count && s18Pass; i++) {
            float expected = (float)(senderRank * 1000 + off + i);
            if (hostRecv[off + i] != expected)
              s18Pass = false;
          }
        }

        // --- Sub-block B: DevPutValue (6 combos, scalar uint64 per slot) ---
        // Slot stride is a shared test parameter (MC952 review P1): the
        // kernel and this oracle must agree, and the historical layouts
        // differ.  Default keeps the packed 8-byte contract every other
        // backend's kernel already implements (offsets 0/8/16/24/32/40).
        // Ascend opts into 128-byte line isolation: dense adjacent u64 slots
        // flushed with whole-cache dcci commands tear rare writebacks
        // (~5e-4 per store, measured 2026-11-17), while one slot per GM line
        // ran zero loss over 80 runs.  That is a TEST-side layout choice on
        // a contributor-reported hardware effect; it is NOT a proof that
        // production putValue into adjacent uint64_t destinations is correct
        // on 910C.
        constexpr size_t kPutValueStrideBytes =
#ifdef FLAGCX_ASCEND_PUTVALUE_LINE_STRIDE
            128;
#else
            sizeof(uint64_t);
#endif
        constexpr size_t kPutValueStrideWords =
            kPutValueStrideBytes / sizeof(uint64_t);
        FLAGCXCHECK(devHandle->deviceMemset(
            recvBuff, 0, 6 * kPutValueStrideBytes, flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevPutValueIntraWorldS(devCommPtr, recvMemPtr, devResults,
                                           bytes, stream);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        FLAGCXCHECK(devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        uint64_t hostRecvV[6 * kPutValueStrideWords];
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRecvV, recvBuff,
                                            6 * kPutValueStrideBytes,
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        if (hostRes != 1)
          s18Pass = false;
        for (int combo = 0; combo < 6 && s18Pass; combo++) {
          int teamIdx = combo % 2;
          int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;
          uint64_t expected = (uint64_t)(senderRank * 100 + combo);
          if (hostRecvV[combo * kPutValueStrideWords] != expected)
            s18Pass = false;
        }

        RPRINTF("S18 DevPut+PutValue(INTRA+WORLD): %s\n",
                s18Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s18Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

      // S19: DevGet — INTRA + WORLD
      // Skipped where the device API cannot read a peer's memory at all (P800:
      // every remote-read path traps), so there is no implementation to check.
#ifdef FLAGCX_TEST_NO_REMOTE_READ
      RPRINTF(
          "S19 DevGet(INTRA+WORLD): SKIP (no remote read on this device)\n");
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

        launchKernelDevGetIntraWorldS(devCommPtr, sendMemPtr, recvMemPtr,
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
        int nextIntra = (intraRank + 1) % intraSize;
        int nextWorld = (proc + 1) % totalProcs;

        for (int combo = 0; combo < 6 && s19Pass; combo++) {
          int teamIdx = combo % 2;
          size_t off = combo * count;
          int sourceRank = (teamIdx == 0) ? nextIntra : nextWorld;

          for (size_t i = 0; i < count && s19Pass; i++) {
            float expected = (float)(sourceRank * 2000 + off + i);
            if (hostRecv[off + i] != expected) {
              s19Pass = false;
            }
          }
        }

        RPRINTF("S19 DevGet(INTRA+WORLD): %s\n", s19Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s19Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

      // =======================================================================
      // S20-S25 capability gate.  Each of these cases ends in a fetchAdd-
      // based completion (remote signal or local counter).  On platforms
      // whose device pass has no global-memory RMW instruction the
      // capability is declared absent in the platform traits, the matching
      // device_ir launchers stay declared-but-not-defined, and this macro
      // compiles the call sites out: a leftover reference would fail the
      // link instead of reporting a pass, and per this suite's contract a
      // skipped case never reaches the report as a pass.
      // =======================================================================
#ifdef FLAGCX_TEST_NO_ATOMIC_COMPLETION
      RPRINTF("S20 DevSignalStandalone(INTRA+WORLD): SKIP (no RMW "
              "completion atomic on this device)\n");
      RPRINTF("S21 DevPutSignalWait(INTRA+WORLD): SKIP (no RMW "
              "completion atomic on this device)\n");
      RPRINTF("S22 DevPut_RSig(INTRA+WORLD): SKIP (no RMW "
              "completion atomic on this device)\n");
      RPRINTF("S23 DevPutCounter(INTRA+WORLD): SKIP (no RMW "
              "completion atomic on this device)\n");
      RPRINTF("S24 DevPutValue_RSig(INTRA+WORLD): SKIP (no RMW "
              "completion atomic on this device)\n");
      RPRINTF("S25 DevSignalShadowFlush(INTRA+WORLD): SKIP (no RMW "
              "completion atomic on this device)\n");
      MPI_Barrier(MPI_COMM_WORLD);
#else
      // =======================================================================
      // S20: DevSignalStandalone — 3 coop × 2 teams × 2 actions = 12 cases
      //      — INTRA + WORLD
      // =======================================================================
      {
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevSignalStandaloneIntraWorldS(devCommPtr, devResults,
                                                   stream, signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s20Pass = flagcxUnifiedSignalResults(hostRes);
        RPRINTF("S20 DevSignalStandalone(INTRA+WORLD): %s\n",
                s20Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s20Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // S21: DevPutSignalWait — INTRA + WORLD (12 full combinations)
      // ResetSignal → Put → SignalInc/SignalAdd → WaitSignal → ReadSignal
      // verify
      {
        for (size_t i = 0;
             i < kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * count;
             i++)
          hostSend[i] = (float)(proc * 3000 + i);
        FLAGCXCHECK(devHandle->deviceMemcpy(
            sendBuff, hostSend,
            kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * bytes,
            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(
            recvBuff, 0,
            kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * bytes,
            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevPutSignalWaitIntraWorldS(devCommPtr, recvMemPtr,
                                                sendMemPtr, devResults, bytes,
                                                stream, signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        FLAGCXCHECK(devHandle->deviceMemcpy(
            hostRecv, recvBuff,
            kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * bytes,
            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s21Pass = flagcxUnifiedSignalResults(hostRes);
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;

        for (int payload = 0;
             payload < kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount &&
             s21Pass;
             ++payload) {
          int combo = payload % kUnifiedIrSignalComboCount;
          if ((expectedPutCoopMask & (uint32_t{1} << combo)) == 0)
            continue;
          int teamIdx = combo % 2;
          size_t off = payload * count;
          int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;

          for (size_t i = 0; i < count && s21Pass; i++) {
            float expected = (float)(senderRank * 3000 + off + i);
            if (hostRecv[off + i] != expected) {
              s21Pass = false;
            }
          }
        }

        RPRINTF("S21 DevPutSignalWait(INTRA+WORLD): %s\n",
                s21Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s21Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // =======================================================================
      // S22: DevPut_RSigInc + DevPut_RSigAdd — INTRA + WORLD
      // ResetSignal → assert ReadSignal==0 → Put_RSigInc/RSigAdd →
      // WaitSignal → assert ReadSignal==expected
      // =======================================================================
      {
        for (size_t i = 0;
             i < kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * count;
             i++)
          hostSend[i] = (float)(proc * 4000 + i);
        FLAGCXCHECK(devHandle->deviceMemcpy(
            sendBuff, hostSend,
            kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * bytes,
            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(
            recvBuff, 0,
            kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * bytes,
            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevPutRSigIntraWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                          devResults, bytes, stream,
                                          signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        FLAGCXCHECK(devHandle->deviceMemcpy(
            hostRecv, recvBuff,
            kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount * bytes,
            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s22Pass = flagcxUnifiedSignalResults(hostRes);
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;

        for (int payload = 0;
             payload < kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount &&
             s22Pass;
             ++payload) {
          int combo = payload % kUnifiedIrSignalComboCount;
          if ((expectedPutCoopMask & (uint32_t{1} << combo)) == 0)
            continue;
          int teamIdx = combo % 2;
          size_t off = payload * count;
          int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;

          for (size_t i = 0; i < count && s22Pass; i++) {
            float expected = (float)(senderRank * 4000 + off + i);
            if (hostRecv[off + i] != expected) {
              s22Pass = false;
            }
          }
        }

        RPRINTF("S22 DevPut_RSig(INTRA+WORLD): %s\n",
                s22Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s22Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // =======================================================================
      // S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc
      //      — INTRA + WORLD (counter mega-scenario)
      // ResetCounter/Signal → assert Read==0 → Put_*LCtrInc variants →
      // WaitCounter → assert ReadCounter==1 → WaitSignal (if applicable) →
      // assert ReadSignal==expected
      // =======================================================================
      {
        for (size_t i = 0;
             i < kFlagcxTestUnifiedCtas * kUnifiedIrS23ComboCount * count; i++)
          hostSend[i] = (float)(proc * 5000 + i);
        FLAGCXCHECK(devHandle->deviceMemcpy(sendBuff, hostSend,
                                            kFlagcxTestUnifiedCtas *
                                                kUnifiedIrS23ComboCount * bytes,
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemset(recvBuff, 0,
                                            kFlagcxTestUnifiedCtas *
                                                kUnifiedIrS23ComboCount * bytes,
                                            flagcxMemDevice, stream));
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevPutCounterIntraWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                             devResults, bytes, stream,
                                             signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        FLAGCXCHECK(devHandle->deviceMemcpy(hostRecv, recvBuff,
                                            kFlagcxTestUnifiedCtas *
                                                kUnifiedIrS23ComboCount * bytes,
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s23Pass = flagcxUnifiedSignalResults(hostRes);
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;

        for (int payload = 0;
             payload < kFlagcxTestUnifiedCtas * kUnifiedIrS23ComboCount &&
             s23Pass;
             ++payload) {
          int combo = payload % kUnifiedIrS23ComboCount;
          if ((expectedPutCounterMask & (uint32_t{1} << combo)) == 0)
            continue;
          int teamIdx = combo % 2;
          size_t off = payload * count;
          int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;

          for (size_t i = 0; i < count && s23Pass; i++) {
            float expected = (float)(senderRank * 5000 + off + i);
            if (hostRecv[off + i] != expected) {
              s23Pass = false;
            }
          }
        }

        RPRINTF("S23 DevPutCounter(INTRA+WORLD): %s\n",
                s23Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s23Pass;
        MPI_Barrier(MPI_COMM_WORLD);
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

        launchKernelDevPutValueRSigIntraWorldS(devCommPtr, recvMemPtr,
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
        int prevIntra = (intraRank + intraSize - 1) % intraSize;
        int prevWorld = (proc + totalProcs - 1) % totalProcs;

        for (int payload = 0;
             payload < kFlagcxTestUnifiedCtas * kUnifiedIrSignalComboCount &&
             s24Pass;
             ++payload) {
          int combo = payload % kUnifiedIrSignalComboCount;
          int teamIdx = combo % 2;
          int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;
          uint64_t expected = (uint64_t)(senderRank * 100 + payload);
          // S24's kernel output stays the packed 12 x uint64_t layout it
          // always had; the S18 stride parameterization does not apply here
          // (indexing combo * 16 read out of bounds from combo 1 on, UBSan
          // "index 16 out of bounds for type uint64_t[12]").
          if (hostRecvV[payload] != expected) {
            s24Pass = false;
          }
        }

        RPRINTF("S24 DevPutValue_RSig(INTRA+WORLD): %s\n",
                s24Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s24Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // =======================================================================
      // S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush
      //      — INTRA + WORLD
      // ResetSignal → assert ReadSignal==0 → IncreaseSignalShadow(5) →
      // SignalInc × 5 → WaitSignalMeetShadow → assert ReadSignal==5 → Flush
      // =======================================================================
      {
        FLAGCXCHECK(devHandle->deviceMemcpy(devResults, zeroResults,
                                            sizeof(zeroResults),
                                            flagcxMemcpyHostToDevice, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        launchKernelDevSignalShadowFlushIntraWorldS(devCommPtr, devResults,
                                                    stream, signalOptions.skew);
        FLAGCXCHECK(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);

        int hostRes[kFlagcxTestUnifiedCtas] = {};
        FLAGCXCHECK(devHandle->deviceMemcpy(hostRes, devResults,
                                            sizeof(hostRes),
                                            flagcxMemcpyDeviceToHost, stream));
        FLAGCXCHECK(devHandle->streamSynchronize(stream));

        bool s25Pass = flagcxUnifiedSignalResults(hostRes);
        RPRINTF("S25 DevSignalShadowFlush(INTRA+WORLD): %s\n",
                s25Pass ? "PASS" : "FAIL");
        gradedCases++;
        allPass &= s25Pass;
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

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
    if (gradedCases == 0)
      // Every case in this build is capability-gated SKIP.  Reporting an
      // empty "ALL PASS" would be exactly the silent green this suite
      // forbids, so name it.  RC stays 0: the SKIP contract itself is
      // what is verified here (MC952 review P1).
      printf("=== Final Result: NO CASES GRADED (all capability-gated "
             "SKIP) ===\n");
    else
      printf("=== Final Result: %s ===\n",
             globalPass ? "ALL PASS" : "SOME FAILED");
  }

  MPI_Finalize();
  return globalPass ? 0 : 1;
}
