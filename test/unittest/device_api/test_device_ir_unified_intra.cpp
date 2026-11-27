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
#include "device_api_report.h"
#include "device_api_test_session.hpp"
#include "device_ir.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

constexpr int kUnifiedIrSignalComboCount = 12;
constexpr int kUnifiedIrS23ComboCount = 18;

static int runDeviceApiTest(int argc, char *argv[],
                            DeviceApiTestSession &resources) {
  auto &devHandle = resources.devHandle;
  auto &comm = resources.comm;
  flagcxUniqueId uniqueId;

  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
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

  // Create DevComm with signal/counter/barrier slots
  // Signal variants use 12 slots; S23 uses 18 signal/counter slots.
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = kUnifiedIrS23ComboCount;
  reqs.interCounterCount = kUnifiedIrS23ComboCount;

  auto &devComm = resources.devComm;
  resources.check(flagcxDevCommCreate(comm, &reqs, &devComm));

  // Allocate send/recv buffers for S23, the largest combination matrix.
  size_t bufSize = maxBytes * kUnifiedIrS23ComboCount;
  auto &sendBuff = resources.sendBuff;
  auto &recvBuff = resources.recvBuff;
#ifdef FLAGCX_TEST_ALLOCATOR_SHMEM
  resources.memAllocator = flagcxMemSHMEM;
  const auto &memAllocator = resources.memAllocator;
#else
  resources.memAllocator = flagcxMemCCL;
  const auto &memAllocator = resources.memAllocator;
#endif
  resources.check(flagcxMemAlloc(&recvBuff, bufSize, memAllocator));
  resources.check(flagcxMemAlloc(&sendBuff, bufSize, memAllocator));

  // Register symmetric windows
  auto &sendWin = resources.sendWin;
  auto &recvWin = resources.recvWin;
  resources.check(flagcxCommWindowRegister(comm, sendBuff, bufSize, &sendWin,
                                           flagcxTestWindowFlags(),
                                           memAllocator));
  resources.check(flagcxCommWindowRegister(comm, recvBuff, bufSize, &recvWin,
                                           flagcxTestWindowFlags(),
                                           memAllocator));

  // Create DevMem handles
  auto &sendMem = resources.sendMem;
  auto &recvMem = resources.recvMem;
  resources.check(
      flagcxDevMemCreate(comm, sendBuff, bufSize, sendWin, &sendMem));
  resources.check(
      flagcxDevMemCreate(comm, recvBuff, bufSize, recvWin, &recvMem));

  // Get device pointers for IR functions
  void *devCommPtr = nullptr;
  resources.check(flagcxDevCommGetDevicePtr(devComm, &devCommPtr));
  void *sendMemPtr = nullptr, *recvMemPtr = nullptr;
  resources.check(flagcxDevMemGetDevicePtr(sendMem, &sendMemPtr));
  resources.check(flagcxDevMemGetDevicePtr(recvMem, &recvMemPtr));

#ifdef FLAGCX_ASCEND_PUTVALUE_TAIL
  // MC952 review P2: sub-word putValue destinations whose last byte is the
  // last byte of their registered allocation.  14 % 4 != 0 keeps the window
  // ending mid-word, so a 32-bit read-back at the tail crosses the
  // allocation (and the byte at offset 13 is also unaligned).
  constexpr size_t kSubwordTailBytes = 14;
  auto &tailBuff = resources.tailBuff;
  resources.check(flagcxMemAlloc(&tailBuff, kSubwordTailBytes, flagcxMemSHMEM));
  auto &tailWin = resources.tailWin;
  resources.check(flagcxCommWindowRegister(comm, tailBuff, kSubwordTailBytes,
                                           &tailWin, flagcxTestWindowFlags(),
                                           flagcxMemSHMEM));
  auto &tailMem = resources.tailMem;
  resources.check(
      flagcxDevMemCreate(comm, tailBuff, kSubwordTailBytes, tailWin, &tailMem));
  void *tailMemPtr = nullptr;
  resources.check(flagcxDevMemGetDevicePtr(tailMem, &tailMemPtr));
#endif

  // Allocate results buffer
  auto &devResults = resources.devResults;
  resources.check(devHandle->deviceMalloc(
      (void **)&devResults, 256 * sizeof(int), flagcxMemDevice, NULL));

  // Host scratch mirrors the already combination-scaled device buffers.
  auto &hostSend = resources.hostSend;
  hostSend = new float[bufSize / sizeof(float)];
  auto &hostRecv = resources.hostRecv;
  hostRecv = new float[bufSize / sizeof(float)];

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
  // S21-S25 atomically clear this value if any device context fails.
  int passResult = 1;

  const DeviceApiReporter reporter("test_device_ir_unified_intra", proc);

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
    reporter.report("S16", size, "SKIP");
    MPI_Barrier(MPI_COMM_WORLD);
#else
    const char *skipS16 = std::getenv("FLAGCX_CI_UNIFIED_IR_SKIP_S16");
    if (skipS16 != nullptr && skipS16[0] != '\0' && skipS16[0] != '0') {
      reporter.report("S16", size, "SKIP");
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
      resources.check(devHandle->deviceMemset(
          devResults, 0, FLAGCX_DEVICE_CTA_COUNT * sizeof(int), flagcxMemDevice,
          stream));
      launchKernelDevBarrierIntraWorldS(devCommPtr, devResults, stream);
      resources.check(devHandle->streamSynchronize(stream));
      resources.check(
          devHandle->deviceMemcpy(hostResults, devResults, 4 * sizeof(int),
                                  flagcxMemcpyDeviceToHost, stream));
      for (int i = 0; i < 4; i++) {
        if (hostResults[i] != 1) {
          s16Pass = false;
          break;
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // Sub-block B: BarrierArrive + BarrierWait (split)
      resources.check(devHandle->deviceMemset(
          devResults, 0, FLAGCX_DEVICE_CTA_COUNT * sizeof(int), flagcxMemDevice,
          stream));
      launchKernelDevBarrierArriveWaitIntraWorldS(devCommPtr, devResults,
                                                  stream);
      resources.check(devHandle->streamSynchronize(stream));
      resources.check(
          devHandle->deviceMemcpy(hostResults, devResults, 4 * sizeof(int),
                                  flagcxMemcpyDeviceToHost, stream));
      for (int i = 0; i < 4; i++) {
        if (hostResults[i] != 1) {
          s16Pass = false;
          break;
        }
      }

      RPRINTF("S16 DevBarrier(INTRA+WORLD): %s\n", s16Pass ? "PASS" : "FAIL");
      reporter.report("S16", size, s16Pass ? "PASS" : "FAIL");
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
    reporter.report("S17", size, "SKIP");
    MPI_Barrier(MPI_COMM_WORLD);
#else
    {
      int maxRanks = intraSize;
      if (totalProcs > maxRanks)
        maxRanks = totalProcs;
      size_t s17Size = 6 * maxRanks * sizeof(float);

      // Keep rank 0 distinguishable from the zero-initialized receive buffer.
      float myTag = (float)(proc + 1);
      resources.check(devHandle->deviceMemcpy(
          sendBuff, &myTag, sizeof(float), flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->deviceMemset(recvBuff, 0, s17Size,
                                              flagcxMemDevice, stream));
      resources.check(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                              flagcxMemDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevTeamResolutionIntraWorldS(devCommPtr, recvMemPtr,
                                               sendMemPtr, devResults, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      float *s17Recv = new float[6 * maxRanks];
      resources.check(devHandle->deviceMemcpy(
          s17Recv, recvBuff, s17Size, flagcxMemcpyDeviceToHost, stream));

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
      reporter.report("S17", size, s17Pass ? "PASS" : "FAIL");
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
    reporter.report("S18", size, "SKIP");
    MPI_Barrier(MPI_COMM_WORLD);
#else
    {
      // --- Sub-block A: DevPut ---
      for (size_t i = 0; i < 6 * count; i++)
        hostSend[i] = (float)(proc * 1000 + i);
      resources.check(devHandle->deviceMemcpy(
          sendBuff, hostSend, 6 * bytes, flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                              flagcxMemDevice, stream));
      resources.check(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                              flagcxMemDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutIntraWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                    devResults, bytes, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));
      resources.check(devHandle->deviceMemcpy(
          hostRecv, recvBuff, 6 * bytes, flagcxMemcpyDeviceToHost, stream));

      bool s18Pass = (hostRes == 1);
      int prevIntra = (intraRank + intraSize - 1) % intraSize;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;

      for (int combo = 0; combo < 6 && s18Pass; combo++) {
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;
        for (size_t i = 0; i < count && s18Pass; i++) {
          float expected = (float)(senderRank * 1000 + off + i);
          if (hostRecv[off + i] != expected) {
            RPRINTF("  S18A mismatch combo=%d i=%zu exp=%.1f got=%.1f\n", combo,
                    i, (double)expected, (double)hostRecv[off + i]);
            s18Pass = false;
          }
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
      // 0xAA sentinel, not 0: an untouched slot must be distinguishable from
      // a transfer that landed carrying a zero.
      resources.check(devHandle->deviceMemset(
          recvBuff, 0xAA, 6 * kPutValueStrideBytes, flagcxMemDevice, stream));
      resources.check(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                              flagcxMemDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutValueIntraWorldS(devCommPtr, recvMemPtr, devResults,
                                         bytes, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));
      uint64_t hostRecvV[6 * kPutValueStrideWords];
      resources.check(
          devHandle->deviceMemcpy(hostRecvV, recvBuff, 6 * kPutValueStrideBytes,
                                  flagcxMemcpyDeviceToHost, stream));
      if (hostRes != 1) {
        RPRINTF("  S18 results[0]=%d (expected 1)\n", hostRes);
        s18Pass = false;
      }
      // Every slot is checked, not just up to the first mismatch: a loss
      // confined to combo 0 and a scattered one are different faults.
      bool s18bOk = true;
      for (int combo = 0; combo < 6; combo++) {
        int teamIdx = combo % 2;
        int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;
        uint64_t expected = (uint64_t)(senderRank * 100 + combo);
        if (hostRecvV[combo * kPutValueStrideWords] != expected) {
          RPRINTF("  S18B mismatch bytes=%zu combo=%d exp=%llu got=0x%llx\n",
                  bytes, combo, (unsigned long long)expected,
                  (unsigned long long)hostRecvV[combo * kPutValueStrideWords]);
          s18bOk = false;
        }
      }
      // A slot that is correct a moment later was late, not lost. The host
      // copy that reads it is not ordered against the SDMA queue the put was
      // posted on, so a transfer still in flight shows up as the sentinel.
      if (!s18bOk) {
        usleep(200000);
        resources.check(devHandle->deviceMemcpy(
            hostRecvV, recvBuff, 6 * kPutValueStrideBytes,
            flagcxMemcpyDeviceToHost, stream));
        for (int combo = 0; combo < 6; combo++) {
          int teamIdx = combo % 2;
          int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;
          uint64_t expected = (uint64_t)(senderRank * 100 + combo);
          if (hostRecvV[combo * kPutValueStrideWords] == expected) {
            RPRINTF("  S18B late bytes=%zu combo=%d\n", bytes, combo);
          }
        }
      }
      s18Pass &= s18bOk;

#ifdef FLAGCX_ASCEND_PUTVALUE_TAIL
      // --- Sub-block C: sub-word values ending on the allocation tail ---
      // MC952 review P2: a 1-byte and a 2-byte putValue whose last byte is
      // the last byte of the registered window, so the read-back must never
      // load a wider word.  bytes = 0 selects the kernel's tail mode.
      {
        resources.check(devHandle->deviceMemset(
            tailBuff, 0xAA, kSubwordTailBytes, flagcxMemDevice, stream));
        resources.check(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                                flagcxMemDevice, stream));
        resources.check(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        launchKernelDevPutValueIntraWorldS(devCommPtr, tailMemPtr, devResults,
                                           0, stream);
        resources.check(devHandle->streamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        resources.check(
            devHandle->deviceMemcpy(&hostRes, devResults, sizeof(int),
                                    flagcxMemcpyDeviceToHost, stream));
        unsigned char tailHost[kSubwordTailBytes];
        resources.check(
            devHandle->deviceMemcpy(tailHost, tailBuff, kSubwordTailBytes,
                                    flagcxMemcpyDeviceToHost, stream));
        // The 2-byte value lands last: [12] = 0x10 + sender, [13] = 0x5A.
        bool s18cOk = (hostRes == 1);
        for (size_t i = 0; i < kSubwordTailBytes; i++) {
          unsigned char want = 0xAA;
          if (i + 2 == kSubwordTailBytes)
            want = (unsigned char)(0x10 + prevIntra);
          if (i + 1 == kSubwordTailBytes)
            want = 0x5A;
          if (tailHost[i] != want) {
            RPRINTF("  S18C mismatch i=%zu exp=0x%02x got=0x%02x\n", i,
                    (unsigned)want, (unsigned)tailHost[i]);
            s18cOk = false;
          }
        }
        s18Pass &= s18cOk;
      }
#endif

      RPRINTF("S18 DevPut+PutValue(INTRA+WORLD): %s\n",
              s18Pass ? "PASS" : "FAIL");
      reporter.report("S18", size, s18Pass ? "PASS" : "FAIL");
      gradedCases++;
      allPass &= s18Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    // S19: DevGet — INTRA + WORLD
    // Skipped where the device API cannot read a peer's memory at all (P800:
    // every remote-read path traps), so there is no implementation to check.
#ifdef FLAGCX_TEST_NO_REMOTE_READ
    RPRINTF("S19 DevGet(INTRA+WORLD): SKIP (no remote read on this device)\n");
    reporter.report("S19", size, "SKIP");
    MPI_Barrier(MPI_COMM_WORLD);
#else
    {
      for (size_t i = 0; i < 6 * count; i++)
        hostSend[i] = (float)(proc * 2000 + i);
      resources.check(devHandle->deviceMemcpy(
          sendBuff, hostSend, 6 * bytes, flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->deviceMemset(recvBuff, 0, 6 * bytes,
                                              flagcxMemDevice, stream));
      resources.check(devHandle->deviceMemset(devResults, 0, sizeof(int),
                                              flagcxMemDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevGetIntraWorldS(devCommPtr, sendMemPtr, recvMemPtr,
                                    devResults, bytes, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      resources.check(devHandle->deviceMemcpy(
          hostRecv, recvBuff, 6 * bytes, flagcxMemcpyDeviceToHost, stream));

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
      reporter.report("S19", size, s19Pass ? "PASS" : "FAIL");
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
    reporter.report("S20", size, "SKIP");
    RPRINTF("S21 DevPutSignalWait(INTRA+WORLD): SKIP (no RMW "
            "completion atomic on this device)\n");
    reporter.report("S21", size, "SKIP");
    RPRINTF("S22 DevPut_RSig(INTRA+WORLD): SKIP (no RMW "
            "completion atomic on this device)\n");
    reporter.report("S22", size, "SKIP");
    RPRINTF("S23 DevPutCounter(INTRA+WORLD): SKIP (no RMW "
            "completion atomic on this device)\n");
    reporter.report("S23", size, "SKIP");
    RPRINTF("S24 DevPutValue_RSig(INTRA+WORLD): SKIP (no RMW "
            "completion atomic on this device)\n");
    reporter.report("S24", size, "SKIP");
    RPRINTF("S25 DevSignalShadowFlush(INTRA+WORLD): SKIP (no RMW "
            "completion atomic on this device)\n");
    reporter.report("S25", size, "SKIP");
    MPI_Barrier(MPI_COMM_WORLD);
#else
    // =======================================================================
    // S20: DevSignalStandalone — 3 coop × 2 teams × 2 actions = 12 cases
    //      — INTRA + WORLD
    // =======================================================================
    {
      int hostInit = 1;
      resources.check(
          devHandle->deviceMemcpy(devResults, &hostInit, sizeof(int),
                                  flagcxMemcpyHostToDevice, stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevSignalStandaloneIntraWorldS(devCommPtr, devResults,
                                                 stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      bool s20Pass = (hostRes == 1);
      RPRINTF("S20 DevSignalStandalone(INTRA+WORLD): %s\n",
              s20Pass ? "PASS" : "FAIL");
      reporter.report("S20", size, s20Pass ? "PASS" : "FAIL");
      gradedCases++;
      allPass &= s20Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // S21: DevPutSignalWait — INTRA + WORLD (12 full combinations)
    // ResetSignal → Put → SignalInc/SignalAdd → WaitSignal → ReadSignal verify
    {
      for (size_t i = 0; i < kUnifiedIrSignalComboCount * count; i++)
        hostSend[i] = (float)(proc * 3000 + i);
      resources.check(devHandle->deviceMemcpy(
          sendBuff, hostSend, kUnifiedIrSignalComboCount * bytes,
          flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->deviceMemset(
          recvBuff, 0, kUnifiedIrSignalComboCount * bytes, flagcxMemDevice,
          stream));
      resources.check(
          devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                  flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutSignalWaitIntraWorldS(
          devCommPtr, recvMemPtr, sendMemPtr, devResults, bytes, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      resources.check(devHandle->deviceMemcpy(
          hostRecv, recvBuff, kUnifiedIrSignalComboCount * bytes,
          flagcxMemcpyDeviceToHost, stream));

      bool s21Pass = (hostRes == 1);
      int prevIntra = (intraRank + intraSize - 1) % intraSize;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;

      for (int combo = 0; combo < kUnifiedIrSignalComboCount && s21Pass;
           combo++) {
        if ((expectedPutCoopMask & (uint32_t{1} << combo)) == 0)
          continue;
        int teamIdx = combo % 2;
        size_t off = combo * count;
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
      reporter.report("S21", size, s21Pass ? "PASS" : "FAIL");
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
      for (size_t i = 0; i < kUnifiedIrSignalComboCount * count; i++)
        hostSend[i] = (float)(proc * 4000 + i);
      resources.check(devHandle->deviceMemcpy(
          sendBuff, hostSend, kUnifiedIrSignalComboCount * bytes,
          flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->deviceMemset(
          recvBuff, 0, kUnifiedIrSignalComboCount * bytes, flagcxMemDevice,
          stream));
      resources.check(
          devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                  flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutRSigIntraWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                        devResults, bytes, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      resources.check(devHandle->deviceMemcpy(
          hostRecv, recvBuff, kUnifiedIrSignalComboCount * bytes,
          flagcxMemcpyDeviceToHost, stream));

      bool s22Pass = (hostRes == 1);
      int prevIntra = (intraRank + intraSize - 1) % intraSize;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;

      for (int combo = 0; combo < kUnifiedIrSignalComboCount && s22Pass;
           combo++) {
        if ((expectedPutCoopMask & (uint32_t{1} << combo)) == 0)
          continue;
        int teamIdx = combo % 2;
        size_t off = combo * count;
        int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;

        for (size_t i = 0; i < count && s22Pass; i++) {
          float expected = (float)(senderRank * 4000 + off + i);
          if (hostRecv[off + i] != expected) {
            s22Pass = false;
          }
        }
      }

      RPRINTF("S22 DevPut_RSig(INTRA+WORLD): %s\n", s22Pass ? "PASS" : "FAIL");
      reporter.report("S22", size, s22Pass ? "PASS" : "FAIL");
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
      for (size_t i = 0; i < kUnifiedIrS23ComboCount * count; i++)
        hostSend[i] = (float)(proc * 5000 + i);
      resources.check(devHandle->deviceMemcpy(
          sendBuff, hostSend, kUnifiedIrS23ComboCount * bytes,
          flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->deviceMemset(recvBuff, 0,
                                              kUnifiedIrS23ComboCount * bytes,
                                              flagcxMemDevice, stream));
      resources.check(
          devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                  flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutCounterIntraWorldS(devCommPtr, recvMemPtr, sendMemPtr,
                                           devResults, bytes, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      resources.check(devHandle->deviceMemcpy(
          hostRecv, recvBuff, kUnifiedIrS23ComboCount * bytes,
          flagcxMemcpyDeviceToHost, stream));

      bool s23Pass = (hostRes == 1);
      int prevIntra = (intraRank + intraSize - 1) % intraSize;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;

      for (int combo = 0; combo < kUnifiedIrS23ComboCount && s23Pass; combo++) {
        if ((expectedPutCounterMask & (uint32_t{1} << combo)) == 0)
          continue;
        int teamIdx = combo % 2;
        size_t off = combo * count;
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
      reporter.report("S23", size, s23Pass ? "PASS" : "FAIL");
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
      resources.check(devHandle->deviceMemset(
          recvBuff, 0, kUnifiedIrSignalComboCount * sizeof(uint64_t),
          flagcxMemDevice, stream));
      resources.check(
          devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                  flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevPutValueRSigIntraWorldS(devCommPtr, recvMemPtr, devResults,
                                             bytes, stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      uint64_t hostRecvV[kUnifiedIrSignalComboCount];
      resources.check(devHandle->deviceMemcpy(
          hostRecvV, recvBuff, kUnifiedIrSignalComboCount * sizeof(uint64_t),
          flagcxMemcpyDeviceToHost, stream));

      bool s24Pass = (hostRes == 1);
      int prevIntra = (intraRank + intraSize - 1) % intraSize;
      int prevWorld = (proc + totalProcs - 1) % totalProcs;

      for (int combo = 0; combo < kUnifiedIrSignalComboCount && s24Pass;
           combo++) {
        int teamIdx = combo % 2;
        int senderRank = (teamIdx == 0) ? prevIntra : prevWorld;
        uint64_t expected = (uint64_t)(senderRank * 100 + combo);
        // S24's kernel output stays the packed 12 x uint64_t layout it
        // always had; the S18 stride parameterization does not apply here
        // (indexing combo * 16 read out of bounds from combo 1 on, UBSan
        // "index 16 out of bounds for type uint64_t[12]").
        if (hostRecvV[combo] != expected) {
          s24Pass = false;
        }
      }

      RPRINTF("S24 DevPutValue_RSig(INTRA+WORLD): %s\n",
              s24Pass ? "PASS" : "FAIL");
      reporter.report("S24", size, s24Pass ? "PASS" : "FAIL");
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
      resources.check(
          devHandle->deviceMemcpy(devResults, &passResult, sizeof(int),
                                  flagcxMemcpyHostToDevice, stream));
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      launchKernelDevSignalShadowFlushIntraWorldS(devCommPtr, devResults,
                                                  stream);
      resources.check(devHandle->streamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int hostRes = 0;
      resources.check(devHandle->deviceMemcpy(
          &hostRes, devResults, sizeof(int), flagcxMemcpyDeviceToHost, stream));

      bool s25Pass = (hostRes == 1);
      RPRINTF("S25 DevSignalShadowFlush(INTRA+WORLD): %s\n",
              s25Pass ? "PASS" : "FAIL");
      reporter.report("S25", size, s25Pass ? "PASS" : "FAIL");
      gradedCases++;
      allPass &= s25Pass;
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    if (proc == 0)
      printf("\n");
  }

  int localPass = allPass ? 1 : 0;
  int globalPass = 0;
  MPI_Allreduce(&localPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  // One checked cleanup path serves successful and failed runs.
  resources.check(resources.finish());
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

  return globalPass ? 0 : 1;
}

int main(int argc, char *argv[]) {
  return deviceApiTestMain(argc, argv, runDeviceApiTest);
}
