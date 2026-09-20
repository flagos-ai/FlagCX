/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Test-only Device IR kernel declarations.
 * These kernels exercise S-suffixed (scalar) IR functions.
 *
 * Intra-node tests (S1–S10) are aligned with device_api_intra K1–K10.
 * Inter-node tests (S1–S15) are aligned with device_api_inter K1–K15.
 *
 * Compiled from device_ir.cu in test/kernel/[platform]/.
 ************************************************************************/

#ifndef TEST_KERNEL_DEVICE_IR_H_
#define TEST_KERNEL_DEVICE_IR_H_

#include <stdint.h>

#include "device_api/flagcx_device_enums.h"
#include "device_utils.h"
#include "flagcx.h"

// S21-S22 use the full 12-slot product of cooperation kind, signal variant,
// and team. S23 uses the corresponding 18-slot product with three completion
// variants. Keep the expected sets in test-only code so a platform cannot
// silently turn an unsupported case into a passing one. NVIDIA expects every
// slot. A platform that cannot issue a cross-node THREAD put may opt out of the
// leading THREAD slots without changing public Device API or CommTraits
// contracts.
#define FLAGCX_TEST_UNIFIED_PUT_COOP_ALL_MASK ((uint32_t)0xfff)
#define FLAGCX_TEST_UNIFIED_PUT_COOP_NO_THREAD_MASK ((uint32_t)0xff0)
#define FLAGCX_TEST_UNIFIED_PUT_COUNTER_ALL_MASK ((uint32_t)0x3ffff)
#define FLAGCX_TEST_UNIFIED_PUT_COUNTER_NO_THREAD_MASK ((uint32_t)0x3ffc0)
#define FLAGCX_TEST_UNIFIED_INTRA_PUT_COOP_EXPECTED_MASK                       \
  FLAGCX_TEST_UNIFIED_PUT_COOP_ALL_MASK
#define FLAGCX_TEST_UNIFIED_INTRA_PUT_COUNTER_EXPECTED_MASK                    \
  FLAGCX_TEST_UNIFIED_PUT_COUNTER_ALL_MASK

#if defined(FLAGCX_TEST_NO_REMOTE_THREAD_PUT)
#define FLAGCX_TEST_UNIFIED_INTER_PUT_COOP_EXPECTED_MASK                       \
  FLAGCX_TEST_UNIFIED_PUT_COOP_NO_THREAD_MASK
#define FLAGCX_TEST_UNIFIED_INTER_PUT_COUNTER_EXPECTED_MASK                    \
  FLAGCX_TEST_UNIFIED_PUT_COUNTER_NO_THREAD_MASK
#else
#define FLAGCX_TEST_UNIFIED_INTER_PUT_COOP_EXPECTED_MASK                       \
  FLAGCX_TEST_UNIFIED_PUT_COOP_ALL_MASK
#define FLAGCX_TEST_UNIFIED_INTER_PUT_COUNTER_EXPECTED_MASK                    \
  FLAGCX_TEST_UNIFIED_PUT_COUNTER_ALL_MASK
#endif

// =========================================================================
// Intra-Node Scalar IR kernel launchers (S1–S10)
// =========================================================================

// S1: Comm Queries — rank, size, intraRank, intraSize
void launchKernelCommQueriesS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                              int *devResults, flagcxStream_t stream);

// S2: Coop Groups — block, tile_span, lanes (results[0..2] = pass flags)
void launchKernelCoopGroupsS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                             int *devResults, flagcxStream_t stream);

// S3: Team Queries — writes intraRank, worldRank to results[0..1]
void launchKernelTeamQueriesS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                              int *devResults, flagcxStream_t stream);

// S4: Local Pointer — verifies localPtr == rawBuff
void launchKernelLocalPointerS(const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                               void *rawBuff, int *devResults,
                               flagcxStream_t stream);

// S5: Intra Pointer — reads peer's data via intra pointer
void launchKernelIntraPointerS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                               const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                               float *devOutput, int count,
                               flagcxStream_t stream);

// S6: Peer Pointer — team-based peer memory access
void launchKernelPeerPointerS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                              const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                              float *devOutput, int count,
                              flagcxStream_t stream);

// S7: Multicast Pointer — NVLS-dependent, commented out
// void launchKernelMulticastPointerS(const void FLAGCX_IR_GLOBAL_PTR
// *devCommPtr,
//                                    const void FLAGCX_IR_GLOBAL_PTR
//                                    *devMemPtr, float *devOutput, int nBlocks,
//                                    int nThreads, flagcxStream_t stream);

// S8: Intra Barrier Sync — write buffer, barrier, read peer
void launchKernelIntraBarrierSyncS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                                   float *buffer, float *output, int N,
                                   flagcxStream_t stream);

// S9: Intra Barrier Arrive/Wait — SyncS(Release) + read + SyncS(Acquire)
void launchKernelIntraBarrierArriveWaitS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *devMemPtr, float *buffer, float *output,
    int N, flagcxStream_t stream);

// S10: Intra AllReduce — composite using barriers + pointers
void launchKernelIntraAllReduceS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                 const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                                 float *buffer, int count,
                                 flagcxStream_t stream);

// =========================================================================
// Inter-Node Transport Tests (S1–S15, aligned with device_api_inter K1–K15)
// =========================================================================

// S1: Transport Handle — verify NetGetFromCommS non-null
void launchKernelNetGetFromCommS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                 int *devResults, flagcxStream_t stream);

// S2: Signal/Counter Reset — read/reset/shadow
void launchKernelNetResetS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                           int *devResults, flagcxStream_t stream);

// S3: Put + SigInc — PutS_RSigInc + WaitSignalS + FlushS
void launchKernelNetPutSignalIncS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                                  size_t countPerPeer, flagcxStream_t stream);

// S4: Put + SigAdd — PutS_RSigAdd + WaitSignalS + FlushS
void launchKernelNetPutSignalAddS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                                  size_t countPerPeer, flagcxStream_t stream);

// S5: Put + SigInc + CtrInc — PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS
// + FlushS
void launchKernelNetCounterPipelineS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr, size_t countPerPeer,
    flagcxStream_t stream);

// S6: Put(None) + Flush + Signal (FlushDecouple) — PutS + FlushS +
// SignalSigIncS + WaitSignalS + FlushS
void launchKernelNetFlushDecoupleS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                                   size_t countPerPeer, flagcxStream_t stream);

// S7: PutValue — PutValueS(None)+Signal then PutValueS_RSigInc (both in one
// kernel)
void launchKernelNetPutValueS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                              const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream);

// S8: Get — GetS + FlushS
void launchKernelNetGetS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                         const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                         const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                         size_t countPerPeer, flagcxStream_t stream);

// S9: Signal — SignalSigIncS + SignalSigAddS + WaitSignalS (both in one kernel)
void launchKernelNetSignalS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                            flagcxStream_t stream);

// S10: Shadow (commented — MeetShadowS)
void launchKernelNetWaitSignalMeetShadowS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, flagcxStream_t stream);

// S11: WaitSignal + Flush (standalone)
void launchKernelNetWaitSignalFlushS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, flagcxStream_t stream);

// S12: Inter Barrier — stress test
void launchKernelInterBarrierS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                               int *devResults, int nIters,
                               flagcxStream_t stream);

// S13: World Barrier — sync + arrive/wait split
void launchKernelWorldBarrierS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                               flagcxStream_t stream);

// S14: AlltoAll (one-sided composite) — put + signal + wait + flush + world
// barrier
void launchKernelNetOneSidedAlltoAllS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr, size_t countPerPeer,
    flagcxStream_t stream);

// S15: AlltoAll (two-sided, commented)
// void launchKernelNetTwoSidedS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
// const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
//                               const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
//                               size_t countPerPeer, flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — INTRA Suite (S16–S25)
// Tests INTRA + WORLD teams across THREAD, WARP, and BLOCK coop kinds.
// =========================================================================

// S18: DevPut — INTRA + WORLD
void launchKernelDevPutIntraWorldS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr,
                                   int *devResult, size_t bytes,
                                   flagcxStream_t stream);

// S19: DevGet — INTRA + WORLD
void launchKernelDevGetIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *remoteMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *localMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S21: DevPutSignalWait — INTRA + WORLD
void launchKernelDevPutSignalWaitIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S16: DevBarrier — INTRA + WORLD
void launchKernelDevBarrierIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S16: DevBarrierArriveWait — INTRA + WORLD
void launchKernelDevBarrierArriveWaitIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S18: DevPutValue — INTRA + WORLD
void launchKernelDevPutValueIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S20: DevSignalStandalone — INTRA + WORLD
void launchKernelDevSignalStandaloneIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S17: DevTeamResolution — INTRA + WORLD
void launchKernelDevTeamResolutionIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult,
    flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — INTER Suite (S16–S25)
// Tests INTER + WORLD teams across THREAD, WARP, and BLOCK coop kinds.
// =========================================================================

// S18: DevPut — INTER + WORLD
void launchKernelDevPutInterWorldS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr,
                                   int *devResult, size_t bytes,
                                   flagcxStream_t stream);

// S19: DevGet — INTER + WORLD
void launchKernelDevGetInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *remoteMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *localMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S21: DevPutSignalWait — INTER + WORLD
void launchKernelDevPutSignalWaitInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S16: DevBarrier — INTER + WORLD
void launchKernelDevBarrierInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S16: DevBarrierArriveWait — INTER + WORLD
void launchKernelDevBarrierArriveWaitInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S18: DevPutValue — INTER + WORLD
void launchKernelDevPutValueInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S20: DevSignalStandalone — INTER + WORLD
void launchKernelDevSignalStandaloneInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S17: DevTeamResolution — INTER + WORLD
void launchKernelDevTeamResolutionInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult,
    flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — completion variants (S22–S25)
// =========================================================================

// S22: DevPut_RSigInc + DevPut_RSigAdd — INTRA + WORLD
void launchKernelDevPutRSigIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S22: DevPut_RSigInc + DevPut_RSigAdd — INTER + WORLD
void launchKernelDevPutRSigInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc — INTRA
// + WORLD
void launchKernelDevPutCounterIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc — INTER
// + WORLD
void launchKernelDevPutCounterInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S24: DevPutValue_RSigInc + DevPutValue_RSigAdd — INTRA + WORLD
void launchKernelDevPutValueRSigIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S24: DevPutValue_RSigInc + DevPutValue_RSigAdd — INTER + WORLD
void launchKernelDevPutValueRSigInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush — INTRA +
// WORLD
void launchKernelDevSignalShadowFlushIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush — INTER +
// WORLD
void launchKernelDevSignalShadowFlushInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_IR_H_
