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
extern "C" void launchKernelCommQueriesS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                              int *devResults, flagcxStream_t stream);

// S2: Coop Groups — block, tile_span, lanes (results[0..2] = pass flags)
extern "C" void launchKernelCoopGroupsS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                             int *devResults, flagcxStream_t stream);

// S3: Team Queries — writes intraRank, worldRank to results[0..1]
extern "C" void launchKernelTeamQueriesS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                              int *devResults, flagcxStream_t stream);

// S4: Local Pointer — verifies localPtr == rawBuff
extern "C" void launchKernelLocalPointerS(const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                               void *rawBuff, int *devResults,
                               flagcxStream_t stream);

// S5: Intra Pointer — reads peer's data via intra pointer
extern "C" void launchKernelIntraPointerS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                               const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                               float *devOutput, int count,
                               flagcxStream_t stream);

// S6: Peer Pointer — team-based peer memory access
extern "C" void launchKernelPeerPointerS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
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
extern "C" void launchKernelIntraBarrierSyncS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                                   float *buffer, float *output, int N,
                                   flagcxStream_t stream);

// S9: two full scalar barriers — SyncS(Release) + read + SyncS(Acquire)
extern "C" void launchKernelIntraBarrierArriveWaitS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *devMemPtr, float *buffer, float *output,
    int N, flagcxStream_t stream);

// S10: Intra AllReduce — composite using barriers + pointers
extern "C" void launchKernelIntraAllReduceS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                 const void FLAGCX_IR_GLOBAL_PTR *devMemPtr,
                                 float *buffer, int count,
                                 flagcxStream_t stream);

// =========================================================================
// Inter-Node Transport Tests (S1–S15, aligned with device_api_inter K1–K15)
// =========================================================================

// S1: Transport Handle — verify NetGetFromCommS non-null
extern "C" void launchKernelNetGetFromCommS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                 int *devResults, flagcxStream_t stream);

// S2: Signal/Counter Reset — read/reset/shadow
extern "C" void launchKernelNetResetS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                           int *devResults, flagcxStream_t stream);

// S3: Put + SigInc — PutS_RSigInc + WaitSignalS + FlushS
extern "C" void launchKernelNetPutSignalIncS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                                  size_t countPerPeer, flagcxStream_t stream);

#ifdef USE_NVIDIA_ADAPTOR
// S3b: one independent put+signal+wait+flush chain per transport context.
extern "C" void launchKernelNetMultiContextPutSignalIncS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr, size_t dataBase,
    int *devResults, flagcxStream_t stream);
#endif

// S4: Put + SigAdd — PutS_RSigAdd + WaitSignalS + FlushS
extern "C" void launchKernelNetPutSignalAddS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                                  const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                                  size_t countPerPeer, flagcxStream_t stream);

// S5: Put + SigInc + CtrInc — PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS
// + FlushS
extern "C" void launchKernelNetCounterPipelineS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr, size_t countPerPeer,
    flagcxStream_t stream);

// S6: Put(None) + Flush + Signal (FlushDecouple) — PutS + FlushS +
// SignalSigIncS + WaitSignalS + FlushS
extern "C" void launchKernelNetFlushDecoupleS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                                   size_t countPerPeer, flagcxStream_t stream);

// S7: PutValue — PutValueS(None)+Signal then PutValueS_RSigInc (both in one
// kernel)
extern "C" void launchKernelNetPutValueS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                              const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream);

// S8: Get — GetS + FlushS
extern "C" void launchKernelNetGetS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                         const void FLAGCX_IR_GLOBAL_PTR *sendMemPtr,
                         const void FLAGCX_IR_GLOBAL_PTR *recvMemPtr,
                         size_t countPerPeer, flagcxStream_t stream);

// S9: Signal — SignalSigIncS + SignalSigAddS + WaitSignalS (both in one kernel)
extern "C" void launchKernelNetSignalS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                            flagcxStream_t stream);

// S10: Shadow (commented — MeetShadowS)
extern "C" void launchKernelNetWaitSignalMeetShadowS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, flagcxStream_t stream);

// S11: WaitSignal + Flush (standalone)
extern "C" void launchKernelNetWaitSignalFlushS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, flagcxStream_t stream);

// S12: Inter Barrier — stress test
extern "C" void launchKernelInterBarrierS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                               int *devResults, int nIters,
                               flagcxStream_t stream);

// S13: World Barrier — sync + arrive/wait split
extern "C" void launchKernelWorldBarrierS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                               flagcxStream_t stream);

// S14: AlltoAll (one-sided composite) — put + signal + wait + flush + world
// barrier
extern "C" void launchKernelNetOneSidedAlltoAllS(
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
extern "C" void launchKernelDevPutIntraWorldS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr,
                                   int *devResult, size_t bytes,
                                   flagcxStream_t stream);

// S19: DevGet — INTRA + WORLD
extern "C" void launchKernelDevGetIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *remoteMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *localMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S21: DevPutSignalWait — INTRA + WORLD
extern "C" void launchKernelDevPutSignalWaitIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S16: DevBarrier — INTRA + WORLD
extern "C" void launchKernelDevBarrierIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S16: DevBarrierArriveWait — INTRA + WORLD
extern "C" void launchKernelDevBarrierArriveWaitIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S18: DevPutValue — INTRA + WORLD
extern "C" void launchKernelDevPutValueIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S20: DevSignalStandalone — INTRA + WORLD
extern "C" void launchKernelDevSignalStandaloneIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S17: DevTeamResolution — INTRA + WORLD
extern "C" void launchKernelDevTeamResolutionIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult,
    flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — INTER Suite (S16–S25)
// Tests INTER + WORLD teams across THREAD, WARP, and BLOCK coop kinds.
// =========================================================================

// S18: DevPut — INTER + WORLD
extern "C" void launchKernelDevPutInterWorldS(const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
                                   const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr,
                                   int *devResult, size_t bytes,
                                   flagcxStream_t stream);

// S19: DevGet — INTER + WORLD
extern "C" void launchKernelDevGetInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *remoteMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *localMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S21: DevPutSignalWait — INTER + WORLD
extern "C" void launchKernelDevPutSignalWaitInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S16: DevBarrier — INTER + WORLD
extern "C" void launchKernelDevBarrierInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S16: DevBarrierArriveWait — INTER + WORLD
extern "C" void launchKernelDevBarrierArriveWaitInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S18: DevPutValue — INTER + WORLD
extern "C" void launchKernelDevPutValueInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S20: DevSignalStandalone — INTER + WORLD
extern "C" void launchKernelDevSignalStandaloneInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S17: DevTeamResolution — INTER + WORLD
extern "C" void launchKernelDevTeamResolutionInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult,
    flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — completion variants (S22–S25)
// =========================================================================

// S22: DevPut_RSigInc + DevPut_RSigAdd — INTRA + WORLD
extern "C" void launchKernelDevPutRSigIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S22: DevPut_RSigInc + DevPut_RSigAdd — INTER + WORLD
extern "C" void launchKernelDevPutRSigInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc — INTRA
// + WORLD
extern "C" void launchKernelDevPutCounterIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc — INTER
// + WORLD
extern "C" void launchKernelDevPutCounterInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr,
    const void FLAGCX_IR_GLOBAL_PTR *srcMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S24: DevPutValue_RSigInc + DevPutValue_RSigAdd — INTRA + WORLD
extern "C" void launchKernelDevPutValueRSigIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S24: DevPutValue_RSigInc + DevPutValue_RSigAdd — INTER + WORLD
extern "C" void launchKernelDevPutValueRSigInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr,
    const void FLAGCX_IR_GLOBAL_PTR *dstMemPtr, int *devResult, size_t bytes,
    flagcxStream_t stream);

// S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush — INTRA +
// WORLD
extern "C" void launchKernelDevSignalShadowFlushIntraWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush — INTER +
// WORLD
extern "C" void launchKernelDevSignalShadowFlushInterWorldS(
    const void FLAGCX_IR_GLOBAL_PTR *devCommPtr, int *devResult,
    flagcxStream_t stream);

// Launch bridge for DeviceAdaptor launchKernel verification: expose
// kernelCommQueriesS (S1) as a plain kernel address so a host test can launch it
// through the slot's (void *func, void **args) contract instead of <<<>>>.
void *flagcxTestKernelCommQueriesSPtr(void);

// Reference-path bridge for the same kernel. The host test cannot call
// launchKernelCommQueriesS() directly: that launcher's `const void
// FLAGCX_IR_GLOBAL_PTR *` parameter is address_space(1) in the device
// translation unit and generic in the host one, so the two mangled names differ
// and the direct call would link against nothing. This wrapper takes the
// host-side generic pointer and adds the address space where the definition is
// visible.
void flagcxTestLaunchCommQueriesS(const void *devCommPtr, int *devResults,
                                 flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_IR_H_
