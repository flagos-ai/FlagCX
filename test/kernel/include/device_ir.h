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

#include "device_api/flagcx_device_enums.h"
#include "flagcx.h"

// =========================================================================
// Intra-Node Scalar IR kernel launchers (S1–S10)
// =========================================================================

// S1: Comm Queries — rank, size, intraRank, intraSize
void launchKernelCommQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream);

// S2: Coop Groups — block, tile_span, lanes (results[0..2] = pass flags)
void launchKernelCoopGroupsS(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream);

// S3: Team Queries — writes intraRank, worldRank to results[0..1]
void launchKernelTeamQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream);

// S4: Local Pointer — verifies localPtr == rawBuff
void launchKernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                               int *devResults, flagcxStream_t stream);

// S5: Intra Pointer — reads peer's data via intra pointer
void launchKernelIntraPointerS(const void *devCommPtr, const void *devMemPtr,
                               float *devOutput, int count,
                               flagcxStream_t stream);

// S6: Peer Pointer — team-based peer memory access
void launchKernelPeerPointerS(const void *devCommPtr, const void *devMemPtr,
                              float *devOutput, int count,
                              flagcxStream_t stream);

// S7: Multicast Pointer — NVLS-dependent, commented out
// void launchKernelMulticastPointerS(const void *devCommPtr,
//                                    const void *devMemPtr, float *devOutput,
//                                    int nBlocks, int nThreads,
//                                    flagcxStream_t stream);

// S8: Intra Barrier Sync — write buffer, barrier, read peer
void launchKernelIntraBarrierSyncS(const void *devCommPtr,
                                   const void *devMemPtr, float *buffer,
                                   float *output, int N, flagcxStream_t stream);

// S9: Intra Barrier Arrive/Wait — SyncS(Release) + read + SyncS(Acquire)
void launchKernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                         const void *devMemPtr, float *buffer,
                                         float *output, int N,
                                         flagcxStream_t stream);

// S10: Intra AllReduce — composite using barriers + pointers
void launchKernelIntraAllReduceS(const void *devCommPtr, const void *devMemPtr,
                                 float *buffer, int count,
                                 flagcxStream_t stream);

// =========================================================================
// Inter-Node Transport Tests (S1–S15, aligned with device_api_inter K1–K15)
// =========================================================================

// S1: Transport Handle — verify NetGetFromCommS non-null
void launchKernelNetGetFromCommS(const void *devCommPtr, int *devResults,
                                 flagcxStream_t stream);

// S2: Signal/Counter Reset — read/reset/shadow
void launchKernelNetResetS(const void *devCommPtr, int *devResults,
                           flagcxStream_t stream);

// S3: Put + SigInc — PutS_RSigInc + WaitSignalS + FlushS
void launchKernelNetPutSignalIncS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream);

// S4: Put + SigAdd — PutS_RSigAdd + WaitSignalS + FlushS
void launchKernelNetPutSignalAddS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream);

// S5: Put + SigInc + CtrInc — PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS
// + FlushS
void launchKernelNetCounterPipelineS(const void *devCommPtr,
                                     const void *sendMemPtr,
                                     const void *recvMemPtr,
                                     size_t countPerPeer,
                                     flagcxStream_t stream);

// S6: Put(None) + Flush + Signal (FlushDecouple) — PutS + FlushS +
// SignalSigIncS + WaitSignalS + FlushS
void launchKernelNetFlushDecoupleS(const void *devCommPtr,
                                   const void *sendMemPtr,
                                   const void *recvMemPtr, size_t countPerPeer,
                                   flagcxStream_t stream);

// S7: PutValue — PutValueS(None)+Signal then PutValueS_RSigInc (both in one
// kernel)
void launchKernelNetPutValueS(const void *devCommPtr, const void *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream);

// S8: Get — GetS + FlushS
void launchKernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream);

// S9: Signal — SignalSigIncS + SignalSigAddS + WaitSignalS (both in one kernel)
void launchKernelNetSignalS(const void *devCommPtr, flagcxStream_t stream);

// S10: Shadow (commented — MeetShadowS)
void launchKernelNetWaitSignalMeetShadowS(const void *devCommPtr,
                                          flagcxStream_t stream);

// S11: WaitSignal + Flush (standalone)
void launchKernelNetWaitSignalFlushS(const void *devCommPtr,
                                     flagcxStream_t stream);

// S12: Inter Barrier — stress test
void launchKernelInterBarrierS(const void *devCommPtr, int *devResults,
                               int nIters, flagcxStream_t stream);

// S13: World Barrier — sync + arrive/wait split
void launchKernelWorldBarrierS(const void *devCommPtr, flagcxStream_t stream);

// S14: AlltoAll (one-sided composite) — put + signal + wait + flush + world
// barrier
void launchKernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                      const void *sendMemPtr,
                                      const void *recvMemPtr,
                                      size_t countPerPeer,
                                      flagcxStream_t stream);

// S15: AlltoAll (two-sided, commented)
// void launchKernelNetTwoSidedS(const void *devCommPtr, const void *sendMemPtr,
//                               const void *recvMemPtr, size_t countPerPeer,
//                               flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — INTRA Suite (S16–S21)
// Tests INTRA + WORLD teams on single-node (8 combinations: 4 coop × 2 teams)
// =========================================================================

// S16: DevPut — INTRA + WORLD
void launchKernelDevPutIntraWorldS(const void *devCommPtr,
                                   const void *dstMemPtr, const void *srcMemPtr,
                                   int *devResult, size_t bytes,
                                   flagcxStream_t stream);

// S17: DevGet — INTRA + WORLD
void launchKernelDevGetIntraWorldS(const void *devCommPtr,
                                   const void *remoteMemPtr,
                                   const void *localMemPtr, int *devResult,
                                   size_t bytes, flagcxStream_t stream);

// S18: DevPutSignalWait — INTRA + WORLD
void launchKernelDevPutSignalWaitIntraWorldS(const void *devCommPtr,
                                             const void *dstMemPtr,
                                             const void *srcMemPtr,
                                             int *devResult, size_t bytes,
                                             flagcxStream_t stream);

// S19: DevBarrier — INTRA + WORLD (merged)
void launchKernelDevBarrierIntraWorldS(const void *devCommPtr, int *devResult,
                                       flagcxStream_t stream);

// S16 sub-block: DevBarrierArriveWait — INTRA + WORLD
void launchKernelDevBarrierArriveWaitIntraWorldS(const void *devCommPtr,
                                                 int *devResult,
                                                 flagcxStream_t stream);

// S18 sub-block: DevPutValue — INTRA + WORLD
void launchKernelDevPutValueIntraWorldS(const void *devCommPtr,
                                        const void *dstMemPtr, int *devResult,
                                        size_t bytes, flagcxStream_t stream);

// S20: DevSignalStandalone — INTRA + WORLD
void launchKernelDevSignalStandaloneIntraWorldS(const void *devCommPtr,
                                                int *devResult,
                                                flagcxStream_t stream);

// S21: DevTeamResolution — INTRA + WORLD
void launchKernelDevTeamResolutionIntraWorldS(const void *devCommPtr,
                                              const void *dstMemPtr,
                                              const void *srcMemPtr,
                                              int *devResult,
                                              flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — INTER Suite (S16–S21)
// Tests INTER + WORLD teams on multi-node (8 combinations: 4 coop × 2 teams)
// =========================================================================

// S16: DevPut — INTER + WORLD
void launchKernelDevPutInterWorldS(const void *devCommPtr,
                                   const void *dstMemPtr, const void *srcMemPtr,
                                   int *devResult, size_t bytes,
                                   flagcxStream_t stream);

// S17: DevGet — INTER + WORLD
void launchKernelDevGetInterWorldS(const void *devCommPtr,
                                   const void *remoteMemPtr,
                                   const void *localMemPtr, int *devResult,
                                   size_t bytes, flagcxStream_t stream);

// S18: DevPutSignalWait — INTER + WORLD
void launchKernelDevPutSignalWaitInterWorldS(const void *devCommPtr,
                                             const void *dstMemPtr,
                                             const void *srcMemPtr,
                                             int *devResult, size_t bytes,
                                             flagcxStream_t stream);

// S19: DevBarrier — INTER + WORLD (merged)
void launchKernelDevBarrierInterWorldS(const void *devCommPtr, int *devResult,
                                       flagcxStream_t stream);

// S16 sub-block: DevBarrierArriveWait — INTER + WORLD
void launchKernelDevBarrierArriveWaitInterWorldS(const void *devCommPtr,
                                                 int *devResult,
                                                 flagcxStream_t stream);

// S18 sub-block: DevPutValue — INTER + WORLD
void launchKernelDevPutValueInterWorldS(const void *devCommPtr,
                                        const void *dstMemPtr, int *devResult,
                                        size_t bytes, flagcxStream_t stream);

// S20: DevSignalStandalone — INTER + WORLD
void launchKernelDevSignalStandaloneInterWorldS(const void *devCommPtr,
                                                int *devResult,
                                                flagcxStream_t stream);

// S21: DevTeamResolution — INTER + WORLD
void launchKernelDevTeamResolutionInterWorldS(const void *devCommPtr,
                                              const void *dstMemPtr,
                                              const void *srcMemPtr,
                                              int *devResult,
                                              flagcxStream_t stream);

// =========================================================================
// Unified One-Sided IR Tests — S22–S25 (new scenarios)
// =========================================================================

// S22: DevPut_RSigInc + DevPut_RSigAdd — INTRA + WORLD
void launchKernelDevPutRSigIntraWorldS(const void *devCommPtr,
                                       const void *dstMemPtr,
                                       const void *srcMemPtr, int *devResult,
                                       size_t bytes, flagcxStream_t stream);

// S22: DevPut_RSigInc + DevPut_RSigAdd — INTER + WORLD
void launchKernelDevPutRSigInterWorldS(const void *devCommPtr,
                                       const void *dstMemPtr,
                                       const void *srcMemPtr, int *devResult,
                                       size_t bytes, flagcxStream_t stream);

// S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc — INTRA
// + WORLD
void launchKernelDevPutCounterIntraWorldS(const void *devCommPtr,
                                          const void *dstMemPtr,
                                          const void *srcMemPtr, int *devResult,
                                          size_t bytes, flagcxStream_t stream);

// S23: DevPut_LCtrInc + DevPut_RSigInc_LCtrInc + DevPut_RSigAdd_LCtrInc — INTER
// + WORLD
void launchKernelDevPutCounterInterWorldS(const void *devCommPtr,
                                          const void *dstMemPtr,
                                          const void *srcMemPtr, int *devResult,
                                          size_t bytes, flagcxStream_t stream);

// S24: DevPutValue_RSigInc + DevPutValue_RSigAdd — INTRA + WORLD
void launchKernelDevPutValueRSigIntraWorldS(const void *devCommPtr,
                                            const void *dstMemPtr,
                                            int *devResult, size_t bytes,
                                            flagcxStream_t stream);

// S24: DevPutValue_RSigInc + DevPutValue_RSigAdd — INTER + WORLD
void launchKernelDevPutValueRSigInterWorldS(const void *devCommPtr,
                                            const void *dstMemPtr,
                                            int *devResult, size_t bytes,
                                            flagcxStream_t stream);

// S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush — INTRA +
// WORLD
void launchKernelDevSignalShadowFlushIntraWorldS(const void *devCommPtr,
                                                 int *devResult,
                                                 flagcxStream_t stream);

// S25: DevIncreaseSignalShadow + DevWaitSignalMeetShadow + DevFlush — INTER +
// WORLD
void launchKernelDevSignalShadowFlushInterWorldS(const void *devCommPtr,
                                                 int *devResult,
                                                 flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_IR_H_
