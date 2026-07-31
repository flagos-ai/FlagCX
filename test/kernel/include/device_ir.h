/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Test-only Device IR kernel declarations.
 * These kernels exercise both API paths:
 *   - Struct-based IR wrappers: K1–K8
 *   - S-suffixed (scalar) IR functions:      S1–S10
 *
 * Compiled from device_ir.cu in test/kernel/[platform]/.
 ************************************************************************/

#ifndef TEST_KERNEL_DEVICE_IR_H_
#define TEST_KERNEL_DEVICE_IR_H_

#include "flagcx.h"

// K1: Comm Queries — writes rank, size, intraRank, intraSize to results[0..3]
void launchKernelCommQueries(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream);

// K2: Cooperative Group — writes threadRank, coopSize per thread
void launchKernelCoopGroup(const void *devCommPtr, int *devResults, int nBlocks,
                           int nThreads, flagcxStream_t stream);

// K3: Team Queries — writes intraRank, worldRank to results[0..1]
void launchKernelTeamQueries(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream);

// K4: Local Pointer — verifies localPtr == rawBuff
void launchKernelLocalPointer(const void *devMemPtr, void *rawBuff,
                              int *devResults, flagcxStream_t stream);

// K5: Intra Pointer — reads peer's data via LSA
void launchKernelIntraPointer(const void *devCommPtr, const void *devMemPtr,
                              float *devOutput, int nBlocks, int nThreads,
                              flagcxStream_t stream);

// K6: Data Type Size — writes sizeof for 5 types to results[0..4]
void launchKernelDataTypeSize(int *devResults, flagcxStream_t stream);

// K7: Intra Barrier Sync — write buffer, barrier, read peer
void launchKernelIntraBarrierSync(const void *devCommPtr, const void *devMemPtr,
                                  float *buffer, float *output, int N,
                                  flagcxStream_t stream);

// K8: Intra Barrier Arrive/Wait — write buffer, arrive, wait, read peer
void launchKernelIntraBarrierArriveWait(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream);

// =========================================================================
// Scalar IR (S-suffixed) kernel launchers
// =========================================================================

// S1: Cooperative Group (Scalar) — writes threadRank, coopSize per thread
void launchKernelCoopGroupS(const void *devCommPtr, int *devResults,
                            int nBlocks, int nThreads, flagcxStream_t stream);

// S2: Team Queries (Scalar) — writes intraRank, worldRank to results[0..1]
void launchKernelTeamQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream);

// S3: Local Pointer (Scalar) — verifies localPtr == rawBuff
void launchKernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                               int *devResults, flagcxStream_t stream);

// S4: Intra Pointer (Scalar) — reads peer's data via LSA
void launchKernelIntraPointerS(const void *devCommPtr, const void *devMemPtr,
                               float *devOutput, int nBlocks, int nThreads,
                               flagcxStream_t stream);

// S5: Intra Barrier Sync (Scalar) — write buffer, barrier, read peer
void launchKernelIntraBarrierSyncS(const void *devCommPtr,
                                   const void *devMemPtr, float *buffer,
                                   float *output, int N, flagcxStream_t stream);

// S6: Intra Barrier Arrive/Wait (Scalar) — write, arrive, wait, read peer
void launchKernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                         const void *devMemPtr, float *buffer,
                                         float *output, int N,
                                         flagcxStream_t stream);

// =========================================================================
// Barrier Ordering Variant Launchers
// =========================================================================

// K7b: Intra Barrier Sync(AcqRel) — single sync call
void launchKernelIntraBarrierSyncAcqRel(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream);

// K8b: Arrive(Release) + Wait(AcqRel)
void launchKernelIntraBarrierArriveWaitAcqRel(const void *devCommPtr,
                                              const void *devMemPtr,
                                              float *buffer, float *output,
                                              int N, flagcxStream_t stream);

// S5b: ArriveS(Release) + WaitS(Acquire)
void launchKernelIntraBarrierArriveWaitSplitS(const void *devCommPtr,
                                              const void *devMemPtr,
                                              float *buffer, float *output,
                                              int N, flagcxStream_t stream);

// S5c: SyncS(Release) + read + SyncS(Acquire) — matches K7 pattern
void launchKernelIntraBarrierSyncSplitS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream);

// =========================================================================
// Extended Coop Kinds (S-suffixed)
// =========================================================================

// S7: TILE_SPAN coop — threadRankEx, sizeEx, syncEx
void launchKernelCoopTileSpanS(int *devResults, int nBlocks, int nThreads,
                               flagcxStream_t stream);

// S8: LANES coop — threadRankEx, sizeEx, syncEx (full warp mask)
void launchKernelCoopLanesS(int *devResults, flagcxStream_t stream);

// =========================================================================
// S-API Transport Tests
// =========================================================================

// S9: GetFromCommS — verify transport handle non-null
void launchKernelNetGetFromCommS(const void *devCommPtr, int *devResults,
                                 flagcxStream_t stream);

// S10: Signal/Counter local read/reset/shadow
void launchKernelNetSignalCounterS(const void *devCommPtr, int *devResults,
                                   flagcxStream_t stream);

#endif // TEST_KERNEL_DEVICE_IR_H_
