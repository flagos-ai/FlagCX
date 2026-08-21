/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device IR kernel implementations — CUDA kernels exercising FlagCX
 * Device API IR functions via device pointers.
 *
 * Intra-node (S1–S10): aligned with device_api_intra K1–K10.
 * Inter-node transport: separate section.
 *
 * Compiled by nvcc into device_ir.o, linked by g++ into test_device_ir.
 ************************************************************************/

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "nvidia_adaptor.h"
#include "flagcx_device_internal.h"

// IR wrapper declarations + implementations (needed for nvcc inline compilation)
#include "flagcx_device_wrapper.h"
#include "flagcx_device_wrapper_impl.h" // also pulls in scalar_ir_impl.h

#include "device_ir.h"

// ===========================================================================
// Scalar IR (S-suffixed) kernels — Intra-Node (S1–S10)
// ===========================================================================

// ---------------------------------------------------------------------------
// S1: Comm Queries (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelCommQueriesS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    results[0] = flagcxDevCommGetRank(devCommPtr);
    results[1] = flagcxDevCommGetSize(devCommPtr);
    results[2] = flagcxDevCommGetIntraRank(devCommPtr);
    results[3] = flagcxDevCommGetIntraSize(devCommPtr);
  }
}

void launchKernelCommQueriesS(const void *devCommPtr, int *devResults,
                              flagcxStream_t stream) {
  kernelCommQueriesS<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// S2: Coop Groups (Scalar) — block, tile_span, lanes in one kernel
// ---------------------------------------------------------------------------

// Sub-kernel: block-level coop check (1 block, 32 threads)
__global__ void kernelCoopGroupsS_block(int *results) {
  int rank = flagcxCoopThreadRankS(FLAGCX_COOP_BLOCK);
  int size = flagcxCoopSizeS(FLAGCX_COOP_BLOCK);
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Thread 0 checks all threads got correct rank/size
  __shared__ int pass;
  if (FLAGCX_THREAD_IDX_X == 0) pass = 1;
  __syncthreads();
  if (rank != (int)FLAGCX_THREAD_IDX_X || size != (int)FLAGCX_BLOCK_DIM_X)
    atomicExch(&pass, 0);
  __syncthreads();
  if (FLAGCX_THREAD_IDX_X == 0) results[0] = pass;
}

// Sub-kernel: tile_span coop check (1 block, 128 threads = 4 tiles of 32)
__global__ void kernelCoopGroupsS_tileSpan(int *results) {
  int tileIdx = FLAGCX_THREAD_IDX_X / 32;
  uint32_t t0 = (uint32_t)tileIdx;
  uint32_t nTiles = 1;
  uint32_t id = 0;

  int rank = flagcxCoopThreadRankExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);
  flagcxCoopSyncExS(FLAGCX_COOP_TILE_SPAN, t0, nTiles, id);

  // Expected: rank = threadIdx % 32, size = 32
  __shared__ int pass;
  if (FLAGCX_THREAD_IDX_X == 0) pass = 1;
  __syncthreads();
  if (rank != (int)(FLAGCX_THREAD_IDX_X % 32) || size != 32)
    atomicExch(&pass, 0);
  __syncthreads();
  if (FLAGCX_THREAD_IDX_X == 0) results[1] = pass;
}

// Sub-kernel: lanes coop check (1 block, 32 threads, full warp mask)
__global__ void kernelCoopGroupsS_lanes(int *results) {
  uint32_t laneMask = 0xFFFFFFFF;

  int rank = flagcxCoopThreadRankExS(FLAGCX_COOP_LANES, laneMask, 0, 0);
  int size = flagcxCoopSizeExS(FLAGCX_COOP_LANES, laneMask, 0, 0);
  flagcxCoopSyncExS(FLAGCX_COOP_LANES, laneMask, 0, 0);

  // Expected: rank = lane index, size = 32
  __shared__ int pass;
  if (FLAGCX_THREAD_IDX_X == 0) pass = 1;
  __syncthreads();
  if (rank != (int)FLAGCX_THREAD_IDX_X || size != 32)
    atomicExch(&pass, 0);
  __syncthreads();
  if (FLAGCX_THREAD_IDX_X == 0) results[2] = pass;
}

void launchKernelCoopGroupsS(const void *devCommPtr, int *devResults,
                             flagcxStream_t stream) {
  kernelCoopGroupsS_block<<<1, 32, 0, stream->base>>>(devResults);
  kernelCoopGroupsS_tileSpan<<<1, 128, 0, stream->base>>>(devResults);
  kernelCoopGroupsS_lanes<<<1, 32, 0, stream->base>>>(devResults);
}

// ---------------------------------------------------------------------------
// S3: Team Queries (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelTeamQueriesS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
    int worldRank =
        flagcxTeamRankToWorldS(devCommPtr, FLAGCX_TEAM_INTRA, intraRank);

    results[0] = intraRank;
    results[1] = worldRank;
  }
}

void launchKernelTeamQueriesS(const void *devCommPtr, int *devResults,
                                   flagcxStream_t stream) {
  kernelTeamQueriesS<<<1, 1, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// S4: Local Pointer (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                                         int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    void *localPtr = flagcxGetLocalPointerS(devMemPtr, 0);
    // Verify local pointer is non-null and points to same data as rawBuff
    // (may be a different VA due to VMM flat-mapping)
    if (localPtr == nullptr) {
      results[0] = 0;
    } else {
      float val = *((volatile float *)localPtr);
      float expected = *((volatile float *)rawBuff);
      results[0] = (val == expected) ? 1 : 0;
    }
  }
}

void launchKernelLocalPointerS(const void *devMemPtr, void *rawBuff,
                                    int *devResults, flagcxStream_t stream) {
  kernelLocalPointerS<<<1, 1, 0, stream->base>>>(devMemPtr, rawBuff,
                                                       devResults);
}

// ---------------------------------------------------------------------------
// S5: Intra Pointer (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelIntraPointerS(const void *devCommPtr,
                                    const void *devMemPtr,
                                    float *output, int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;

  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[i] = *peerPtr;
  }
}

void launchKernelIntraPointerS(const void *devCommPtr,
                                    const void *devMemPtr, float *devOutput,
                                    int count,
                                    flagcxStream_t stream) {
  kernelIntraPointerS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, devOutput, count);
}

// ---------------------------------------------------------------------------
// S8: Intra Barrier Sync (Scalar)
// ---------------------------------------------------------------------------

__global__ void kernelIntraBarrierSyncS(const void *devCommPtr,
                                        const void *devMemPtr,
                                        float *buffer, float *output,
                                        int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int i = tid; i < count; i += nthreads) {
    buffer[i] = (float)(myRank + 1);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcqRel);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[i] = *peerPtr;
  }
}

void launchKernelIntraBarrierSyncS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream) {
  kernelIntraBarrierSyncS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, output, N);
}

// ---------------------------------------------------------------------------
// S9: Intra Barrier Sync Split (Release + read + Acquire)
// ---------------------------------------------------------------------------

__global__ void kernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                              const void *devMemPtr,
                                              float *buffer, float *output,
                                              int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int i = tid; i < count; i += nthreads) {
    buffer[i] = (float)(myRank + 500);
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelease);

  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
    output[i] = *peerPtr;
  }

  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcquire);
}

void launchKernelIntraBarrierArriveWaitS(const void *devCommPtr,
                                        const void *devMemPtr, float *buffer,
                                        float *output, int N,
                                        flagcxStream_t stream) {
  kernelIntraBarrierArriveWaitS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, output, N);
}

// ---------------------------------------------------------------------------
// S6: Peer Pointer (Scalar) — team-based peer memory access
// ---------------------------------------------------------------------------

__global__ void kernelPeerPointerS(const void *devCommPtr,
                                   const void *devMemPtr,
                                   float *output, int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);
  int peer = (myRank + 1) % nRanks;

  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  for (int i = tid; i < count; i += nthreads) {
    size_t offset = i * sizeof(float);
    float *peerPtr = (float *)flagcxGetPeerPointerS(
        devMemPtr, offset, devCommPtr, FLAGCX_TEAM_INTRA, peer);
    output[i] = *peerPtr;
  }
}

void launchKernelPeerPointerS(const void *devCommPtr,
                              const void *devMemPtr, float *devOutput,
                              int count,
                              flagcxStream_t stream) {
  kernelPeerPointerS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, devOutput, count);
}

// ---------------------------------------------------------------------------
// S10: Intra AllReduce (Scalar) — composite using barriers + pointers
// ---------------------------------------------------------------------------

__global__ void kernelIntraAllReduceS(const void *devCommPtr,
                                           const void *devMemPtr,
                                           float *buffer, int count) {
  int myRank = flagcxDevCommGetIntraRank(devCommPtr);
  int nRanks = flagcxDevCommGetIntraSize(devCommPtr);

  // Cooperative indexing: partition elements across all ranks so each element
  // is processed by exactly one rank (eliminates cross-GPU race).
  int localNthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  int globalTid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_DIM_X * (myRank + FLAGCX_BLOCK_IDX_X * nRanks);
  int globalNthreads = localNthreads * nRanks;

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcquire);

  // Reduce + write: each rank handles a disjoint subset of elements,
  // reads from all peers, writes result to all peers.
  for (int i = globalTid; i < count; i += globalNthreads) {
    float sum = 0.0f;
    for (int peer = 0; peer < nRanks; peer++) {
      size_t offset = i * sizeof(float);
      float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
      sum += *peerPtr;
    }
    for (int peer = 0; peer < nRanks; peer++) {
      size_t offset = i * sizeof(float);
      float *peerPtr = (float *)flagcxGetIntraPointerS(devMemPtr, offset, peer);
      *peerPtr = sum;
    }
  }

  // Post-reduce barrier (release — ensure writes are visible)
  flagcxIntraBarrierSyncS(devCommPtr, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelease);
}

void launchKernelIntraAllReduceS(const void *devCommPtr,
                                  const void *devMemPtr, float *buffer,
                                  int count, flagcxStream_t stream) {
  kernelIntraAllReduceS<<<4, 256, 0, stream->base>>>(
      devCommPtr, devMemPtr, buffer, count);
}

// ---------------------------------------------------------------------------
// S7: Multicast Pointer (Scalar) — NVLS-dependent, commented out
// ---------------------------------------------------------------------------

// __global__ void kernelScalarMulticastPointer(const void *devCommPtr,
//                                              const void *devMemPtr,
//                                              float *output, int nElems) {
//   int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
//   if (tid < nElems) {
//     size_t offset = tid * sizeof(float);
//     float *mcPtr = (float *)flagcxGetMulticastPointerS(
//         devMemPtr, offset, devCommPtr);
//     output[tid] = *mcPtr;
//   }
// }
//
// void launchKernelMulticastPointerS(const void *devCommPtr,
//                                    const void *devMemPtr, float *devOutput,
//                                    int nBlocks, int nThreads,
//                                    flagcxStream_t stream) {
//   int nElems = nBlocks * nThreads;
//   kernelScalarMulticastPointer<<<nBlocks, nThreads, 0, stream->base>>>(
//       devCommPtr, devMemPtr, devOutput, nElems);
// }

// ===========================================================================
// Inter-Node Transport Tests (S1–S15, aligned with device_api_inter K1–K15)
// ===========================================================================

// ---------------------------------------------------------------------------
// S1: Transport Handle — GetFromCommS
// ---------------------------------------------------------------------------

__global__ void kernelNetGetFromCommS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
    results[0] = (net != nullptr) ? 1 : 0;
    results[1] = flagcxDevCommGetIntraSize(devCommPtr);
  }
}

void launchKernelNetGetFromCommS(const void *devCommPtr, int *devResults,
                                 flagcxStream_t stream) {
  kernelNetGetFromCommS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResults);
}

// ---------------------------------------------------------------------------
// S2: Signal/Counter Reset
// ---------------------------------------------------------------------------

__global__ void kernelNetResetS(const void *devCommPtr, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
    if (net == nullptr) {
      results[0] = 0;
      return;
    }

    const flagcxDevNet *netObj = (const flagcxDevNet *)net;
    if (!netObj->isValid()) {
      results[0] = 0;
      return;
    }

    // Reset signal slot 0
    flagcxDevNetResetSignal(net, (flagcxDevSignal_t)0);
    // Read it — should be 0
    uint64_t sig0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                            flagcxDeviceMemoryOrderRelaxed);
    results[0] = (sig0 == 0) ? 1 : 0;

    // Increase shadow by 5, read signal (still 0, shadow is separate)
    flagcxDevNetIncreaseSignalShadow(net, (flagcxDevSignal_t)0, 5);
    uint64_t sig1 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                            flagcxDeviceMemoryOrderRelaxed);
    results[1] = (sig1 == 0) ? 1 : 0;

    // Reset counter slot 0
    flagcxDevNetResetCounter(net, (flagcxDevCounter_t)0);
    // Read counter — should be 0
    uint64_t ctr0 = flagcxDevNetReadCounterS(net, (flagcxDevCounter_t)0, 64,
                                             flagcxDeviceMemoryOrderRelaxed);
    results[2] = (ctr0 == 0) ? 1 : 0;
  }
}

void launchKernelNetResetS(const void *devCommPtr, int *devResults,
                           flagcxStream_t stream) {
  kernelNetResetS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResults);
}

// ===========================================================================
// S3–S8: One-sided transport kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// S11: WaitSignalS + FlushS (standalone)
// Each rank signals all inter peers, waits for signals from all inter peers.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S11: WaitSignal + Flush (standalone)
// Reset signal, signal all inter peers, wait for signals, then flush.
// ---------------------------------------------------------------------------

__global__ void kernelNetWaitSignalFlushS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // Reset signal slot 0 (aligned with K11:1410) — no guard, matches K11
  flagcxDevNetResetSignal(net, (flagcxDevSignal_t)0);


  // Read baseline signal (aligned with K11:1411)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K11:1412)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Signal all inter peers (aligned with K11:1417-1420)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer < intraBase || peer >= intraBase + intraSize) {
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevSignal_t)0);
    }
  }


  // Wait for signals from all inter peers (aligned with K11:1423-1424)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                            s0 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }


  // Flush (aligned with K11:1427)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K11:1429)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetWaitSignalFlushS(const void *devCommPtr,
                                     flagcxStream_t stream) {
  kernelNetWaitSignalFlushS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// WaitCounterS (COMMENTED — standalone SignalCtrIncS is not supported
// by the GIN protocol. Counter wait is tested in S5 via PutS_RSigInc_LCtrInc.)
// ---------------------------------------------------------------------------

// __global__ void kernelNetWaitCounterS(const void *devCommPtr) {
//   if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
//     int myRank = flagcxDevCommGetRank(devCommPtr);
//     int nRanks = flagcxDevCommGetSize(devCommPtr);
//     int next = (myRank + 1) % nRanks;
//
//     const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
//     if (!net) return;
//
//     uint64_t c0 = flagcxDevNetReadCounterS(net, (flagcxDevCounter_t)0, 64,
//                                            flagcxDeviceMemoryOrderRelaxed);
//
//     flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevCounter_t)0,
//                              c0 + 1, 64, flagcxDeviceMemoryOrderAcquire);
//   }
// }
//
// void launchKernelNetWaitCounterS(const void *devCommPtr,
//                                  flagcxStream_t stream) {
//   kernelNetWaitCounterS<<<1, 32, 0, stream->base>>>(devCommPtr);
// }

// ---------------------------------------------------------------------------
// S10: Shadow (MeetShadowS — commented in test driver)
// increaseSignalShadow + signalSigInc to inter peers + waitSignalMeetShadow
// ---------------------------------------------------------------------------

__global__ void kernelNetWaitSignalMeetShadowS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) return;

  int nInterPeers = nRanks - intraSize;

  // Reset signal slot 2 and increase shadow
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevNetResetSignal(net, (flagcxDevSignal_t)2);
    flagcxDevNetIncreaseSignalShadow(net, (flagcxDevSignal_t)2,
                                     (uint64_t)nInterPeers);
  }

  // First world barrier: ensure all ranks have reset + increased shadow
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 0, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Second world barrier: ensure all ranks are ready before signaling
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, 1, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);

  // Single-thread: signal all inter peers, then wait
  if (FLAGCX_THREAD_IDX_X == 0) {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevSignal_t)2);
    }

    // Wait until signal meets shadow
    flagcxDevNetWaitSignalMeetShadowS(net, FLAGCX_COOP_THREAD,
                                      (flagcxDevSignal_t)2, 64,
                                      flagcxDeviceMemoryOrderAcquire);
  }
}

void launchKernelNetWaitSignalMeetShadowS(const void *devCommPtr,
                                          flagcxStream_t stream) {
  kernelNetWaitSignalMeetShadowS<<<1, 32, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S12: Inter-Barrier Test
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S12: Inter Barrier (stress test)
// Tests inter-node barrier synchronization with multiple iterations.
// ---------------------------------------------------------------------------

__global__ void kernelInterBarrierStress(const void *devCommPtr,
                                         int *devResults, int nIters) {
  int myRank = flagcxDevCommGetRank(devCommPtr);


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
      devResults[0] = -1; // no net context
    }
    return;
  }


  // Inter barrier loop (aligned with K12:1461-1463)
  for (int i = 0; i < nIters; i++) {
    flagcxInterBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X,
                            flagcxDeviceMemoryOrderAcqRel,
                            flagcxDevNetFenceLevel::Relaxed);
  }

  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    devResults[0] = 1; // success (aligned with K12:1466)
  }
}

void launchKernelInterBarrierS(const void *devCommPtr, int *devResults,
                               int nIters, flagcxStream_t stream) {
  kernelInterBarrierStress<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, devResults,
                                                         nIters);
}

// ---------------------------------------------------------------------------
// S6: FlushDecouple — PutS(None,None) + FlushS + SignalSigIncS + WaitSignalS + FlushS
// AlltoAll: put with no signal, flush, then signal separately, wait, flush.
// ---------------------------------------------------------------------------

__global__ void kernelNetFlushDecoupleS(const void *devCommPtr,
                                        const void *sendMemPtr,
                                        const void *recvMemPtr,
                                        size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);
  int nInterRanks = nRanks - intraSize;

  // Read baseline signal (aligned with K6:692)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // Pre-barrier (aligned with K6:693)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // Thread-parallelized put with None+None (aligned with K6:701-708)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     recvMemPtr, (size_t)myRank * chunkBytes,
                     sendMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }


  // Flush BEFORE signaling (aligned with K6:709)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Thread-parallelized signal loop (aligned with K6:712-716)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                              FLAGCX_COOP_THREAD, (flagcxDevSignal_t)0);
  }


  // WaitSignal (aligned with K6:717)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush after wait (aligned with K6:718)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final barrier (aligned with K6:719)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetFlushDecoupleS(const void *devCommPtr,
                                   const void *sendMemPtr,
                                   const void *recvMemPtr, size_t countPerPeer,
                                   flagcxStream_t stream) {
  kernelNetFlushDecoupleS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                        recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S3: PutS_RSigInc + WaitSignalS + FlushS
// AlltoAll with fused remote signal increment.
// ---------------------------------------------------------------------------

__global__ void kernelNetPutSignalIncS(const void *devCommPtr,
                                       const void *sendMemPtr,
                                       const void *recvMemPtr,
                                       size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;

  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }

  size_t chunkBytes = countPerPeer * sizeof(float);

  // World barrier before reading baseline signal (aligned with K3:386-387)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  // Read baseline signal (aligned with K3:388)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);

  // World barrier sync (aligned with K3:395)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

  // Thread-parallelized put loop (aligned with K3:411-422)
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevSignal_t)0);
  }

  // WaitSignal + Flush (aligned with K3:429-430)
  int nInterRanks = nRanks - intraSize;
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);

  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);

  // Final world barrier (aligned with K3:436)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);
}

void launchKernelNetPutSignalIncS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream) {
  kernelNetPutSignalIncS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                       recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S4: PutS_RSigAdd + WaitSignalS + FlushS
// AlltoAll with remote signal add (value = 1 per peer).
// ---------------------------------------------------------------------------

__global__ void kernelNetPutSignalAddS(const void *devCommPtr,
                                       const void *sendMemPtr,
                                       const void *recvMemPtr,
                                       size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);

  // World barrier before reading baseline signal (aligned with K4:470-471)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal (aligned with K4:472)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K4:473)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // Thread-parallelized put + separate signal loop (aligned with K4:477-486)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    // Put with None+None (aligned with K4:479-483)
    flagcxDevNetPutS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     recvMemPtr, (size_t)myRank * chunkBytes,
                     sendMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
    // Separate SignalAdd with value=2 (aligned with K4:484-485)
    flagcxDevNetSignalSigAddS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                              FLAGCX_COOP_THREAD, (flagcxDevSignal_t)0, 2);
  }


  // WaitSignal for s0 + nInterRanks * 2 (aligned with K4:487)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nInterRanks * 2, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush (aligned with K4:488)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K4:489)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetPutSignalAddS(const void *devCommPtr,
                                  const void *sendMemPtr,
                                  const void *recvMemPtr, size_t countPerPeer,
                                  flagcxStream_t stream) {
  kernelNetPutSignalAddS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                                       recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S5: PutS_RSigInc_LCtrInc + WaitSignalS + WaitCounterS + FlushS
// AlltoAll with both remote signal inc and local counter inc.
// ---------------------------------------------------------------------------

__global__ void kernelNetCounterPipelineS(const void *devCommPtr,
                                          const void *sendMemPtr,
                                          const void *recvMemPtr,
                                          size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);
  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier before reading baselines (aligned with K5:521-522)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal and counter (aligned with K5:523-524)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);
  uint64_t c0 = flagcxDevNetReadCounterS(net, (flagcxDevCounter_t)0, 64,
                                          flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K5:525)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Round 1: Put with SignalInc + CounterInc (aligned with K5:529-536)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc_LCtrInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                     recvMemPtr, (size_t)myRank * chunkBytes,
                                     sendMemPtr, (size_t)peer * chunkBytes,
                                     chunkBytes, FLAGCX_COOP_THREAD,
                                     (flagcxDevSignal_t)0,
                                     (flagcxDevCounter_t)0);
  }


  // WaitCounter (aligned with K5:537)
  flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevCounter_t)0,
                           c0 + (uint64_t)nInterRanks, 64,
                           flagcxDeviceMemoryOrderAcquire);


  // Stamp sentinel (aligned with K5:540-541)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    float *slot = (float *)flagcxGetLocalPointerS(sendMemPtr, (size_t)peer * chunkBytes);
    *slot = 999.0f;
  }

  // Barrier between rounds (aligned with K5:542)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Round 2: Put with SignalInc + CounterInc again (aligned with K5:546-553)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetPutS_RSigInc_LCtrInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                     recvMemPtr, (size_t)myRank * chunkBytes,
                                     sendMemPtr, (size_t)peer * chunkBytes,
                                     chunkBytes, FLAGCX_COOP_THREAD,
                                     (flagcxDevSignal_t)0,
                                     (flagcxDevCounter_t)0);
  }


  // WaitCounter for c0 + 2*nInterRanks (aligned with K5:554)
  flagcxDevNetWaitCounterS(net, FLAGCX_COOP_BLOCK, (flagcxDevCounter_t)0,
                           c0 + 2 * (uint64_t)nInterRanks, 64,
                           flagcxDeviceMemoryOrderAcquire);


  // WaitSignal for s0 + 2*nInterRanks (aligned with K5:555)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + 2 * (uint64_t)nInterRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush (aligned with K5:556)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K5:562)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetCounterPipelineS(const void *devCommPtr,
                                     const void *sendMemPtr,
                                     const void *recvMemPtr,
                                     size_t countPerPeer,
                                     flagcxStream_t stream) {
  kernelNetCounterPipelineS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(
      devCommPtr, sendMemPtr, recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S9: Signal (SigInc + SigAdd) — merged into single kernel
// Tests both SignalSigIncS and SignalSigAddS + WaitSignalS in sequence.
// ---------------------------------------------------------------------------

__global__ void kernelNetSignalS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier before reading baseline (aligned with K9:653-654)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal on slot 1 (aligned with K9:655)
  uint64_t s1 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)1, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K9:656)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Signal loop (aligned with K9:659-662)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer != myRank && (peer < intraBase || peer >= intraBase + intraSize)) {
      flagcxDevNetSignalSigIncS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                FLAGCX_COOP_THREAD, (flagcxDevSignal_t)1);
    }
  }


  // WaitSignal (aligned with K9:663-664)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)1,
                            s1 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }


  // Final world barrier (aligned with K9:665)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetSignalS(const void *devCommPtr, flagcxStream_t stream) {
  kernelNetSignalS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S7: PutValue — tests both PutValueS(None)+Signal and PutValueS_RSigInc
// Each rank writes uint64_t value = myRank*1000 + peer to peer's recv area.
// Phase 1: PutValueS(None) + SignalSigIncS + WaitSignalS
// Phase 2: PutValueS_RSigInc + WaitSignalS (fused putValue + signal)
// ---------------------------------------------------------------------------

__global__ void kernelNetPutValueS(const void *devCommPtr,
                                   const void *recvMemPtr,
                                   size_t putValBase) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier before reading baseline (aligned with K7:604-605)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Read baseline signal on slot 1 (aligned with K7:606)
  uint64_t s1 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)1, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // World barrier sync (aligned with K7:607)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // PutValue loop (aligned with K7:608-620)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
    flagcxDevNetPutValueS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                                  recvMemPtr, putValBase + (size_t)myRank * sizeof(uint64_t),
                                  val, FLAGCX_COOP_THREAD, (flagcxDevSignal_t)1);
  }


  // WaitSignal (aligned with K7:622-623)
  if (nInterRanks > 0) {
    flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)1,
                            s1 + (uint64_t)nInterRanks, 64,
                            flagcxDeviceMemoryOrderAcquire);
  }


  // Final world barrier (aligned with K7:624)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetPutValueS(const void *devCommPtr, const void *recvMemPtr,
                              size_t putValBase, flagcxStream_t stream) {
  kernelNetPutValueS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, recvMemPtr,
                                                  putValBase);
}

// ---------------------------------------------------------------------------
// S8: GetS + FlushS
// AlltoAll via one-sided get: each rank pulls from every inter peer.
// ---------------------------------------------------------------------------

__global__ void kernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                              const void *recvMemPtr, size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraBase = myRank - intraRank;


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // World barrier (aligned with K8:975)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Get loop (aligned with K8:981-989)
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    if (peer >= intraBase && peer < intraBase + intraSize) continue;
    flagcxDevNetGetS(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                     sendMemPtr, (size_t)myRank * chunkBytes,
                     recvMemPtr, (size_t)peer * chunkBytes,
                     chunkBytes, FLAGCX_COOP_THREAD);
  }


  // Flush (aligned with K8:990)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Final world barrier (aligned with K8:991)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed, flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetGetS(const void *devCommPtr, const void *sendMemPtr,
                         const void *recvMemPtr, size_t countPerPeer,
                         flagcxStream_t stream) {
  kernelNetGetS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
                                              recvMemPtr, countPerPeer);
}

// ---------------------------------------------------------------------------
// S15: Two-sided (COMMENTED)
// ---------------------------------------------------------------------------
// __global__ void kernelNetTwoSidedS(const void *devCommPtr,
//                                    const void *sendMemPtr,
//                                    const void *recvMemPtr,
//                                    size_t countPerPeer) {
//   int myRank = flagcxDevCommGetRank(devCommPtr);
//   int nRanks = flagcxDevCommGetSize(devCommPtr);
//   const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
//   if (!net) return;
//   size_t chunkBytes = countPerPeer * sizeof(float);
//   // Post receives from all peers
//   for (int peer = 0; peer < nRanks; peer++) {
//     if (peer == myRank) continue;
//     flagcxDevNetRecvS(net, FLAGCX_COOP_BLOCK, recvMemPtr,
//                       (size_t)peer * chunkBytes, countPerPeer,
//                       flagcxFloat, peer);
//   }
//   // Send to all peers
//   for (int peer = 0; peer < nRanks; peer++) {
//     if (peer == myRank) continue;
//     flagcxDevNetSendS(net, FLAGCX_COOP_BLOCK, sendMemPtr,
//                       (size_t)peer * chunkBytes, countPerPeer,
//                       flagcxFloat, peer);
//   }
//   flagcxDevNetTermS(net, FLAGCX_COOP_BLOCK);
//   flagcxDevNetWaitS(net, FLAGCX_COOP_BLOCK);
// }
//
// void launchKernelNetTwoSidedS(const void *devCommPtr, const void *sendMemPtr,
//                               const void *recvMemPtr, size_t countPerPeer,
//                               flagcxStream_t stream) {
//   kernelNetTwoSidedS<<<1, 128, 0, stream->base>>>(devCommPtr, sendMemPtr,
//                                                    recvMemPtr, countPerPeer);
// }

// ---------------------------------------------------------------------------
// S13: WorldBarrierS — sync + arrive/wait split in one kernel
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S13: WorldBarrierS — sync + arrive/wait split in one kernel
// Tests world barrier synchronization in both sync and split (arrive/wait) modes.
// ---------------------------------------------------------------------------

__global__ void kernelWorldBarrierS(const void *devCommPtr) {
  int myRank = flagcxDevCommGetRank(devCommPtr);


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  // Test sync (aligned with K13:1496)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcqRel,
                          flagcxDevNetFenceLevel::Relaxed);


  // Test arrive + wait (split) (aligned with K13:1499-1500)
  flagcxWorldBarrierArriveS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                            flagcxDeviceMemoryOrderRelease,
                            flagcxDevNetFenceLevel::Relaxed);


  flagcxWorldBarrierWaitS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderAcquire,
                          flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelWorldBarrierS(const void *devCommPtr, flagcxStream_t stream) {
  kernelWorldBarrierS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(devCommPtr);
}

// ---------------------------------------------------------------------------
// S14: OneSidedAlltoAll (composite) — put + signal + wait + flush + world barrier
// Each rank puts its chunk to every inter peer using PutS_RSigInc,
// waits for signals, flushes, then world barrier for completion.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// S14: OneSidedAlltoAll — composite put + signal + wait + flush + world barrier
// One-sided alltoall pattern using put with signal increment.
// ---------------------------------------------------------------------------

__global__ void kernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                           const void *sendMemPtr,
                                           const void *recvMemPtr,
                                           size_t countPerPeer) {
  int myRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);


  const void *net = flagcxDevNetGetFromCommS(devCommPtr, 0);
  if (!net) {
    return;
  }


  size_t chunkBytes = countPerPeer * sizeof(float);

  // Read signal baseline (aligned with K14:210)
  uint64_t s0 = flagcxDevNetReadSignalS(net, (flagcxDevSignal_t)0, 64,
                                        flagcxDeviceMemoryOrderRelaxed);


  // Pre-communication barrier (aligned with K14:213)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);


  // Thread-parallelized put loop (aligned with K14:217-221)
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (int peer = tid; peer < nRanks; peer += nthreads) {
    flagcxDevNetPutS_RSigInc(net, devCommPtr, FLAGCX_TEAM_WORLD, peer,
                             recvMemPtr, (size_t)myRank * chunkBytes,
                             sendMemPtr, (size_t)peer * chunkBytes,
                             chunkBytes, FLAGCX_COOP_THREAD,
                             (flagcxDevSignal_t)0);
  }


  // Wait for all incoming signals (aligned with K14:223)
  flagcxDevNetWaitSignalS(net, FLAGCX_COOP_BLOCK, (flagcxDevSignal_t)0,
                          s0 + (uint64_t)nRanks, 64,
                          flagcxDeviceMemoryOrderAcquire);


  // Flush to ensure data visibility (aligned with K14:224)
  flagcxDevNetFlushS(net, FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderRelaxed);


  // Post-communication barrier (aligned with K14:227)
  flagcxWorldBarrierSyncS(net, FLAGCX_COOP_BLOCK, FLAGCX_BLOCK_IDX_X, false,
                          flagcxDeviceMemoryOrderRelaxed,
                          flagcxDevNetFenceLevel::Relaxed);

}

void launchKernelNetOneSidedAlltoAllS(const void *devCommPtr,
                                      const void *sendMemPtr,
                                      const void *recvMemPtr,
                                      size_t countPerPeer,
                                      flagcxStream_t stream) {
  kernelNetOneSidedAlltoAllS<<<FLAGCX_DEVICE_CTA_COUNT, 128, 0, stream->base>>>(
      devCommPtr, sendMemPtr, recvMemPtr, countPerPeer);
}

FLAGCX_DEVICE_INLINE_DECORATOR bool
flagcxUnifiedIrTestCoopActive(flagcxDevCoopKind_t coopKind) {
  if (coopKind == FLAGCX_COOP_THREAD)
    return FLAGCX_THREAD_IDX_X == 0;
  if (coopKind == FLAGCX_COOP_WARP)
    return FLAGCX_THREAD_IDX_X < 32;
  return coopKind == FLAGCX_COOP_BLOCK;
}

// ---------------------------------------------------------------------------
// S18: flagcxDevPut (includes DevPutValue) — INTRA + WORLD teams
// Tests 4 cooperation levels × 2 teams = 8 combinations
// Buffer layout (8× base size):
//   Combination index = coopLevel * 2 + teamIdx
//   [0, bytes):        THREAD + INTRA    (idx 0)
//   [bytes, 2*bytes):  THREAD + WORLD    (idx 1)
//   [2*bytes, 3*bytes): WARP + INTRA     (idx 2)
//   [3*bytes, 4*bytes): WARP + WORLD     (idx 3)
//   [4*bytes, 5*bytes): BLOCK + INTRA    (idx 4)
//   [5*bytes, 6*bytes): BLOCK + WORLD    (idx 5)
//   [6*bytes, 7*bytes): GRID + INTRA     (idx 6)
//   [7*bytes, 8*bytes): GRID + WORLD     (idx 7)
// ---------------------------------------------------------------------------
__global__ void kernelDevPutIntraWorldS(const void *devCommPtr,
                                         const void *dstMemPtr,
                                         const void *srcMemPtr,
                                         int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int nBlocks = FLAGCX_GRID_DIM_X;
  int myBlockIdx = FLAGCX_BLOCK_IDX_X;

  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // === Combination 0: THREAD + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (intraRank + 1) % intraSize;
    size_t off = 0 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 1 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (intraRank + 1) % intraSize;
    size_t off = 2 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 3 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTRA ===
  {
    int peer = (intraRank + 1) % intraSize;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 4 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevPut(devCommPtr, dstMemPtr, baseOff + myOffset,
                   srcMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  {
    int peer = (worldRank + 1) % nRanks;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 5 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevPut(devCommPtr, dstMemPtr, baseOff + myOffset,
                   srcMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Flush to ensure all operations complete
  if (myBlockIdx < nContexts) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  }

  // Write result
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    result[0] = 1;
  }
}

void launchKernelDevPutIntraWorldS(const void *devCommPtr, const void *dstMemPtr,
                                    const void *srcMemPtr, int *devResult, size_t bytes,
                                    flagcxStream_t stream) {
  kernelDevPutIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S18: flagcxDevPutValue — INTRA + WORLD teams
// Writes a scalar value (uint64) to a remote peer's buffer.
// Uses 6 combinations: 3 coop kinds × 2 teams (INTRA, WORLD).
// Buffer layout: slot combo = coopIdx*2 + teamIdx, each holds 1 uint64_t.
// Expected value at slot combo = (uint64_t)(proc * 100 + combo).
// ---------------------------------------------------------------------------
__global__ void kernelDevPutValueIntraWorldS(const void *devCommPtr,
                                              const void *dstMemPtr,
                                              int *result, size_t /*bytes*/) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // === Combination 0: THREAD + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (intraRank + 1) % intraSize;
    uint64_t val = (uint64_t)(worldRank * 100 + 0);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)0 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_THREAD,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    uint64_t val = (uint64_t)(worldRank * 100 + 1);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)1 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (intraRank + 1) % intraSize;
    uint64_t val = (uint64_t)(worldRank * 100 + 2);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)2 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_WARP,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    uint64_t val = (uint64_t)(worldRank * 100 + 3);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)3 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTRA ===
  if (myBlockIdx == 0) {
    int peer = (intraRank + 1) % intraSize;
    uint64_t val = (uint64_t)(worldRank * 100 + 4);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)4 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  if (myBlockIdx == 0) {
    int peer = (worldRank + 1) % nRanks;
    uint64_t val = (uint64_t)(worldRank * 100 + 5);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)5 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  if (myBlockIdx < nContexts) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  }

  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) result[0] = 1;
}

void launchKernelDevPutValueIntraWorldS(const void *devCommPtr,
                                        const void *dstMemPtr, int *devResult,
                                        size_t bytes, flagcxStream_t stream) {
  kernelDevPutValueIntraWorldS<<<4, 128, 0,
                                 stream->base>>>(devCommPtr, dstMemPtr,
                                                 devResult, bytes);
}

// ---------------------------------------------------------------------------
// S19: flagcxDevGet — INTRA + WORLD teams
// Tests 4 cooperation levels × 2 teams = 8 combinations
// Buffer layout: same as S16 (8× base size)
// ---------------------------------------------------------------------------
__global__ void kernelDevGetIntraWorldS(const void *devCommPtr,
                                         const void *remoteMemPtr,
                                         const void *localMemPtr,
                                         int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int nBlocks = FLAGCX_GRID_DIM_X;
  int myBlockIdx = FLAGCX_BLOCK_IDX_X;

  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // === Combination 0: THREAD + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (intraRank + 1) % intraSize;
    size_t off = 0 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 1 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (intraRank + 1) % intraSize;
    size_t off = 2 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 3 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTRA ===
  {
    int peer = (intraRank + 1) % intraSize;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 4 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevGet(devCommPtr, remoteMemPtr, baseOff + myOffset,
                   localMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  {
    int peer = (worldRank + 1) % nRanks;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 5 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevGet(devCommPtr, remoteMemPtr, baseOff + myOffset,
                   localMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Flush to ensure all operations complete
  if (myBlockIdx < nContexts) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  }

  if (FLAGCX_THREAD_IDX_X == 0 && myBlockIdx == 0) result[0] = 1;
}

void launchKernelDevGetIntraWorldS(const void *devCommPtr, const void *remoteMemPtr,
                                    const void *localMemPtr, int *devResult, size_t bytes,
                                    flagcxStream_t stream) {
  kernelDevGetIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, remoteMemPtr, localMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S16: flagcxDevBarrierSync (includes ArriveWait) — INTRA + WORLD (merged)
// Part 1: INTRA barrier
// Part 2: WORLD barrier
// ---------------------------------------------------------------------------
__global__ void kernelDevBarrierIntraWorldS(const void *devCommPtr, int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? FLAGCX_BLOCK_IDX_X % nContexts : 0;

  // Part 1: INTRA barrier
  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTRA, /*index=*/FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);

  // Sync between parts
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Part 2: WORLD barrier
  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_WORLD, /*index=*/FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);

  if (FLAGCX_THREAD_IDX_X == 0) result[FLAGCX_BLOCK_IDX_X] = 1;
}

void launchKernelDevBarrierIntraWorldS(const void *devCommPtr, int *devResult,
                                        flagcxStream_t stream) {
  kernelDevBarrierIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S16: flagcxDevBarrierArrive + flagcxDevBarrierWait — INTRA + WORLD
// Verifies split arrive/wait semantics for both INTRA and WORLD teams.
// ---------------------------------------------------------------------------
__global__ void kernelDevBarrierArriveWaitIntraWorldS(const void *devCommPtr,
                                                      int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId =
      nContexts > 0 ? FLAGCX_BLOCK_IDX_X % nContexts : 0;

  // Part 1: INTRA barrier — split Arrive + Wait
  flagcxDevBarrierArrive(devCommPtr, FLAGCX_TEAM_INTRA, FLAGCX_BLOCK_IDX_X,
                         contextId, FLAGCX_COOP_BLOCK,
                         flagcxDeviceMemoryOrderRelease,
                         flagcxDeviceScopeSystem);
  flagcxDevBarrierWait(devCommPtr, FLAGCX_TEAM_INTRA, FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcquire,
                       flagcxDeviceScopeSystem);

  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Part 2: WORLD barrier — split Arrive + Wait
  flagcxDevBarrierArrive(devCommPtr, FLAGCX_TEAM_WORLD, FLAGCX_BLOCK_IDX_X,
                         contextId, FLAGCX_COOP_BLOCK,
                         flagcxDeviceMemoryOrderRelease,
                         flagcxDeviceScopeSystem);
  flagcxDevBarrierWait(devCommPtr, FLAGCX_TEAM_WORLD, FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcquire,
                       flagcxDeviceScopeSystem);

  if (FLAGCX_THREAD_IDX_X == 0) result[FLAGCX_BLOCK_IDX_X] = 1;
}

void launchKernelDevBarrierArriveWaitIntraWorldS(const void *devCommPtr,
                                                 int *devResult,
                                                 flagcxStream_t stream) {
  kernelDevBarrierArriveWaitIntraWorldS<<<4, 128, 0,
                                         stream->base>>>(devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S20: flagcxDevSignalInc + flagcxDevSignalAdd + flagcxDevWaitSignal +
//      flagcxDevReadSignal + flagcxDevResetSignal — INTRA + WORLD (signal-only)
// 6 combos: 3 coop kinds × 2 teams (INTRA, WORLD), slot = combo.
// Per combo:
//   Reset(slot) → assert ReadSignal==0
//   Leg A: SignalInc(peer=next) → WaitSignal(slot,1) → assert ReadSignal==1
//   Reset(slot)
//   Leg B: SignalAdd(peer=next, value=5) → WaitSignal(slot,5) → assert ReadSignal==5
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevSignalStandaloneIntraWorldS(const void *devCommPtr,
                                                      int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  uint64_t expectedInc = 1;  // leg A: 4 blocks, 4 contexts, 1 block per context
  uint64_t expectedAdd = 5;  // leg B: 1 block per context does +5

  bool ok = true;

#define S20_INTRA_COMBO(slot, teamKind, peer, coopKind, waitOrder)             \
  do {                                                                         \
    /* Reset local signal (local op), then verify zero, then barrier */        \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != 0) ok = false;                                                  \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    /* Barrier: all ranks have reset before any rank sends signal */           \
    flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTRA, myBlockIdx,            \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    /* Leg A: SignalInc */                                                     \
    if (flagcxUnifiedIrTestCoopActive(coopKind)) {                             \
      flagcxDevSignalInc(devCommPtr, teamKind, peer,                           \
                         (flagcxDevSignal_t)(slot), contextId,                 \
                         coopKind, flagcxDeviceScopeSystem);                   \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    if (flagcxUnifiedIrTestCoopActive(coopKind)) {                             \
      flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expectedInc,  \
                          64, contextId, coopKind, waitOrder);                 \
    }                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != expectedInc) ok = false;                                        \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    /* Reset before Leg B (local), then verify, then barrier */                \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != 0) ok = false;                                                  \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    /* Barrier: all ranks have reset before any rank sends Leg B signal */     \
    flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTRA, myBlockIdx,            \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    /* Leg B: SignalAdd(value=5) */                                            \
    if (flagcxUnifiedIrTestCoopActive(coopKind)) {                             \
      flagcxDevSignalAdd(devCommPtr, teamKind, peer,                           \
                         (flagcxDevSignal_t)(slot), (uint64_t)5, contextId,    \
                         coopKind, flagcxDeviceScopeSystem);                   \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    if (flagcxUnifiedIrTestCoopActive(coopKind)) {                             \
      flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expectedAdd,  \
                          64, contextId, coopKind, waitOrder);                 \
    }                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != expectedAdd) ok = false;                                        \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
  } while (0)

  // combo 0: THREAD + INTRA  (slot 0)
  S20_INTRA_COMBO(0, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_THREAD, flagcxDeviceMemoryOrderRelaxed);
  // combo 1: THREAD + WORLD  (slot 1)
  S20_INTRA_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_THREAD, flagcxDeviceMemoryOrderAcquire);
  // combo 2: WARP + INTRA    (slot 2)
  S20_INTRA_COMBO(2, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_WARP, flagcxDeviceMemoryOrderSeqCst);
  // combo 3: WARP + WORLD    (slot 3)
  S20_INTRA_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_WARP, flagcxDeviceMemoryOrderRelaxed);
  // combo 4: BLOCK + INTRA   (slot 4)
  S20_INTRA_COMBO(4, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderSeqCst);
  // combo 5: BLOCK + WORLD   (slot 5)
  S20_INTRA_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderAcquire);

#undef S20_INTRA_COMBO

  if (FLAGCX_THREAD_IDX_X == 0)
    atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevSignalStandaloneIntraWorldS(const void *devCommPtr, int *devResult,
                                                 flagcxStream_t stream) {
  kernelDevSignalStandaloneIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S17: flagcxDevTeamResolution — INTRA + WORLD teams
// Tests 4 cooperation levels × 2 teams = 8 combinations
// Each rank writes sizeof(float) to peer's buffer at deterministic offset
// Buffer layout: [i*maxRanks*sizeof(float), (i+1)*maxRanks*sizeof(float)) for combo i
// Within region: rank writes at rankInTeam * sizeof(float) offset
// ---------------------------------------------------------------------------
__global__ void kernelDevTeamResolutionIntraWorldS(const void *devCommPtr,
                                                    const void *dstMemPtr,
                                                    const void *srcMemPtr,
                                                    int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;

  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // Determine max ranks for buffer sizing
  int maxRanks = intraSize;
  if (nRanks > maxRanks) maxRanks = nRanks;

  // === Combination 0: THREAD + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (intraRank + 1) % intraSize;
    size_t dstOff = 0 * maxRanks * sizeof(float) + intraRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    size_t dstOff = 1 * maxRanks * sizeof(float) + worldRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTRA ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (intraRank + 1) % intraSize;
    size_t dstOff = 2 * maxRanks * sizeof(float) + intraRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    size_t dstOff = 3 * maxRanks * sizeof(float) + worldRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTRA ===
  {
    int peer = (intraRank + 1) % intraSize;
    size_t dstOff = 4 * maxRanks * sizeof(float) + intraRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_BLOCK,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  {
    int peer = (worldRank + 1) % nRanks;
    size_t dstOff = 5 * maxRanks * sizeof(float) + worldRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Flush to ensure all FIFO operations complete before barrier
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Barrier to ensure all puts land before host reads
  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTRA, /*index=*/myBlockIdx,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);

  // Flush after barrier
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  if (FLAGCX_THREAD_IDX_X == 0 && myBlockIdx == 0) result[0] = 1;
}

void launchKernelDevTeamResolutionIntraWorldS(const void *devCommPtr,
                                               const void *dstMemPtr,
                                               const void *srcMemPtr, int *devResult,
                                               flagcxStream_t stream) {
  kernelDevTeamResolutionIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult);
}

// ---------------------------------------------------------------------------
// S22: flagcxDevPut_RSigInc + flagcxDevWaitSignal — INTRA + WORLD teams
// Tests 4 cooperation levels × 2 teams = 8 combinations
// Buffer layout: 8× base size
// Signal slots: 0-7, one per combination
// NOTE: Requires concurrent multi-rank launch (ring dependency).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// S21: flagcxDevPut + flagcxDevSignalInc + flagcxDevSignalAdd +
//      flagcxDevWaitSignal + flagcxDevReadSignal + flagcxDevResetSignal
//      — INTRA + WORLD (split put+signal form)
// 6 combos: 3 coop kinds × 2 teams.
// Per combo:
//   ResetSignal(slot=combo) → Put(bytes) →
//   even combo: SignalInc → WaitSignal(slot, 1) → assert ReadSignal==1
//   odd combo:  SignalAdd(value=3) → WaitSignal(slot, 3) → assert ReadSignal==3
//   Then verify payload.
// result[0] = 1 iff all signal reads and payload checks pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutSignalWaitIntraWorldS(const void *devCommPtr,
                                                   const void *dstMemPtr,
                                                   const void *srcMemPtr,
                                                   int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S21_INTRA_COMBO(slot, teamKind, peer, expected)                        \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      size_t off = (slot)*bytes;                                                \
      flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,          \
                   teamKind, peer, contextId, FLAGCX_COOP_THREAD,               \
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);    \
      if ((slot) % 2 == 0) {                                                    \
        flagcxDevSignalInc(devCommPtr, teamKind, peer,                          \
                           (flagcxDevSignal_t)(slot), contextId,                \
                           FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);        \
      } else {                                                                  \
        flagcxDevSignalAdd(devCommPtr, teamKind, peer,                          \
                           (flagcxDevSignal_t)(slot), (uint64_t)3, contextId,   \
                           FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);        \
      }                                                                         \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expected, 64,    \
                        contextId, FLAGCX_COOP_BLOCK,                           \
                        flagcxDeviceMemoryOrderAcquire);                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != expected) ok = false;                                            \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
  } while (0)

  // combo 0: THREAD + INTRA (even → Inc, expected=1)
  S21_INTRA_COMBO(0, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize, 1);
  // combo 1: THREAD + WORLD (odd → Add(3), expected=3)
  S21_INTRA_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 2: WARP + INTRA (even → Inc, expected=1)
  S21_INTRA_COMBO(2, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize, 1);
  // combo 3: WARP + WORLD (odd → Add(3), expected=3)
  S21_INTRA_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 4: BLOCK + INTRA (even → Inc, expected=1)
  S21_INTRA_COMBO(4, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize, 1);
  // combo 5: BLOCK + WORLD (odd → Add(3), expected=3)
  S21_INTRA_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);

#undef S21_INTRA_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutSignalWaitIntraWorldS(const void *devCommPtr,
                                              const void *dstMemPtr,
                                              const void *srcMemPtr, int *devResult,
                                              size_t bytes, flagcxStream_t stream) {
  kernelDevPutSignalWaitIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S22: flagcxDevPut_RSigInc + flagcxDevPut_RSigAdd — INTRA + WORLD teams
// 6 combos: 3 coop kinds × 2 teams (INTRA, WORLD), slot = combo.
// Per combo:
//   ResetSignal(slot) → assert ReadSignal==0
//   even combo: Put_RSigInc (expected=1)
//   odd combo: Put_RSigAdd(value=3) (expected=3)
//   WaitSignal(slot, expected) → assert ReadSignal==expected
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutRSigIntraWorldS(const void *devCommPtr,
                                            const void *dstMemPtr,
                                            const void *srcMemPtr,
                                            int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S22_INTRA_COMBO(slot, teamKind, peer, expected)                        \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 0) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      size_t off = (slot)*bytes;                                                \
      if ((slot) % 2 == 0)                                                      \
        flagcxDevPut_RSigInc(devCommPtr, dstMemPtr, off, srcMemPtr, off,       \
                             bytes, teamKind, peer, contextId,                  \
                             FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,       \
                             flagcxDeviceMemoryOrderRelease,                    \
                             (flagcxDevSignal_t)(slot));                        \
      else                                                                      \
        flagcxDevPut_RSigAdd(devCommPtr, dstMemPtr, off, srcMemPtr, off,       \
                             bytes, teamKind, peer, contextId,                  \
                             FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,       \
                             flagcxDeviceMemoryOrderRelease,                    \
                             (flagcxDevSignal_t)(slot), (uint64_t)3);           \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expected, 64,    \
                        contextId, FLAGCX_COOP_BLOCK,                           \
                        flagcxDeviceMemoryOrderAcquire);                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != expected) ok = false;                                            \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
  } while (0)

  // combo 0: THREAD + INTRA (even → RSigInc, expected=1)
  S22_INTRA_COMBO(0, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize, 1);
  // combo 1: THREAD + WORLD (odd → RSigAdd(3), expected=3)
  S22_INTRA_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 2: WARP + INTRA (even → RSigInc, expected=1)
  S22_INTRA_COMBO(2, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize, 1);
  // combo 3: WARP + WORLD (odd → RSigAdd(3), expected=3)
  S22_INTRA_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 4: BLOCK + INTRA (even → RSigInc, expected=1)
  S22_INTRA_COMBO(4, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize, 1);
  // combo 5: BLOCK + WORLD (odd → RSigAdd(3), expected=3)
  S22_INTRA_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);

#undef S22_INTRA_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutRSigIntraWorldS(const void *devCommPtr,
                                        const void *dstMemPtr,
                                        const void *srcMemPtr, int *devResult,
                                        size_t bytes, flagcxStream_t stream) {
  kernelDevPutRSigIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S23: flagcxDevPut_LCtrInc + flagcxDevPut_RSigInc_LCtrInc +
//      flagcxDevPut_RSigAdd_LCtrInc — INTRA + WORLD teams
// 6 combos: 3 coop kinds × 2 teams, slot = combo, counter = combo.
// Per combo:
//   ResetCounter(ctr=combo) → assert ReadCounter==0
//   ResetSignal(sig=combo) → assert ReadSignal==0
//   combo%3==0: Put_LCtrInc (counter-only)
//   combo%3==1: Put_RSigInc_LCtrInc (signal+counter, sig expected=1)
//   combo%3==2: Put_RSigAdd_LCtrInc (signal+counter, sig expected=3)
//   WaitCounter(ctr=combo, 1) → assert ReadCounter==1
//   If combo%3!=0: WaitSignal(sig=combo, expected) → assert ReadSignal==expected
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutCounterIntraWorldS(const void *devCommPtr,
                                                const void *dstMemPtr,
                                                const void *srcMemPtr,
                                                int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S23_INTRA_COMBO(slot, teamKind, peer)                                  \
  do {                                                                          \
    flagcxDevCounter_t ctr = (flagcxDevCounter_t)(slot);                       \
    flagcxDevSignal_t sig = (flagcxDevSignal_t)(slot);                         \
    int variant = (slot) % 3;                                                   \
    /* Reset counter and signal */                                              \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      flagcxDevResetCounter(devCommPtr, contextId, ctr);                        \
      flagcxDevResetSignal(devCommPtr, contextId, sig);                         \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Assert both are zero */                                                  \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t cv = flagcxDevReadCounter(devCommPtr, ctr, 64, contextId,       \
                                          flagcxDeviceMemoryOrderAcquire);      \
      uint64_t sv = flagcxDevReadSignal(devCommPtr, sig, 64, contextId,        \
                                         flagcxDeviceMemoryOrderAcquire);       \
      if (cv != 0 || sv != 0) ok = false;                                      \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    /* Put operation with counter (and optionally signal) */                    \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      size_t off = (slot)*bytes;                                                \
      if (variant == 0)                                                         \
        flagcxDevPut_LCtrInc(devCommPtr, dstMemPtr, off, srcMemPtr, off,       \
                             bytes, teamKind, peer, contextId,                  \
                             FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,       \
                             flagcxDeviceMemoryOrderRelease, ctr);              \
      else if (variant == 1)                                                    \
        flagcxDevPut_RSigInc_LCtrInc(devCommPtr, dstMemPtr, off, srcMemPtr,    \
                                     off, bytes, teamKind, peer, contextId,     \
                                     FLAGCX_COOP_THREAD,                        \
                                     flagcxDeviceScopeSystem,                   \
                                     flagcxDeviceMemoryOrderRelease, sig, ctr); \
      else                                                                      \
        flagcxDevPut_RSigAdd_LCtrInc(devCommPtr, dstMemPtr, off, srcMemPtr,    \
                                     off, bytes, teamKind, peer, contextId,     \
                                     FLAGCX_COOP_THREAD,                        \
                                     flagcxDeviceScopeSystem,                   \
                                     flagcxDeviceMemoryOrderRelease, sig,       \
                                     (uint64_t)3, ctr);                         \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Wait and verify counter */                                               \
    flagcxDevWaitCounter(devCommPtr, ctr, 1, 64, contextId,                     \
                         FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderAcquire);    \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t cv = flagcxDevReadCounter(devCommPtr, ctr, 64, contextId,       \
                                          flagcxDeviceMemoryOrderAcquire);      \
      if (cv != 1) ok = false;                                                  \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Wait and verify signal (if variant != 0) */                              \
    if (variant != 0) {                                                         \
      uint64_t expectedSig = (variant == 1) ? 1 : 3;                            \
      flagcxDevWaitSignal(devCommPtr, sig, expectedSig, 64, contextId,          \
                          FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderAcquire);   \
      if (FLAGCX_THREAD_IDX_X == 0) {                                          \
        uint64_t sv = flagcxDevReadSignal(devCommPtr, sig, 64, contextId,      \
                                           flagcxDeviceMemoryOrderAcquire);     \
        if (sv != expectedSig) ok = false;                                      \
      }                                                                         \
      flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                       \
    }                                                                           \
  } while (0)

  // combo 0: THREAD + INTRA (variant=0: LCtrInc only)
  S23_INTRA_COMBO(0, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize);
  // combo 1: THREAD + WORLD (variant=1: RSigInc_LCtrInc)
  S23_INTRA_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);
  // combo 2: WARP + INTRA (variant=2: RSigAdd_LCtrInc)
  S23_INTRA_COMBO(2, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize);
  // combo 3: WARP + WORLD (variant=0: LCtrInc only)
  S23_INTRA_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);
  // combo 4: BLOCK + INTRA (variant=1: RSigInc_LCtrInc)
  S23_INTRA_COMBO(4, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize);
  // combo 5: BLOCK + WORLD (variant=2: RSigAdd_LCtrInc)
  S23_INTRA_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);

#undef S23_INTRA_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutCounterIntraWorldS(const void *devCommPtr,
                                           const void *dstMemPtr,
                                           const void *srcMemPtr, int *devResult,
                                           size_t bytes, flagcxStream_t stream) {
  kernelDevPutCounterIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S24: flagcxDevPutValue_RSigInc + flagcxDevPutValue_RSigAdd — INTRA + WORLD
// 6 combos: 3 coop kinds × 2 teams (INTRA, WORLD), slot = combo.
// Per combo:
//   ResetSignal(slot) → assert ReadSignal==0
//   even combo: PutValue_RSigInc (expected=1)
//   odd combo: PutValue_RSigAdd(value=3) (expected=3)
//   WaitSignal(slot, expected) → assert ReadSignal==expected
// result[0] = 1 iff all assertions pass.
// Buffer layout: each combo writes 1 uint64_t at slot offset.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutValueRSigIntraWorldS(const void *devCommPtr,
                                                  const void *dstMemPtr,
                                                  int *result, size_t /*bytes*/) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S24_INTRA_COMBO(slot, teamKind, peer, coopKind, expected)              \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 0) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (flagcxUnifiedIrTestCoopActive(coopKind)) {                             \
      size_t off = (slot) * sizeof(uint64_t);                                  \
      uint64_t val = (uint64_t)(worldRank * 100 + (slot));                     \
      if ((slot) % 2 == 0)                                                      \
        flagcxDevPutValue_RSigInc(devCommPtr, dstMemPtr, off, val, teamKind,   \
                                  peer, contextId, coopKind,                    \
                                  flagcxDeviceScopeSystem,                      \
                                  flagcxDeviceMemoryOrderRelease,               \
                                  (flagcxDevSignal_t)(slot));                   \
      else                                                                      \
        flagcxDevPutValue_RSigAdd(devCommPtr, dstMemPtr, off, val, teamKind,   \
                                  peer, contextId, coopKind,                    \
                                  flagcxDeviceScopeSystem,                      \
                                  flagcxDeviceMemoryOrderRelease,               \
                                  (flagcxDevSignal_t)(slot), (uint64_t)3);      \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expected, 64,    \
                        contextId, FLAGCX_COOP_BLOCK,                           \
                        flagcxDeviceMemoryOrderAcquire);                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != expected) ok = false;                                            \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
  } while (0)

  // combo 0: THREAD + INTRA (even → RSigInc, expected=1)
  S24_INTRA_COMBO(0, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_THREAD, 1);
  // combo 1: THREAD + WORLD (odd → RSigAdd(3), expected=3)
  S24_INTRA_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_THREAD, 3);
  // combo 2: WARP + INTRA (even → RSigInc, expected=1)
  S24_INTRA_COMBO(2, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_WARP, 1);
  // combo 3: WARP + WORLD (odd → RSigAdd(3), expected=3)
  S24_INTRA_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_WARP, 3);
  // combo 4: BLOCK + INTRA (even → RSigInc, expected=1)
  S24_INTRA_COMBO(4, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_BLOCK, 1);
  // combo 5: BLOCK + WORLD (odd → RSigAdd(3), expected=3)
  S24_INTRA_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_BLOCK, 3);

#undef S24_INTRA_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutValueRSigIntraWorldS(const void *devCommPtr,
                                             const void *dstMemPtr,
                                             int *devResult, size_t bytes,
                                             flagcxStream_t stream) {
  kernelDevPutValueRSigIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S25: flagcxDevIncreaseSignalShadow + flagcxDevWaitSignalMeetShadow +
//      flagcxDevFlush — INTRA + WORLD teams
// 6 combos: 3 coop kinds × 2 teams (INTRA, WORLD), slot = combo.
// Per combo:
//   ResetSignal(slot) → assert ReadSignal==0
//   IncreaseSignalShadow(slot, increment=5)
//   SignalInc(peer=next, slot) → repeat 5 times
//   WaitSignalMeetShadow(slot) → assert ReadSignal==5
//   Flush(contextId)
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevSignalShadowFlushIntraWorldS(const void *devCommPtr,
                                                       int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S25_INTRA_COMBO(slot, teamKind, peer, coopKind)                        \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 0) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    /* Increase shadow by 5 */                                                  \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevIncreaseSignalShadow(devCommPtr, contextId,                     \
                                    (flagcxDevSignal_t)(slot),                 \
                                    (uint64_t)5);                               \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Signal 5 times to meet shadow */                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      for (int i = 0; i < 5; i++)                                               \
        flagcxDevSignalInc(devCommPtr, teamKind, peer,                          \
                           (flagcxDevSignal_t)(slot), contextId,                \
                           FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);        \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Wait for shadow to be met */                                             \
    if (flagcxUnifiedIrTestCoopActive(coopKind)) {                              \
      flagcxDevWaitSignalMeetShadow(devCommPtr, contextId,                      \
                                    (flagcxDevSignal_t)(slot), 64,              \
                                    coopKind,                                   \
                                    flagcxDeviceMemoryOrderAcquire);            \
    }                                                                           \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 5) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Flush */                                                                 \
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,                    \
                   flagcxDeviceMemoryOrderAcquire);                             \
  } while (0)

  // combo 0: THREAD + INTRA
  S25_INTRA_COMBO(0, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_THREAD);
  // combo 1: THREAD + WORLD
  S25_INTRA_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_THREAD);
  // combo 2: WARP + INTRA
  S25_INTRA_COMBO(2, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_WARP);
  // combo 3: WARP + WORLD
  S25_INTRA_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_WARP);
  // combo 4: BLOCK + INTRA
  S25_INTRA_COMBO(4, FLAGCX_TEAM_INTRA, (intraRank + 1) % intraSize,
                  FLAGCX_COOP_BLOCK);
  // combo 5: BLOCK + WORLD
  S25_INTRA_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_BLOCK);

#undef S25_INTRA_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevSignalShadowFlushIntraWorldS(const void *devCommPtr,
                                                  int *devResult,
                                                  flagcxStream_t stream) {
  kernelDevSignalShadowFlushIntraWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, devResult);
}


// ===========================================================================
// Unified One-Sided IR Tests — INTER Suite (S16–S21)
// INTER + WORLD teams (8 combinations: 4 coop × 2 teams)
// ===========================================================================

// ---------------------------------------------------------------------------
// S18: flagcxDevPut — INTER + WORLD teams
// Tests 4 cooperation levels × 2 teams = 8 combinations
// Buffer layout (8× base size):
//   Combination index = coopLevel * 2 + teamIdx
//   [0, bytes):        THREAD + INTER    (idx 0)
//   [bytes, 2*bytes):  THREAD + WORLD    (idx 1)
//   [2*bytes, 3*bytes): WARP + INTER     (idx 2)
//   [3*bytes, 4*bytes): WARP + WORLD     (idx 3)
//   [4*bytes, 5*bytes): BLOCK + INTER    (idx 4)
//   [5*bytes, 6*bytes): BLOCK + WORLD    (idx 5)
//   [6*bytes, 7*bytes): GRID + INTER     (idx 6)
//   [7*bytes, 8*bytes): GRID + WORLD     (idx 7)
// ---------------------------------------------------------------------------
__global__ void kernelDevPutInterWorldS(const void *devCommPtr,
                                        const void *dstMemPtr,
                                        const void *srcMemPtr,
                                        int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int nBlocks = FLAGCX_GRID_DIM_X;
  int myBlockIdx = FLAGCX_BLOCK_IDX_X;

  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // === Combination 0: THREAD + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (nodeIdx + 1) % nNodes;
    size_t off = 0 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 1 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (nodeIdx + 1) % nNodes;
    size_t off = 2 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 3 * bytes;
    flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTER ===
  {
    int peer = (nodeIdx + 1) % nNodes;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 4 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevPut(devCommPtr, dstMemPtr, baseOff + myOffset,
                   srcMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  {
    int peer = (worldRank + 1) % nRanks;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 5 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevPut(devCommPtr, dstMemPtr, baseOff + myOffset,
                   srcMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Flush to ensure all operations complete
  if (myBlockIdx < nContexts) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  }

  // Write result
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    result[0] = 1;
  }
}

void launchKernelDevPutInterWorldS(const void *devCommPtr, const void *dstMemPtr,
                                   const void *srcMemPtr, int *devResult, size_t bytes,
                                   flagcxStream_t stream) {
  kernelDevPutInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S18: flagcxDevPutValue — INTER + WORLD teams
// Writes a scalar uint64 value to a remote peer's buffer.
// Uses 6 combinations: 3 coop kinds × 2 teams (INTER, WORLD).
// Expected value at slot combo = (uint64_t)(proc * 100 + combo).
// ---------------------------------------------------------------------------
__global__ void kernelDevPutValueInterWorldS(const void *devCommPtr,
                                              const void *dstMemPtr,
                                              int *result, size_t /*bytes*/) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // === Combination 0: THREAD + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (nodeIdx + 1) % nNodes;
    uint64_t val = (uint64_t)(worldRank * 100 + 0);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)0 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_THREAD,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    uint64_t val = (uint64_t)(worldRank * 100 + 1);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)1 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (nodeIdx + 1) % nNodes;
    uint64_t val = (uint64_t)(worldRank * 100 + 2);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)2 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_WARP,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    uint64_t val = (uint64_t)(worldRank * 100 + 3);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)3 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTER ===
  if (myBlockIdx == 0) {
    int peer = (nodeIdx + 1) % nNodes;
    uint64_t val = (uint64_t)(worldRank * 100 + 4);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)4 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  if (myBlockIdx == 0) {
    int peer = (worldRank + 1) % nRanks;
    uint64_t val = (uint64_t)(worldRank * 100 + 5);
    flagcxDevPutValue(devCommPtr, dstMemPtr, (size_t)5 * sizeof(uint64_t), val,
                      FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                      flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  if (myBlockIdx < nContexts) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  }

  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) result[0] = 1;
}

void launchKernelDevPutValueInterWorldS(const void *devCommPtr,
                                        const void *dstMemPtr, int *devResult,
                                        size_t bytes, flagcxStream_t stream) {
  kernelDevPutValueInterWorldS<<<4, 128, 0,
                                 stream->base>>>(devCommPtr, dstMemPtr,
                                                 devResult, bytes);
}

// ---------------------------------------------------------------------------
// S19: flagcxDevGet — INTER + WORLD teams
// Tests 4 cooperation levels × 2 teams = 8 combinations
// Buffer layout: same as S16 (8× base size)
// ---------------------------------------------------------------------------
__global__ void kernelDevGetInterWorldS(const void *devCommPtr,
                                        const void *remoteMemPtr,
                                        const void *localMemPtr,
                                        int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int nBlocks = FLAGCX_GRID_DIM_X;
  int myBlockIdx = FLAGCX_BLOCK_IDX_X;

  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // === Combination 0: THREAD + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (nodeIdx + 1) % nNodes;
    size_t off = 0 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 1 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (nodeIdx + 1) % nNodes;
    size_t off = 2 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    size_t off = 3 * bytes;
    flagcxDevGet(devCommPtr, remoteMemPtr, off, localMemPtr, off, bytes,
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTER ===
  {
    int peer = (nodeIdx + 1) % nNodes;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 4 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevGet(devCommPtr, remoteMemPtr, baseOff + myOffset,
                   localMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  {
    int peer = (worldRank + 1) % nRanks;
    size_t blockBytes = (bytes + nBlocks - 1) / nBlocks;
    size_t myOffset = myBlockIdx * blockBytes;
    size_t baseOff = 5 * bytes;
    size_t copyBytes = (myOffset + blockBytes > bytes) ? (bytes - myOffset) : blockBytes;
    if (myOffset < bytes) {
      flagcxDevGet(devCommPtr, remoteMemPtr, baseOff + myOffset,
                   localMemPtr, baseOff + myOffset, copyBytes,
                   FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderAcquire);
    }
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Flush to ensure all operations complete
  if (myBlockIdx < nContexts) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,
                   flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  }

  if (FLAGCX_THREAD_IDX_X == 0 && myBlockIdx == 0) result[0] = 1;
}

void launchKernelDevGetInterWorldS(const void *devCommPtr, const void *remoteMemPtr,
                                   const void *localMemPtr, int *devResult, size_t bytes,
                                   flagcxStream_t stream) {
  kernelDevGetInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, remoteMemPtr, localMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S16: flagcxDevBarrierSync (includes ArriveWait) — INTER + WORLD (merged)
// Part 1: INTER barrier
// Part 2: WORLD barrier
// ---------------------------------------------------------------------------
__global__ void kernelDevBarrierInterWorldS(const void *devCommPtr, int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? FLAGCX_BLOCK_IDX_X % nContexts : 0;

  // Part 1: INTER barrier
  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTER, /*index=*/FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);

  // Sync between parts
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Part 2: WORLD barrier
  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_WORLD, /*index=*/FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);

  if (FLAGCX_THREAD_IDX_X == 0) result[FLAGCX_BLOCK_IDX_X] = 1;
}

void launchKernelDevBarrierInterWorldS(const void *devCommPtr, int *devResult,
                                       flagcxStream_t stream) {
  kernelDevBarrierInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S16: flagcxDevBarrierArrive + flagcxDevBarrierWait — INTER + WORLD
// Verifies split arrive/wait semantics for both INTER and WORLD teams.
// ---------------------------------------------------------------------------
__global__ void kernelDevBarrierArriveWaitInterWorldS(const void *devCommPtr,
                                                      int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId =
      nContexts > 0 ? FLAGCX_BLOCK_IDX_X % nContexts : 0;

  // Part 1: INTER barrier — split Arrive + Wait
  flagcxDevBarrierArrive(devCommPtr, FLAGCX_TEAM_INTER, FLAGCX_BLOCK_IDX_X,
                         contextId, FLAGCX_COOP_BLOCK,
                         flagcxDeviceMemoryOrderRelease,
                         flagcxDeviceScopeSystem);
  flagcxDevBarrierWait(devCommPtr, FLAGCX_TEAM_INTER, FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcquire,
                       flagcxDeviceScopeSystem);

  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Part 2: WORLD barrier — split Arrive + Wait
  flagcxDevBarrierArrive(devCommPtr, FLAGCX_TEAM_WORLD, FLAGCX_BLOCK_IDX_X,
                         contextId, FLAGCX_COOP_BLOCK,
                         flagcxDeviceMemoryOrderRelease,
                         flagcxDeviceScopeSystem);
  flagcxDevBarrierWait(devCommPtr, FLAGCX_TEAM_WORLD, FLAGCX_BLOCK_IDX_X,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcquire,
                       flagcxDeviceScopeSystem);

  if (FLAGCX_THREAD_IDX_X == 0) result[FLAGCX_BLOCK_IDX_X] = 1;
}

void launchKernelDevBarrierArriveWaitInterWorldS(const void *devCommPtr,
                                                 int *devResult,
                                                 flagcxStream_t stream) {
  kernelDevBarrierArriveWaitInterWorldS<<<4, 128, 0,
                                         stream->base>>>(devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S20: flagcxDevSignalInc + flagcxDevSignalAdd + flagcxDevWaitSignal +
//      flagcxDevReadSignal + flagcxDevResetSignal — INTER + WORLD (signal-only)
// 6 combos: 3 coop kinds × 2 teams (INTER, WORLD), slot = combo.
// Per combo:
//   Reset(slot) → assert ReadSignal==0
//   Leg A: SignalInc(peer=next) → WaitSignal(slot,1) → assert ReadSignal==1
//   Reset(slot)
//   Leg B: SignalAdd(peer=next, value=5) → WaitSignal(slot,5) → assert ReadSignal==5
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevSignalStandaloneInterWorldS(const void *devCommPtr,
                                                      int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  uint64_t expectedInc = 1;  // leg A: 1 block per context does +1
  uint64_t expectedAdd = 5;  // leg B: 1 block per context does +5

  bool ok = true;

#define S20_INTER_COMBO(slot, teamKind, peer, coopKind)                        \
  do {                                                                         \
    /* Reset this context's local signal and verify it before sending. */      \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot), \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != 0) ok = false;                                                  \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    /* Do not let a remote signal race ahead of another rank's reset. */       \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (flagcxUnifiedIrTestCoopActive(coopKind))                               \
      flagcxDevSignalInc(devCommPtr, teamKind, peer,                           \
                         (flagcxDevSignal_t)(slot), contextId,                 \
                         coopKind, flagcxDeviceScopeSystem);                   \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expectedInc,    \
                        64, contextId, FLAGCX_COOP_BLOCK,                      \
                        flagcxDeviceMemoryOrderAcquire);                       \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot), \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != expectedInc) ok = false;                                        \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    /* Reset and order the second leg independently. */                        \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot), \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != 0) ok = false;                                                  \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (flagcxUnifiedIrTestCoopActive(coopKind))                               \
      flagcxDevSignalAdd(devCommPtr, teamKind, peer,                           \
                         (flagcxDevSignal_t)(slot), expectedAdd, contextId,    \
                         coopKind, flagcxDeviceScopeSystem);                   \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expectedAdd,    \
                        64, contextId, FLAGCX_COOP_BLOCK,                      \
                        flagcxDeviceMemoryOrderAcquire);                       \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot), \
                                       64, contextId,                          \
                                       flagcxDeviceMemoryOrderAcquire);        \
      if (v != expectedAdd) ok = false;                                        \
    }                                                                          \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                        \
  } while (0)

  // combo 0: THREAD + INTER  (slot 0)
  S20_INTER_COMBO(0, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes,
                  FLAGCX_COOP_THREAD);
  // combo 1: THREAD + WORLD  (slot 1)
  S20_INTER_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_THREAD);
  // combo 2: WARP + INTER    (slot 2)
  S20_INTER_COMBO(2, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes,
                  FLAGCX_COOP_WARP);
  // combo 3: WARP + WORLD    (slot 3)
  S20_INTER_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_WARP);
  // combo 4: BLOCK + INTER   (slot 4)
  S20_INTER_COMBO(4, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes,
                  FLAGCX_COOP_BLOCK);
  // combo 5: BLOCK + WORLD   (slot 5)
  S20_INTER_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_BLOCK);

#undef S20_INTER_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevSignalStandaloneInterWorldS(const void *devCommPtr, int *devResult,
                                                 flagcxStream_t stream) {
  kernelDevSignalStandaloneInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, devResult);
}

// ---------------------------------------------------------------------------
// S17: flagcxDevTeamResolution — INTER + WORLD teams
// Six INTER/WORLD combinations plus one INTRA regression on the same
// two-virtual-node communicator.  The INTRA case is what verifies that ranks
// on a nonzero node use the caller's world rank as the team-conversion base.
// Each rank writes sizeof(float) to peer's buffer at deterministic offset
// Buffer layout: [i*maxRanks*sizeof(float), (i+1)*maxRanks*sizeof(float)) for combo i
// Within region: rank writes at rankInTeam * sizeof(float) offset
// ---------------------------------------------------------------------------
__global__ void kernelDevTeamResolutionInterWorldS(const void *devCommPtr,
                                                    const void *dstMemPtr,
                                                    const void *srcMemPtr,
                                                    int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraRank = flagcxDevCommGetIntraRank(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;

  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  // Determine max ranks for buffer sizing
  int maxRanks = nRanks;
  if (nNodes > maxRanks) maxRanks = nNodes;

  // === Combination 0: THREAD + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (nodeIdx + 1) % nNodes;
    size_t dstOff = 0 * maxRanks * sizeof(float) + nodeIdx * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 1: THREAD + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (worldRank + 1) % nRanks;
    size_t dstOff = 1 * maxRanks * sizeof(float) + worldRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 2: WARP + INTER ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (nodeIdx + 1) % nNodes;
    size_t dstOff = 2 * maxRanks * sizeof(float) + nodeIdx * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 3: WARP + WORLD ===
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X < 32) {
    int peer = (worldRank + 1) % nRanks;
    size_t dstOff = 3 * maxRanks * sizeof(float) + worldRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_WARP,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 4: BLOCK + INTER ===
  {
    int peer = (nodeIdx + 1) % nNodes;
    size_t dstOff = 4 * maxRanks * sizeof(float) + nodeIdx * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_INTER, peer, contextId, FLAGCX_COOP_BLOCK,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 5: BLOCK + WORLD ===
  {
    int peer = (worldRank + 1) % nRanks;
    size_t dstOff = 5 * maxRanks * sizeof(float) + worldRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_WORLD, peer, contextId, FLAGCX_COOP_BLOCK,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // === Combination 6: THREAD + INTRA on the multi-node communicator ===
  // On node 1, worldRank != intraRank.  Using team.rank as the world-rank
  // conversion base makes flagcxValidateAndDispatch silently drop this put.
  if (myBlockIdx == 0 && FLAGCX_THREAD_IDX_X == 0) {
    int peer = (intraRank + 1) % intraSize;
    size_t dstOff = 6 * maxRanks * sizeof(float) + intraRank * sizeof(float);
    flagcxDevPut(devCommPtr, dstMemPtr, dstOff, srcMemPtr, 0, sizeof(float),
                 FLAGCX_TEAM_INTRA, peer, contextId, FLAGCX_COOP_THREAD,
                 flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Flush to ensure all FIFO operations complete before barrier
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  // Barrier to ensure all puts land before host reads
  flagcxDevBarrierSync(devCommPtr, FLAGCX_TEAM_INTER, /*index=*/myBlockIdx,
                       contextId, FLAGCX_COOP_BLOCK,
                       flagcxDeviceMemoryOrderAcqRel,
                       flagcxDeviceScopeSystem);

  // Flush after barrier
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);
  if (FLAGCX_THREAD_IDX_X == 0) {
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_THREAD,
                   flagcxDeviceMemoryOrderRelaxed);
  }
  flagcxCoopSyncS(FLAGCX_COOP_BLOCK);

  if (FLAGCX_THREAD_IDX_X == 0 && myBlockIdx == 0) result[0] = 1;
}

void launchKernelDevTeamResolutionInterWorldS(const void *devCommPtr,
                                               const void *dstMemPtr,
                                               const void *srcMemPtr, int *devResult,
                                               flagcxStream_t stream) {
  kernelDevTeamResolutionInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult);
}

// ---------------------------------------------------------------------------
// S21: flagcxDevPut_RSigInc + flagcxDevWaitSignal — INTER + WORLD teams
// Tests 4 cooperation levels × 2 teams = 8 combinations
// Buffer layout: 8× base size
// Signal slots: 0-7, one per combination
// NOTE: Requires concurrent multi-rank launch (ring dependency).
// ---------------------------------------------------------------------------
__global__ void kernelDevPutSignalWaitInterWorldS(const void *devCommPtr,
                                                   const void *dstMemPtr,
                                                   const void *srcMemPtr,
                                                   int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S21_INTER_COMBO(slot, teamKind, peer, expected)                        \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      size_t off = (slot)*bytes;                                                \
      flagcxDevPut(devCommPtr, dstMemPtr, off, srcMemPtr, off, bytes,          \
                   teamKind, peer, contextId, FLAGCX_COOP_THREAD,               \
                   flagcxDeviceScopeSystem, flagcxDeviceMemoryOrderRelease);    \
      if ((slot) % 2 == 0)                                                      \
        flagcxDevSignalInc(devCommPtr, teamKind, peer,                          \
                           (flagcxDevSignal_t)(slot), contextId,                \
                           FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);        \
      else                                                                      \
        flagcxDevSignalAdd(devCommPtr, teamKind, peer,                          \
                           (flagcxDevSignal_t)(slot), (uint64_t)3, contextId,   \
                           FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);        \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expected, 64,    \
                        contextId, FLAGCX_COOP_BLOCK,                           \
                        flagcxDeviceMemoryOrderAcquire);                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != expected) ok = false;                                            \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
  } while (0)

  // combo 0: THREAD + INTER (even → Inc, expected=1)
  S21_INTER_COMBO(0, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes, 1);
  // combo 1: THREAD + WORLD (odd → Add(3), expected=3)
  S21_INTER_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 2: WARP + INTER (even → Inc, expected=1)
  S21_INTER_COMBO(2, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes, 1);
  // combo 3: WARP + WORLD (odd → Add(3), expected=3)
  S21_INTER_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 4: BLOCK + INTER (even → Inc, expected=1)
  S21_INTER_COMBO(4, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes, 1);
  // combo 5: BLOCK + WORLD (odd → Add(3), expected=3)
  S21_INTER_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);

#undef S21_INTER_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutSignalWaitInterWorldS(const void *devCommPtr,
                                              const void *dstMemPtr,
                                              const void *srcMemPtr, int *devResult,
                                              size_t bytes, flagcxStream_t stream) {
  kernelDevPutSignalWaitInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S22: flagcxDevPut_RSigInc + flagcxDevPut_RSigAdd — INTER + WORLD teams
// 6 combos: 3 coop kinds × 2 teams (INTER, WORLD), slot = combo.
// Per combo:
//   ResetSignal(slot) → assert ReadSignal==0
//   even combo: Put_RSigInc (expected=1)
//   odd combo: Put_RSigAdd(value=3) (expected=3)
//   WaitSignal(slot, expected) → assert ReadSignal==expected
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutRSigInterWorldS(const void *devCommPtr,
                                            const void *dstMemPtr,
                                            const void *srcMemPtr,
                                            int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S22_INTER_COMBO(slot, teamKind, peer, expected)                        \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 0) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      size_t off = (slot)*bytes;                                                \
      if ((slot) % 2 == 0)                                                      \
        flagcxDevPut_RSigInc(devCommPtr, dstMemPtr, off, srcMemPtr, off,       \
                             bytes, teamKind, peer, contextId,                  \
                             FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,       \
                             flagcxDeviceMemoryOrderRelease,                    \
                             (flagcxDevSignal_t)(slot));                        \
      else                                                                      \
        flagcxDevPut_RSigAdd(devCommPtr, dstMemPtr, off, srcMemPtr, off,       \
                             bytes, teamKind, peer, contextId,                  \
                             FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,       \
                             flagcxDeviceMemoryOrderRelease,                    \
                             (flagcxDevSignal_t)(slot), (uint64_t)3);           \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expected, 64,    \
                        contextId, FLAGCX_COOP_BLOCK,                           \
                        flagcxDeviceMemoryOrderAcquire);                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != expected) ok = false;                                            \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
  } while (0)

  // combo 0: THREAD + INTER (even → RSigInc, expected=1)
  S22_INTER_COMBO(0, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes, 1);
  // combo 1: THREAD + WORLD (odd → RSigAdd(3), expected=3)
  S22_INTER_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 2: WARP + INTER (even → RSigInc, expected=1)
  S22_INTER_COMBO(2, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes, 1);
  // combo 3: WARP + WORLD (odd → RSigAdd(3), expected=3)
  S22_INTER_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);
  // combo 4: BLOCK + INTER (even → RSigInc, expected=1)
  S22_INTER_COMBO(4, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes, 1);
  // combo 5: BLOCK + WORLD (odd → RSigAdd(3), expected=3)
  S22_INTER_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks, 3);

#undef S22_INTER_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutRSigInterWorldS(const void *devCommPtr,
                                        const void *dstMemPtr,
                                        const void *srcMemPtr, int *devResult,
                                        size_t bytes, flagcxStream_t stream) {
  kernelDevPutRSigInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S23: flagcxDevPut_LCtrInc + flagcxDevPut_RSigInc_LCtrInc +
//      flagcxDevPut_RSigAdd_LCtrInc — INTER + WORLD teams
// 6 combos: 3 coop kinds × 2 teams, slot = combo, counter = combo.
// Per combo:
//   ResetCounter(ctr=combo) → assert ReadCounter==0
//   ResetSignal(sig=combo) → assert ReadSignal==0
//   combo%3==0: Put_LCtrInc (counter-only)
//   combo%3==1: Put_RSigInc_LCtrInc (signal+counter, sig expected=1)
//   combo%3==2: Put_RSigAdd_LCtrInc (signal+counter, sig expected=3)
//   WaitCounter(ctr=combo, 1) → assert ReadCounter==1
//   If combo%3!=0: WaitSignal(sig=combo, expected) → assert ReadSignal==expected
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutCounterInterWorldS(const void *devCommPtr,
                                                const void *dstMemPtr,
                                                const void *srcMemPtr,
                                                int *result, size_t bytes) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S23_INTER_COMBO(slot, teamKind, peer)                                  \
  do {                                                                          \
    flagcxDevCounter_t ctr = (flagcxDevCounter_t)(slot);                       \
    flagcxDevSignal_t sig = (flagcxDevSignal_t)(slot);                         \
    int variant = (slot) % 3;                                                   \
    /* Reset counter and signal */                                              \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      flagcxDevResetCounter(devCommPtr, contextId, ctr);                        \
      flagcxDevResetSignal(devCommPtr, contextId, sig);                         \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Assert both are zero */                                                  \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t cv = flagcxDevReadCounter(devCommPtr, ctr, 64, contextId,       \
                                          flagcxDeviceMemoryOrderAcquire);      \
      uint64_t sv = flagcxDevReadSignal(devCommPtr, sig, 64, contextId,        \
                                         flagcxDeviceMemoryOrderAcquire);       \
      if (cv != 0 || sv != 0) ok = false;                                      \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    /* Put operation with counter (and optionally signal) */                    \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      size_t off = (slot)*bytes;                                                \
      if (variant == 0)                                                         \
        flagcxDevPut_LCtrInc(devCommPtr, dstMemPtr, off, srcMemPtr, off,       \
                             bytes, teamKind, peer, contextId,                  \
                             FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem,       \
                             flagcxDeviceMemoryOrderRelease, ctr);              \
      else if (variant == 1)                                                    \
        flagcxDevPut_RSigInc_LCtrInc(devCommPtr, dstMemPtr, off, srcMemPtr,    \
                                     off, bytes, teamKind, peer, contextId,     \
                                     FLAGCX_COOP_THREAD,                        \
                                     flagcxDeviceScopeSystem,                   \
                                     flagcxDeviceMemoryOrderRelease, sig, ctr); \
      else                                                                      \
        flagcxDevPut_RSigAdd_LCtrInc(devCommPtr, dstMemPtr, off, srcMemPtr,    \
                                     off, bytes, teamKind, peer, contextId,     \
                                     FLAGCX_COOP_THREAD,                        \
                                     flagcxDeviceScopeSystem,                   \
                                     flagcxDeviceMemoryOrderRelease, sig,       \
                                     (uint64_t)3, ctr);                         \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Wait and verify counter */                                               \
    flagcxDevWaitCounter(devCommPtr, ctr, 1, 64, contextId,                     \
                         FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderAcquire);    \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t cv = flagcxDevReadCounter(devCommPtr, ctr, 64, contextId,       \
                                          flagcxDeviceMemoryOrderAcquire);      \
      if (cv != 1) ok = false;                                                  \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Wait and verify signal (if variant != 0) */                              \
    if (variant != 0) {                                                         \
      uint64_t expectedSig = (variant == 1) ? 1 : 3;                            \
      flagcxDevWaitSignal(devCommPtr, sig, expectedSig, 64, contextId,          \
                          FLAGCX_COOP_BLOCK, flagcxDeviceMemoryOrderAcquire);   \
      if (FLAGCX_THREAD_IDX_X == 0) {                                          \
        uint64_t sv = flagcxDevReadSignal(devCommPtr, sig, 64, contextId,      \
                                           flagcxDeviceMemoryOrderAcquire);     \
        if (sv != expectedSig) ok = false;                                      \
      }                                                                         \
      flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                       \
    }                                                                           \
  } while (0)

  // combo 0: THREAD + INTER (variant=0: LCtrInc only)
  S23_INTER_COMBO(0, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes);
  // combo 1: THREAD + WORLD (variant=1: RSigInc_LCtrInc)
  S23_INTER_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);
  // combo 2: WARP + INTER (variant=2: RSigAdd_LCtrInc)
  S23_INTER_COMBO(2, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes);
  // combo 3: WARP + WORLD (variant=0: LCtrInc only)
  S23_INTER_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);
  // combo 4: BLOCK + INTER (variant=1: RSigInc_LCtrInc)
  S23_INTER_COMBO(4, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes);
  // combo 5: BLOCK + WORLD (variant=2: RSigAdd_LCtrInc)
  S23_INTER_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);

#undef S23_INTER_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutCounterInterWorldS(const void *devCommPtr,
                                           const void *dstMemPtr,
                                           const void *srcMemPtr, int *devResult,
                                           size_t bytes, flagcxStream_t stream) {
  kernelDevPutCounterInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, srcMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S24: flagcxDevPutValue_RSigInc + flagcxDevPutValue_RSigAdd — INTER + WORLD
// 6 combos: 3 coop kinds × 2 teams (INTER, WORLD), slot = combo.
// Per combo:
//   ResetSignal(slot) → assert ReadSignal==0
//   even combo: PutValue_RSigInc (expected=1)
//   odd combo: PutValue_RSigAdd(value=3) (expected=3)
//   WaitSignal(slot, expected) → assert ReadSignal==expected
// result[0] = 1 iff all assertions pass.
// Buffer layout: each combo writes 1 uint64_t at slot offset.
// ---------------------------------------------------------------------------
__global__ void kernelDevPutValueRSigInterWorldS(const void *devCommPtr,
                                                  const void *dstMemPtr,
                                                  int *result, size_t /*bytes*/) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S24_INTER_COMBO(slot, teamKind, peer, coopKind, expected)              \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 0) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    if (flagcxUnifiedIrTestCoopActive(coopKind)) {                             \
      size_t off = (slot) * sizeof(uint64_t);                                  \
      uint64_t val = (uint64_t)(worldRank * 100 + (slot));                     \
      if ((slot) % 2 == 0)                                                      \
        flagcxDevPutValue_RSigInc(devCommPtr, dstMemPtr, off, val, teamKind,   \
                                  peer, contextId, coopKind,                    \
                                  flagcxDeviceScopeSystem,                      \
                                  flagcxDeviceMemoryOrderRelease,               \
                                  (flagcxDevSignal_t)(slot));                   \
      else                                                                      \
        flagcxDevPutValue_RSigAdd(devCommPtr, dstMemPtr, off, val, teamKind,   \
                                  peer, contextId, coopKind,                    \
                                  flagcxDeviceScopeSystem,                      \
                                  flagcxDeviceMemoryOrderRelease,               \
                                  (flagcxDevSignal_t)(slot), (uint64_t)3);      \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevWaitSignal(devCommPtr, (flagcxDevSignal_t)(slot), expected, 64,    \
                        contextId, FLAGCX_COOP_BLOCK,                           \
                        flagcxDeviceMemoryOrderAcquire);                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != expected) ok = false;                                            \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
  } while (0)

  // combo 0: THREAD + INTER (even → RSigInc, expected=1)
  S24_INTER_COMBO(0, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes,
                  FLAGCX_COOP_THREAD, 1);
  // combo 1: THREAD + WORLD (odd → RSigAdd(3), expected=3)
  S24_INTER_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_THREAD, 3);
  // combo 2: WARP + INTER (even → RSigInc, expected=1)
  S24_INTER_COMBO(2, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes,
                  FLAGCX_COOP_WARP, 1);
  // combo 3: WARP + WORLD (odd → RSigAdd(3), expected=3)
  S24_INTER_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_WARP, 3);
  // combo 4: BLOCK + INTER (even → RSigInc, expected=1)
  S24_INTER_COMBO(4, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes,
                  FLAGCX_COOP_BLOCK, 1);
  // combo 5: BLOCK + WORLD (odd → RSigAdd(3), expected=3)
  S24_INTER_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks,
                  FLAGCX_COOP_BLOCK, 3);

#undef S24_INTER_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevPutValueRSigInterWorldS(const void *devCommPtr,
                                             const void *dstMemPtr,
                                             int *devResult, size_t bytes,
                                             flagcxStream_t stream) {
  kernelDevPutValueRSigInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, dstMemPtr, devResult, bytes);
}

// ---------------------------------------------------------------------------
// S25: flagcxDevIncreaseSignalShadow + flagcxDevWaitSignalMeetShadow +
//      flagcxDevFlush — INTER + WORLD teams
// 6 combos: 3 coop kinds × 2 teams (INTER, WORLD), slot = combo.
// Per combo:
//   ResetSignal(slot) → assert ReadSignal==0
//   IncreaseSignalShadow(slot, increment=5)
//   SignalInc(peer=next, slot) → repeat 5 times
//   WaitSignalMeetShadow(slot) → assert ReadSignal==5
//   Flush(contextId)
// result[0] = 1 iff all assertions pass.
// ---------------------------------------------------------------------------
__global__ void kernelDevSignalShadowFlushInterWorldS(const void *devCommPtr,
                                                       int *result) {
  const flagcxDevComm *comm = (const flagcxDevComm *)devCommPtr;
  int worldRank = flagcxDevCommGetRank(devCommPtr);
  int nRanks = flagcxDevCommGetSize(devCommPtr);
  int intraSize = flagcxDevCommGetIntraSize(devCommPtr);
  int nNodes = nRanks / intraSize;
  int nodeIdx = worldRank / intraSize;

  int myBlockIdx = FLAGCX_BLOCK_IDX_X;
  int nContexts = comm->getContextCount();
  flagcxDevContext_t contextId = nContexts > 0 ? myBlockIdx % nContexts : 0;

  bool ok = true;

#define S25_INTER_COMBO(slot, teamKind, peer)                                  \
  do {                                                                          \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevResetSignal(devCommPtr, contextId, (flagcxDevSignal_t)(slot));  \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 0) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    flagcxDevBarrierSync(devCommPtr, teamKind, myBlockIdx,                     \
                         contextId, FLAGCX_COOP_BLOCK,                         \
                         flagcxDeviceMemoryOrderAcqRel,                        \
                         flagcxDeviceScopeSystem);                             \
    /* Increase shadow by 5 */                                                  \
    if (FLAGCX_THREAD_IDX_X == 0)                                              \
      flagcxDevIncreaseSignalShadow(devCommPtr, contextId,                     \
                                    (flagcxDevSignal_t)(slot),                 \
                                    (uint64_t)5);                               \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Signal 5 times to meet shadow */                                         \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      for (int i = 0; i < 5; i++)                                               \
        flagcxDevSignalInc(devCommPtr, teamKind, peer,                          \
                           (flagcxDevSignal_t)(slot), contextId,                \
                           FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);        \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Wait for shadow to be met */                                             \
    flagcxDevWaitSignalMeetShadow(devCommPtr, contextId,                        \
                                  (flagcxDevSignal_t)(slot), 64,                \
                                  FLAGCX_COOP_BLOCK,                            \
                                  flagcxDeviceMemoryOrderAcquire);              \
    if (FLAGCX_THREAD_IDX_X == 0) {                                            \
      uint64_t v = flagcxDevReadSignal(devCommPtr, (flagcxDevSignal_t)(slot),  \
                                       64, contextId,                           \
                                       flagcxDeviceMemoryOrderAcquire);         \
      if (v != 5) ok = false;                                                   \
    }                                                                           \
    flagcxCoopSyncS(FLAGCX_COOP_BLOCK);                                         \
    /* Flush */                                                                 \
    flagcxDevFlush(devCommPtr, contextId, FLAGCX_COOP_BLOCK,                    \
                   flagcxDeviceMemoryOrderAcquire);                             \
  } while (0)

  // combo 0: THREAD + INTER
  S25_INTER_COMBO(0, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes);
  // combo 1: THREAD + WORLD
  S25_INTER_COMBO(1, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);
  // combo 2: WARP + INTER
  S25_INTER_COMBO(2, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes);
  // combo 3: WARP + WORLD
  S25_INTER_COMBO(3, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);
  // combo 4: BLOCK + INTER
  S25_INTER_COMBO(4, FLAGCX_TEAM_INTER, (nodeIdx + 1) % nNodes);
  // combo 5: BLOCK + WORLD
  S25_INTER_COMBO(5, FLAGCX_TEAM_WORLD, (worldRank + 1) % nRanks);

#undef S25_INTER_COMBO

  if (FLAGCX_THREAD_IDX_X == 0) atomicAnd(result, ok ? 1 : 0);
}

void launchKernelDevSignalShadowFlushInterWorldS(const void *devCommPtr,
                                                  int *devResult,
                                                  flagcxStream_t stream) {
  kernelDevSignalShadowFlushInterWorldS<<<4, 128, 0, stream->base>>>(
      devCommPtr, devResult);
}
