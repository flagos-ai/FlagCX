/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Device API demo kernels.
 *
 * 1. Intra-node AllReduce — peer pointer + barrier based.
 *    Tier 1 (NCCL > 2.28): wraps ncclDevComm + ncclWindow_t + ncclLsaBarrier.
 *    Tier 2 (fallback):    IPC peer pointers + atomics barrier.
 *    Same kernel code compiles for both tiers.
 *
 * 2. Inter-node AlltoAll — unified kernel, same structure for both paths.
 *    Thread 0 block-stride loop dispatches communication ops.
 *    One-sided path (Tier 1 + window): put + waitSignal + flush.
 *    Two-sided path (all tiers):       send + recv via FIFO.
 *    Both paths wrapped by bar.sync() pre/post barriers.
 *
 * Host-side flagcxDevCommCreate/Destroy are in flagcx_device.cc.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "nvidia_adaptor.h"
#include "global_comm.h"
#include "flagcx_kernel.h"
#include <cuda_runtime.h>

// ==========================================================================
// 1. Intra-node AllReduce
// ==========================================================================

// Intra-node AllReduce: each block reads from all peers via team-based
// flagcxGetPeerPointer, reduces (sum), and writes result back to all peers.
template <typename T>
__global__ void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraAllReduceKernel(flagcxDevComm devComm, flagcxDevMem mem,
                               size_t offset, size_t count) {
  // AllReduce requires peer pointer access (window or IPC)
  if (!mem._hasWindow && mem.peerPtrs == nullptr) {
    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
      printf("flagcxIntraAllReduceKernel: no peer access (no window, no IPC), "
             "skipping\n");
    }
    return;
  }

  flagcxTeam_t intra = flagcxTeamIntra(devComm);

  // Create barrier session using simplified FlagCX API (4 params).
  flagcxIntraBarrierSession<flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X};

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderAcquire);

  const int rank = devComm.getIntraRank();
  const int nRanks = devComm.getIntraSize();
  const int globalTid =
      FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_DIM_X * (rank + FLAGCX_BLOCK_IDX_X * nRanks);
  const int globalNthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X * nRanks;

  // Phase 1: Reduce — sum data from all intra-node peers
  // Phase 2: Write — store result to all intra-node peers
  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = T(0);
    for (int peer = 0; peer < nRanks; peer++) {
      T* inputPtr = (T*)flagcxGetPeerPointer(mem, offset, intra, peer);
      v += inputPtr[o];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* outputPtr = (T*)flagcxGetPeerPointer(mem, offset, intra, peer);
      outputPtr[o] = v;
    }
  }

  // Post-reduce barrier (release ordering — ensure writes are visible)
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelease);
}

// Host-side launcher
template <typename T>
static cudaError_t launchFlagcxIntraAllReduce(flagcxDevComm devComm,
                                              flagcxDevMem mem,
                                              size_t offset, size_t count,
                                              cudaStream_t stream) {
  flagcxIntraAllReduceKernel<T>
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         stream>>>(devComm, mem, offset, count);
  return cudaGetLastError();
}

// Explicit instantiations for common types
template cudaError_t launchFlagcxIntraAllReduce<float>(flagcxDevComm,
                                                       flagcxDevMem, size_t,
                                                       size_t, cudaStream_t);
template cudaError_t launchFlagcxIntraAllReduce<double>(flagcxDevComm,
                                                        flagcxDevMem, size_t,
                                                        size_t, cudaStream_t);

// Host-side demo function — launches the kernel using caller-provided
// registered buffer and device communicator.
flagcxResult_t flagcxIntraAllReduceDemo(flagcxDevMem_t devMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (devComm == nullptr || devMem == nullptr) {
    return flagcxInternalError;
  }

  cudaStream_t cudaStream = *(cudaStream_t *)stream;

  // Unified constructors — work for both Tier 1 and Tier 2
  flagcxDevComm devCommKernel(*devComm);
  flagcxDevMem devMemKernel(*devMem);

  cudaError_t err;
  switch (datatype) {
  case flagcxFloat32:
    err = launchFlagcxIntraAllReduce<float>(devCommKernel, devMemKernel, 0,
                                            count, cudaStream);
    break;
  case flagcxFloat64:
    err = launchFlagcxIntraAllReduce<double>(devCommKernel, devMemKernel, 0,
                                             count, cudaStream);
    break;
  default:
    return flagcxInvalidArgument;
  }

  // Advance barrier epoch for next launch (2 syncs, each += nLocalRanks-1)
  devComm->barrierEpoch += 2 * (devComm->intraSize - 1);

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// 2. Inter-node AlltoAll — Unified kernel
//
// Both one-sided (put) and two-sided (send/recv) paths share the same
// structure: thread-0 block-stride loop over peers, wrapped by bar.sync().
//
// Buffer layout: [rank0_data][rank1_data]...[rankN_data], each of size `count`
// sendMem: data at offset peerRank * count * elementSize is sent to peerRank
// recvMem: data from peerRank is stored at offset peerRank * count * elementSize
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterAlltoAllKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                              size_t count, flagcxDataType_t datatype,
                              flagcxDevComm devComm) {

  flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
  flagcxBarrierSession<flagcxCoopBlock> bar(
      flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);

  // Pre-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

  int nRanks = devComm.getSize();
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);
  bool oneSided = devComm._hasBase && sendMem._isSymmetric;

  // Thread 0 dispatches all communication ops (block-stride over peers).
  // Both put() and send()/recv() are control-plane descriptors — NIC or
  // proxy does the actual data movement.
  if (FLAGCX_THREAD_IDX_X == 0) {
    int myRank = devComm.getRank();
    uint64_t signalValue = 0;
    if (oneSided) signalValue = net.readSignal(0);

    for (int peer = FLAGCX_BLOCK_IDX_X; peer < nRanks;
         peer += FLAGCX_GRID_DIM_X) {
      if (oneSided) {
        net.put(flagcxTeamWorld(devComm), peer, recvMem, myRank * size,
                sendMem, peer * size, size, flagcxDevNet_SignalInc{0},
                flagcxDevNet_None{}, flagcxCoopThread{});
      } else {
        size_t offset = peer * size;
        net.send(sendMem, offset, count, datatype, peer);
        net.recv(recvMem, offset, count, datatype, peer);
      }
    }

    if (oneSided) {
      net.waitSignal(flagcxCoopThread{}, 0, signalValue + nRanks);
      net.flush(flagcxCoopThread{});
    } else {
      // Two-sided: all CTAs must enqueue term/wait, even idle ones
      // (blockIdx >= nRanks), because the proxy counts term entries from all
      // CTA_COUNT channels to trigger groupEnd. Idle CTAs' term/wait are
      // benign — no data was enqueued so wait completes instantly.
      net.term();
      net.wait();
    }
  }

  // Post-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
}

// Host-side unified alltoall demo function.
// Runtime dispatch: one-sided (put) if window available, two-sided (send/recv)
// otherwise. Uses same launch config for both paths.
flagcxResult_t flagcxInterAlltoAllDemo(flagcxDevMem_t sendMem,
                                       flagcxDevMem_t recvMem, size_t count,
                                       flagcxDataType_t datatype,
                                       flagcxDevComm_t devComm,
                                       flagcxStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return flagcxInternalError;
  }

  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);

  flagcxInterAlltoAllKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);

  cudaError_t err = cudaGetLastError();

  // Advance barrier epoch:
  // - Single-node: 2 syncs × 1 intra arrive × (intraSize-1) = 2*(intraSize-1)
  // - Multi-node:  2 syncs × 2 intra arrives × (intraSize-1) = 4*(intraSize-1)
  int intraArrivesPerSync = (devComm->nInterPeers > 0) ? 2 : 1;
  devComm->barrierEpoch +=
      2 * intraArrivesPerSync * (devComm->intraSize - 1);

  // Advance inter-node barrier epoch (2 syncs per kernel, each += nInterPeers)
  // Only meaningful on leader, but safe to advance on all ranks.
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}
