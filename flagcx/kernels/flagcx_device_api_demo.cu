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
 * 2. Inter-node AlltoAll — unified kernel with runtime dispatch.
 *    One-sided path (Tier 1 + window): put + signals via flagcxDevNet.
 *    Two-sided path (all tiers):       send/recv + FIFO via flagcxDevNet,
 *      with atomic last-block grid sync using flagcxDeviceAtomicFetchAdd.
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

  // Advance barrier epoch for next launch (2 syncs per kernel invocation)
  devComm->barrierEpoch += 2;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// 2. Inter-node AlltoAll — Unified kernel with runtime dispatch
//
// One-sided path (Tier 1 + window): put + signals via flagcxDevNet.
// Two-sided path (all tiers): send/recv via FIFO + atomic last-block grid sync.
//
// Buffer layout: [rank0_data][rank1_data]...[rankN_data], each of size `count`
// sendMem: data at offset peerRank * count * elementSize is sent to peerRank
// recvMem: data from peerRank is stored at offset peerRank * count * elementSize
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterAlltoAllKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                              size_t count, flagcxDataType_t datatype,
                              flagcxDevComm devComm,
                              unsigned int *gridDoneCounter) {

  if (devComm._hasBase && sendMem._isSymmetric) {
    // ======== One-sided path (Tier 1 with windows) ========
    flagcxDevNet net(devComm, 0);
    uint64_t signalValue = net.readSignal(0);

    flagcxBarrierSession<flagcxCoopBlock> bar(flagcxCoopBlock(),
                                              flagcxTeamTagWorld{}, net,
                                              FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    int myRank = devComm.getRank();
    int nRanks = devComm.getSize();
    size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

    for (int r = tid; r < nRanks; r += nthreads) {
      net.put(flagcxTeamWorld(devComm), r, recvMem, myRank * size, sendMem,
              r * size, size, flagcxDevNet_SignalInc{0});
    }

    net.waitSignal(flagcxCoopBlock(), 0, signalValue + nRanks);
    net.flush(flagcxCoopBlock());

  } else {
    // ======== Two-sided path (all tiers, send/recv + FIFO) ========
    flagcxDevNet net(devComm);
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    size_t elementSize = getFlagcxDataTypeSizeDevice(datatype);
    int nRanks = devComm.getSize();

    // Each thread handles one or more peers (grid-stride loop)
    for (int peerRank = tid; peerRank < nRanks; peerRank += nthreads) {
      size_t offset = peerRank * count * elementSize;
      net.send(sendMem, offset, count, datatype, peerRank);
      net.recv(recvMem, offset, count, datatype, peerRank);
    }

    // Grid sync: atomic last-block pattern
    // All threads in this block sync first, then thread 0 atomically
    // increments the counter. The last block to arrive calls term() + wait().
    FLAGCX_DEVICE_SYNC_THREADS();

    if (FLAGCX_THREAD_IDX_X == 0) {
      // AcqRel ordering provides threadfence semantics — ensures all
      // FIFO writes from this block are visible before last block acts
      unsigned int arrived =
          flagcxDeviceAtomicFetchAdd<unsigned int, flagcxDeviceScopeDevice>(
              gridDoneCounter, 1u, flagcxDeviceMemoryOrderAcqRel);

      if (arrived == FLAGCX_GRID_DIM_X - 1) {
        // Last block: send termination and wait for all FIFO entries
        net.term();
        net.wait();

        // Reset counter for next kernel launch
        flagcxDeviceAtomicStore<unsigned int, flagcxDeviceScopeDevice>(
            gridDoneCounter, 0u, flagcxDeviceMemoryOrderRelaxed);
      }
    }
  }
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
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc,
                                    devComm->gridDoneCounter);

  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}
