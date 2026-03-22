/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Device API kernels.
 *
 * 1. Intra-node AllReduce — peer pointer + barrier based.
 *    Vendor (NCCL > 2.28): wraps ncclDevComm + ncclWindow_t + ncclLsaBarrier.
 *    Fallback:    IPC peer pointers + atomics barrier.
 *    Same kernel code compiles for both paths.
 *
 * 2. Inter-node AlltoAll — two separate kernels:
 *    a) One-sided (put): thread-stride loop, put + waitSignal + flush.
 *    b) Two-sided (send/recv): thread-0 block-stride loop, FIFO + term/wait.
 *    Both paths wrapped by bar.sync() pre/post barriers.
 *
 * Host-side flagcxDevCommCreate/Destroy are in flagcx_device.cc.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "nvidia_adaptor.h"
#include "global_comm.h"
#include "flagcx_kernel.h"
#include <cuda_runtime.h>

// Helper: advance intra-barrier epoch by nBarSyncs, accounting for topology.
// Each bar.sync does syncsPerBarrier intra syncs (1 single-node, 2 multi-node).
static inline void advanceIntraEpoch(flagcxDevComm_t devComm, int nBarSyncs) {
  int syncsPerBarrier = (devComm->nInterPeers > 0) ? 2 : 1;
  devComm->intraBarrierEpoch += nBarSyncs * syncsPerBarrier;
}

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

// Host-side function — launches the kernel using caller-provided
// registered buffer and device communicator.
flagcxResult_t flagcxIntraAllReduce(flagcxDevMem_t devMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (devComm == nullptr || devMem == nullptr) {
    return flagcxInternalError;
  }

  cudaStream_t cudaStream = *(cudaStream_t *)stream;

  // Unified constructors — work for both Vendor and Fallback
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

  // Advance barrier epoch for next launch (2 syncs, each epoch += 1)
  devComm->intraBarrierEpoch += 2;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// Inter-node One-sided AlltoAll
//
// Thread-stride loop: each thread dispatches put ops to different peers.
// put() posts FIFO descriptor (Fallback) or GIN descriptor (Vendor).
// After all puts, waitSignal + flush ensure completion.
//
// Buffer layout: [rank0_data][rank1_data]...[rankN_data], each of size `count`
// sendMem: data at offset peerRank * count * elementSize is sent to peerRank
// recvMem: data from peerRank is stored at offset peerRank * count * elementSize
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterOneSidedAlltoAllKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxDevComm devComm) {

  // contextIndex=0: all CTAs share signal slot 0. readSignal is taken before
  // bar.sync so the baseline is captured before any signals from this round arrive.
  flagcxDevNet net(devComm, 0);
  // Unified barrier: intra (IPC) + inter (FIFO signal relay).
  // Single-node: intra sync only.  Multi-node: three-phase intra/inter/intra.
  flagcxBarrierSession<flagcxCoopBlock> bar(
      flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  // Read signal baseline before pre-barrier so it reflects the pre-round state.
  uint64_t signalValue = net.readSignal(0);

  // Pre-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    net.put(flagcxTeamWorld(devComm), peer, recvMem, myRank * size,
            sendMem, peer * size, size, flagcxDevNet_SignalInc{0},
            flagcxDevNet_None{}, flagcxCoopThread{});
  }

  net.waitSignal(flagcxCoopBlock{}, 0, signalValue + nRanks);
  net.flush(flagcxCoopBlock{});

  // Post-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
}

// ==========================================================================
// Inter-node Two-sided AlltoAll
//
// Thread-0 block-stride loop dispatches send/recv via FIFO.
// term() + wait() for group semantic completion.
//
// Buffer layout: same as one-sided.
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTwoSidedAlltoAllKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxDevComm devComm) {

  flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
  // Unified barrier: intra (IPC) + inter (FIFO signal relay).
  // Single-node: intra sync.  Multi-node: three-phase intra/inter/intra.
  flagcxBarrierSession<flagcxCoopBlock> bar(
      flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  // Pre-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

  // Thread 0 dispatches all communication ops (block-stride over peers).
  if (FLAGCX_THREAD_IDX_X == 0) {
    for (int peer = FLAGCX_BLOCK_IDX_X; peer < nRanks;
         peer += FLAGCX_GRID_DIM_X) {
      size_t offset = peer * size;
      net.send(sendMem, offset, count, datatype, peer);
      net.recv(recvMem, offset, count, datatype, peer);
    }

    // Two-sided: all CTAs must enqueue term/wait, even idle ones
    // (blockIdx >= nRanks), because the proxy counts term entries from all
    // CTA_COUNT channels to trigger groupEnd. Idle CTAs' term/wait are
    // benign — no data was enqueued so wait completes instantly.
    net.term();
    net.wait();
  }

  // Post-communication barrier
  bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
}

// Host-side one-sided AlltoAll function.
flagcxResult_t flagcxInterOneSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return flagcxInternalError;
  }

  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);

  flagcxInterOneSidedAlltoAllKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);

  cudaError_t err = cudaGetLastError();

  // Advance barrier epochs: 2 bar.syncs per kernel.
  // flagcxBarrierSession::sync does (nInterPeers>0): 2 intra + 1 inter,
  //                              or (nInterPeers==0): 1 intra.
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// Host-side two-sided AlltoAll function.
flagcxResult_t flagcxInterTwoSidedAlltoAll(flagcxDevMem_t sendMem,
                                            flagcxDevMem_t recvMem, size_t count,
                                            flagcxDataType_t datatype,
                                            flagcxDevComm_t devComm,
                                            flagcxStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return flagcxInternalError;
  }

  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);

  flagcxInterTwoSidedAlltoAllKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);

  cudaError_t err = cudaGetLastError();

  // Advance barrier epochs (2 bar.syncs per kernel)
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}


// ==========================================================================
// Inter-node One-sided Device API Tests (K2-K9)
//
// Eight focused kernels, each covering one Device API facet.
// Signal/counter slot assignments:
//   slot 0: SignalAdd, CounterPipeline, FlushDecouple
//   slot 1: PutValue, SignalOnly
//   slot 2: FollowShadow, MeetShadow
//   counter 0: CounterInc
// DevCommRequirements: interSignalCount=3, interCounterCount=1
// All kernels follow: pre-bar.sync / core logic / post-bar.sync.
// ==========================================================================

// Shared IPC alltoall helper: each thread copies its portion of all intra peers.
// NOTE: assumes chunkSize is a multiple of 4 bytes (i.e., element type is
// >= 4-byte aligned -- float, int32, etc.). Sub-4-byte types (half, int8)
// with odd counts will lose tail bytes.
FLAGCX_DEVICE_INLINE_DECORATOR void
ipcAlltoAll(const flagcxDevMem &sendMem, const flagcxDevMem &recvMem,
            flagcxTeam_t intra, int intraSize, int intraBase,
            int myWorldRank, size_t chunkSize) {
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  size_t nWords = chunkSize / sizeof(uint32_t);
  for (int lr = 0; lr < intraSize; lr++) {
    int worldPeer = intraBase + lr;
    uint32_t *src = (uint32_t *)flagcxGetLocalPointer(
        sendMem, (size_t)worldPeer * chunkSize);
    uint32_t *dst = (uint32_t *)flagcxGetPeerPointer(
        recvMem, (size_t)myWorldRank * chunkSize, intra, lr);
    for (size_t w = (size_t)tid; w < nWords; w += (size_t)nthreads)
      dst[w] = src[w];
  }
}

// put + SignalInc
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestSignalIncKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam_t intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    // Hybrid: DevNet for inter + IPC for intra
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(flagcxTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              flagcxDevNet_SignalInc{0}, flagcxDevNet_None{},
              flagcxCoopThread{});
    }
    net.waitSignal(flagcxCoopBlock{}, 0, s0 + (uint64_t)nInterRanks);
    net.flush(flagcxCoopBlock{});
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    // Single-node: IPC only, no DevNet
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// put data + separate SignalAdd: decouples data transfer from signalling.
// GIN path:      two NIC ops (WRITE then ATOMIC+2 on slot 0).
// Fallback path: PrimPut + PrimSignal(value=2) (two FIFO entries, slot 0).
// Contrast with K1 where both paths fuse into a single chained WR.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestSignalAddKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam_t intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(flagcxTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              flagcxDevNet_None{}, flagcxDevNet_None{},
              flagcxCoopThread{});
      net.signal(flagcxTeamWorld(devComm), peer,
                 flagcxDevNet_SignalAdd{0, 2}, flagcxCoopThread{});
    }
    net.waitSignal(flagcxCoopBlock{}, 0, s0 + (uint64_t)nInterRanks * 2);
    net.flush(flagcxCoopBlock{});
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// put + CounterInc two-round pipeline
// Round 1: put with CounterInc; waitCounter; stamp sentinel; Round 2: put again.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestCounterPipelineKernel(flagcxDevMem sendMem,
                                         flagcxDevMem recvMem, size_t count,
                                         flagcxDataType_t datatype,
                                         flagcxDevComm devComm,
                                         uint64_t *resultBuf) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam_t intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  float *sendRaw = (float *)sendMem.rawPtr;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    uint64_t c0 = net.readCounter(0);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

    // Round 1: IPC for intra + DevNet for inter
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(flagcxTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              flagcxDevNet_SignalInc{0}, flagcxDevNet_CounterInc{0},
              flagcxCoopThread{});
    }
    net.waitCounter(flagcxCoopBlock{}, 0, c0 + (uint64_t)nInterRanks);

    // Stamp sentinel
    for (int peer = tid; peer < nRanks; peer += nthreads)
      sendRaw[(ptrdiff_t)peer * (ptrdiff_t)count] = 999.0f;
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

    // Round 2: same
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(flagcxTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              flagcxDevNet_SignalInc{0}, flagcxDevNet_CounterInc{0},
              flagcxCoopThread{});
    }
    net.waitCounter(flagcxCoopBlock{}, 0, c0 + 2 * (uint64_t)nInterRanks);
    net.waitSignal(flagcxCoopBlock{}, 0, s0 + 2 * (uint64_t)nInterRanks);
    net.flush(flagcxCoopBlock{});

    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      resultBuf[0] = net.readCounter(0);
      resultBuf[1] = (uint64_t)nInterRanks;
    }
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

    // Round 1
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    // Stamp sentinel
    for (int peer = tid; peer < nRanks; peer += nthreads)
      sendRaw[(ptrdiff_t)peer * (ptrdiff_t)count] = 999.0f;
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

    // Round 2
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      resultBuf[0] = 0; // no counter in IPC mode
      resultBuf[1] = 0; // nInterRanks = 0
    }
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// putValue
// Each rank writes value=myRank*1000+peer to peer's recvBuff[putValBase+myRank*8].
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestPutValueKernel(flagcxDevMem recvMem, flagcxDevComm devComm,
                                  size_t putValBase) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam_t intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s1 = net.readSignal(1);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      if (peer >= intraBase && peer < intraBase + intraSize) {
        // IPC: direct write to peer's recvBuff
        int lr = peer - intraBase;
        uint64_t *dst = (uint64_t *)flagcxGetPeerPointer(
            recvMem, putValBase + (size_t)myRank * sizeof(uint64_t), intra, lr);
        *dst = val;
      } else {
        net.putValue(flagcxTeamWorld(devComm), peer,
                     recvMem, putValBase + (size_t)myRank * sizeof(uint64_t),
                     val, flagcxDevNet_SignalInc{1}, flagcxCoopThread{});
      }
    }
    if (nInterRanks > 0)
      net.waitSignal(flagcxCoopBlock{}, 1, s1 + (uint64_t)nInterRanks);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      int lr = peer - intraBase;
      uint64_t *dst = (uint64_t *)flagcxGetPeerPointer(
          recvMem, putValBase + (size_t)myRank * sizeof(uint64_t), intra, lr);
      *dst = val;
    }
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// signal standalone
// Each rank signals all other peers on slot 1; waits for nRanks-1 incoming.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestSignalOnlyKernel(flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  int nInterRanks = nRanks - intraSize;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s1 = net.readSignal(1);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads)
      if (peer != myRank && (peer < intraBase || peer >= intraBase + intraSize))
        net.signal(flagcxTeamWorld(devComm), peer,
                   flagcxDevNet_SignalInc{1}, flagcxCoopThread{});
    if (nInterRanks > 0)
      net.waitSignal(flagcxCoopBlock{}, 1, s1 + (uint64_t)nInterRanks);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// put + flush decoupled
// put(None,None) → flush (src drain) → signal → waitSignal → flush (dst).
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestFlushDecoupleKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam_t intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);

    // IPC for intra peers
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    // DevNet puts (None,None) for inter peers only
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(flagcxTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              flagcxDevNet_None{}, flagcxDevNet_None{},
              flagcxCoopThread{});
    }
    net.flush(flagcxCoopBlock{});

    // Signal inter peers only
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.signal(flagcxTeamWorld(devComm), peer,
                 flagcxDevNet_SignalInc{0}, flagcxCoopThread{});
    }
    net.waitSignal(flagcxCoopBlock{}, 0, s0 + (uint64_t)nInterRanks);
    net.flush(flagcxCoopBlock{});
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// waitSignalFollowShadow
// All ranks signal all peers on slot 2; FollowShadow advances shadow by nRanks.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestFollowShadowKernel(flagcxDevComm devComm) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    int nRanks = devComm.getSize();
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads)
      net.signal(flagcxTeamWorld(devComm), peer,
                 flagcxDevNet_SignalInc{2}, flagcxCoopThread{});
    uint64_t before, delta;
    net.waitSignalFollowShadow(flagcxCoopBlock{}, (flagcxDevNetSignal_t)2,
                                (uint64_t)nRanks, &before, &delta);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// increaseSignalShadow + waitSignalMeetShadow
// Block 0 thread 0 advances shadow; all blocks signal peers then waitMeetShadow.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestMeetShadowKernel(flagcxDevComm devComm) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    int nRanks = devComm.getSize();
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      net.increaseSignalShadow((flagcxDevNetSignal_t)2, (uint64_t)nRanks);
      __threadfence();
    }
    for (int peer = tid; peer < nRanks; peer += nthreads)
      net.signal(flagcxTeamWorld(devComm), peer,
                 flagcxDevNet_SignalInc{2}, flagcxCoopThread{});
    net.waitSignalMeetShadow(flagcxCoopBlock{}, (flagcxDevNetSignal_t)2);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// resetSignal + resetCounter + 32-bit readSignal
// Resets all used signal/counter slots; records post-reset values in resultBuf.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestResetKernel(flagcxDevComm devComm, uint64_t *resultBuf) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      net.resetSignal(0);
      net.resetSignal(1);
      net.resetSignal(2);
      net.resetCounter(0);
      *net.getSignalShadowPtr(2) = 0;
      (void)net.readSignal(0, 32);
      resultBuf[0] = net.readSignal(0);
      resultBuf[1] = net.readSignal(1);
      resultBuf[2] = net.readSignal(2);
      resultBuf[3] = net.readCounter(0);
    }
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxBarrierSession<flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, devComm, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      resultBuf[0] = 0;
      resultBuf[1] = 0;
      resultBuf[2] = 0;
      resultBuf[3] = 0;
    }
    bar.sync(flagcxCoopBlock(), flagcxDeviceMemoryOrderRelaxed);
  }
}

// --------------------------------------------------------------------------
// Host wrappers
// --------------------------------------------------------------------------

flagcxResult_t flagcxInterTestSignalInc(flagcxDevMem_t sendMem,
                                        flagcxDevMem_t recvMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);
  flagcxInterTestSignalIncKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestSignalAdd(flagcxDevMem_t sendMem,
                                        flagcxDevMem_t recvMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);
  flagcxInterTestSignalAddKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestCounterPipeline(flagcxDevMem_t sendMem,
                                              flagcxDevMem_t recvMem,
                                              size_t count,
                                              flagcxDataType_t datatype,
                                              flagcxDevComm_t devComm,
                                              flagcxStream_t stream,
                                              uint64_t *resultBuf) {
  if (!devComm || !sendMem || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);
  flagcxInterTestCounterPipelineKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc, resultBuf);
  cudaError_t err = cudaGetLastError();
  // K3 CounterPipeline: 3 bar.syncs (pre-barrier + post-round1 + post-round2).
  // Each bar.sync does syncsPerBarrier intra syncs (1 single-node, 2 multi-node).
  advanceIntraEpoch(devComm, 3);
  devComm->interBarrierEpoch += 3 * devComm->nInterPeers;
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestPutValue(flagcxDevMem_t recvMem,
                                       flagcxDevComm_t devComm,
                                       flagcxStream_t stream,
                                       size_t putValBase) {
  if (!devComm || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem rm(*recvMem);
  flagcxInterTestPutValueKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(rm, dc, putValBase);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestSignalOnly(flagcxDevComm_t devComm,
                                         flagcxStream_t stream) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestSignalOnlyKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  // K5 SignalOnly: multi-node = 2 bar.syncs (pre + post), single-node = 1 bar.sync
  // (no inter-peer ops to bracket, barrier-only). Epoch accounting reflects this.
  if (devComm->nInterPeers > 0) {
    devComm->intraBarrierEpoch += 4; // 2 bar.syncs × 2 intra syncs each
    devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  } else {
    devComm->intraBarrierEpoch += 1; // 1 bar.sync × 1 intra sync
  }
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestFlushDecouple(flagcxDevMem_t sendMem,
                                            flagcxDevMem_t recvMem,
                                            size_t count,
                                            flagcxDataType_t datatype,
                                            flagcxDevComm_t devComm,
                                            flagcxStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);
  flagcxInterTestFlushDecoupleKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestFollowShadow(flagcxDevComm_t devComm,
                                           flagcxStream_t stream) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestFollowShadowKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  // K8 FollowShadow: multi-node = 2 bar.syncs, single-node = 1 bar.sync.
  // Asymmetric epoch accounting same as K5.
  if (devComm->nInterPeers > 0) {
    devComm->intraBarrierEpoch += 4;
    devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  } else {
    devComm->intraBarrierEpoch += 1;
  }
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestMeetShadow(flagcxDevComm_t devComm,
                                         flagcxStream_t stream) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestMeetShadowKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  // K9 MeetShadow: multi-node = 2 bar.syncs, single-node = 1 bar.sync.
  // Asymmetric epoch accounting same as K5.
  if (devComm->nInterPeers > 0) {
    devComm->intraBarrierEpoch += 4;
    devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  } else {
    devComm->intraBarrierEpoch += 1;
  }
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t flagcxInterTestReset(flagcxDevComm_t devComm,
                                    flagcxStream_t stream,
                                    uint64_t *resultBuf) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestResetKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc, resultBuf);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}
