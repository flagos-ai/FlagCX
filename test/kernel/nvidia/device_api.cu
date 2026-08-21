/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API kernels.
 *
 * 1. Intra-node AllReduce — peer pointer + barrier based.
 *    Vendor:   wraps vendor specific DevComm + Window + barrier.
 *    Default:  IPC peer pointers + atomics barrier.
 *    Same kernel code compiles for both paths.
 *
 * 2. Inter-node AlltoAll — two separate kernels:
 *    a) One-sided (put): thread-stride loop, put + waitSignal + flush.
 *    b) Two-sided (send/recv): thread-0 block-stride loop, FIFO + term/wait.
 *    Both paths wrapped by bar.sync() pre/post barriers.
 * 
 * 3. Inter-node One-sided Device API Tests
 *    Eight focused kernels, each covering one Device API facet.
 *    Signal/counter slot assignments:
 *      slot 0: SignalAdd, CounterPipeline, FlushDecouple
 *      slot 1: PutValue, SignalOnly
 *      slot 2: FollowShadow, MeetShadow
 *      counter 0: CounterInc
 *    DevCommRequirements: interSignalCount=3, interCounterCount=1
 *    All kernels follow: pre-bar.sync / core logic / post-bar.sync.
 *
 * Host-side flagcxDevCommCreate/Destroy are in flagcx_device.cc.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "nvidia_adaptor.h"
#include "global_comm.h"
#include "flagcx_kernel.h"
#include <cuda_runtime.h>

// Datatype size helper (moved from flagcx/kernels/flagcx_kernel_device.cu)
FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype) {
  switch (dtype) {
    case flagcxChar:
      return sizeof(char);
    case flagcxUint8:
      return sizeof(unsigned char);
    case flagcxInt:
      return sizeof(int);
    case flagcxUint32:
      return sizeof(unsigned int);
    case flagcxInt64:
      return sizeof(long long);
    case flagcxUint64:
      return sizeof(unsigned long long);
    case flagcxHalf:
      return 2;
    case flagcxFloat:
      return sizeof(float);
    case flagcxDouble:
      return sizeof(double);
    case flagcxBfloat16:
      return 2;
    default:
      return 0;
  }
}



// ==========================================================================
// 1. Intra-node AllReduce
// ==========================================================================

// Intra-node AllReduce: each block reads from all peers via
// flagcxGetIntraPointer, reduces (sum), and writes result back to all peers.
template <typename T>
__global__ void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraAllReduceKernel(flagcxDevComm devComm, flagcxDevMem mem,
                               size_t offset, size_t count) {
  // AllReduce requires peer pointer access (window or IPC)
  if (!mem.hasWindow())
    return;

  flagcxTeam intra = flagcxTeamIntra(devComm);
  flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);

  // Create barrier session using simplified FlagCX API (4 params).
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X};

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  bar.sync(flagcxDeviceMemoryOrderAcquire);

  const int rank = devComm.getIntraRank();
  const int nRanks = devComm.getIntraSize();
  const int globalTid =
      FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_DIM_X * (rank + FLAGCX_BLOCK_IDX_X * nRanks);
  const int globalNthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X * nRanks;

  // Phase 1: Reduce — sum data from all intra-node peers
  // Phase 2: Write — store result to all intra-node peers
  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = T(0);
    for (int peer = 0; peer < nRanks; peer++)
      v += *((T *)flagcxGetIntraPointer(mem, offset + o * sizeof(T), peer));
    for (int peer = 0; peer < nRanks; peer++)
      *((T *)flagcxGetIntraPointer(mem, offset + o * sizeof(T), peer)) = v;
  }

  // Post-reduce barrier (release ordering — ensure writes are visible)
  bar.sync(flagcxDeviceMemoryOrderRelease);
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
flagcxResult_t launchKernelIntraAllReduce(flagcxDevMem_t devMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (devComm == nullptr || devMem == nullptr) {
    return flagcxInternalError;
  }

  cudaStream_t cudaStream = *(cudaStream_t *)stream;

  // Unified constructors — work for both Vendor and Default
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

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// Inter-node One-sided AlltoAll
//
// Thread-stride loop: each thread dispatches put ops to different peers.
// put() posts FIFO descriptor (Default) or one-sided descriptor (Vendor).
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
  flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
      flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  // Read signal baseline before pre-barrier so it reflects the pre-round state.
  uint64_t signalValue = net.readSignal(0);

  // Pre-communication barrier
  bar.sync(flagcxDeviceMemoryOrderRelaxed);

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
  bar.sync(flagcxDeviceMemoryOrderRelaxed);
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
  flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
      flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  // Pre-communication barrier
  bar.sync(flagcxDeviceMemoryOrderRelaxed);

  // All operations are Coop-scope: every thread participates,
  // only threadRank==0 touches FIFO.
  for (int peer = FLAGCX_BLOCK_IDX_X; peer < nRanks;
       peer += FLAGCX_GRID_DIM_X) {
    size_t offset = peer * size;
    net.send(flagcxCoopBlock{}, sendMem, offset, count, datatype, peer);
    net.recv(flagcxCoopBlock{}, recvMem, offset, count, datatype, peer);
  }

  net.term(flagcxCoopBlock{});
  net.wait(flagcxCoopBlock{});

  // Post-communication barrier
  bar.sync(flagcxDeviceMemoryOrderRelaxed);
}

// Host-side one-sided AlltoAll function.
flagcxResult_t launchKernelNetOneSidedAlltoAll(flagcxDevMem_t sendMem,
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

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// Host-side two-sided AlltoAll function.
flagcxResult_t launchKernelNetTwoSidedAlltoAll(flagcxDevMem_t sendMem,
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

  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}


// Shared IPC alltoall helper: each CTA copies its own partition of all intra
// peers.  Work is partitioned by word range so that the per-CTA barrier
// (indexed by FLAGCX_BLOCK_IDX_X) correctly synchronizes the writes each CTA
// performs — no cross-CTA grid barrier needed.
// NOTE: assumes chunkSize is a multiple of 4 bytes (i.e., element type is
// >= 4-byte aligned -- float, int32, etc.). Sub-4-byte types (half, int8)
// with odd counts will lose tail bytes.
FLAGCX_DEVICE_INLINE_DECORATOR void
ipcAlltoAll(const flagcxDevMem &sendMem, const flagcxDevMem &recvMem,
            flagcxTeam intra, int intraSize, int intraBase,
            int myWorldRank, size_t chunkSize) {
  size_t nWords = chunkSize / sizeof(uint32_t);
  size_t wordsPerCta =
      (nWords + (size_t)FLAGCX_GRID_DIM_X - 1) / (size_t)FLAGCX_GRID_DIM_X;
  size_t wStart = (size_t)FLAGCX_BLOCK_IDX_X * wordsPerCta;
  size_t wEnd = wStart + wordsPerCta;
  if (wEnd > nWords)
    wEnd = nWords;
  int ctaTid = FLAGCX_THREAD_IDX_X;
  int ctaThreads = FLAGCX_BLOCK_DIM_X;
  for (int lr = 0; lr < intraSize; lr++) {
    int worldPeer = intraBase + lr;
    uint32_t *src = (uint32_t *)flagcxGetLocalPointer(
        sendMem, (size_t)worldPeer * chunkSize);
    uint32_t *dst = (uint32_t *)flagcxGetPeerPointer(
        recvMem, (size_t)myWorldRank * chunkSize, intra, lr);
    for (size_t w = wStart + (size_t)ctaTid; w < wEnd; w += (size_t)ctaThreads)
      dst[w] = src[w];
  }
}

// put + SignalInc
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestPutSignalIncKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    // Hybrid: DevNet for inter + IPC for intra
    flagcxDevNet net(devComm, 0);

    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);

    bar.sync(flagcxDeviceMemoryOrderRelaxed);

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

    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    // Single-node: IPC only, no DevNet
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

// put data + separate SignalAdd: decouples data transfer from signalling.
// One-sided path: two NIC ops (WRITE then ATOMIC+2 on slot 0).
// Default path: PrimPut + PrimSignal(value=2) (two FIFO entries, slot 0).
// Contrast with K1 where both paths fuse into a single chained WR.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestPutSignalAddDecoupledKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
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
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
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
  flagcxTeam intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    uint64_t c0 = net.readCounter(0);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);

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
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      float *slot = (float *)flagcxGetLocalPointer(sendMem, (size_t)peer * size);
      *slot = 999.0f;
    }
    bar.sync(flagcxDeviceMemoryOrderRelaxed);

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
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);

    // Round 1
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    // Stamp sentinel
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      float *slot = (float *)flagcxGetLocalPointer(sendMem, (size_t)peer * size);
      *slot = 999.0f;
    }
    bar.sync(flagcxDeviceMemoryOrderRelaxed);

    // Round 2
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      resultBuf[0] = 0; // no counter in IPC mode
      resultBuf[1] = 0; // nInterRanks = 0
    }
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
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
  flagcxTeam intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s1 = net.readSignal(1);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
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
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      int lr = peer - intraBase;
      uint64_t *dst = (uint64_t *)flagcxGetPeerPointer(
          recvMem, putValBase + (size_t)myRank * sizeof(uint64_t), intra, lr);
      *dst = val;
    }
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

// signal standalone
// Each rank signals all other peers on slot 1; waits for nRanks-1 incoming.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestSignalKernel(flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  int nInterRanks = nRanks - intraSize;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s1 = net.readSignal(1);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads)
      if (peer != myRank && (peer < intraBase || peer >= intraBase + intraSize))
        net.signal(flagcxTeamWorld(devComm), peer,
                   flagcxDevNet_SignalInc{1}, flagcxCoopThread{});
    if (nInterRanks > 0)
      net.waitSignal(flagcxCoopBlock{}, 1, s1 + (uint64_t)nInterRanks);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
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
  flagcxTeam intra = flagcxTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);

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
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

// waitSignalFollowShadow
// All ranks signal all peers on slot 2; FollowShadow advances shadow by nRanks.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestFollowShadowKernel(flagcxDevComm devComm) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    int nRanks = devComm.getSize();
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads)
      net.signal(flagcxTeamWorld(devComm), peer,
                 flagcxDevNet_SignalInc{2}, flagcxCoopThread{});
    uint64_t before, delta;
    net.waitSignalFollowShadow(flagcxCoopBlock{}, (flagcxDevSignal_t)2,
                                (uint64_t)nRanks, &before, &delta);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

// increaseSignalShadow + waitSignalMeetShadow
// Block 0 thread 0 advances shadow; all blocks signal peers then waitMeetShadow.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestMeetShadowKernel(flagcxDevComm devComm) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    int nRanks = devComm.getSize();
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      net.increaseSignalShadow((flagcxDevSignal_t)2, (uint64_t)nRanks);
      __threadfence();
    }
    for (int peer = tid; peer < nRanks; peer += nthreads)
      net.signal(flagcxTeamWorld(devComm), peer,
                 flagcxDevNet_SignalInc{2}, flagcxCoopThread{});
    net.waitSignalMeetShadow(flagcxCoopBlock{}, (flagcxDevSignal_t)2);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

// resetSignal + resetCounter + 32-bit readSignal
// Resets all used signal/shadow/counter slots; records post-reset values.
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestResetKernel(flagcxDevComm devComm, uint64_t *resultBuf) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      net.resetSignal(0);
      net.resetSignal(1);
      net.resetSignal(2);
      net.resetCounter(0);
      (void)net.readSignal(0, 32);
      resultBuf[0] = net.readSignal(0);
      resultBuf[1] = net.readSignal(1);
      resultBuf[2] = net.readSignal(2);
      resultBuf[3] = net.readCounter(0);
    }
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      resultBuf[0] = 0;
      resultBuf[1] = 0;
      resultBuf[2] = 0;
      resultBuf[3] = 0;
    }
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

// --------------------------------------------------------------------------
// Host wrappers
// --------------------------------------------------------------------------

flagcxResult_t launchKernelNetPutSignalInc(flagcxDevMem_t sendMem,
                                        flagcxDevMem_t recvMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);

  flagcxInterTestPutSignalIncKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t launchKernelNetPutSignalAdd(flagcxDevMem_t sendMem,
                                        flagcxDevMem_t recvMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);
  flagcxInterTestPutSignalAddDecoupledKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t launchKernelNetCounterPipeline(flagcxDevMem_t sendMem,
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
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t launchKernelNetPutValue(flagcxDevMem_t recvMem,
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
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t launchKernelNetSignal(flagcxDevComm_t devComm,
                                         flagcxStream_t stream) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestSignalKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t launchKernelNetFlushDecouple(flagcxDevMem_t sendMem,
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
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t launchKernelNetFollowShadow(flagcxDevComm_t devComm,
                                           flagcxStream_t stream) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestFollowShadowKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t launchKernelNetMeetShadow(flagcxDevComm_t devComm,
                                         flagcxStream_t stream) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestMeetShadowKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t launchKernelNetReset(flagcxDevComm_t devComm,
                                    flagcxStream_t stream,
                                    uint64_t *resultBuf) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestResetKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc, resultBuf);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// K8: get AlltoAll
//
// Each rank RDMA-READs peer's sendBuff[myRank*size..] into local
// recvBuff[peer*size..].  Producer fills sendBuff, consumer pulls via get().
// Synchronized by barrier.  No fused signal — get() has no signal action;
// the post-barrier ensures completion visibility.
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestGetKernel(flagcxDevMem sendMem, flagcxDevMem recvMem,
                                     size_t count, flagcxDataType_t datatype,
                                     flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  flagcxTeam intra = flagcxTeamIntra(devComm);
  size_t size = count * getFlagcxDataTypeSizeDevice(datatype);
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);

    // IPC for intra-node peers
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    // RDMA READ for inter-node peers: pull from peer's sendMem into my recvMem
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      // src: peer's sendBuff at offset myRank*size (peer's data for me)
      // dst: my recvBuff at offset peer*size (my slot for peer's data)
      net.get(flagcxTeamWorld(devComm), peer,
              sendMem, (size_t)myRank * size,
              recvMem, (size_t)peer * size, size,
              flagcxCoopThread{});
    }
    net.flush(flagcxCoopBlock{});
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    // Intra-only: use IPC
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

flagcxResult_t launchKernelNetGet(flagcxDevMem_t sendMem,
                                          flagcxDevMem_t recvMem, size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxDevComm_t devComm,
                                          flagcxStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevMem sm(*sendMem), rm(*recvMem);
  flagcxInterTestGetKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// Intra-node Device API test kernels
// ==========================================================================

// K1: Local Pointer
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestLocalPointerKernel(flagcxDevMem devMem, void *rawPtr,
                                      int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    void *localPtr = flagcxGetLocalPointer(devMem, 0);
    // Verify local pointer is non-null and points to same data as rawPtr
    // (may be a different VA due to VMM flat-mapping)
    if (localPtr == nullptr) {
      results[0] = 0;
    } else {
      // Read the value written by host through rawPtr, verify via localPtr
      float val = *((volatile float *)localPtr);
      float expected = *((volatile float *)rawPtr);
      results[0] = (val == expected) ? 1 : 0;
    }
  }
}

flagcxResult_t launchKernelLocalPointer(flagcxDevMem_t devMem,
                                           void *rawPtr, int *results,
                                           flagcxStream_t stream) {
  if (!devMem || !results) return flagcxInternalError;
  flagcxDevMem dm(*devMem);
  flagcxIntraTestLocalPointerKernel
      <<<1, 32, 0, *(cudaStream_t *)stream>>>(dm, rawPtr, results);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K2: Intra Pointer
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestIntraPointerKernel(flagcxDevMem devMem,
                                      flagcxDevComm devComm, float *output,
                                      size_t count) {
  int myRank = devComm.getIntraRank();
  int nRanks = devComm.getIntraSize();
  int peer = (myRank + 1) % nRanks;
  flagcxTeam intra = flagcxTeamIntra(devComm);

  // Barrier before reading — ensures peer's host memcpy is visible via P2P
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X};
  bar.sync(flagcxDeviceMemoryOrderAcquire);

  // Read peer's data
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  // Check first pointer before any thread dereferences
  float *firstPtr = (float *)flagcxGetIntraPointer(devMem, tid * sizeof(float), peer);
  if (firstPtr == nullptr)
    return;

  for (size_t i = tid; i < count; i += nthreads) {
    float *peerPtr = (float *)flagcxGetIntraPointer(devMem, i * sizeof(float), peer);
    output[i] = *peerPtr;
  }
}

flagcxResult_t launchKernelIntraPointer(flagcxDevMem_t devMem,
                                           flagcxDevComm_t devComm,
                                           float *output, size_t count,
                                           flagcxStream_t stream) {
  if (!devMem || !devComm || !output) return flagcxInternalError;
  flagcxDevMem dm(*devMem);
  flagcxDevComm dc(*devComm);
  flagcxIntraTestIntraPointerKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dm, dc, output, count);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K3: Peer Pointer (team-based)
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestPeerPointerKernel(flagcxDevMem devMem, flagcxDevComm devComm,
                                     float *output, size_t count) {
  int myRank = devComm.getIntraRank();
  int nRanks = devComm.getIntraSize();
  int peer = (myRank + 1) % nRanks;
  flagcxTeam intra = flagcxTeamIntra(devComm);

  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (size_t i = tid; i < count; i += nthreads) {
    float *peerPtr =
        (float *)flagcxGetPeerPointer(devMem, i * sizeof(float), intra, peer);
    output[i] = *peerPtr;
  }
}

flagcxResult_t launchKernelPeerPointer(flagcxDevMem_t devMem,
                                          flagcxDevComm_t devComm,
                                          float *output, size_t count,
                                          flagcxStream_t stream) {
  if (!devMem || !devComm || !output) return flagcxInternalError;
  flagcxDevMem dm(*devMem);
  flagcxDevComm dc(*devComm);
  flagcxIntraTestPeerPointerKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dm, dc, output, count);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K5: Intra Barrier Sync
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestBarrierSyncKernel(flagcxDevMem devMem, flagcxDevComm devComm,
                                     float *output, size_t count) {
  int myRank = devComm.getIntraRank();
  int nRanks = devComm.getIntraSize();
  int peer = (myRank + 1) % nRanks;
  flagcxTeam intra = flagcxTeamIntra(devComm);

  // Pre-barrier (acquire — ensure peer's host-written data is visible)
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X};

  bar.sync(flagcxDeviceMemoryOrderAcquire);

  // Read peer's data
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;

  for (size_t i = tid; i < count; i += nthreads) {
    float *peerPtr =
        (float *)flagcxGetIntraPointer(devMem, i * sizeof(float), peer);
    output[i] = *peerPtr;
  }

  // Post-barrier
  bar.sync(flagcxDeviceMemoryOrderRelease);
}

flagcxResult_t launchKernelIntraBarrierSync(flagcxDevMem_t devMem,
                                          flagcxDevComm_t devComm,
                                          float *output, size_t count,
                                          flagcxStream_t stream) {
  if (!devMem || !devComm || !output) return flagcxInternalError;
  flagcxDevMem dm(*devMem);
  flagcxDevComm dc(*devComm);
  flagcxIntraTestBarrierSyncKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dm, dc, output, count);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K6: Intra Barrier Arrive/Wait
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestBarrierArriveWaitKernel(flagcxDevMem devMem,
                                           flagcxDevComm devComm, float *output,
                                           size_t count) {
  int myRank = devComm.getIntraRank();
  int nRanks = devComm.getIntraSize();
  int peer = (myRank + 1) % nRanks;
  flagcxTeam intra = flagcxTeamIntra(devComm);

  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X};

  // Arrive (release — signal readiness)
  bar.arrive(flagcxDeviceMemoryOrderRelease);

  // Wait (acquire — ensure peer's host-written data is visible)
  bar.wait(flagcxDeviceMemoryOrderAcquire);

  // Read peer's data
  int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
  int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  for (size_t i = tid; i < count; i += nthreads) {
    float *peerPtr =
        (float *)flagcxGetIntraPointer(devMem, i * sizeof(float), peer);
    output[i] = *peerPtr;
  }

  // Post-barrier
  bar.sync(flagcxDeviceMemoryOrderRelease);
}

flagcxResult_t launchKernelIntraBarrierArriveWait(flagcxDevMem_t devMem,
                                                flagcxDevComm_t devComm,
                                                float *output, size_t count,
                                                flagcxStream_t stream) {
  if (!devMem || !devComm || !output) return flagcxInternalError;
  flagcxDevMem dm(*devMem);
  flagcxDevComm dc(*devComm);
  flagcxIntraTestBarrierArriveWaitKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dm, dc, output, count);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K7: SymPtr
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestSymPtrKernel(flagcxDevMem devMem, flagcxDevComm devComm,
                                int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    int myRank = devComm.getIntraRank();
    int nRanks = devComm.getIntraSize();
    int peer = (myRank + 1) % nRanks;

    // Create SymPtr
    flagcxSymPtr<float> ptr(devMem, 0);

    // Test localPtr
    float *local = ptr.localPtr();
    bool localOk = (local != nullptr);

    // Test intraPtr
    float *intra = ptr.intraPtr(peer);
    bool intraOk = (intra != nullptr);

    // Test arithmetic
    flagcxSymPtr<float> ptr2 = ptr + 10;
    bool arithOk = ((ptr2.offset - ptr.offset) == 10 * sizeof(float));

    // Test subtraction
    ptrdiff_t diff = ptr2 - ptr;
    bool diffOk = (diff == 10);

    results[0] = (localOk && intraOk && arithOk && diffOk) ? 1 : 0;
  }
}

flagcxResult_t launchKernelIntraSymPtr(flagcxDevMem_t devMem,
                                     flagcxDevComm_t devComm, int *results,
                                     flagcxStream_t stream) {
  if (!devMem || !devComm || !results) return flagcxInternalError;
  flagcxDevMem dm(*devMem);
  flagcxDevComm dc(*devComm);
  flagcxIntraTestSymPtrKernel
      <<<1, 32, 0, *(cudaStream_t *)stream>>>(dm, dc, results);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K9: DevMem & Comm Queries
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestCommQueriesKernel(flagcxDevMem devMem, flagcxDevComm devComm,
                                     int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    // DevMem queries
    results[0] = devMem.hasWindow() ? 1 : 0;

    // Comm queries
    results[1] = devComm.getIntraRank();
    results[2] = devComm.getIntraSize();
    results[3] = devComm.getRank();
    results[4] = devComm.getSize();

    // DevMem peer ptrs (optional)
    void **peerPtrs = devMem.getDevPeerPtrs();
    results[5] = (peerPtrs != nullptr) ? 1 : 0;
  }
}

flagcxResult_t launchKernelCommQueries(flagcxDevMem_t devMem,
                                          flagcxDevComm_t devComm,
                                          int *results,
                                          flagcxStream_t stream) {
  if (!devMem || !devComm || !results) return flagcxInternalError;
  flagcxDevMem dm(*devMem);
  flagcxDevComm dc(*devComm);
  flagcxIntraTestCommQueriesKernel
      <<<1, 32, 0, *(cudaStream_t *)stream>>>(dm, dc, results);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K10: Coop Groups
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(256)
    flagcxIntraTestCoopGroupsKernel(int *results) {
  // Block
  {
    flagcxCoopBlock block;
    int rank = block.threadRank();
    int size = block.size();
    if (FLAGCX_BLOCK_IDX_X == 0 && rank == 0) {
      results[0] = (size == 256) ? 1 : 0;
    }
    block.sync();
  }

  // Warp (Tile<32>)
  {
    flagcxCoopWarp warp;
    int rank = warp.threadRank();
    int size = warp.size();
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      results[1] = (size == 32 && rank == 0) ? 1 : 0;
    }
    warp.sync();
  }

  // Thread (Tile<1>)
  {
    flagcxCoopThread thread;
    int rank = thread.threadRank();
    int size = thread.size();
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      results[2] = (size == 1 && rank == 0) ? 1 : 0;
    }
  }

  // Tile<8>
  {
    flagcxCoopTile<8> tile;
    int rank = tile.threadRank();
    int size = tile.size();
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      results[3] = (size == 8 && rank == 0) ? 1 : 0;
    }
    tile.sync();
  }

  // Lanes (full warp mask)
  {
    flagcxCoopLanes lanes(0xffffffffu);
    int rank = lanes.threadRank();
    int size = lanes.size();
    if (FLAGCX_BLOCK_IDX_X == 0 && FLAGCX_THREAD_IDX_X == 0) {
      results[4] = (size == 32 && rank == 0) ? 1 : 0;
    }
    lanes.sync();
  }
}

flagcxResult_t launchKernelCoopGroups(int *results, flagcxStream_t stream) {
  if (!results) return flagcxInternalError;
  flagcxIntraTestCoopGroupsKernel
      <<<4, 256, 0, *(cudaStream_t *)stream>>>(results);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// K11: Team
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxIntraTestTeamKernel(flagcxDevComm devComm, int *results) {
  if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0) {
    int myRank = devComm.getIntraRank();
    int nRanks = devComm.getIntraSize();

    // Test flagcxTeamIntra
    flagcxTeam intra = flagcxTeamIntra(devComm);
    results[0] = (intra.nRanks() == nRanks) ? 1 : 0;

    // Test rankToWorld
    int worldRank = flagcxTeamRankToWorld(devComm, intra, myRank);
    results[1] = (worldRank == myRank) ? 1 : 0; // single-node: intra == world

    // Test rankToIntra
    int intraRank = flagcxTeamRankToIntra(devComm, intra, myRank);
    results[2] = (intraRank == myRank) ? 1 : 0;

    // Test rankIsMember
    bool isMember = flagcxTeamRankIsMember(intra, intra, myRank);
    results[3] = isMember ? 1 : 0;
  }
}

flagcxResult_t launchKernelTeam(flagcxDevComm_t devComm, int *results,
                                   flagcxStream_t stream) {
  if (!devComm || !results) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxIntraTestTeamKernel
      <<<1, 32, 0, *(cudaStream_t *)stream>>>(dc, results);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ===========================================================================
// Inter-node K11/K12/K13 — dedicated test kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// K11: WaitSignal + Flush (standalone)
// Signal all inter peers, wait for signals from all inter peers, flush.
// ---------------------------------------------------------------------------
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestWaitSignalFlushKernel(flagcxDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  int nInterRanks = nRanks - intraSize;

  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);

    // Reset signal slot 0
    net.resetSignal(0);
    uint64_t s0 = net.readSignal(0);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);

    // Signal all inter peers
    int tid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_IDX_X * FLAGCX_BLOCK_DIM_X;
    int nthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads)
      if (peer < intraBase || peer >= intraBase + intraSize)
        net.signal(flagcxTeamWorld(devComm), peer,
                   flagcxDevNet_SignalInc{0}, flagcxCoopThread{});

    // Wait for signals from all inter peers
    if (nInterRanks > 0)
      net.waitSignal(flagcxCoopBlock{}, 0, s0 + (uint64_t)nInterRanks);

    // Flush
    net.flush(flagcxCoopBlock{});

    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  } else {
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);
    bar.sync(flagcxDeviceMemoryOrderRelaxed);
  }
}

flagcxResult_t launchKernelNetWaitSignalFlush(flagcxDevComm_t devComm,
                                              flagcxStream_t stream) {
  if (!devComm) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestWaitSignalFlushKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ---------------------------------------------------------------------------
// K12: Inter Barrier — stress test (N iterations of inter barrier sync)
// ---------------------------------------------------------------------------
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestInterBarrierKernel(flagcxDevComm devComm, int *results,
                                      int nIters) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxTeam team = flagcxTeamInter(devComm);
    flagcxDevBarrier<flagcxTeamTagInter, flagcxCoopBlock> bar(
        flagcxCoopBlock(), net, team, FLAGCX_BLOCK_IDX_X);

    for (int i = 0; i < nIters; i++) {
      bar.sync(flagcxDeviceMemoryOrderAcqRel);
    }

    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0)
      results[0] = 1; // success
  } else {
    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0)
      results[0] = -1; // no inter peers, skip
  }
}

flagcxResult_t launchKernelInterBarrier(flagcxDevComm_t devComm,
                                           int *results, int nIters,
                                           flagcxStream_t stream) {
  if (!devComm || !results) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestInterBarrierKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc, results, nIters);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ---------------------------------------------------------------------------
// K13: World Barrier — sync + arrive/wait split
// ---------------------------------------------------------------------------
FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
    flagcxInterTestWorldBarrierKernel(flagcxDevComm devComm, int *results) {
  if (devComm._nInterPeers > 0) {
    flagcxDevNet net(devComm, 0);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagWorld{}, net, FLAGCX_BLOCK_IDX_X);

    // Test sync
    bar.sync(flagcxDeviceMemoryOrderAcqRel);

    // Test arrive + wait (split)
    bar.arrive(flagcxDeviceMemoryOrderRelease);
    bar.wait(flagcxDeviceMemoryOrderAcquire);

    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0)
      results[0] = 1; // success
  } else {
    // Single-node: use intra barrier as world barrier
    flagcxDevNet net(devComm, FLAGCX_BLOCK_IDX_X);
    flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> bar(
        flagcxCoopBlock(), flagcxTeamTagIntra{}, net, FLAGCX_BLOCK_IDX_X);

    bar.sync(flagcxDeviceMemoryOrderAcqRel);
    bar.arrive(flagcxDeviceMemoryOrderRelease);
    bar.wait(flagcxDeviceMemoryOrderAcquire);

    if (FLAGCX_THREAD_IDX_X == 0 && FLAGCX_BLOCK_IDX_X == 0)
      results[0] = 1;
  }
}

flagcxResult_t launchKernelWorldBarrier(flagcxDevComm_t devComm,
                                           int *results,
                                           flagcxStream_t stream) {
  if (!devComm || !results) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxInterTestWorldBarrierKernel
      <<<FLAGCX_DEVICE_CTA_COUNT, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc, results);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ==========================================================================
// K1: DevNetGetFromComm
// Verifies that constructing flagcxDevNet from DevComm yields a valid
// transport handle (contextCount > 0).
// results[0] = 1 if valid, 0 if no contexts available
// results[1] = intraSize
// ==========================================================================

FLAGCX_GLOBAL_DECORATOR void __launch_bounds__(1)
    flagcxDevNetGetFromCommKernel(flagcxDevComm devComm, int *results) {
  flagcxDevNet net(devComm, 0);
  results[0] = (devComm.getContextCount() > 0) ? 1 : 0;
  results[1] = devComm.getIntraSize();
}

flagcxResult_t launchKernelNetGetFromComm(flagcxDevComm_t devComm, int *results,
                                       flagcxStream_t stream) {
  if (!devComm || !results) return flagcxInternalError;
  flagcxDevComm dc(*devComm);
  flagcxDevNetGetFromCommKernel
      <<<1, 1, 0, *(cudaStream_t *)stream>>>(dc, results);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}
