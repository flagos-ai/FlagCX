/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * AlltoAll over the FlagCX Device API Net and Barrier contracts. Provider
 * selection, NIC posting, and signal completion belong to those contracts.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "device_api/net_gda_alltoall.h"

#if defined(FLAGCX_COMM_TRAITS_DEFAULT)

#include "nvidia_adaptor.h" // flagcxStream::base
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

FLAGCX_GLOBAL_DECORATOR void
flagcxNetGdaAlltoAllKernel(flagcxNetGdaAlltoAllWork work) {
  // One CTA owns one Barrier channel. All its threads enter sync().
  if (blockIdx.x != 0 || gridDim.x != 1 || work.comm == nullptr ||
      work.errorFlag == nullptr || work.contextIndex < 0 ||
      work.chunkBytes == 0) {
    FLAGCX_DEVICE_TRAP();
    return;
  }

  const flagcxDevComm &comm = *work.comm;
  const int rank = comm.getRank();
  const int nranks = comm.getSize();
  if (nranks <= 0 || rank < 0 || rank >= nranks ||
      work.contextIndex >= comm.getContextCount() ||
      work.bytesPerPeer > SIZE_MAX / static_cast<size_t>(nranks) ||
      (work.bytesPerPeer != 0 &&
       (work.send == nullptr || work.recv == nullptr))) {
    FLAGCX_DEVICE_TRAP();
    return;
  }

  flagcxCoopBlock block;
  flagcxDevNet net(comm, work.contextIndex);
  if (!net.isValid() || net.context().provider != DefaultNetProviderGda) {
    FLAGCX_DEVICE_TRAP();
    return;
  }

  const flagcxTeam team = flagcxTeamWorld(comm);
  const size_t rankOffset = static_cast<size_t>(rank) * work.bytesPerPeer;

  if (work.bytesPerPeer != 0) {
    uint8_t *dst = static_cast<uint8_t *>(work.recv->getRawPtr());
    const uint8_t *src = static_cast<const uint8_t *>(work.send->getRawPtr());
    if (dst == nullptr || src == nullptr) {
      FLAGCX_DEVICE_TRAP();
      return;
    }
    DeviceAPI::CoopOps::copy(block._base, dst + rankOffset, src + rankOffset,
                             work.bytesPerPeer);
  }
  block.sync();

  // The first thread submits each peer's chunks in order. The GDA adapter
  // maps each Net::put to the peer's QP and reports posting failures.
  if (block.threadRank() == 0 && work.bytesPerPeer != 0) {
    const flagcxDevMem &send = *work.send;
    const flagcxDevMem &recv = *work.recv;
    for (int peer = 0; peer < nranks; ++peer) {
      if (peer == rank)
        continue;
      const size_t peerOffset = static_cast<size_t>(peer) * work.bytesPerPeer;
      for (size_t done = 0; done < work.bytesPerPeer;) {
        const size_t remaining = work.bytesPerPeer - done;
        const size_t bytes =
            remaining < work.chunkBytes ? remaining : work.chunkBytes;
        net.put(team, peer, recv, rankOffset + done, send, peerOffset + done,
                bytes, flagcxDevNet_None{}, flagcxDevNet_None{},
                flagcxCoopThread{});
        if (DeviceAPI::Atomic::load(work.errorFlag,
                                    flagcxDeviceMemoryOrderAcquire) != 0)
          break;
        done += bytes;
      }
      if (DeviceAPI::Atomic::load(work.errorFlag,
                                  flagcxDeviceMemoryOrderAcquire) != 0)
        break;
    }
  }
  block.sync();
  if (DeviceAPI::Atomic::load(work.errorFlag, flagcxDeviceMemoryOrderAcquire) !=
      0)
    return;

  // The GDA World Barrier exchanges one arrival with every rank, including
  // ranks on the same node. The current XPU NetIO profile permits only the
  // Relaxed fence; transport ordering and target visibility require a real
  // two-node hardware check before treating this as a general Put fence.
  flagcxDevBarrier<flagcxTeamTagWorld, flagcxCoopBlock> barrier(
      block, flagcxTeamTagWorld{}, net, work.barrierIndex);
  barrier.sync(flagcxDeviceMemoryOrderAcqRel, work.fence);
}

} // namespace

flagcxResult_t flagcxNetGdaAlltoAllLaunch(const flagcxNetGdaAlltoAllWork &work,
                                          flagcxStream_t stream) {
  if (work.comm == nullptr || work.errorFlag == nullptr ||
      work.contextIndex < 0 || work.chunkBytes == 0 ||
      work.barrierIndex >= FLAGCX_DEVICE_CTA_COUNT ||
      work.fence != flagcxDevNetFenceLevel::Relaxed ||
      (work.bytesPerPeer != 0 &&
       (work.send == nullptr || work.recv == nullptr)))
    return flagcxInvalidArgument;

  cudaStream_t cudaStream = stream ? stream->base : cudaStream_t{};
  flagcxNetGdaAlltoAllKernel<<<1, FLAGCX_DEVICE_THREADS_PER_CTA, 0,
                               cudaStream>>>(work);
  return cudaGetLastError() == cudaSuccess ? flagcxSuccess
                                           : flagcxUnhandledDeviceError;
}

#endif // FLAGCX_COMM_TRAITS_DEFAULT
