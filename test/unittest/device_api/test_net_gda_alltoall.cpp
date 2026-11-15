/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Standalone MPI test of the Device API Net GDA AlltoAll kernel. Run with
 * USE_XPUNETIO=1 and one NVIDIA GPU per rank; this does not call the public
 * flagcxAlltoAll runner.
 ************************************************************************/

#include "device_api/net_gda_alltoall.h"
#include "device_api/nvidia_xpu_netio_host.h"
#include "flagcx.h"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int worldRank = 0;

void check(flagcxResult_t status, const char *operation) {
  if (status == flagcxSuccess)
    return;
  std::fprintf(stderr, "[rank %d] %s failed: %d\n", worldRank, operation,
               static_cast<int>(status));
  MPI_Abort(MPI_COMM_WORLD, 1);
}

uint8_t payload(int source, int target, size_t offset) {
  return static_cast<uint8_t>((source * 37 + target * 53 + offset) % 251);
}

} // namespace

#define CHECK(call) check((call), #call)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks < 2) {
    if (worldRank == 0)
      std::fprintf(stderr, "Net GDA AlltoAll needs at least two ranks\n");
    MPI_Finalize();
    return 2;
  }

  flagcxDeviceHandle_t device = nullptr;
  CHECK(flagcxDeviceHandleInit(&device));
  int gpuCount = 0;
  CHECK(device->getDeviceCount(&gpuCount));
  if (gpuCount <= 0)
    MPI_Abort(MPI_COMM_WORLD, 2);
  CHECK(device->setDevice(worldRank % gpuCount));

  flagcxUniqueId id{};
  if (worldRank == 0)
    CHECK(flagcxGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  flagcxComm_t comm = nullptr;
  CHECK(flagcxCommInitRank(&comm, nranks, &id, worldRank));
  flagcxStream_t stream = nullptr;
  CHECK(device->streamCreate(&stream));

  flagcxDevCommRequirements requirements =
      FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.intraBarrierCount = 1;
  requirements.interBarrierCount = 1;
  requirements.interContextCount = 1;
  requirements.interForceEnable = true;
  requirements.netGdaPolicy = flagcxDevNetGdaRequired;
  requirements.netGdaOperations =
      flagcxDevNetGdaOpPut | flagcxDevNetGdaOpPutValue |
      flagcxDevNetGdaOpFlush | flagcxDevNetGdaOpBarrier;
  flagcxDevComm_t devComm = nullptr;
  CHECK(flagcxDevCommCreate(comm, &requirements, &devComm));

  unsigned capabilities = 0;
  uint32_t *errorFlag = nullptr;
  uint64_t maxChunkBytes = 0;
  unsigned alignment = 0;
  CHECK(DefaultNvidiaGda::queryComm(devComm, &capabilities, &errorFlag,
                                    &maxChunkBytes, &alignment));
  if ((capabilities & requirements.netGdaOperations) !=
          requirements.netGdaOperations ||
      errorFlag == nullptr || alignment == 0 || alignment > (1u << 20) ||
      maxChunkBytes < alignment) {
    std::fprintf(stderr, "[rank %d] invalid GDA capability profile\n",
                 worldRank);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }

  const size_t bytesPerPeer =
      ((size_t{4096} + alignment - 1) / alignment) * alignment;
  if (static_cast<size_t>(nranks) >
      std::numeric_limits<size_t>::max() / bytesPerPeer)
    MPI_Abort(MPI_COMM_WORLD, 4);
  const size_t totalBytes = static_cast<size_t>(nranks) * bytesPerPeer;
  const size_t chunkBytes =
      (std::min<uint64_t>(maxChunkBytes, bytesPerPeer) / alignment) * alignment;

  void *sendBuffer = nullptr;
  void *recvBuffer = nullptr;
  CHECK(flagcxMemAlloc(&sendBuffer, totalBytes, flagcxMemCCL));
  CHECK(flagcxMemAlloc(&recvBuffer, totalBytes, flagcxMemCCL));
  flagcxWindow_t sendWindow = nullptr;
  flagcxWindow_t recvWindow = nullptr;
  CHECK(flagcxCommWindowRegister(comm, sendBuffer, totalBytes, &sendWindow,
                                 FLAGCX_WIN_COLL_SYMMETRIC, flagcxMemCCL));
  CHECK(flagcxCommWindowRegister(comm, recvBuffer, totalBytes, &recvWindow,
                                 FLAGCX_WIN_COLL_SYMMETRIC, flagcxMemCCL));

  flagcxDevMem_t sendMem = nullptr;
  flagcxDevMem_t recvMem = nullptr;
  CHECK(flagcxDevMemCreate(comm, sendBuffer, totalBytes, sendWindow, &sendMem));
  CHECK(flagcxDevMemCreate(comm, recvBuffer, totalBytes, recvWindow, &recvMem));

  std::vector<uint8_t> send(totalBytes);
  std::vector<uint8_t> recv(totalBytes, 0xff);
  for (int peer = 0; peer < nranks; ++peer)
    for (size_t byte = 0; byte < bytesPerPeer; ++byte)
      send[static_cast<size_t>(peer) * bytesPerPeer + byte] =
          payload(worldRank, peer, byte);
  CHECK(device->deviceMemcpy(sendBuffer, send.data(), totalBytes,
                             flagcxMemcpyHostToDevice, stream));
  CHECK(device->deviceMemcpy(recvBuffer, recv.data(), totalBytes,
                             flagcxMemcpyHostToDevice, stream));
  CHECK(device->streamSynchronize(stream));
  CHECK(DefaultNvidiaGda::clearError(devComm));

  void *commDevice = nullptr;
  void *sendDevice = nullptr;
  void *recvDevice = nullptr;
  CHECK(flagcxDevCommGetDevicePtr(devComm, &commDevice));
  CHECK(flagcxDevMemGetDevicePtr(sendMem, &sendDevice));
  CHECK(flagcxDevMemGetDevicePtr(recvMem, &recvDevice));

  flagcxNetGdaAlltoAllWork work{};
  work.comm = static_cast<const flagcxDevComm *>(commDevice);
  work.send = static_cast<const flagcxDevMem *>(sendDevice);
  work.recv = static_cast<const flagcxDevMem *>(recvDevice);
  work.errorFlag = errorFlag;
  work.bytesPerPeer = bytesPerPeer;
  work.chunkBytes = chunkBytes;
  work.contextIndex = 0;
  work.barrierIndex = 0;
  work.fence = flagcxDevNetFenceLevel::Relaxed;
  CHECK(flagcxNetGdaAlltoAllLaunch(work, stream));
  CHECK(device->streamSynchronize(stream));
  CHECK(DefaultNvidiaGda::pollError(devComm));
  CHECK(device->deviceMemcpy(recv.data(), recvBuffer, totalBytes,
                             flagcxMemcpyDeviceToHost, stream));
  CHECK(device->streamSynchronize(stream));

  int localPass = 1;
  for (int peer = 0; peer < nranks && localPass; ++peer)
    for (size_t byte = 0; byte < bytesPerPeer; ++byte)
      if (recv[static_cast<size_t>(peer) * bytesPerPeer + byte] !=
          payload(peer, worldRank, byte)) {
        std::fprintf(stderr, "[rank %d] mismatch from rank %d at byte %zu\n",
                     worldRank, peer, byte);
        localPass = 0;
        break;
      }
  int globalPass = 0;
  MPI_Allreduce(&localPass, &globalPass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (worldRank == 0)
    std::printf("Device API Net GDA AlltoAll: %s\n",
                globalPass ? "PASS" : "FAIL");

  CHECK(flagcxDevMemDestroy(comm, sendMem));
  CHECK(flagcxDevMemDestroy(comm, recvMem));
  CHECK(flagcxCommWindowDeregister(comm, sendWindow, flagcxMemCCL));
  CHECK(flagcxCommWindowDeregister(comm, recvWindow, flagcxMemCCL));
  CHECK(flagcxMemFree(sendBuffer, flagcxMemCCL));
  CHECK(flagcxMemFree(recvBuffer, flagcxMemCCL));
  CHECK(flagcxDevCommDestroy(comm, devComm));
  CHECK(device->streamDestroy(stream));
  CHECK(flagcxCommDestroy(comm));
  CHECK(flagcxDeviceHandleFree(device));
  MPI_Finalize();
  return globalPass ? 0 : 1;
}
