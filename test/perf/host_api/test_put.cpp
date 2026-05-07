#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "tools.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>

namespace {

void fatal(flagcxResult_t res, const char *msg, int rank) {
  if (res != flagcxSuccess) {
    fprintf(stderr, "[rank %d] %s (err=%d)\n", rank, msg, int(res));
    MPI_Abort(MPI_COMM_WORLD, res);
  }
}
} // namespace

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();
  int printBuffer = args.isPrintBuffer();
  uint64_t splitMask = args.getSplitMask();
  int localRegister = args.getLocalRegister();

  // RMA requires flagcxMemAlloc (GDR memory with SYNC_MEMOPS)
  if (localRegister < 1) {
    fprintf(stderr,
            "test_put requires -R 1 or -R 2 for GDR buffer allocation.\n");
    return 1;
  }

  flagcxHandlerGroup_t handler;
  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  int nGpu;
  devHandle->getDeviceCount(&nGpu);
  devHandle->setDevice(worldRank % nGpu);

  if (proc == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, totalProcs, uniqueId, proc);

  int isHomo = 0;
  flagcxIsHomoComm(comm, &isHomo);
  if (isHomo) {
    if (proc == 0)
      printf("Skipping put benchmark: hetero communicator not initialised "
             "(isHomo=%d).\n",
             isHomo);
    flagcxCommDestroy(comm);
    flagcxHandleFree(handler);
    MPI_Finalize();
    return 0;
  }

  if (totalProcs < 2) {
    if (proc == 0)
      printf("test_put requires at least 2 MPI processes\n");
    MPI_Finalize();
    return 0;
  }

  const int senderRank = 0;
  const int receiverRank = 1;
  if (totalProcs != 2) {
    if (proc == 0)
      printf("test_put requires exactly 2 ranks (sender=0, receiver=1).\n");
    MPI_Finalize();
    return 0;
  }

  bool isSender = (proc == senderRank);
  bool isReceiver = (proc == receiverRank);

  flagcxResult_t res;

  size_t signalBytes = sizeof(uint64_t);
  size_t totalItersPerSize = numWarmupIters + numIters;
  size_t maxDataIters = std::max(numWarmupIters, numIters);
  size_t dataBytes = maxBytes * maxDataIters;
  size_t signalTotalBytes = signalBytes * totalItersPerSize;

  // Data buffer: GDR memory (SYNC_MEMOPS ensures NIC visibility via GDR BAR)
  void *dataWindow = nullptr;
  res = flagcxMemAlloc(&dataWindow, dataBytes);
  if (res != flagcxSuccess || dataWindow == nullptr) {
    fprintf(stderr, "[rank %d] flagcxMemAlloc failed for data (size=%zu)\n",
            proc, dataBytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(dataWindow, 0, dataBytes, flagcxMemDevice, NULL);

  // Signal buffer: GDR memory (SYNC_MEMOPS for RDMA ATOMIC visibility)
  void *signalWindow = nullptr;
  res = flagcxMemAlloc(&signalWindow, signalTotalBytes);
  if (res != flagcxSuccess || signalWindow == nullptr) {
    fprintf(stderr, "[rank %d] flagcxMemAlloc failed for signal (size=%zu)\n",
            proc, signalTotalBytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(signalWindow, 0, signalTotalBytes, flagcxMemDevice,
                          NULL);

  // Register data buffer in global reg pool
  void *dataHandle = nullptr;
  res = flagcxCommRegister(comm, dataWindow, dataBytes, &dataHandle);
  fatal(res, "flagcxCommRegister (data) failed", proc);

  // Register data buffer for one-sided operations
  res = flagcxOneSideRegister(comm, dataWindow, dataBytes);
  if (res == flagcxNotSupported) {
    if (proc == 0)
      printf("Skipping put benchmark: net adaptor does not support iput.\n");
    flagcxCommDeregister(comm, dataHandle);
    flagcxMemFree(dataWindow);
    flagcxMemFree(signalWindow);
    flagcxCommDestroy(comm);
    flagcxHandleFree(handler);
    MPI_Finalize();
    return 0;
  }
  fatal(res, "flagcxOneSideRegister (data) failed", proc);

  // Register signal buffer for one-sided operations
  res = flagcxOneSideSignalRegister(comm, signalWindow, signalTotalBytes,
                                    FLAGCX_PTR_CUDA);
  fatal(res, "flagcxOneSideSignalRegister failed", proc);

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);
  void *dummyBuff = nullptr;
  devHandle->deviceMalloc(&dummyBuff, 1, flagcxMemDevice, NULL);

  // Both sides must call GroupStart/GroupEnd together to ensure synchronization
  flagcxGroupStart(comm);
  if (isSender) {
    flagcxSend(dummyBuff, 1, flagcxChar, receiverRank, comm, stream);
  } else if (isReceiver) {
    flagcxRecv(dummyBuff, 1, flagcxChar, senderRank, comm, stream);
  }
  flagcxGroupEnd(comm);

  // Wait for the connection to be fully established
  devHandle->streamSynchronize(stream);
  devHandle->deviceFree(dummyBuff, flagcxMemDevice, NULL);
  devHandle->streamDestroy(stream);

  // Additional barrier to ensure connection is ready
  MPI_Barrier(MPI_COMM_WORLD);

  // Host staging buffer for sender data fill and receiver verification
  void *hostStaging = nullptr;
  if (posix_memalign(&hostStaging, 64, maxBytes) != 0 ||
      hostStaging == nullptr) {
    fprintf(stderr, "[rank %d] posix_memalign failed for staging (size=%zu)\n",
            proc, maxBytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Create stream for receiver-side wait operations
  flagcxStream_t waitStream = nullptr;
  if (isReceiver) {
    devHandle->streamCreate(&waitStream);
  }

  // Benchmark loop
  timer tim;
  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    if (size == 0)
      break;

    // Reset signal buffer before each size iteration
    devHandle->deviceMemset(signalWindow, 0, signalTotalBytes, flagcxMemDevice,
                            NULL);
    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup iterations (signal slots [0 .. numWarmupIters-1])
    for (int i = 0; i < numWarmupIters; ++i) {
      size_t signalOffset = i * signalBytes;
      size_t currentSendOffset = i * size;
      size_t currentRecvOffset = i * size;

      if (isSender) {
        // Fill host staging, then copy H2D to device data buffer
        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + currentSendOffset,
                                hostStaging, size, flagcxMemcpyHostToDevice,
                                NULL);

        res = flagcxPutSignal(comm, receiverRank, currentSendOffset,
                              currentRecvOffset, size, signalOffset, 0, 0, 1);
        fatal(res, "flagcxPutSignal warmup failed", proc);
      } else if (isReceiver) {
        res = flagcxWaitSignal(comm, senderRank, signalOffset, 1, waitStream);
        fatal(res, "flagcxWaitSignal warmup failed", proc);
        devHandle->streamSynchronize(waitStream);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tim.reset();

    // Benchmark iterations (signal slots [numWarmupIters .. totalIters-1])
    for (int i = 0; i < numIters; ++i) {
      size_t signalOffset = (numWarmupIters + i) * signalBytes;
      size_t currentSendOffset = i * size;
      size_t currentRecvOffset = i * size;

      if (isSender) {
        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + currentSendOffset,
                                hostStaging, size, flagcxMemcpyHostToDevice,
                                NULL);

        res = flagcxPutSignal(comm, receiverRank, currentSendOffset,
                              currentRecvOffset, size, signalOffset, 0, 0, 1);
        fatal(res, "flagcxPutSignal failed", proc);
      } else if (isReceiver) {
        res = flagcxWaitSignal(comm, senderRank, signalOffset, 1, waitStream);
        fatal(res, "flagcxWaitSignal failed", proc);
        devHandle->streamSynchronize(waitStream);

        if (printBuffer) {
          // Copy device data to host for verification
          devHandle->deviceMemcpy(
              hostStaging, (char *)dataWindow + currentRecvOffset,
              std::min(size, (size_t)64), flagcxMemcpyDeviceToHost, NULL);
          printf("[rank %d] Received data at offset %zu, size %zu:\n", proc,
                 currentRecvOffset, size);
          for (size_t j = 0; j < size && j < 64; ++j) {
            printf("%02x ", ((unsigned char *)hostStaging)[j]);
            if ((j + 1) % 16 == 0)
              printf("\n");
          }
          if (size > 64)
            printf("... (truncated)\n");
          else
            printf("\n");
        }
      }
    }

    if (numIters > 0) {
      double elapsedTime = tim.elapsed() / numIters;
      MPI_Allreduce(MPI_IN_PLACE, &elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsedTime /= worldSize;

      double bandwidth = (double)size / 1.0e9 / elapsedTime;
      if (proc == 0 && color == 0) {
        printf("Size: %zu bytes; Avg time: %lf sec; Bandwidth: %lf GB/s\n",
               size, elapsedTime, bandwidth);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);
  res = flagcxCommDeregister(comm, dataHandle);
  fatal(res, "flagcxCommDeregister failed", proc);

  flagcxOneSideSignalDeregister(comm->heteroComm);
  flagcxOneSideDeregister(comm->heteroComm);
  flagcxMemFree(dataWindow);
  flagcxMemFree(signalWindow);
  free(hostStaging);

  if (waitStream != nullptr) {
    devHandle->streamDestroy(waitStream);
  }

  fatal(flagcxCommDestroy(comm), "flagcxCommDestroy failed", proc);
  fatal(flagcxHandleFree(handler), "flagcxHandleFree failed", proc);

  MPI_Finalize();
  return 0;
}
