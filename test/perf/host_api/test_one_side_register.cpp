#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "tools.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>

namespace {

void fatal(flagcxResult_t res, const char *msg, int rank) {
  if (res != flagcxSuccess) {
    fprintf(stderr, "[rank %d] FATAL: %s (err=%d)\n", rank, msg, int(res));
    MPI_Abort(MPI_COMM_WORLD, res);
  }
}

void warmupConnect(flagcxComm_t comm, flagcxDeviceHandle_t devHandle, int proc,
                   int totalProcs) {
  flagcxStream_t stream;
  devHandle->streamCreate(&stream);
  void *dummyBuff = nullptr;
  devHandle->deviceMalloc(&dummyBuff, 1, flagcxMemDevice, NULL);

  flagcxGroupStart(comm);
  flagcxSend(dummyBuff, 1, flagcxChar, (proc + 1) % totalProcs, comm, stream);
  flagcxRecv(dummyBuff, 1, flagcxChar, (proc - 1 + totalProcs) % totalProcs,
             comm, stream);
  flagcxGroupEnd(comm);

  devHandle->streamSynchronize(stream);
  devHandle->deviceFree(dummyBuff, flagcxMemDevice, NULL);
  devHandle->streamDestroy(stream);
}

bool doPutRound(flagcxComm_t comm, flagcxDeviceHandle_t devHandle,
                void *dataWindow, void *signalWindow, void *hostStaging,
                size_t size, size_t signalOffset, size_t dataOffset,
                uint64_t signalValue, int proc, int senderRank,
                int receiverRank, flagcxStream_t waitStream) {
  flagcxResult_t res;
  if (proc == senderRank) {
    uint8_t fillVal = (uint8_t)(signalValue & 0xff);
    std::memset(hostStaging, fillVal, size);
    devHandle->deviceMemcpy((char *)dataWindow + dataOffset, hostStaging, size,
                            flagcxMemcpyHostToDevice, NULL);
    res = flagcxPutSignal(comm, receiverRank, dataOffset, dataOffset, size,
                          signalOffset, 0, 0, signalValue);
    if (res != flagcxSuccess) {
      fprintf(stderr, "[rank %d] flagcxPutSignal failed (err=%d)\n", proc,
              int(res));
      return false;
    }
  } else {
    res = flagcxWaitSignal(comm, senderRank, signalOffset, signalValue,
                           waitStream);
    if (res != flagcxSuccess) {
      fprintf(stderr, "[rank %d] flagcxWaitSignal failed (err=%d)\n", proc,
              int(res));
      return false;
    }
    devHandle->streamSynchronize(waitStream);
  }
  return true;
}

} // namespace

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmup = args.getWarmupIters();
  int numIters = args.getTestIters();
  int localRegister = args.getLocalRegister();

  if (localRegister < 1) {
    fprintf(stderr,
            "test_multi_comm_put requires -R 1 or -R 2 for GDR allocation.\n");
    return 1;
  }

  int worldRank = 0, worldSize = 1, totalProcs = 1, proc = 0, color = 0;
  uint64_t splitMask = args.getSplitMask();
  MPI_Comm splitComm;

  flagcxHandlerGroup_t handler;
  flagcxHandleInit(&handler);
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  if (totalProcs != 2) {
    if (proc == 0)
      printf("test_multi_comm_put requires exactly 2 MPI processes.\n");
    flagcxHandleFree(handler);
    MPI_Finalize();
    return 0;
  }

  int nGpu;
  devHandle->getDeviceCount(&nGpu);
  devHandle->setDevice(worldRank % nGpu);

  const int senderRank = 0;
  const int receiverRank = 1;

  flagcxUniqueId_t uid1 = NULL;
  flagcxCalloc(&uid1, 1);
  if (proc == 0)
    flagcxGetUniqueId(&uid1);
  MPI_Bcast((void *)uid1, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxComm_t comm1 = NULL;
  fatal(flagcxCommInitRank(&comm1, totalProcs, uid1, proc),
        "flagcxCommInitRank (comm1) failed", proc);

  flagcxUniqueId_t uid2 = NULL;
  flagcxCalloc(&uid2, 1);
  if (proc == 0)
    flagcxGetUniqueId(&uid2);
  MPI_Bcast((void *)uid2, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxComm_t comm2 = NULL;
  fatal(flagcxCommInitRank(&comm2, totalProcs, uid2, proc),
        "flagcxCommInitRank (comm2) failed", proc);

  free(uid1);
  free(uid2);

  if (proc == 0)
    printf("[test] comm1=%p  comm2=%p\n", (void *)comm1, (void *)comm2);

  size_t signalBytes = sizeof(uint64_t);
  size_t totalIters = (size_t)(numWarmup + numIters);
  size_t dataBytes = maxBytes * std::max(numWarmup, numIters);
  size_t signalTotalBytes = signalBytes * totalIters;

  void *dataWin1 = nullptr, *sigWin1 = nullptr;
  fatal(flagcxMemAlloc(&dataWin1, dataBytes), "MemAlloc dataWin1", proc);
  fatal(flagcxMemAlloc(&sigWin1, signalTotalBytes), "MemAlloc sigWin1", proc);
  devHandle->deviceMemset(dataWin1, 0, dataBytes, flagcxMemDevice, NULL);
  devHandle->deviceMemset(sigWin1, 0, signalTotalBytes, flagcxMemDevice, NULL);

  void *dataWin2 = nullptr, *sigWin2 = nullptr;
  fatal(flagcxMemAlloc(&dataWin2, dataBytes), "MemAlloc dataWin2", proc);
  fatal(flagcxMemAlloc(&sigWin2, signalTotalBytes), "MemAlloc sigWin2", proc);
  devHandle->deviceMemset(dataWin2, 0, dataBytes, flagcxMemDevice, NULL);
  devHandle->deviceMemset(sigWin2, 0, signalTotalBytes, flagcxMemDevice, NULL);

  flagcxResult_t r1 = flagcxOneSideRegister(comm1, dataWin1, dataBytes);
  if (r1 == flagcxNotSupported) {
    if (proc == 0)
      printf("[SKIP] flagcxOneSideRegister returned NotSupported; "
             "set FLAGCX_USE_HETERO_COMM=1 and ensure IB net adaptor.\n");
    goto cleanup_no_onesided;
  }
  fatal(r1, "OneSideRegister (comm1)", proc);

  {
    flagcxResult_t r2 = flagcxOneSideRegister(comm2, dataWin2, dataBytes);
    if (r2 == flagcxNotSupported) {
      if (proc == 0)
        printf("[SKIP] comm2 OneSideRegister NotSupported.\n");
      flagcxOneSideDeregister(comm1->heteroComm);
      goto cleanup_no_onesided;
    }
    fatal(r2, "OneSideRegister (comm2)", proc);
  }

  fatal(flagcxOneSideSignalRegister(comm1, sigWin1, signalTotalBytes,
                                    FLAGCX_PTR_CUDA),
        "SignalRegister (comm1)", proc);
  fatal(flagcxOneSideSignalRegister(comm2, sigWin2, signalTotalBytes,
                                    FLAGCX_PTR_CUDA),
        "SignalRegister (comm2)", proc);

  if (proc == 0)
    printf("[test] OneSideRegister for comm1 and comm2 succeeded.\n");

  warmupConnect(comm1, devHandle, proc, totalProcs);
  MPI_Barrier(MPI_COMM_WORLD);
  warmupConnect(comm2, devHandle, proc, totalProcs);
  MPI_Barrier(MPI_COMM_WORLD);

  {
    flagcxStream_t waitStream = nullptr;
    if (proc == receiverRank)
      devHandle->streamCreate(&waitStream);

    void *hostStaging = nullptr;
    fatal((flagcxResult_t)(posix_memalign(&hostStaging, 64, maxBytes) == 0
                               ? flagcxSuccess
                               : flagcxInternalError),
          "posix_memalign", proc);

    if (proc == 0)
      printf("\n[Phase 1] put via comm1\n");
    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      devHandle->deviceMemset(sigWin1, 0, signalTotalBytes, flagcxMemDevice,
                              NULL);
      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < numWarmup + numIters; i++) {
        size_t signalOffset = (size_t)i * signalBytes;
        size_t dataOffset = (size_t)(i % std::max(numWarmup, numIters)) * size;
        uint64_t sigVal = (uint64_t)(i + 1);

        bool ok = doPutRound(comm1, devHandle, dataWin1, sigWin1, hostStaging,
                             size, signalOffset, dataOffset, sigVal, proc,
                             senderRank, receiverRank, waitStream);
        if (!ok) {
          fprintf(stderr, "[rank %d] Phase 1 FAILED at iter %d size %zu\n",
                  proc, i, size);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (proc == 0)
        printf("[Phase 1] size=%zu PASS\n", size);
    }

    if (proc == 0)
      printf("\n[Phase 2] put via comm2\n");
    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      devHandle->deviceMemset(sigWin2, 0, signalTotalBytes, flagcxMemDevice,
                              NULL);
      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < numWarmup + numIters; i++) {
        size_t signalOffset = (size_t)i * signalBytes;
        size_t dataOffset = (size_t)(i % std::max(numWarmup, numIters)) * size;
        uint64_t sigVal = (uint64_t)(i + 1);

        bool ok = doPutRound(comm2, devHandle, dataWin2, sigWin2, hostStaging,
                             size, signalOffset, dataOffset, sigVal, proc,
                             senderRank, receiverRank, waitStream);
        if (!ok) {
          fprintf(stderr, "[rank %d] Phase 2 FAILED at iter %d size %zu\n",
                  proc, i, size);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (proc == 0)
        printf("[Phase 2] size=%zu PASS\n", size);
    }

    if (proc == 0)
      printf("\n[Phase 3] deregister comm1, then put via comm2\n");

    flagcxOneSideDeregister(comm1->heteroComm);
    flagcxOneSideSignalDeregister(comm1->heteroComm);
    MPI_Barrier(MPI_COMM_WORLD);

    if (proc == 0)
      printf("[Phase 3] comm1 deregistered.\n");

    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      devHandle->deviceMemset(sigWin2, 0, signalTotalBytes, flagcxMemDevice,
                              NULL);
      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < numWarmup + numIters; i++) {
        size_t signalOffset = (size_t)i * signalBytes;
        size_t dataOffset = (size_t)(i % std::max(numWarmup, numIters)) * size;
        uint64_t sigVal = (uint64_t)(i + 1);

        bool ok = doPutRound(comm2, devHandle, dataWin2, sigWin2, hostStaging,
                             size, signalOffset, dataOffset, sigVal, proc,
                             senderRank, receiverRank, waitStream);
        if (!ok) {
          fprintf(stderr,
                  "[rank %d] Phase 3 FAILED at iter %d size %zu "
                  "(comm1 deregister broke comm2!)\n",
                  proc, i, size);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (proc == 0)
        printf(
            "[Phase 3] size=%zu PASS (comm2 unaffected by comm1 deregister)\n",
            size);
    }

    if (proc == 0)
      printf("\n[test_multi_comm_put] ALL PHASES PASSED\n");

    free(hostStaging);
    if (waitStream)
      devHandle->streamDestroy(waitStream);

    flagcxOneSideDeregister(comm2->heteroComm);
    flagcxOneSideSignalDeregister(comm2->heteroComm);
  }

cleanup_no_onesided:
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  flagcxMemFree(dataWin1);
  flagcxMemFree(dataWin2);
  flagcxMemFree(sigWin1);
  flagcxMemFree(sigWin2);

  flagcxCommDestroy(comm1);
  flagcxCommDestroy(comm2);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}