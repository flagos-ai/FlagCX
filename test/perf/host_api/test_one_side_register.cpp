#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "onesided.h"
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
                void *dataWindow, flagcxWindow_t dataWin, void *hostStaging,
                size_t size, size_t dataOffset, int proc, int senderRank,
                int receiverRank, flagcxStream_t waitStream,
                uint64_t expectedSignal) {
  flagcxResult_t res;
  if (proc == senderRank) {
    uint8_t fillVal = (uint8_t)(dataOffset & 0xff);
    std::memset(hostStaging, fillVal, size);
    devHandle->deviceMemcpy((char *)dataWindow + dataOffset, hostStaging, size,
                            flagcxMemcpyHostToDevice, NULL);
    res =
        flagcxPutSignal((char *)dataWindow + dataOffset, size, flagcxChar,
                        receiverRank, dataWin, dataOffset, 0, comm, waitStream);
    if (res != flagcxSuccess) {
      fprintf(stderr, "[rank %d] flagcxPutSignal failed (err=%d)\n", proc,
              int(res));
      return false;
    }
  } else {
    flagcxWaitSignalDesc_t desc = {expectedSignal, senderRank};
    res = flagcxWaitSignal(1, &desc, comm, waitStream);
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

  // RMA with window registration requires -R 2
  if (localRegister != 2) {
    fprintf(stderr,
            "test_multi_comm_put requires -R 2 for window-based RMA.\n");
    return 1;
  }

  int worldRank = 0, worldSize = 1, totalProcs = 1, proc = 0, color = 0;
  uint64_t splitMask = args.getSplitMask();
  MPI_Comm splitComm;

  flagcxDeviceHandle_t devHandle;
  flagcxDeviceHandleInit(&devHandle);

  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  if (totalProcs != 2) {
    if (proc == 0)
      printf("test_multi_comm_put requires exactly 2 MPI processes.\n");
    flagcxDeviceHandleFree(devHandle);
    MPI_Finalize();
    return 0;
  }

  int nGpu;
  devHandle->getDeviceCount(&nGpu);
  devHandle->setDevice(worldRank % nGpu);

  const int senderRank = 0;
  const int receiverRank = 1;

  flagcxUniqueId uid1;
  if (proc == 0)
    flagcxGetUniqueId(&uid1);
  MPI_Bcast((void *)&uid1, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxComm_t comm1 = NULL;
  fatal(flagcxCommInitRank(&comm1, totalProcs, &uid1, proc),
        "flagcxCommInitRank (comm1) failed", proc);

  flagcxUniqueId uid2;
  if (proc == 0)
    flagcxGetUniqueId(&uid2);
  MPI_Bcast((void *)&uid2, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxComm_t comm2 = NULL;
  fatal(flagcxCommInitRank(&comm2, totalProcs, &uid2, proc),
        "flagcxCommInitRank (comm2) failed", proc);

  if (proc == 0)
    printf("[test] comm1=%p  comm2=%p\n", (void *)comm1, (void *)comm2);

  size_t signalBytes = sizeof(uint64_t);
  size_t totalIters = (size_t)(numWarmup + numIters);
  size_t dataBytes = maxBytes * std::max(numWarmup, numIters);
  size_t signalTotalBytes = signalBytes * totalIters;

  void *dataBuf1 = nullptr, *sigBuf1 = nullptr;
  fatal(flagcxMemAlloc(&dataBuf1, dataBytes), "MemAlloc dataBuf1", proc);
  fatal(flagcxMemAlloc(&sigBuf1, signalTotalBytes), "MemAlloc sigBuf1", proc);
  devHandle->deviceMemset(dataBuf1, 0, dataBytes, flagcxMemDevice, NULL);
  devHandle->deviceMemset(sigBuf1, 0, signalTotalBytes, flagcxMemDevice, NULL);

  void *dataBuf2 = nullptr, *sigBuf2 = nullptr;
  fatal(flagcxMemAlloc(&dataBuf2, dataBytes), "MemAlloc dataBuf2", proc);
  fatal(flagcxMemAlloc(&sigBuf2, signalTotalBytes), "MemAlloc sigBuf2", proc);
  devHandle->deviceMemset(dataBuf2, 0, dataBytes, flagcxMemDevice, NULL);
  devHandle->deviceMemset(sigBuf2, 0, signalTotalBytes, flagcxMemDevice, NULL);

  flagcxWindow_t dataWin1 = nullptr;
  flagcxWindow_t dataWin2 = nullptr;

  flagcxResult_t r1, r2;

  r1 = flagcxCommWindowRegister(comm1, dataBuf1, dataBytes, &dataWin1,
                                FLAGCX_WIN_COLL_SYMMETRIC);
  if (r1 == flagcxNotSupported || dataWin1 == nullptr) {
    if (proc == 0)
      printf("[SKIP] flagcxCommWindowRegister returned NotSupported; "
             "set FLAGCX_USE_HETERO_COMM=1 and ensure IB net adaptor.\n");
    goto cleanup_no_onesided;
  }
  fatal(r1, "CommWindowRegister (comm1)", proc);

  r2 = flagcxCommWindowRegister(comm2, dataBuf2, dataBytes, &dataWin2,
                                FLAGCX_WIN_COLL_SYMMETRIC);
  if (r2 == flagcxNotSupported || dataWin2 == nullptr) {
    if (proc == 0)
      printf("[SKIP] comm2 CommWindowRegister NotSupported.\n");
    flagcxCommWindowDeregister(comm1, dataWin1);
    goto cleanup_no_onesided;
  }
  fatal(r2, "CommWindowRegister (comm2)", proc);

  fatal(flagcxOneSideSignalRegister(comm1, sigBuf1, signalTotalBytes,
                                    FLAGCX_PTR_CUDA),
        "SignalRegister (comm1)", proc);
  fatal(flagcxOneSideSignalRegister(comm2, sigBuf2, signalTotalBytes,
                                    FLAGCX_PTR_CUDA),
        "SignalRegister (comm2)", proc);

  if (proc == 0)
    printf("[test] CommWindowRegister for comm1 and comm2 succeeded.\n");

  warmupConnect(comm1, devHandle, proc, totalProcs);
  MPI_Barrier(MPI_COMM_WORLD);
  warmupConnect(comm2, devHandle, proc, totalProcs);
  MPI_Barrier(MPI_COMM_WORLD);

  {
    flagcxStream_t waitStream = nullptr;
    devHandle->streamCreate(&waitStream);

    void *hostStaging = nullptr;
    fatal((flagcxResult_t)(posix_memalign(&hostStaging, 64, maxBytes) == 0
                               ? flagcxSuccess
                               : flagcxInternalError),
          "posix_memalign", proc);

    if (proc == 0)
      printf("\n[Phase 1] put via comm1\n");
    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      devHandle->deviceMemset(sigBuf1, 0, signalTotalBytes, flagcxMemDevice,
                              NULL);
      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < numWarmup + numIters; i++) {
        size_t dataOffset = (size_t)(i % std::max(numWarmup, numIters)) * size;

        bool ok = doPutRound(comm1, devHandle, dataBuf1, dataWin1, hostStaging,
                             size, dataOffset, proc, senderRank, receiverRank,
                             waitStream, (uint64_t)(i + 1));
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
      devHandle->deviceMemset(sigBuf2, 0, signalTotalBytes, flagcxMemDevice,
                              NULL);
      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < numWarmup + numIters; i++) {
        size_t dataOffset = (size_t)(i % std::max(numWarmup, numIters)) * size;

        bool ok = doPutRound(comm2, devHandle, dataBuf2, dataWin2, hostStaging,
                             size, dataOffset, proc, senderRank, receiverRank,
                             waitStream, (uint64_t)(i + 1));
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

    flagcxCommWindowDeregister(comm1, dataWin1);
    flagcxOneSideSignalDeregister(comm1);
    dataWin1 = nullptr;
    MPI_Barrier(MPI_COMM_WORLD);

    if (proc == 0)
      printf("[Phase 3] comm1 deregistered.\n");

    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      devHandle->deviceMemset(sigBuf2, 0, signalTotalBytes, flagcxMemDevice,
                              NULL);
      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < numWarmup + numIters; i++) {
        size_t dataOffset = (size_t)(i % std::max(numWarmup, numIters)) * size;

        bool ok = doPutRound(comm2, devHandle, dataBuf2, dataWin2, hostStaging,
                             size, dataOffset, proc, senderRank, receiverRank,
                             waitStream, (uint64_t)(i + 1));
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
    devHandle->streamDestroy(waitStream);

    flagcxCommWindowDeregister(comm2, dataWin2);
    flagcxOneSideSignalDeregister(comm2);
    dataWin2 = nullptr;
  }

cleanup_no_onesided:
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  flagcxMemFree(dataBuf1);
  flagcxMemFree(dataBuf2);
  flagcxMemFree(sigBuf1);
  flagcxMemFree(sigBuf2);

  flagcxCommDestroy(comm1);
  flagcxCommDestroy(comm2);
  flagcxDeviceHandleFree(devHandle);

  MPI_Finalize();
  return 0;
}