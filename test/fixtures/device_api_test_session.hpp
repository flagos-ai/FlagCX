#pragma once

#include "comm_lifecycle_check.hpp"
#include "flagcx_kernel.h"
#include <cstdlib>
#include <initializer_list>

struct DeviceApiTestFailure {
  flagcxResult_t status;
};

// Test-owned handles survive every failed cleanup attempt. The test body uses
// references to these slots, so an exception never loses an accepted owner.
struct DeviceApiTestSession {
  MPI_Comm group = MPI_COMM_WORLD;
  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxComm_t comm = nullptr;
  flagcxStream_t stream = nullptr;
  flagcxDevComm_t devComm = nullptr, interOnlyComm = nullptr;
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr, devMem = nullptr,
                 tailMem = nullptr;
  flagcxWindow_t sendWin = nullptr, recvWin = nullptr, win = nullptr,
                 tailWin = nullptr;
  flagcxMemAllocator_t memAllocator = flagcxMemCCL;
  void *sendBuff = nullptr, *recvBuff = nullptr, *regBuff = nullptr,
       *tailBuff = nullptr, *sendHandle = nullptr, *recvHandle = nullptr;
  uint64_t *dResultBuf = nullptr;
  int *devResults = nullptr;
  float *devOutput = nullptr, *hostSend = nullptr, *hostRecv = nullptr,
        *hostBuf = nullptr;
  void *hostBuff = nullptr;
  bool initialized = false;
  int phase = 0;

  void check(flagcxResult_t result) {
    int mpi = 0;
    MPI_Initialized(&mpi);
    if (mpi)
      result = flagcxTestAgreeLifecycleStatus(result, MPI_COMM_WORLD);
    if (result != flagcxSuccess) {
      std::fprintf(stderr, "Device API test operation failed: result=%d\n",
                   (int)result);
      throw DeviceApiTestFailure{result};
    }
  }

  void checkInit(flagcxResult_t result) {
    check(result);
    initialized = true;
  }

  template <typename T>
  flagcxResult_t freeDevice(T *&pointer) {
    if (pointer == nullptr)
      return flagcxSuccess;
    flagcxResult_t result =
        devHandle->deviceFree(pointer, flagcxMemDevice, nullptr);
    if (result == flagcxSuccess)
      pointer = nullptr;
    return result;
  }

  flagcxResult_t freeBuffer(void *&pointer, flagcxMemAllocator_t allocator) {
    if (pointer == nullptr)
      return flagcxSuccess;
    flagcxResult_t result = flagcxMemFree(pointer, allocator);
    if (result == flagcxSuccess)
      pointer = nullptr;
    return result;
  }

  flagcxResult_t closePhase() {
    switch (phase) {
      case 0:
        return stream ? devHandle->streamSynchronize(stream) : flagcxSuccess;
      case 1: {
        for (auto *view : {&sendMem, &recvMem, &devMem, &tailMem}) {
          if (*view == nullptr)
            continue;
          flagcxResult_t result = flagcxDevMemDestroy(comm, *view);
          if (result != flagcxSuccess)
            return result;
          *view = nullptr;
        }
        for (auto *window : {&sendWin, &recvWin, &win, &tailWin}) {
          if (*window == nullptr)
            continue;
          flagcxResult_t result = flagcxCommWindowDeregister(
              comm, *window,
              window == &tailWin ? flagcxMemSHMEM : memAllocator);
          if (result != flagcxSuccess)
            return result;
          *window = nullptr;
        }
        for (auto *handle : {&sendHandle, &recvHandle}) {
          if (*handle == nullptr)
            continue;
          flagcxResult_t result =
              flagcxCommDeregister(comm, *handle, memAllocator);
          if (result != flagcxSuccess)
            return result;
          *handle = nullptr;
        }
        return flagcxSuccess;
      }
      case 2: {
        // Phase1 is agreed by all importers before any caller storage is freed.
        for (auto *buffer : {&sendBuff, &recvBuff, &regBuff, &tailBuff}) {
          flagcxResult_t result = freeBuffer(
              *buffer, buffer == &tailBuff ? flagcxMemSHMEM : memAllocator);
          if (result != flagcxSuccess)
            return result;
        }
        flagcxResult_t result = freeDevice(dResultBuf);
        if (result == flagcxSuccess)
          result = freeDevice(devResults);
        if (result == flagcxSuccess)
          result = freeDevice(devOutput);
        return result;
      }
      case 3:
        for (auto *view : {&devComm, &interOnlyComm}) {
          if (*view == nullptr)
            continue;
          flagcxResult_t result = flagcxDevCommDestroy(comm, *view);
          if (result != flagcxSuccess)
            return result;
          *view = nullptr;
        }
        return flagcxSuccess;
      case 4:
        if (comm != nullptr && initialized) {
          flagcxTestLifecycleCalls calls;
          return flagcxTestFinalizeAcrossRanks(comm, group, calls,
                                               MPI_Wtime() + 30.0);
        }
        return flagcxSuccess;
      case 5: {
        if (comm == nullptr)
          return flagcxSuccess;
        // Failed initialization retains a local rollback owner; it must not
        // enter normal collective Finalize on a partially initialized Comm.
        flagcxTestLifecycleCalls calls;
        return flagcxTestDestroyLocally(&comm, calls, MPI_Wtime() + 30.0);
      }
      case 6:
        if (stream != nullptr) {
          flagcxResult_t result = devHandle->streamDestroy(stream);
          if (result != flagcxSuccess)
            return result;
          stream = nullptr;
        }
        return flagcxSuccess;
      case 7:
        if (devHandle != nullptr) {
          flagcxResult_t result = flagcxDeviceHandleFree(devHandle);
          if (result != flagcxSuccess)
            return result;
          devHandle = nullptr;
        }
        return flagcxSuccess;
      default:
        return flagcxSuccess;
    }
  }

  flagcxResult_t close() {
    int mpi = 0;
    MPI_Initialized(&mpi);
    while (phase < 8) {
      flagcxResult_t result = closePhase();
      if (mpi)
        result = flagcxTestAgreeLifecycleStatus(result, group);
      if (result != flagcxSuccess)
        return result;
      ++phase;
    }
    delete[] hostSend;
    delete[] hostRecv;
    delete[] hostBuf;
    std::free(hostBuff);
    hostSend = hostRecv = hostBuf = nullptr;
    hostBuff = nullptr;
    return flagcxSuccess;
  }

  flagcxResult_t finish() {
    int mpi = 0;
    MPI_Initialized(&mpi);
    double deadline = mpi ? MPI_Wtime() + 30.0 : 0;
    flagcxResult_t result = close();
    while (result == flagcxInProgress && mpi && MPI_Wtime() < deadline) {
      std::this_thread::yield();
      result = close();
    }
    return result;
  }
};

static int deviceApiTestMain(int argc, char **argv,
                             int (*run)(int, char **, DeviceApiTestSession &)) {
  DeviceApiTestSession resources;
  int status = 0;
  try {
    status = run(argc, argv, resources);
  } catch (const DeviceApiTestFailure &error) {
    status = (int)error.status;
  }
  int mpi = 0;
  MPI_Initialized(&mpi);
  flagcxResult_t cleanup = resources.finish();
  if (cleanup != flagcxSuccess) {
    std::fprintf(stderr,
                 "Device API test cleanup phase=%d result=%d; "
                 "owners retained, original status=%d\n",
                 resources.phase, (int)cleanup, status);
    // A failed hardware test must not let SDK shutdown unmap storage while
    // retained workers still access it. Cleanup itself remains retryable.
    if (mpi)
      MPI_Abort(MPI_COMM_WORLD, status ? status : (int)cleanup);
    return status ? status : (int)cleanup;
  }
  if (mpi) {
    if (resources.group != MPI_COMM_WORLD && resources.group != MPI_COMM_NULL)
      MPI_Comm_free(&resources.group);
    MPI_Finalize();
  }
  return status;
}
