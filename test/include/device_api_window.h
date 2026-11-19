#ifndef FLAGCX_TEST_DEVICE_API_WINDOW_MODE_H_
#define FLAGCX_TEST_DEVICE_API_WINDOW_MODE_H_

#include "flagcx_kernel.h"
#include "tools.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

// Configure before the first FlagCX call, since allocator parameters are
// cached.
static bool deviceApiWindowMode(const parser &args, int *mode) {
  if (args.getLocalRegister() != 2) {
    fprintf(stderr, "Device API tests and benchmarks require -R 2.\n");
    return false;
  }
  const char *name = std::getenv("FLAGCX_TEST_WIN_MODE");
  if (name == nullptr || std::strcmp(name, "asymmetric") == 0) {
    *mode = FLAGCX_WIN_ASYMMETRIC;
    // Ordinary exporter storage is required by legacy IPC. Honor an explicit
    // allocator setting so incompatible user configurations report an error.
    if (std::getenv("FLAGCX_VMM_ENABLE") == nullptr)
      setenv("FLAGCX_VMM_ENABLE", "0", 0);
  } else if (std::strcmp(name, "symmetric") == 0) {
    *mode = FLAGCX_WIN_SYMMETRIC;
  } else {
    fprintf(stderr, "FLAGCX_TEST_WIN_MODE must be symmetric or asymmetric.\n");
    return false;
  }
  return true;
}

static flagcxResult_t deviceApiReportWindowMode(flagcxComm_t comm, int mode) {
  int homo = 0;
  flagcxResult_t result = flagcxIsHomoComm(comm, &homo);
  if (result != flagcxSuccess)
    return result;
  const char *forced = std::getenv("FLAGCX_USE_HETERO_COMM");
  if (forced != nullptr && std::strtol(forced, nullptr, 10) == 1)
    homo = 0;
#if defined(FLAGCX_TEST_ALLOCATOR_SHMEM)
  const char *backend = "shmem";
#elif defined(FLAGCX_COMM_TRAITS_CCL)
  const char *backend = "vendor";
#else
  const char *backend = "default";
#endif
  printf("# backend=%s path=%s window=%s\n", backend, homo ? "homo" : "hetero",
         mode == FLAGCX_WIN_SYMMETRIC ? "symmetric" : "asymmetric");
  return flagcxSuccess;
}

static flagcxResult_t deviceApiCleanupStatus(MPI_Comm group,
                                             flagcxResult_t local) {
  int value = static_cast<int>(local), common = 0;
  if (MPI_Allreduce(&value, &common, 1, MPI_INT, MPI_MAX, group) != MPI_SUCCESS)
    return flagcxRemoteError;
  return static_cast<flagcxResult_t>(common);
}

// Backing storage remains caller-owned. Every rank must close imports before
// any exporter is freed; an error leaves the caller's buffers alive for retry.
static flagcxResult_t
deviceApiCloseWindows(flagcxComm_t comm, MPI_Comm group,
                      flagcxDeviceHandle_t device, flagcxStream_t stream,
                      std::initializer_list<flagcxDevMem_t *> views,
                      std::initializer_list<flagcxWindow_t *> windows,
                      flagcxMemAllocator_t allocator) {
  flagcxResult_t result =
      deviceApiCleanupStatus(group, device->streamSynchronize(stream));
  if (result != flagcxSuccess)
    return result;
  flagcxResult_t local = flagcxSuccess;
  for (auto view : views) {
    flagcxResult_t status = flagcxDevMemDestroy(comm, *view);
    if (status == flagcxSuccess)
      *view = nullptr;
    else if (local == flagcxSuccess)
      local = status;
  }
  for (auto window : windows) {
    flagcxResult_t status =
        flagcxCommWindowDeregister(comm, *window, allocator);
    if (status == flagcxSuccess)
      *window = nullptr;
    else if (local == flagcxSuccess)
      local = status;
  }
  return deviceApiCleanupStatus(group, local);
}

#endif
