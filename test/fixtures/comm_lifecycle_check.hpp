#pragma once
#include "flagcx.h"
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <thread>

// Test diagnostics only: no device operations, completion events or threads.
// Log the first entry and each status change, rather than every Progress poll.
struct flagcxTestLifecyclePhase {
  const char *name;
  int previous = -1;

  void trace(const char *edge, flagcxResult_t status) {
    int initialized = 0, rank = -1;
    MPI_Initialized(&initialized);
    if (initialized)
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *test = std::getenv("FLAGCX_CI_LIFECYCLE_CASE");
    std::fprintf(stderr,
                 "[lifecycle] rank=%d case=%s phase=%s edge=%s result=%d\n",
                 rank, test ? test : "unspecified", name, edge, (int)status);
    std::fflush(stderr);
  }

  void enter() {
    if (previous == -1)
      trace("enter", flagcxSuccess);
  }

  void returned(flagcxResult_t status) {
    if (previous != (int)status)
      trace("return", status);
    previous = (int)status;
  }
};

static inline flagcxResult_t
flagcxTestSynchronizeStream(flagcxDeviceHandle_t device,
                            flagcxStream_t stream) {
  flagcxTestLifecyclePhase phase{"stream"};
  phase.enter();
  flagcxResult_t result = device->streamSynchronize(stream);
  phase.returned(result);
  return result;
}

struct flagcxTestLifecycleCalls {
  flagcxResult_t (*finalize)(flagcxComm_t) = flagcxCommFinalize;
  flagcxResult_t (*destroy)(flagcxComm_t) = flagcxCommDestroy;
};

// Reduce errors separately: numeric InProgress (7) must not mask errors (1..6).
static inline flagcxResult_t
flagcxTestAgreeLifecycleStatus(flagcxResult_t local, MPI_Comm group) {
  int input[2] = {local != flagcxSuccess && local != flagcxInProgress ? local
                                                                      : 0,
                  local == flagcxInProgress};
  int common[2] = {};
  if (MPI_Allreduce(input, common, 2, MPI_INT, MPI_MAX, group) != MPI_SUCCESS)
    return flagcxSystemError;
  return common[0]   ? (flagcxResult_t)common[0]
         : common[1] ? flagcxInProgress
                     : flagcxSuccess;
}

static inline flagcxResult_t
flagcxTestFinalizeAcrossRanks(flagcxComm_t comm, MPI_Comm group,
                              const flagcxTestLifecycleCalls &calls,
                              double deadline) {
  bool complete = comm == nullptr;
  flagcxTestLifecyclePhase operation{"finalize"}, agreement{"finalize-agree"};
  while (true) {
    operation.enter();
    flagcxResult_t local = complete ? flagcxSuccess : calls.finalize(comm);
    operation.returned(local);
    if (local == flagcxSuccess)
      complete = true;
    if (local == flagcxInProgress && MPI_Wtime() >= deadline) {
      std::fprintf(stderr, "Finalize retry deadline expired; retaining Comm\n");
      local = flagcxSystemError;
    }
    agreement.enter();
    flagcxResult_t common = flagcxTestAgreeLifecycleStatus(local, group);
    agreement.returned(common);
    if (common != flagcxInProgress)
      return common;
    std::this_thread::yield();
  }
}

static inline flagcxResult_t
flagcxTestDestroyLocally(flagcxComm_t *comm,
                         const flagcxTestLifecycleCalls &calls,
                         double deadline) {
  flagcxTestLifecyclePhase operation{"destroy"};
  while (*comm != nullptr) {
    operation.enter();
    flagcxResult_t result = calls.destroy(*comm);
    operation.returned(result);
    if (result == flagcxSuccess) {
      *comm = nullptr;
      return result;
    }
    if (result != flagcxInProgress)
      return result;
    if (MPI_Wtime() >= deadline) {
      std::fprintf(stderr, "Destroy retry deadline expired; retaining Comm\n");
      return flagcxSystemError;
    }
    std::this_thread::yield();
  }
  return flagcxSuccess;
}

// Native Destroy can require peers even after successful Finalize. Every rank
// enters each retry round; completed owners stay null and are never consumed
// twice. A blocking SDK call still requires the outer process watchdog.
static inline flagcxResult_t
flagcxTestDestroyAcrossRanks(flagcxComm_t *comm, MPI_Comm group,
                             const flagcxTestLifecycleCalls &calls,
                             double deadline) {
  flagcxTestLifecyclePhase operation{"destroy"}, agreement{"destroy-agree"};
  while (true) {
    operation.enter();
    flagcxResult_t local = *comm ? calls.destroy(*comm) : flagcxSuccess;
    operation.returned(local);
    if (local == flagcxSuccess)
      *comm = nullptr;
    if (local == flagcxInProgress && MPI_Wtime() >= deadline) {
      std::fprintf(stderr, "Destroy retry deadline expired; retaining Comm\n");
      local = flagcxSystemError;
    }
    agreement.enter();
    flagcxResult_t common = flagcxTestAgreeLifecycleStatus(local, group);
    agreement.returned(common);
    if (common != flagcxInProgress)
      return common;
    std::this_thread::yield();
  }
}

static inline flagcxResult_t flagcxTestFinalizeAndDestroyAcrossRanks(
    flagcxComm_t *comm, MPI_Comm group = MPI_COMM_WORLD,
    const flagcxTestLifecycleCalls &calls = {}, double maxWaitSeconds = 30.0) {
  double deadline = MPI_Wtime() + maxWaitSeconds;
  flagcxResult_t result =
      flagcxTestFinalizeAcrossRanks(*comm, group, calls, deadline);
  if (result != flagcxSuccess)
    return result;
  return flagcxTestDestroyAcrossRanks(comm, group, calls, deadline);
}
