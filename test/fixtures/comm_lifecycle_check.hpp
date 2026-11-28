#pragma once
#include "flagcx.h"
#include <cstdio>
#include <mpi.h>
#include <thread>

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
  while (true) {
    flagcxResult_t local = complete ? flagcxSuccess : calls.finalize(comm);
    if (local == flagcxSuccess)
      complete = true;
    if (local == flagcxInProgress && MPI_Wtime() >= deadline) {
      std::fprintf(stderr, "Finalize retry deadline expired; retaining Comm\n");
      local = flagcxSystemError;
    }
    flagcxResult_t common = flagcxTestAgreeLifecycleStatus(local, group);
    if (common != flagcxInProgress)
      return common;
    std::this_thread::yield();
  }
}

static inline flagcxResult_t
flagcxTestDestroyLocally(flagcxComm_t *comm,
                         const flagcxTestLifecycleCalls &calls,
                         double deadline) {
  while (*comm != nullptr) {
    flagcxResult_t result = calls.destroy(*comm);
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
  while (true) {
    flagcxResult_t local = *comm ? calls.destroy(*comm) : flagcxSuccess;
    if (local == flagcxSuccess)
      *comm = nullptr;
    if (local == flagcxInProgress && MPI_Wtime() >= deadline) {
      std::fprintf(stderr, "Destroy retry deadline expired; retaining Comm\n");
      local = flagcxSystemError;
    }
    flagcxResult_t common = flagcxTestAgreeLifecycleStatus(local, group);
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
