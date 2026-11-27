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

// Other ranks wait outside FlagCX while each selected rank destroys its Comm.
// All ranks must have completed communicator initialization before entering.
// A null Comm denotes an owner already destroyed successfully on an earlier
// try.
static inline flagcxResult_t flagcxTestFinalizeAndDestroyRankLocally(
    flagcxComm_t *comm, int rank, int nranks, MPI_Comm group = MPI_COMM_WORLD,
    const flagcxTestLifecycleCalls &calls = {}, double maxWaitSeconds = 30.0) {
  double deadline = MPI_Wtime() + maxWaitSeconds;
  flagcxResult_t result =
      flagcxTestFinalizeAcrossRanks(*comm, group, calls, deadline);
  if (result != flagcxSuccess)
    return result;
  for (int selected = 0; selected < nranks; ++selected) {
    int local = (int)flagcxSuccess;
    if (rank == selected)
      local = (int)flagcxTestDestroyLocally(comm, calls, deadline);
    if (MPI_Bcast(&local, 1, MPI_INT, selected, group) != MPI_SUCCESS)
      return flagcxSystemError;
    if (local != (int)flagcxSuccess)
      return (flagcxResult_t)local;
  }
  return flagcxSuccess;
}
