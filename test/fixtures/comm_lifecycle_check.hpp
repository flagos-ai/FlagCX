#pragma once
#include "flagcx.h"
#include <mpi.h>

// Other ranks wait outside FlagCX while each selected rank destroys its Comm.
static inline flagcxResult_t flagcxTestFinalizeAndDestroyRankLocally(
    flagcxComm_t *comm, int rank, int nranks, MPI_Comm group = MPI_COMM_WORLD) {
  int local = (int)flagcxCommFinalize(*comm), common = 0;
  MPI_Allreduce(&local, &common, 1, MPI_INT, MPI_MAX, group);
  if (common != (int)flagcxSuccess)
    return (flagcxResult_t)common;
  MPI_Barrier(group);
  flagcxResult_t first = flagcxSuccess;
  for (int selected = 0; selected < nranks; ++selected) {
    local = (int)flagcxSuccess;
    if (rank == selected) {
      local = (int)flagcxCommDestroy(*comm);
      if (local == (int)flagcxSuccess)
        *comm = nullptr;
    }
    MPI_Bcast(&local, 1, MPI_INT, selected, group);
    if (first == flagcxSuccess && local != (int)flagcxSuccess)
      first = (flagcxResult_t)local;
  }
  return first;
}
