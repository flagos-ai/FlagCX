#pragma once
#include "flagcx.h"
#include <mpi.h>

// Every other rank waits outside FlagCX until the selected rank's Destroy
// returns. Exercise every rank, including after earlier peers freed their Comm.
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
        *comm = NULL;
    }
    MPI_Bcast(&local, 1, MPI_INT, selected, group);
    if (first == flagcxSuccess && local != (int)flagcxSuccess)
      first = (flagcxResult_t)local;
  }
  return first;
}
