#pragma once
#include "flagcx.h"
#include <mpi.h>

// Every nonzero rank waits outside FlagCX until rank zero's local Destroy has
// returned. A hidden bootstrap/IPC rendezvous in Destroy therefore times out.
static inline flagcxResult_t flagcxTestFinalizeAndDestroyRankLocally(
    flagcxComm_t *comm, int rank, int nranks, MPI_Comm group = MPI_COMM_WORLD) {
  int local = (int)flagcxCommFinalize(*comm), common = 0;
  MPI_Allreduce(&local, &common, 1, MPI_INT, MPI_MAX, group);
  if (common != (int)flagcxSuccess)
    return (flagcxResult_t)common;
  MPI_Barrier(group);
  int first = 0;
  if (rank == 0) {
    local = (int)flagcxCommDestroy(*comm);
    if (local == (int)flagcxSuccess)
      *comm = NULL;
    for (int peer = 1; peer < nranks; peer++)
      MPI_Send(&local, 1, MPI_INT, peer, 0x6c11, group);
  } else {
    MPI_Recv(&first, 1, MPI_INT, 0, 0x6c11, group, MPI_STATUS_IGNORE);
    local = (int)flagcxCommDestroy(*comm);
    if (local == (int)flagcxSuccess)
      *comm = NULL;
    if (local == (int)flagcxSuccess)
      local = first;
  }
  MPI_Allreduce(&local, &common, 1, MPI_INT, MPI_MAX, group);
  return (flagcxResult_t)common;
}
