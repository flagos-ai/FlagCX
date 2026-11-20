#include "device_api/flagcx_device.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "runner_fixtures.hpp"
TEST_F(FlagCXCollTest, FinalizeThenRankLocalDestroy) {
  ASSERT_EQ(devHandle->deviceMemset(sendbuff, 0, size, flagcxMemDevice, stream),
            flagcxSuccess);
  ASSERT_EQ(flagcxAllReduce(sendbuff, recvbuff, count, flagcxFloat, flagcxSum,
                            comm, stream),
            flagcxSuccess);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
  int released = 1;
  if (rank == 0) {
    ASSERT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    comm = nullptr;
    for (int peer = 1; peer < nranks; peer++)
      MPI_Send(&released, 1, MPI_INT, peer, 0x6c10, MPI_COMM_WORLD);
  } else {
    // Rank zero must finish while every peer waits outside FlagCX.
    MPI_Recv(&released, 1, MPI_INT, 0, 0x6c10, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    ASSERT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    comm = nullptr;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
