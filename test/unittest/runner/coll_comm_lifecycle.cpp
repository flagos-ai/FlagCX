#include "device_api/flagcx_device.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "runner_fixtures.hpp"
#include <cstdio>
TEST_F(FlagCXCollTest, FinalizeThenRankLocalDestroy) {
  ASSERT_EQ(devHandle->deviceMemset(sendbuff, 0, size, flagcxMemDevice, stream),
            flagcxSuccess);
  // AllGather is supported in the platform modes used by this lifecycle test.
  // Keep the existing AllReduce conformance test responsible for that
  // operation.
  fprintf(stderr, "LifecycleTest rank=%d allgather submit begin\n", rank);
  ASSERT_EQ(flagcxAllGather(sendbuff, recvbuff, count / nranks, flagcxFloat,
                            comm, stream),
            flagcxSuccess);
  fprintf(stderr, "LifecycleTest rank=%d synchronize begin\n", rank);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
  fprintf(stderr, "LifecycleTest rank=%d finalize begin\n", rank);
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  fprintf(stderr, "LifecycleTest rank=%d finalize done; MPI barrier begin\n",
          rank);
  MPI_Barrier(MPI_COMM_WORLD);
  fprintf(stderr, "LifecycleTest rank=%d MPI barrier done\n", rank);
  int released = 1;
  if (rank == 0) {
    fprintf(stderr, "LifecycleTest rank=%d destroy begin\n", rank);
    ASSERT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    fprintf(stderr, "LifecycleTest rank=%d destroy done\n", rank);
    comm = nullptr;
    for (int peer = 1; peer < nranks; peer++)
      MPI_Send(&released, 1, MPI_INT, peer, 0x6c10, MPI_COMM_WORLD);
  } else {
    // Rank zero must finish while every peer waits outside FlagCX.
    MPI_Recv(&released, 1, MPI_INT, 0, 0x6c10, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    fprintf(stderr, "LifecycleTest rank=%d destroy begin\n", rank);
    ASSERT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    fprintf(stderr, "LifecycleTest rank=%d destroy done\n", rank);
    comm = nullptr;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
