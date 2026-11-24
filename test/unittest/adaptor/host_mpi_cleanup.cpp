#include "global_comm.h"
#include "mpi_adaptor.h"
#include <gtest/gtest.h>

TEST(HostMpiCleanup, EmptyAndStagedOwnersDestroyIndependently) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  flagcxInnerComm_t empty = nullptr, a = nullptr, b = nullptr;
  ASSERT_EQ(mpiAdaptor.commInitRank(&empty, size, nullptr, rank, nullptr),
            flagcxSuccess);
  ASSERT_EQ(mpiAdaptor.commFinalize(empty), flagcxSuccess);
  ASSERT_EQ(mpiAdaptor.commFinalize(empty), flagcxSuccess);
  EXPECT_EQ(mpiAdaptor.commDestroy(empty), flagcxSuccess);
  ASSERT_EQ(mpiAdaptor.commInitRank(&a, size, nullptr, rank, nullptr),
            flagcxSuccess);
  ASSERT_EQ(mpiAdaptor.commInitRank(&b, size, nullptr, rank, nullptr),
            flagcxSuccess);
  void *as, *bs, *br;
  ASSERT_EQ(mpiAdaptor.getStagedBuffer(a, &as, 64, 0), flagcxSuccess);
  ASSERT_EQ(mpiAdaptor.getStagedBuffer(b, &bs, 64, 0), flagcxSuccess);
  ASSERT_EQ(mpiAdaptor.getStagedBuffer(b, &br, 64, 1), flagcxSuccess);
  ASSERT_NE(as, bs);
  EXPECT_EQ(mpiAdaptor.commDestroy(a), flagcxSuccess);
  *(int *)bs = rank + 1;
  EXPECT_EQ(mpiAdaptor.allReduce(bs, br, 1, flagcxInt, flagcxSum, b, nullptr),
            flagcxSuccess);
  EXPECT_EQ(*(int *)br, size * (size + 1) / 2);
  EXPECT_EQ(mpiAdaptor.commDestroy(b), flagcxSuccess);
}

TEST(HostMpiCleanup, AutomaticFallbackOwnerDestroysWhilePeersWait) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_GE(size, 2);
  auto *saved = cclAdaptors[flagcxCCLAdaptorHost];
  cclAdaptors[flagcxCCLAdaptorHost] = &mpiAdaptor;
  struct Restore {
    flagcxCCLAdaptor *saved;
    ~Restore() { cclAdaptors[flagcxCCLAdaptorHost] = saved; }
  } restore{saved};
  EXPECT_EQ(getenv("FLAGCX_USE_HOST_COMM"), nullptr);
  flagcxComm_t comm = nullptr;
  ASSERT_EQ(flagcxCalloc(&comm, 1), flagcxSuccess);
  comm->commType = flagcxCommunicatorHybrid;
  comm->rank = rank;
  comm->nranks = size;
  comm->hasSingleRankHomoComm = 1;
  // Recreate the automatic fallback owner's populated state, using the
  // actual MPI adaptor. No accelerator or heterogeneous transport is needed.
  ASSERT_EQ(
      mpiAdaptor.commInitRank(&comm->hostComm, size, nullptr, rank, nullptr),
      flagcxSuccess);
  comm->destroyPhase =
      7; // No IPC owners; test the existing local release path.
  comm->ipcTable = comm->ipcTableInline;
  comm->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
  MPI_Barrier(MPI_COMM_WORLD);
  int released = 1;
  if (rank != 0)
    MPI_Recv(&released, 1, MPI_INT, 0, 0x6b70, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  if (rank == 0)
    for (int peer = 1; peer < size; peer++)
      MPI_Send(&released, 1, MPI_INT, peer, 0x6b70, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  unsetenv("FLAGCX_USE_HOST_COMM");
  unsetenv("FLAGCX_USE_HETERO_COMM");
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
