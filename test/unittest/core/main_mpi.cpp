// MPI test runner entry point for core tests.
// Provides main(), MPIEnvironment, FlagCXTest::SetUp(), and FlagCXTopoTest
// fixture.

#include "core_fixtures.hpp"
#include <cstring>

void FlagCXTest::SetUp() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
}

// ---------- FlagCXTopoTest ----------

void FlagCXTopoTest::SetUp() {
  FlagCXTest::SetUp();

  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  if (rank == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

void FlagCXTopoTest::TearDown() {
  if (handler->comm) {
    flagcxCommDestroy(handler->comm);
  }
  flagcxHandleFree(handler);
}

// ---------- main ----------

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
