// MPI test runner entry point.
// Provides main(), MPIEnvironment, and all fixture implementations
// for the coll_*.cpp test files.

#include "runner_fixtures.hpp"
#include <cstring>

// ---------- MPIEnvironment ----------

class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    int argc = 0;
    char **argv = nullptr;
    int mpiError = MPI_Init(&argc, &argv);
    ASSERT_FALSE(mpiError);
  }
  void TearDown() override {
    int mpiError = MPI_Finalize();
    ASSERT_FALSE(mpiError);
  }
};

// ---------- FlagCXTest ----------

void FlagCXTest::SetUp() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
}

// ---------- FlagCXCollTest ----------

void FlagCXCollTest::SetUp() {
  FlagCXTest::SetUp();

  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;
  sendbuff = nullptr;
  recvbuff = nullptr;
  hostsendbuff = nullptr;
  hostrecvbuff = nullptr;
  size = 4ULL * 1024 * 1024; // 4MB
  count = size / sizeof(float);

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  if (rank == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, nranks, uniqueId, rank);
  devHandle->streamCreate(&stream);

  devHandle->deviceMalloc(&sendbuff, size, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, size, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&hostsendbuff, size, flagcxMemHost, NULL);
  devHandle->deviceMemset(hostsendbuff, 0, size, flagcxMemHost, NULL);
  devHandle->deviceMalloc(&hostrecvbuff, size, flagcxMemHost, NULL);
  devHandle->deviceMemset(hostrecvbuff, 0, size, flagcxMemHost, NULL);
}

void FlagCXCollTest::TearDown() {
  flagcxCommDestroy(handler->comm);

  flagcxDeviceHandle_t &devHandle = handler->devHandle;
  devHandle->streamDestroy(stream);
  devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
  devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  devHandle->deviceFree(hostsendbuff, flagcxMemHost, NULL);
  devHandle->deviceFree(hostrecvbuff, flagcxMemHost, NULL);

  flagcxHandleFree(handler);
  FlagCXTest::TearDown();
}

// ---------- main ----------

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
