// MPI test runner entry point.
// Provides main(), MPIEnvironment, and all fixture implementations
// for the coll_*.cpp test files.

#include "adaptor.h"
#include "runner_fixtures.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

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

  flagcxDeviceHandleInit(&devHandle);
  size = 4ULL * 1024 * 1024; // 4MB
  count = size / sizeof(float);
  const char *registerEnv = std::getenv("FLAGCX_TEST_REGISTER_BUFFERS");
  useRegisteredBuffers =
      registerEnv != nullptr && std::strcmp(registerEnv, "0") != 0;

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  flagcxUniqueId uniqueId;
  if (rank == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, nranks, &uniqueId, rank);
  devHandle->streamCreate(&stream);

  if (useRegisteredBuffers) {
    ASSERT_NE(deviceAdaptor, nullptr);
    ASSERT_NE(deviceAdaptor->gdrMemAlloc, nullptr);
    ASSERT_NE(deviceAdaptor->gdrMemFree, nullptr);
    ASSERT_EQ(deviceAdaptor->gdrMemAlloc(&sendbuff, size, nullptr),
              flagcxSuccess);
    ASSERT_EQ(deviceAdaptor->gdrMemAlloc(&recvbuff, size, nullptr),
              flagcxSuccess);
    // These tests need global registration-pool entries for the P2P/NET
    // transports. A communicator-bound registration also creates optional
    // one-sided full-mesh state, which is unrelated to runner collectives and
    // can block during fixture setup.
    ASSERT_EQ(flagcxCommRegister(nullptr, sendbuff, size, &sendRegHandle),
              flagcxSuccess);
    ASSERT_EQ(flagcxCommRegister(nullptr, recvbuff, size, &recvRegHandle),
              flagcxSuccess);
  } else {
    ASSERT_EQ(devHandle->deviceMalloc(&sendbuff, size, flagcxMemDevice, NULL),
              flagcxSuccess);
    ASSERT_EQ(devHandle->deviceMalloc(&recvbuff, size, flagcxMemDevice, NULL),
              flagcxSuccess);
  }
  ASSERT_EQ(devHandle->deviceMalloc(&hostsendbuff, size, flagcxMemHost, NULL),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemset(hostsendbuff, 0, size, flagcxMemHost, NULL),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMalloc(&hostrecvbuff, size, flagcxMemHost, NULL),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemset(hostrecvbuff, 0, size, flagcxMemHost, NULL),
            flagcxSuccess);

  if (rank == 0) {
    std::cout << "Runner buffer mode: "
              << (useRegisteredBuffers ? "registered" : "unregistered")
              << std::endl;
  }
}

void FlagCXCollTest::TearDown() {
  if (devHandle != nullptr && stream != nullptr)
    devHandle->streamSynchronize(stream);

  if (comm != nullptr) {
    // Communicator teardown releases its transport-specific MR handles while
    // the proxy is alive. The global registration entries are removed below.
    EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    comm = nullptr;
  }

  if (useRegisteredBuffers) {
    if (sendRegHandle != nullptr) {
      EXPECT_EQ(flagcxCommDeregister(nullptr, sendRegHandle), flagcxSuccess);
      sendRegHandle = nullptr;
    }
    if (recvRegHandle != nullptr) {
      EXPECT_EQ(flagcxCommDeregister(nullptr, recvRegHandle), flagcxSuccess);
      recvRegHandle = nullptr;
    }
  }

  // Transport registrations may retain these allocations until communicator
  // teardown, so free the underlying memory only after flagcxCommDestroy.
  if (useRegisteredBuffers && deviceAdaptor != nullptr &&
      deviceAdaptor->gdrMemFree != nullptr) {
    if (sendbuff != nullptr)
      EXPECT_EQ(deviceAdaptor->gdrMemFree(sendbuff, nullptr), flagcxSuccess);
    if (recvbuff != nullptr)
      EXPECT_EQ(deviceAdaptor->gdrMemFree(recvbuff, nullptr), flagcxSuccess);
  } else if (devHandle != nullptr) {
    if (sendbuff != nullptr)
      devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
    if (recvbuff != nullptr)
      devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  }

  if (devHandle != nullptr) {
    if (stream != nullptr)
      devHandle->streamDestroy(stream);
    if (hostsendbuff != nullptr)
      devHandle->deviceFree(hostsendbuff, flagcxMemHost, NULL);
    if (hostrecvbuff != nullptr)
      devHandle->deviceFree(hostrecvbuff, flagcxMemHost, NULL);
    flagcxDeviceHandleFree(devHandle);
  }
  FlagCXTest::TearDown();

  // Synchronize all ranks before the next test to prevent bootstrap hangs
  MPI_Barrier(MPI_COMM_WORLD);
}

// ---------- main ----------

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
