// MPI test runner entry point.
// Provides main(), MPIEnvironment, and all fixture implementations
// for the coll_*.cpp test files.

#include "comm.h"
#include "global_comm.h"
#include "runner_fixtures.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

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
  sendbuff = nullptr;
  recvbuff = nullptr;
  hostsendbuff = nullptr;
  hostrecvbuff = nullptr;
  stream = nullptr;
  size = 4ULL * 1024 * 1024; // 4MB
  count = size / sizeof(float);

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  flagcxUniqueId uniqueId;
  if (rank == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t commInitResult =
      flagcxCommInitRank(&comm, nranks, &uniqueId, rank);
  int localCommReady = commInitResult == flagcxSuccess && comm != nullptr;
  int allCommsReady = 0;
  MPI_Allreduce(&localCommReady, &allCommsReady, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  ASSERT_EQ(allCommsReady, 1)
      << "flagcxCommInitRank failed on at least one rank; local result="
      << commInitResult;

  // Forced-NET CI invocations must prove that they selected the intended
  // hardware adaptor. FLAGCX_P2P_DISABLE only disables the IPC transport; it
  // does not prevent flagcxNetInit() from falling back to Socket when RDMA is
  // unavailable. Converge this check across ranks before any collective so a
  // mixed or unexpected selection fails quickly instead of hanging later.
  const char *expectedNetAdaptor = std::getenv("FLAGCX_CI_EXPECT_NET_ADAPTOR");
  if (expectedNetAdaptor != nullptr && expectedNetAdaptor[0] != '\0') {
    const char *actualNetAdaptor = nullptr;
    if (comm->heteroComm != nullptr &&
        comm->heteroComm->netAdaptor != nullptr) {
      actualNetAdaptor = comm->heteroComm->netAdaptor->name;
    }
    int localAdaptorMatches =
        actualNetAdaptor != nullptr &&
        std::strcmp(actualNetAdaptor, expectedNetAdaptor) == 0;
    int allAdaptorsMatch = 0;
    MPI_Allreduce(&localAdaptorMatches, &allAdaptorsMatch, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    ASSERT_EQ(allAdaptorsMatch, 1)
        << "Forced-NET runner expected adaptor " << expectedNetAdaptor
        << " on every rank, but rank " << rank << " selected "
        << (actualNetAdaptor != nullptr ? actualNetAdaptor : "<none>");
  }

  devHandle->streamCreate(&stream);

  devHandle->deviceMalloc(&sendbuff, size, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, size, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&hostsendbuff, size, flagcxMemHost, NULL);
  devHandle->deviceMemset(hostsendbuff, 0, size, flagcxMemHost, NULL);
  devHandle->deviceMalloc(&hostrecvbuff, size, flagcxMemHost, NULL);
  devHandle->deviceMemset(hostrecvbuff, 0, size, flagcxMemHost, NULL);
}

void FlagCXCollTest::TearDown() {
  if (teardownDelayMs > 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(teardownDelayMs));

  // Collective work is asynchronous with respect to the host.  Drain the
  // stream before communicator teardown so no runtime callback can retain an
  // IPC mapping after the communicator starts releasing transport resources.
  if (devHandle != nullptr && stream != nullptr)
    EXPECT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);

  if (comm != nullptr) {
    EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    comm = nullptr;
  }

  if (devHandle != nullptr) {
    if (stream != nullptr)
      EXPECT_EQ(devHandle->streamDestroy(stream), flagcxSuccess);
    if (sendbuff != nullptr)
      EXPECT_EQ(devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL),
                flagcxSuccess);
    if (recvbuff != nullptr)
      EXPECT_EQ(devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL),
                flagcxSuccess);
    if (hostsendbuff != nullptr)
      EXPECT_EQ(devHandle->deviceFree(hostsendbuff, flagcxMemHost, NULL),
                flagcxSuccess);
    if (hostrecvbuff != nullptr)
      EXPECT_EQ(devHandle->deviceFree(hostrecvbuff, flagcxMemHost, NULL),
                flagcxSuccess);

    EXPECT_EQ(flagcxDeviceHandleFree(devHandle), flagcxSuccess);
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
