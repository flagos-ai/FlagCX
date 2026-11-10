// MPI test runner entry point.
// Provides main(), MPIEnvironment, and all fixture implementations
// for the coll_*.cpp test files.

#include "comm.h"
#include "global_comm.h"
#include "runner_fixtures.hpp"
#include <cstdlib>
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

// A transport failure is communicator-local in the FlagCX runtime, so a user
// may legitimately create another communicator after destroying the failed
// one. This test process is different: continuing with the next GTest case
// after a permanent hardware error only repeats the same route failure and can
// consume the job timeout. Every rank sets this flag from a collective error
// convergence before a later fixture decides to skip initialization.
static bool runnerTransportFailureObserved = false;

void FlagCXCollTest::SetUp() {
  FlagCXTest::SetUp();

  sendbuff = nullptr;
  recvbuff = nullptr;
  hostsendbuff = nullptr;
  hostrecvbuff = nullptr;
  stream = nullptr;
  size = 4ULL * 1024 * 1024; // 4MB
  const char *runnerBytes = std::getenv("FLAGCX_CI_RUNNER_BYTES");
  if (runnerBytes != nullptr && runnerBytes[0] != '\0') {
    const unsigned long long configured =
        std::strtoull(runnerBytes, nullptr, 0);
    if (configured != 0)
      size = static_cast<size_t>(configured);
  }
  count = size / sizeof(float);

  if (runnerTransportFailureObserved) {
    GTEST_SKIP() << "A previous runner test observed a permanent asynchronous "
                    "transport error";
  }

  flagcxDeviceHandleInit(&devHandle);

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

  // Each CI invocation declares the runner mode it intends to exercise.  Do
  // not let an ignored environment variable or a topology fallback turn a
  // heterogeneous run into another homogeneous pass (or vice versa).
  const char *expectedRunnerMode = std::getenv("FLAGCX_CI_EXPECT_RUNNER_MODE");
  if (expectedRunnerMode != nullptr && expectedRunnerMode[0] != '\0') {
    int localModeMatches = 0;
    if (std::strcmp(expectedRunnerMode, "HOMO") == 0) {
      localModeMatches = comm->commType == flagcxCommunicatorHomo;
    } else if (std::strcmp(expectedRunnerMode, "HYBRID") == 0) {
      localModeMatches = comm->commType == flagcxCommunicatorHybrid;
    }
    int allModesMatch = 0;
    MPI_Allreduce(&localModeMatches, &allModesMatch, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    ASSERT_EQ(allModesMatch, 1)
        << "Runner expected communicator mode " << expectedRunnerMode
        << " on every rank, but rank " << rank << " initialized mode "
        << static_cast<int>(comm->commType);
  }

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

flagcxResult_t FlagCXCollTest::synchronizeAndCheckAsyncError() {
  flagcxResult_t localError = devHandle->streamSynchronize(stream);
  flagcxResult_t asyncError = flagcxSuccess;
  if (localError == flagcxSuccess) {
    flagcxResult_t queryResult = flagcxCommGetAsyncError(comm, &asyncError);
    localError = queryResult != flagcxSuccess ? queryResult : asyncError;
  }

  int localErrorCode = static_cast<int>(localError);
  int globalErrorCode = static_cast<int>(flagcxSuccess);
  MPI_Allreduce(&localErrorCode, &globalErrorCode, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (globalErrorCode != static_cast<int>(flagcxSuccess)) {
    runnerTransportFailureObserved = true;
    return static_cast<flagcxResult_t>(globalErrorCode);
  }
  return flagcxSuccess;
}

void FlagCXCollTest::TearDown() {
  // Collective work is asynchronous with respect to the host.  Drain the
  // stream before communicator teardown so no runtime callback can retain an
  // IPC mapping after the communicator starts releasing transport resources.
  if (devHandle != nullptr && stream != nullptr) {
    EXPECT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);
  }

  if (comm != nullptr) {
    EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    comm = nullptr;
  }

  if (devHandle != nullptr) {
    if (stream != nullptr) {
      EXPECT_EQ(devHandle->streamDestroy(stream), flagcxSuccess);
    }
    if (sendbuff != nullptr) {
      EXPECT_EQ(devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL),
                flagcxSuccess);
    }
    if (recvbuff != nullptr) {
      EXPECT_EQ(devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL),
                flagcxSuccess);
    }
    if (hostsendbuff != nullptr) {
      EXPECT_EQ(devHandle->deviceFree(hostsendbuff, flagcxMemHost, NULL),
                flagcxSuccess);
    }
    if (hostrecvbuff != nullptr) {
      EXPECT_EQ(devHandle->deviceFree(hostrecvbuff, flagcxMemHost, NULL),
                flagcxSuccess);
    }

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
