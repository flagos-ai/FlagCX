#include "device_api/flagcx_device.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "runner_fixtures.hpp"
#include <thread>

class FlagCXIntraProcessTest : public FlagCXTest {
protected:
  void run(bool explicitFinalize) {
    if (nranks != 1)
      GTEST_SKIP()
          << "Executed separately with one MPI process and two devices";
    flagcxDeviceHandle_t device = nullptr;
    ASSERT_EQ(flagcxDeviceHandleInit(&device), flagcxSuccess);
    int count = 0;
    ASSERT_EQ(device->getDeviceCount(&count), flagcxSuccess);
    ASSERT_GE(count, 2) << "The lifecycle CI invocation requires two devices";
    flagcxUniqueId id;
    ASSERT_EQ(flagcxGetUniqueId(&id), flagcxSuccess);
    flagcxComm_t comms[2] = {};
    void *buffers[2] = {};
    flagcxResult_t results[2] = {};
    auto initialize = [&](int r) {
      results[r] = device->setDevice(r);
      if (results[r] == flagcxSuccess)
        results[r] = flagcxCommInitRank(&comms[r], 2, &id, r);
      if (results[r] == flagcxSuccess)
        results[r] = flagcxCommAllocInternalBuffer(comms[r], &buffers[r], 64,
                                                   flagcxInternalBufferDevice);
    };
    std::thread first(initialize, 0), second(initialize, 1);
    first.join();
    second.join();
    ASSERT_EQ(results[0], flagcxSuccess);
    ASSERT_EQ(results[1], flagcxSuccess);
    if (explicitFinalize) {
      ASSERT_EQ(flagcxGroupStart(nullptr), flagcxSuccess);
      ASSERT_EQ(flagcxCommFinalize(comms[0]), flagcxSuccess);
      ASSERT_EQ(flagcxCommFinalize(comms[1]), flagcxSuccess);
      ASSERT_EQ(flagcxGroupEnd(nullptr), flagcxSuccess);
    }
    // The main thread deliberately has a different device than the first
    // communicator: cleanup must select each owner's device itself.
    ASSERT_EQ(device->setDevice(1), flagcxSuccess);
    ASSERT_EQ(flagcxCommDestroy(comms[0]), flagcxSuccess);
    ASSERT_EQ(flagcxCommDestroy(comms[1]), flagcxSuccess);
    EXPECT_EQ(flagcxDeviceHandleFree(device), flagcxSuccess);
  }
};

TEST_F(FlagCXIntraProcessTest, SequentialImplicitDestroy) { run(false); }
TEST_F(FlagCXIntraProcessTest, GroupedFinalizeThenSequentialDestroy) {
  run(true);
}

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
