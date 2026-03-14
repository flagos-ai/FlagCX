// Kernel P2P demo test (migrated from test/unittest/main.cpp)
#include "runner_fixtures.hpp"
#include "flagcx_kernel.h"
#include <cstring>
#include <iostream>

TEST_F(FlagCXKernelTest, P2pDemo) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  size_t countPerPeer = count / nranks;

  // Initialize sendbuff: all elements = rank (my rank)
  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = (float)rank;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create device communicator for P2P demo
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  // Create raw device memory handles for send/recv buffers
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(NULL, sendbuff, size, NULL, &sendMem),
            flagcxSuccess);
  ASSERT_EQ(flagcxDevMemCreate(NULL, recvbuff, size, NULL, &recvMem),
            flagcxSuccess);

  // Launch AlltoAll kernel demo
  flagcxResult_t result = flagcxInterAlltoAllDemo(
      sendMem, recvMem, countPerPeer, flagcxFloat, devComm, stream);
  devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Destroy raw device memory handles
  flagcxDevMemDestroy(NULL, sendMem);
  flagcxDevMemDestroy(NULL, recvMem);

  // Destroy device communicator
  flagcxDevCommDestroy(comm, devComm);

  // Copy results back
  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: recvbuff[p*countPerPeer] should equal p for all p
  bool success = true;
  for (int p = 0; p < nranks; p++) {
    float expected = (float)p;
    float actual = ((float *)hostrecvbuff)[p * countPerPeer];
    if (actual != expected) {
      success = false;
      if (rank == 0) {
        std::cout << "Mismatch at peer " << p << ": expected " << expected
                  << ", got " << actual << std::endl;
      }
    }
  }
  EXPECT_TRUE(success);
}
