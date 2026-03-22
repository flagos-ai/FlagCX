#include "flagcx_coll_test.hpp"
#include "flagcx_kernel_test.hpp"
#include "flagcx_topo_test.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

#define BASELINE_FILE "baseline_result.txt"
#define NUM_BASELINE_ENTRIES 1000

TEST_F(FlagCXCollTest, AllReduce) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxAllReduce(sendbuff, recvbuff, count, flagcxFloat, flagcxSum, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  for (size_t i = 0; i < count; i++) {
    ((float *)hostrecvbuff)[i] /= nranks;
  }

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(FlagCXCollTest, AllGather) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                          flagcxMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxAllGather(sendbuff, recvbuff, count / nranks, flagcxFloat, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(FlagCXCollTest, ReduceScatter) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxReduceScatter(sendbuff, recvbuff, count / nranks, flagcxFloat,
                      flagcxSum, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size / nranks,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(FlagCXCollTest, Reduce) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxReduce(sendbuff, recvbuff, count, flagcxFloat, flagcxSum, 0, comm,
               stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(FlagCXCollTest, Gather) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                          flagcxMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff  = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << static_cast<float *>(hostsendbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxGather(sendbuff, recvbuff, count / nranks, flagcxFloat, 0, comm,
               stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "recvbuff  = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << static_cast<float *>(hostrecvbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(FlagCXCollTest, Scatter) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  if (rank == 0) {
    for (size_t i = 0; i < count; i++) {
      ((float *)hostsendbuff)[i] = static_cast<float>(i);
    }

    devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                            flagcxMemcpyHostToDevice, stream);

    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxScatter(sendbuff, recvbuff, count / nranks, flagcxFloat, 0, comm,
                stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size / nranks,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(FlagCXCollTest, Broadcast) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }
  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxBroadcast(sendbuff, recvbuff, count, flagcxFloat, 0, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(FlagCXTopoTest, TopoDetection) {
  flagcxComm_t &comm = handler->comm;
  flagcxUniqueId_t &uniqueId = handler->uniqueId;

  std::cout << "executing flagcxCommInitRank" << std::endl;
  auto result = flagcxCommInitRank(&comm, nranks, uniqueId, rank);
  EXPECT_EQ(result, flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Intra-node AllReduce: each rank fills with (rank+1), verify sum
// ---------------------------------------------------------------------------
TEST_F(FlagCXKernelTest, IntraAllReduce) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  // Allocate a separate buffer for the kernel (aligned with
  // test_kernel_intranode -R 0)
  void *regBuff = nullptr;
  devHandle->deviceMalloc(&regBuff, size, flagcxMemDevice, NULL);

  // Initialize: each rank fills with (rank + 1)
  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = (float)(rank + 1);
  }
  devHandle->deviceMemcpy(regBuff, hostsendbuff, size, flagcxMemcpyHostToDevice,
                          NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create device communicator with intra barriers
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  // Create device memory handle (implicit IPC via flagcxDevMemCreate)
  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, regBuff, size, NULL, &devMem),
            flagcxSuccess);

  // Run AllReduce
  flagcxResult_t result =
      flagcxIntraAllReduce(devMem, count, flagcxFloat, devComm, stream);
  devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Copy results back from regBuff
  devHandle->deviceMemcpy(hostrecvbuff, regBuff, size, flagcxMemcpyDeviceToHost,
                          NULL);

  // Verify: expected = nranks*(nranks+1)/2
  float expected = (float)(nranks * (nranks + 1)) / 2.0f;
  bool success = true;
  for (size_t i = 0; i < count && success; i++) {
    if (fabsf(((float *)hostrecvbuff)[i] - expected) > 1e-3f) {
      success = false;
      if (rank == 0) {
        std::cout << "IntraAllReduce MISMATCH at [" << i << "]: got "
                  << ((float *)hostrecvbuff)[i] << ", expected " << expected
                  << std::endl;
      }
    }
  }
  EXPECT_TRUE(success);

  // Cleanup
  flagcxDevMemDestroy(comm, devMem);
  flagcxDevCommDestroy(comm, devComm);
  devHandle->deviceFree(regBuff, flagcxMemDevice, NULL);
}

// ---------------------------------------------------------------------------
// Inter-node AlltoAll: two-sided send/recv via FIFO
// ---------------------------------------------------------------------------
TEST_F(FlagCXKernelTest, InterTwoSidedAlltoAll) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  // count per peer
  size_t countPerPeer = count / nranks;

  // Initialize sendbuff: all elements = rank (my rank)
  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = (float)rank;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create device communicator
  // Request inter barriers — needed by flagcxInterBarrierSession in the kernel
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  // Create raw device memory handles for send/recv buffers
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, sendbuff, size, NULL, &sendMem),
            flagcxSuccess);
  ASSERT_EQ(flagcxDevMemCreate(comm, recvbuff, size, NULL, &recvMem),
            flagcxSuccess);

  // Launch AlltoAll kernel
  flagcxResult_t result = flagcxInterTwoSidedAlltoAll(
      sendMem, recvMem, countPerPeer, flagcxFloat, devComm, stream);
  devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Destroy raw device memory handles
  flagcxDevMemDestroy(comm, sendMem);
  flagcxDevMemDestroy(comm, recvMem);

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
        std::cout << "InterTwoSidedAlltoAll mismatch at peer " << p
                  << ": expected " << expected << ", got " << actual
                  << std::endl;
      }
    }
  }
  EXPECT_TRUE(success);
}

// ---------------------------------------------------------------------------
// Inter-node one-sided AlltoAll: put + waitSignal + flush
// ---------------------------------------------------------------------------
TEST_F(FlagCXKernelTest, InterOneSidedAlltoAll) {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  size_t countPerPeer = count / nranks;

  // One-sided needs VMM memory + RDMA registration.
  // Allocate separate buffers (fixture's sendbuff/recvbuff stay untouched).
  void *osSend = nullptr, *osRecv = nullptr;
  ASSERT_EQ(flagcxMemAlloc(&osSend, size), flagcxSuccess);
  ASSERT_EQ(flagcxMemAlloc(&osRecv, size), flagcxSuccess);

  void *sendRegHandle = nullptr, *recvRegHandle = nullptr;
  ASSERT_EQ(flagcxCommRegister(comm, osSend, size, &sendRegHandle),
            flagcxSuccess);
  ASSERT_EQ(flagcxCommRegister(comm, osRecv, size, &recvRegHandle),
            flagcxSuccess);

  // Initialize sendbuff: all elements = rank
  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = (float)rank;
  }
  devHandle->deviceMemcpy(osSend, hostsendbuff, size, flagcxMemcpyHostToDevice,
                          NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create device communicator with inter-node barrier + signal
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 1;
  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  // Create device memory handles
  flagcxDevMem_t sendMem = nullptr, recvMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, osSend, size, NULL, &sendMem),
            flagcxSuccess);
  ASSERT_EQ(flagcxDevMemCreate(comm, osRecv, size, NULL, &recvMem),
            flagcxSuccess);

  // Launch one-sided AlltoAll
  flagcxResult_t result = flagcxInterOneSidedAlltoAll(
      sendMem, recvMem, countPerPeer, flagcxFloat, devComm, stream);
  devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, flagcxSuccess);

  // Destroy device memory handles
  flagcxDevMemDestroy(comm, sendMem);
  flagcxDevMemDestroy(comm, recvMem);

  // Destroy device communicator
  flagcxDevCommDestroy(comm, devComm);

  // Copy results back from osRecv
  devHandle->deviceMemcpy(hostrecvbuff, osRecv, size, flagcxMemcpyDeviceToHost,
                          NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: recvbuff[p*countPerPeer] should equal p for all p
  bool success = true;
  for (int p = 0; p < nranks; p++) {
    float expected = (float)p;
    float actual = ((float *)hostrecvbuff)[p * countPerPeer];
    if (actual != expected) {
      success = false;
      if (rank == 0) {
        std::cout << "InterOneSidedAlltoAll mismatch at peer " << p
                  << ": expected " << expected << ", got " << actual
                  << std::endl;
      }
    }
  }
  EXPECT_TRUE(success);

  // Cleanup one-sided buffers
  flagcxCommDeregister(comm, sendRegHandle);
  flagcxCommDeregister(comm, recvRegHandle);
  flagcxMemFree(osSend);
  flagcxMemFree(osRecv);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
