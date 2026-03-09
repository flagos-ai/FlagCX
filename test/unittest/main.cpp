#include "flagcx_coll_test.hpp"
#include "flagcx_kernel_test.hpp"
#include "flagcx_topo_test.hpp"
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

TEST_F(FlagCXKernelTest, P2pDemo) {
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

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
