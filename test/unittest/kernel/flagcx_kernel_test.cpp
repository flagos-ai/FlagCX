#include "flagcx_kernel_test.hpp"
#include <cstring>
#include <iostream>

void FlagCXKernelTest::SetUp() {
  FlagCXTest::SetUp();

  // initialize flagcx handles
  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;
  sendbuff = nullptr;
  recvbuff = nullptr;
  hostsendbuff = nullptr;
  hostrecvbuff = nullptr;
  size = 1ULL * 1024 * 1024; // 1MB for kernel test
  count = size / sizeof(float);

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  // Create and broadcast uniqueId
  if (rank == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // Create comm and stream
  flagcxCommInitRank(&comm, nranks, uniqueId, rank);
  devHandle->streamCreate(&stream);

  // allocate device buffers
  devHandle->deviceMalloc(&sendbuff, size, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, size, flagcxMemDevice, NULL);

  // allocate host buffers
  hostsendbuff = malloc(size);
  memset(hostsendbuff, 0, size);
  hostrecvbuff = malloc(size);
  memset(hostrecvbuff, 0, size);
}

void FlagCXKernelTest::TearDown() {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  // Destroy stream first (sync any pending work)
  devHandle->streamDestroy(stream);

  // Destroy comm to stop kernel proxy thread BEFORE freeing device memory
  // The kernel proxy thread holds a CUDA stream that can interfere with
  // deviceFree
  flagcxCommDestroy(comm);

  // free data
  devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
  devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  free(hostsendbuff);
  free(hostrecvbuff);

  // free handles
  flagcxHandleFree(handler);

  FlagCXTest::TearDown();
}

void FlagCXKernelTest::Run() {}
