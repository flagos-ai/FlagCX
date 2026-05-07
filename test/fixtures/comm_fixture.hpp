#pragma once

#include "flagcx.h"
#include "mpi_environment.hpp"
#include <gtest/gtest.h>

// MPI + FlagCX communicator fixture with configurable buffer size.
// Default buffer size is 4KB (suitable for fast unit iteration).
// Override COMM_FIXTURE_SIZE to use a different size.
#ifndef COMM_FIXTURE_SIZE
#define COMM_FIXTURE_SIZE (4ULL * 1024)
#endif

class CommFixture : public ::testing::Test {
protected:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    flagcxCommInitRank(&handler->comm, nranks, uniqueId, rank);
    devHandle->streamCreate(&stream);

    size = COMM_FIXTURE_SIZE;
    count = size / sizeof(float);

    devHandle->deviceMalloc(&sendbuff, size, flagcxMemDevice, NULL);
    devHandle->deviceMalloc(&recvbuff, size, flagcxMemDevice, NULL);
    devHandle->deviceMalloc(&hostsendbuff, size, flagcxMemHost, NULL);
    devHandle->deviceMemset(hostsendbuff, 0, size, flagcxMemHost, NULL);
    devHandle->deviceMalloc(&hostrecvbuff, size, flagcxMemHost, NULL);
    devHandle->deviceMemset(hostrecvbuff, 0, size, flagcxMemHost, NULL);
  }

  void TearDown() override {
    flagcxDeviceHandle_t &devHandle = handler->devHandle;

    flagcxCommDestroy(handler->comm);
    devHandle->streamDestroy(stream);

    devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
    devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
    devHandle->deviceFree(hostsendbuff, flagcxMemHost, NULL);
    devHandle->deviceFree(hostrecvbuff, flagcxMemHost, NULL);

    flagcxHandleFree(handler);
  }

  int rank;
  int nranks;
  flagcxHandlerGroup_t handler;
  flagcxStream_t stream;
  void *sendbuff;
  void *recvbuff;
  void *hostsendbuff;
  void *hostrecvbuff;
  size_t size;
  size_t count;
};
