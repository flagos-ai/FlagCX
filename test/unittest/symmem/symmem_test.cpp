#include "symmem_test.hpp"
#include <cstring>

void SymMemTest::SetUp() {
  FlagCXTest::SetUp();

  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  devBuff = nullptr;
  devBuff2 = nullptr;
  hostBuff = nullptr;
  size = SYMMEM_TEST_SIZE;
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

  // Allocate GDR-pinned device buffers (required for one-sided MR registration)
  flagcxMemAlloc(&devBuff, size);
  flagcxMemAlloc(&devBuff2, size);

  // Host buffer for verification
  hostBuff = malloc(size);
  memset(hostBuff, 0, size);
}

void SymMemTest::TearDown() {
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  devHandle->streamDestroy(stream);
  flagcxCommDestroy(comm);

  flagcxMemFree(devBuff);
  flagcxMemFree(devBuff2);
  free(hostBuff);

  flagcxHandleFree(handler);

  FlagCXTest::TearDown();
}

bool SymMemTest::hasHeteroComm() const {
  return handler->comm != nullptr && handler->comm->heteroComm != nullptr;
}
