#include "symmem_test.hpp"
#include <cstring>

// Static member definitions
flagcxHandlerGroup_t SymMemTest::handler = nullptr;
flagcxComm_t SymMemTest::comm = nullptr;
flagcxStream_t SymMemTest::stream = nullptr;
void *SymMemTest::devBuff = nullptr;
void *SymMemTest::devBuff2 = nullptr;
void *SymMemTest::hostBuff = nullptr;
size_t SymMemTest::size = 0;
size_t SymMemTest::count = 0;

void SymMemTest::SetUpTestSuite() {
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size = SYMMEM_TEST_SIZE;
  count = size / sizeof(float);

  flagcxHandleInit(&handler);
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  flagcxUniqueId_t uniqueId = nullptr;
  if (rank == 0)
    flagcxGetUniqueId(&uniqueId);
  else
    uniqueId = (flagcxUniqueId_t)calloc(1, sizeof(flagcxUniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, nranks, uniqueId, rank);
  free(uniqueId);

  devHandle->streamCreate(&stream);

  flagcxMemAlloc(&devBuff, size);
  flagcxMemAlloc(&devBuff2, size);

  hostBuff = malloc(size);
  memset(hostBuff, 0, size);
}

void SymMemTest::TearDownTestSuite() {
  if (handler == nullptr)
    return;

  handler->devHandle->streamDestroy(stream);

  if (comm)
    flagcxCommDestroy(comm);

  flagcxMemFree(devBuff);
  flagcxMemFree(devBuff2);
  free(hostBuff);

  flagcxHandleFree(handler);

  handler = nullptr;
  comm = nullptr;
  stream = nullptr;
  devBuff = nullptr;
  devBuff2 = nullptr;
  hostBuff = nullptr;
}

void SymMemTest::SetUp() {
  FlagCXTest::SetUp();
  if (comm == nullptr) {
    GTEST_SKIP() << "SetUpTestSuite failed";
  }
}

bool SymMemTest::hasHeteroComm() const {
  return comm != nullptr && comm->heteroComm != nullptr;
}
