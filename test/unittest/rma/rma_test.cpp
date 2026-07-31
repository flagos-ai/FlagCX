#include "rma_test.hpp"
#include <cstring>

// Static member definitions
flagcxDeviceHandle_t RmaTest::devHandle = nullptr;
flagcxComm_t RmaTest::comm = nullptr;
flagcxStream_t RmaTest::stream = nullptr;
void *RmaTest::dataBuff = nullptr;
void *RmaTest::signalBuff = nullptr;
flagcxWindow_t RmaTest::dataWin = nullptr;
size_t RmaTest::size = 0;
size_t RmaTest::signalSize = 0;

void RmaTest::SetUpTestSuite() {
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size = RMA_TEST_SIZE;
  signalSize = sizeof(uint64_t) * nranks;

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

  flagcxCommInitRank(&comm, nranks, &uniqueId, rank);

  // Skip setup if hetero comm not available
  if (comm == nullptr || comm->heteroComm == nullptr) {
    return;
  }

  devHandle->streamCreate(&stream);

  // Allocate and register data buffer
  flagcxMemAlloc(&dataBuff, size);
  devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);

  flagcxResult_t res = flagcxCommWindowRegister(comm, dataBuff, size, &dataWin,
                                                FLAGCX_WIN_COLL_SYMMETRIC);
  if (res != flagcxSuccess || dataWin == nullptr) {
    // Net adaptor doesn't support one-sided, tests will skip
    dataWin = nullptr;
    return;
  }

  // Allocate and register signal buffer
  flagcxMemAlloc(&signalBuff, signalSize);
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  res = flagcxOneSideSignalRegister(comm, signalBuff, signalSize,
                                    FLAGCX_PTR_CUDA);
  if (res != flagcxSuccess) {
    flagcxMemFree(signalBuff);
    signalBuff = nullptr;
    dataWin = nullptr;
    return;
  }
}

void RmaTest::TearDownTestSuite() {
  if (devHandle == nullptr)
    return;

  if (dataWin) {
    flagcxCommWindowDeregister(comm, dataWin);
    dataWin = nullptr;
  }

  if (signalBuff && comm && comm->heteroComm) {
    flagcxOneSideSignalDeregister(comm);
    flagcxMemFree(signalBuff);
    signalBuff = nullptr;
  }

  if (dataBuff) {
    flagcxMemFree(dataBuff);
    dataBuff = nullptr;
  }

  if (stream) {
    devHandle->streamDestroy(stream);
    stream = nullptr;
  }

  if (comm) {
    flagcxCommDestroy(comm);
    comm = nullptr;
  }

  flagcxDeviceHandleFree(devHandle);
  devHandle = nullptr;
}

void RmaTest::SetUp() {
  FlagCXTest::SetUp();
  if (comm == nullptr || comm->heteroComm == nullptr) {
    GTEST_SKIP() << "Hetero communicator not available";
  }
  if (dataWin == nullptr) {
    GTEST_SKIP() << "Net adaptor does not support one-sided ops (iput/iget)";
  }
}

bool RmaTest::hasHeteroComm() const {
  return comm != nullptr && comm->heteroComm != nullptr;
}
