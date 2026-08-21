#include "rma_test.hpp"
#include "comm.h"
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
bool RmaTest::oneSidedAvailable = false;
const char *RmaTest::oneSidedSkipReason = "RMA one-sided setup not completed";

void RmaTest::SetUpTestSuite() {
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size = RMA_TEST_SIZE;
  signalSize = sizeof(uint64_t) * nranks;
  oneSidedAvailable = false;
  oneSidedSkipReason = "RMA one-sided setup not completed";

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

  flagcxResult_t res = flagcxCommInitRank(&comm, nranks, &uniqueId, rank);
  if (res != flagcxSuccess) {
    comm = nullptr;
    oneSidedSkipReason = "Communicator initialization failed";
    return;
  }

  // Skip setup if hetero comm not available
  if (comm == nullptr || comm->heteroComm == nullptr) {
    oneSidedSkipReason = "Hetero communicator not available";
    return;
  }

  if (comm->heteroComm->rmaProxy == nullptr) {
    oneSidedSkipReason = "RMA proxy not available";
    return;
  }

  if (comm->heteroComm->netAdaptor == nullptr ||
      comm->heteroComm->netAdaptor->iput == nullptr ||
      comm->heteroComm->netAdaptor->iget == nullptr ||
      comm->heteroComm->netAdaptor->iputSignal == nullptr) {
    oneSidedSkipReason = "Net adaptor does not support one-sided RMA";
    return;
  }

  devHandle->streamCreate(&stream);

  // Allocate and register data buffer
  flagcxMemAlloc(&dataBuff, size);
  devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);

  res = flagcxCommWindowRegister(comm, dataBuff, size, &dataWin,
                                 FLAGCX_WIN_COLL_SYMMETRIC);
  if (res != flagcxSuccess || dataWin == nullptr) {
    // Net adaptor doesn't support one-sided, tests will skip
    oneSidedSkipReason = "Net adaptor does not support one-sided ops";
    return;
  }

  // Allocate and register signal buffer
  res = flagcxMemAlloc(&signalBuff, signalSize);
  if (res != flagcxSuccess || signalBuff == nullptr) {
    signalBuff = nullptr;
    oneSidedSkipReason = "Signal buffer allocation is not supported";
    return;
  }
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  res = flagcxOneSideSignalRegister(comm, signalBuff, signalSize,
                                    FLAGCX_PTR_CUDA);
  if (res != flagcxSuccess) {
    flagcxMemFree(signalBuff);
    signalBuff = nullptr;
    oneSidedSkipReason = "Signal buffer registration is not supported";
    return;
  }

  oneSidedAvailable = true;
  oneSidedSkipReason = nullptr;
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
  }
  if (signalBuff) {
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
  if (!oneSidedAvailable) {
    GTEST_SKIP() << oneSidedSkipReason;
  }
  if (dataWin == nullptr) {
    GTEST_SKIP() << "Net adaptor does not support one-sided ops (iput/iget)";
  }
}

bool RmaTest::hasHeteroComm() const {
  return comm != nullptr && comm->heteroComm != nullptr;
}
