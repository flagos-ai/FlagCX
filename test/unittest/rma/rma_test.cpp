#include "rma_test.hpp"
#include "comm.h"
#include "sym_heap.h"
#include <cstdio>
#include <cstdlib>
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
bool RmaTest::requireIpc = false;
bool RmaTest::windowAvailable = false;
bool RmaTest::networkRmaAvailable = false;
bool RmaTest::ipcRmaAvailable = false;
bool RmaTest::dataRmaAvailable = false;
const char *RmaTest::dataRmaSkipReason = "Data RMA setup not completed";
bool RmaTest::signalRmaAvailable = false;
const char *RmaTest::signalRmaSkipReason = "Signal RMA setup not completed";

void RmaTest::SetUpTestSuite() {
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size = RMA_TEST_SIZE;
  signalSize = sizeof(uint64_t) * nranks;
  const char *requireIpcEnv = std::getenv("FLAGCX_RMA_TEST_REQUIRE_IPC");
  requireIpc = requireIpcEnv != nullptr && std::strcmp(requireIpcEnv, "0") != 0;
  const char *forceNetEnv = std::getenv("FLAGCX_RMA_FORCE_NET");
  bool forceNet = forceNetEnv != nullptr && std::strcmp(forceNetEnv, "0") != 0;
  windowAvailable = false;
  networkRmaAvailable = false;
  ipcRmaAvailable = false;
  dataRmaAvailable = false;
  dataRmaSkipReason = "Data RMA setup not completed";
  signalRmaAvailable = false;
  signalRmaSkipReason = "Signal RMA setup not completed";

  if (requireIpc == forceNet) {
    dataRmaSkipReason = requireIpc
                            ? "IPC and forced-network RMA modes conflict"
                            : "RMA test invocation did not select IPC or NET";
    return;
  }
  if (rank == 0)
    std::printf("RMA transport under test: %s\n", requireIpc ? "IPC" : "NET");

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
    dataRmaSkipReason = "Communicator initialization failed";
    return;
  }

  if (comm == nullptr || comm->heteroComm == nullptr) {
    dataRmaSkipReason = "Hetero communicator not available";
    return;
  }

  if (comm->heteroComm->rmaProxy == nullptr) {
    dataRmaSkipReason = "RMA proxy not available";
    return;
  }

  devHandle->streamCreate(&stream);

  // Allocate and register data buffer
  flagcxMemAlloc(&dataBuff, size);
  devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);

  res = flagcxCommWindowRegister(comm, dataBuff, size, &dataWin,
                                 FLAGCX_WIN_COLL_SYMMETRIC);
  windowAvailable = res == flagcxSuccess && dataWin != nullptr &&
                    dataWin->isSymmetricDefault &&
                    dataWin->defaultBase != nullptr;
  if (!windowAvailable) {
    dataRmaSkipReason = "Symmetric window registration failed";
    return;
  }

  const int localNetworkMrReady = res == flagcxSuccess && dataWin != nullptr &&
                                  dataWin->isSymmetricDefault &&
                                  dataWin->defaultBase != nullptr &&
                                  dataWin->defaultBase->mrIndex >= 0;
  int allNetworkMrsReady = 0;
  MPI_Allreduce(&localNetworkMrReady, &allNetworkMrsReady, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  networkRmaAvailable = allNetworkMrsReady != 0;

  if (!requireIpc && !networkRmaAvailable) {
    dataRmaSkipReason = "Symmetric window has no registered network MR";
    return;
  }
  if (!requireIpc && (comm->heteroComm->netAdaptor == nullptr ||
                      comm->heteroComm->netAdaptor->iget == nullptr)) {
    dataRmaSkipReason = "Net adaptor does not support RDMA Get";
    return;
  }
  if (!requireIpc) {
    dataRmaAvailable = true;
    dataRmaSkipReason = nullptr;
  }

  if (comm->heteroComm->netAdaptor == nullptr ||
      comm->heteroComm->netAdaptor->iputSignal == nullptr) {
    signalRmaSkipReason = "Net adaptor does not support one-sided signals";
    return;
  }

  // Allocate and register signal buffer
  res = flagcxMemAlloc(&signalBuff, signalSize);
  if (res != flagcxSuccess || signalBuff == nullptr) {
    signalBuff = nullptr;
    signalRmaSkipReason = "Signal buffer allocation is not supported";
    return;
  }
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  res = flagcxOneSideSignalRegister(comm, signalBuff, signalSize,
                                    FLAGCX_PTR_CUDA);
  if (res != flagcxSuccess) {
    flagcxMemFree(signalBuff);
    signalBuff = nullptr;
    signalRmaSkipReason = "Signal buffer registration is not supported";
    return;
  }

  signalRmaAvailable = true;
  signalRmaSkipReason = nullptr;

  if (requireIpc) {
    res = flagcxHeteroRmaIpcInit(comm->heteroComm);
    int localIpcReady = 0;
    int peer = nranks == 2 ? 1 - rank : -1;
    int mrIndex = dataWin->defaultBase->mrIndex;
    struct flagcxRmaIpcState *ipc = comm->heteroComm->rmaProxy->ipcState;
    bool peerIsLocal =
        peer >= 0 && comm->heteroComm->rankToNode != nullptr &&
        comm->heteroComm->rankToNode[peer] == comm->heteroComm->node;
    if (peerIsLocal && res == flagcxSuccess && ipc != nullptr && mrIndex >= 0 &&
        mrIndex < ipc->dataHandleCount && ipc->peerDataBufs != nullptr &&
        ipc->peerDataBufs[peer] != nullptr &&
        ipc->peerDataBufs[peer][mrIndex] != nullptr &&
        ipc->peerSignalBufs != nullptr &&
        ipc->peerSignalBufs[peer] != nullptr) {
      localIpcReady = 1;
    }
    int allIpcReady = 0;
    MPI_Allreduce(&localIpcReady, &allIpcReady, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    ipcRmaAvailable = allIpcReady != 0;
    if (!ipcRmaAvailable) {
      dataRmaSkipReason =
          "RMA IPC mode requires resolved peer data and signal mappings";
      return;
    }
  }

  dataRmaAvailable = requireIpc ? ipcRmaAvailable : networkRmaAvailable;
  dataRmaSkipReason = nullptr;
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
  ASSERT_TRUE(windowAvailable) << "RMA data window is unavailable";
  ASSERT_TRUE(dataRmaAvailable) << dataRmaSkipReason;
  ASSERT_NE(dataWin, nullptr);
}

bool RmaTest::hasHeteroComm() const {
  return comm != nullptr && comm->heteroComm != nullptr;
}
