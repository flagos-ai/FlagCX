#include "rma_test.hpp"
#include "adaptor.h"
#include "comm.h"
#include "flagcx_hetero.h"
#include "global_comm.h"
#include "sym_heap.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

bool allRanksReady(bool localReady) {
  int local = localReady ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global != 0;
}

} // namespace

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
bool RmaTest::signalRmaSetupFailed = false;
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
  signalRmaSetupFailed = false;
  signalRmaSkipReason = "Signal RMA setup not completed";

  int localMode = requireIpc == forceNet ? 0 : (requireIpc ? 1 : 2);
  int minMode = 0;
  int maxMode = 0;
  MPI_Allreduce(&localMode, &minMode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&localMode, &maxMode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (localMode == 0 || minMode != maxMode) {
    dataRmaSkipReason =
        "All ranks must select the same single RMA transport mode";
    return;
  }
  if (rank == 0)
    std::printf("RMA transport under test: %s\n", requireIpc ? "IPC" : "NET");

  flagcxResult_t res = flagcxDeviceHandleInit(&devHandle);
  int numDevices = 0;
  bool localDeviceReady = res == flagcxSuccess && devHandle != nullptr;
  if (localDeviceReady) {
    res = devHandle->getDeviceCount(&numDevices);
    localDeviceReady = res == flagcxSuccess && numDevices > 0;
  }
  if (localDeviceReady) {
    res = devHandle->setDevice(rank % numDevices);
    localDeviceReady = res == flagcxSuccess;
  }
  if (!allRanksReady(localDeviceReady)) {
    dataRmaSkipReason =
        "Device adaptor initialization failed on at least one rank";
    return;
  }

  flagcxUniqueId uniqueId;
  bool localUniqueIdReady = true;
  if (rank == 0) {
    res = flagcxGetUniqueId(&uniqueId);
    localUniqueIdReady = res == flagcxSuccess;
  }
  if (!allRanksReady(localUniqueIdReady)) {
    dataRmaSkipReason = "Unique ID creation failed";
    return;
  }
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);

  res = flagcxCommInitRank(&comm, nranks, &uniqueId, rank);
  bool localCommReady = res == flagcxSuccess && comm != nullptr;
  if (!allRanksReady(localCommReady)) {
    if (!localCommReady)
      comm = nullptr;
    dataRmaSkipReason =
        "Communicator initialization failed on at least one rank";
    return;
  }

  bool localProxyReady =
      comm->heteroComm != nullptr && comm->heteroComm->rmaProxy != nullptr;
  if (!allRanksReady(localProxyReady)) {
    dataRmaSkipReason =
        "Hetero communicator or RMA proxy unavailable on at least one rank";
    return;
  }

  res = devHandle->streamCreate(&stream);
  bool localDataReady = res == flagcxSuccess && stream != nullptr;
  if (localDataReady) {
    res = flagcxMemAlloc(&dataBuff, size);
    localDataReady = res == flagcxSuccess && dataBuff != nullptr;
  }
  if (localDataReady) {
    res = devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
    localDataReady = res == flagcxSuccess;
  }
  if (!allRanksReady(localDataReady)) {
    dataRmaSkipReason =
        "RMA stream or data buffer setup failed on at least one rank";
    return;
  }

  // Register the data buffer only after every rank has allocated it.
  res = flagcxCommWindowRegister(comm, dataBuff, size, &dataWin,
                                 FLAGCX_WIN_COLL_SYMMETRIC);
  bool localWindowReady = res == flagcxSuccess && dataWin != nullptr &&
                          dataWin->isSymmetricDefault &&
                          dataWin->defaultBase != nullptr;
  windowAvailable = allRanksReady(localWindowReady);
  if (!windowAvailable) {
    dataRmaSkipReason =
        "Symmetric window registration failed on at least one rank";
    return;
  }

  const bool localNetworkMrReady = dataWin->defaultBase->mrIndex >= 0;
  const bool allNetworkMrsReady = allRanksReady(localNetworkMrReady);
  const bool localNetworkGetReady =
      comm->heteroComm->netAdaptor != nullptr &&
      comm->heteroComm->netAdaptor->iget != nullptr;
  const bool allNetworkGetsReady = allRanksReady(localNetworkGetReady);
  networkRmaAvailable = allNetworkMrsReady && allNetworkGetsReady;

  if (!requireIpc) {
    if (!allNetworkMrsReady) {
      dataRmaSkipReason =
          "Symmetric window has no registered network MR on at least one rank";
      return;
    }
    if (!allNetworkGetsReady) {
      dataRmaSkipReason =
          "RDMA Get is unavailable on at least one rank's net adaptor";
      return;
    }
    dataRmaAvailable = true;
    dataRmaSkipReason = nullptr;
  } else {
    int peer = nranks == 2 ? 1 - rank : -1;
    bool peerIsLocal =
        peer >= 0 && comm->heteroComm->rankToNode != nullptr &&
        comm->heteroComm->rankToNode[peer] == comm->heteroComm->node;
    void *peerData = nullptr;
    bool localDataIpcReady =
        peerIsLocal && dataWin->defaultBase->ipcSlot >= 0 &&
        flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm, dataWin->defaultBase,
                                         peer, 0, size,
                                         &peerData) == flagcxSuccess &&
        peerData != nullptr;
    ipcRmaAvailable = allRanksReady(localDataIpcReady);
    if (!ipcRmaAvailable) {
      dataRmaSkipReason = "RMA IPC mode requires a resolved peer data mapping";
      return;
    }
    dataRmaAvailable = true;
    dataRmaSkipReason = nullptr;
  }

  bool localSignalCapable =
      deviceAdaptor != nullptr && deviceAdaptor->streamWaitValue64 != nullptr;
  if (requireIpc) {
    localSignalCapable =
        localSignalCapable && deviceAdaptor->streamWriteValue64 != nullptr;
  } else {
    localSignalCapable = localSignalCapable &&
                         comm->heteroComm->netAdaptor != nullptr &&
                         comm->heteroComm->netAdaptor->iputSignal != nullptr;
  }
  const bool allSignalsCapable = allRanksReady(localSignalCapable);

  // Allocate signal buffers before entering collective registration so every
  // rank either participates or exits at the same phase.
  res = flagcxMemAlloc(&signalBuff, signalSize);
  bool localSignalBufferReady = res == flagcxSuccess && signalBuff != nullptr;
  if (localSignalBufferReady) {
    res = devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice,
                                  nullptr);
    localSignalBufferReady = res == flagcxSuccess;
  }
  if (!allRanksReady(localSignalBufferReady)) {
    signalRmaSetupFailed = true;
    signalRmaSkipReason = "Signal buffer setup failed on at least one rank";
    return;
  }

  res = flagcxOneSideSignalRegister(comm, signalBuff, signalSize,
                                    FLAGCX_PTR_CUDA);
  if (!allRanksReady(res == flagcxSuccess)) {
    signalRmaSetupFailed = true;
    signalRmaSkipReason =
        "Signal buffer registration failed on at least one rank";
    return;
  }

  // Verify the exact acquire semantics required by flagcxWaitSignal before a
  // test can submit a network PUT/signal.  The probe waits on a value already
  // stored in local device memory, so it cannot depend on remote progress.
  int localSignalStatus = allSignalsCapable ? 0 : 1;
  if (localSignalStatus == 0) {
    uint64_t probeValue = 1;
    res = devHandle->deviceMemcpy(signalBuff, &probeValue, sizeof(probeValue),
                                  flagcxMemcpyHostToDevice, nullptr);
    if (res == flagcxSuccess) {
      res = deviceAdaptor->streamWaitValue64(
          stream, signalBuff, probeValue,
          FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES);
    }
    if (res == flagcxSuccess)
      res = devHandle->streamSynchronize(stream);

    localSignalStatus =
        res == flagcxSuccess ? 0 : (res == flagcxNotSupported ? 1 : 2);
  }

  int globalSignalStatus = 0;
  MPI_Allreduce(&localSignalStatus, &globalSignalStatus, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  if (globalSignalStatus == 2) {
    signalRmaSetupFailed = true;
    signalRmaSkipReason =
        "Signal stream capability probe failed on at least one rank";
    return;
  }

  res = devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice,
                                nullptr);
  if (!allRanksReady(res == flagcxSuccess)) {
    signalRmaSetupFailed = true;
    signalRmaSkipReason =
        "Signal buffer reset failed after the capability probe";
    return;
  }

  if (!allSignalsCapable) {
    signalRmaSkipReason =
        "One-sided signal operations are unavailable on at least one rank";
  } else if (globalSignalStatus == 1) {
    signalRmaSkipReason =
        "Remote-write visibility flush is unavailable on at least one rank";
  } else {
    signalRmaAvailable = true;
    signalRmaSkipReason = nullptr;
  }

  if (requireIpc && signalRmaAvailable) {
    res = flagcxHeteroRmaIpcInit(comm->heteroComm);
    int peer = nranks == 2 ? 1 - rank : -1;
    struct flagcxRmaIpcState *ipc = comm->heteroComm->rmaProxy->ipcState;
    bool localSignalIpcReady = res == flagcxSuccess && peer >= 0 &&
                               ipc != nullptr && peer < ipc->nRanks &&
                               ipc->peerSignalBufs != nullptr &&
                               ipc->peerSignalBufs[peer] != nullptr;
    if (!allRanksReady(localSignalIpcReady)) {
      signalRmaAvailable = false;
      signalRmaSkipReason =
          "RMA IPC mode requires a resolved peer signal mapping";
    }
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
