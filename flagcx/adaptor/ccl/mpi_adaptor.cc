#include "mpi_adaptor.h"
#include <functional>

#ifdef USE_MPI_ADAPTOR

static flagcxResult_t validateComm(flagcxInnerComm_t comm) {
  if (!comm || !comm->base) {
    return flagcxInvalidArgument;
  }

  if (!comm->base->isValidContext()) {
    printf("Error: Invalid MPI context: %s\n",
           comm->base->getLastError().c_str());
    return flagcxInternalError;
  }

  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorGetVersion(int *version) {
  int subversion;
  int result = MPI_Get_version(version, &subversion);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  return flagcxSuccess;
}

const char *mpiAdaptorGetErrorString(flagcxResult_t result) {
  switch (result) {
    case flagcxSuccess:
      return "MPI Success";
    case flagcxInternalError:
      return "MPI Internal Error";
    default:
      return "MPI Unknown Error";
  }
}

const char *mpiAdaptorGetLastError(flagcxInnerComm_t comm) {

  return "MPI: No Error";
}

flagcxResult_t mpiAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                             void *buff, size_t size,
                                             void **win, int flags) {
  return flagcxNotSupported;
}
flagcxResult_t mpiAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               void *win) {
  return flagcxNotSupported;
}
flagcxResult_t mpiAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                         void **buff, size_t size,
                                         int isRecv) {
  return flagcxNotSupported;
}

flagcxResult_t mpiAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                      flagcxUniqueId_t /*commId*/, int rank,
                                      bootstrapState *bootstrap) {
  int initialized;
  MPI_Initialized(&initialized);

  if (!initialized) {
    int provided;
    int result = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (result != MPI_SUCCESS) {
      return flagcxInternalError;
    }

    if (provided < MPI_THREAD_SERIALIZED) {
      printf("Warning: MPI does not support required thread level\n");
    }
  }

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // validate parameters if bootstrap is provided
  if (bootstrap != nullptr) {
    if (rank != mpi_rank || nranks != mpi_size) {
      printf("Warning: Expected rank/size (%d/%d) differs from MPI (%d/%d), "
             "using MPI values\n",
             rank, nranks, mpi_rank, mpi_size);
    }
  }

  if (*comm == NULL) {
    FLAGCXCHECK(flagcxCalloc(comm, 1));
  }

  // use actual MPI rank and size to create context
  (*comm)->base =
      std::make_shared<flagcxMpiContext>(mpi_rank, mpi_size, bootstrap);

  // check if context is created successfully
  if (!(*comm)->base || !(*comm)->base->isValidContext()) {
    printf("Error: Failed to create MPI context: %s\n",
           (*comm)->base ? (*comm)->base->getLastError().c_str()
                         : "Unknown error");
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorCommFinalize(flagcxInnerComm_t comm) {
  comm->base.reset();
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorCommDestroy(flagcxInnerComm_t comm) {
  comm->base.reset();
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorCommAbort(flagcxInnerComm_t comm) {
  MPI_Abort(comm->base->getMpiComm(), 1);
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t mpiAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t mpiAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  *count = comm->base->getSize();
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                      int *device) {
  *device = -1;
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorCommUserRank(const flagcxInnerComm_t comm, int *rank) {
  *rank = comm->base->getRank();
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                           flagcxResult_t *asyncError) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t mpiAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t mpiAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t mpiAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                      size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t mpiAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t mpiAdaptorReduce(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                flagcxRedOp_t op, int root,
                                flagcxInnerComm_t comm,
                                flagcxStream_t /*stream*/) {
  int result;
  MPI_Op mpiOp = getFlagcxToMpiOp(op);
  CALL_MPI_REDUCE(datatype, sendbuff, recvbuff, count, mpiOp, root,
                  comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   int root, flagcxInnerComm_t comm,
                                   flagcxStream_t /*stream*/) {
  int result;
  void *buffer =
      (comm->base->getRank() == root) ? const_cast<void *>(sendbuff) : recvbuff;

  CALL_MPI_BCAST(datatype, buffer, count, root, comm->base->getMpiComm(),
                 &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, flagcxInnerComm_t comm,
                                   flagcxStream_t /*stream*/) {
  FLAGCXCHECK(validateComm(comm));

  int result;
  MPI_Op mpiOp = getFlagcxToMpiOp(op);
  CALL_MPI_ALLREDUCE(datatype, sendbuff, recvbuff, count, mpiOp,
                     comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorGather(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root, flagcxInnerComm_t comm,
                                flagcxStream_t /*stream*/) {
  int result;
  CALL_MPI_GATHER(datatype, sendbuff, count, recvbuff, count, root,
                  comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorScatter(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t /*stream*/) {
  int result;
  CALL_MPI_SCATTER(datatype, sendbuff, count, recvbuff, count, root,
                   comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorReduceScatter(const void *sendbuff, void *recvbuff,
                                       size_t recvcount,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxInnerComm_t comm,
                                       flagcxStream_t /*stream*/) {
  int result;
  MPI_Op mpiOp = getFlagcxToMpiOp(op);
  CALL_MPI_REDUCE_SCATTER(datatype, sendbuff, recvbuff, recvcount, mpiOp,
                          comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                   size_t sendcount, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t /*stream*/) {
  int result;
  CALL_MPI_ALLGATHER(datatype, sendbuff, sendcount, recvbuff, sendcount,
                     comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t /*stream*/) {
  int result;
  CALL_MPI_ALLTOALL(datatype, sendbuff, count, recvbuff, count,
                    comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                   size_t *sdispls, void *recvbuff,
                                   size_t *recvcounts, size_t *rdispls,
                                   flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t /*stream*/) {
  FLAGCXCHECK(validateComm(comm));

  // validate parameters
  if (!sendcounts || !sdispls || !recvcounts || !rdispls) {
    printf(
        "Error: AlltoAllv requires non-null count and displacement arrays\n");
    return flagcxInvalidArgument;
  }

  int size = comm->base->getSize();
  MPI_Datatype mpi_datatype = getFlagcxToMpiDataType(datatype);

  std::vector<int> mpi_sendcounts(size), mpi_recvcounts(size);
  std::vector<int> mpi_sdispls(size), mpi_rdispls(size);

  for (int i = 0; i < size; i++) {
    mpi_sendcounts[i] = static_cast<int>(sendcounts[i]);
    mpi_recvcounts[i] = static_cast<int>(recvcounts[i]);
    mpi_sdispls[i] = static_cast<int>(sdispls[i]);
    mpi_rdispls[i] = static_cast<int>(rdispls[i]);
  }

  int result =
      MPI_Alltoallv(sendbuff, mpi_sendcounts.data(), mpi_sdispls.data(),
                    mpi_datatype, recvbuff, mpi_recvcounts.data(),
                    mpi_rdispls.data(), mpi_datatype, comm->base->getMpiComm());

  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorSend(const void *sendbuff, size_t count,
                              flagcxDataType_t datatype, int peer,
                              flagcxInnerComm_t comm,
                              flagcxStream_t /*stream*/) {
  FLAGCXCHECK(validateComm(comm));

  // validate peer range
  if (peer < 0 || peer >= comm->base->getSize()) {
    printf("Error: Invalid peer %d, must be in range [0, %d)\n", peer,
           comm->base->getSize());
    return flagcxInvalidArgument;
  }

  MPI_Datatype mpi_datatype = getFlagcxToMpiDataType(datatype);
  int tag = 0;

  int result = MPI_Send(sendbuff, static_cast<int>(count), mpi_datatype, peer,
                        tag, comm->base->getMpiComm());

  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorRecv(void *recvbuff, size_t count,
                              flagcxDataType_t datatype, int peer,
                              flagcxInnerComm_t comm,
                              flagcxStream_t /*stream*/) {
  FLAGCXCHECK(validateComm(comm));

  // validate peer range (allow MPI_ANY_SOURCE)
  if (peer != MPI_ANY_SOURCE && (peer < 0 || peer >= comm->base->getSize())) {
    printf(
        "Error: Invalid peer %d, must be in range [0, %d) or MPI_ANY_SOURCE\n",
        peer, comm->base->getSize());
    return flagcxInvalidArgument;
  }

  MPI_Datatype mpi_datatype = getFlagcxToMpiDataType(datatype);
  int tag = 0;
  MPI_Status status;

  int result = MPI_Recv(recvbuff, static_cast<int>(count), mpi_datatype, peer,
                        tag, comm->base->getMpiComm(), &status);

  return (result == MPI_SUCCESS) ? flagcxSuccess : flagcxInternalError;
}

flagcxResult_t mpiAdaptorGroupStart() { return flagcxSuccess; }

flagcxResult_t mpiAdaptorGroupEnd() { return flagcxSuccess; }

struct flagcxCCLAdaptor mpiAdaptor = {
    "MPI",
    // Basic functions
    mpiAdaptorGetVersion, mpiAdaptorGetUniqueId, mpiAdaptorGetErrorString,
    mpiAdaptorGetLastError,
    // Symmetric operations
    mpiAdaptorCommWindowRegister, mpiAdaptorCommWindowDeregister,
    mpiAdaptorGetStagedBuffer,
    // Communicator functions
    mpiAdaptorCommInitRank, mpiAdaptorCommFinalize, mpiAdaptorCommDestroy,
    mpiAdaptorCommAbort, mpiAdaptorCommResume, mpiAdaptorCommSuspend,
    mpiAdaptorCommCount, mpiAdaptorCommCuDevice, mpiAdaptorCommUserRank,
    mpiAdaptorCommGetAsyncError, mpiAdaptorMemAlloc, mpiAdaptorMemFree,
    mpiAdaptorCommRegister, mpiAdaptorCommDeregister,
    // Communication functions
    mpiAdaptorReduce, mpiAdaptorGather, mpiAdaptorScatter, mpiAdaptorBroadcast,
    mpiAdaptorAllReduce, mpiAdaptorReduceScatter, mpiAdaptorAllGather,
    mpiAdaptorAlltoAll, mpiAdaptorAlltoAllv, mpiAdaptorSend, mpiAdaptorRecv,
    // Group semantics
    mpiAdaptorGroupStart, mpiAdaptorGroupEnd};

#endif // USE_MPI_ADAPTOR