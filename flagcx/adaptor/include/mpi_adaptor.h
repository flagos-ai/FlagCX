#ifdef USE_MPI_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "check.h"
#include "comm.h"
#include "flagcx.h"
#include "utils.h"

#include <cstring>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

// Predefined handles can be pointers (OpenMPI) or integers (MPICH). Keep
// them as runtime values instead of non-type C++ template arguments.
inline MPI_Datatype getFlagcxToMpiDataType(flagcxDataType_t datatype) {
  switch (datatype) {
    case flagcxChar:
      return MPI_CHAR;
    case flagcxUint8:
      return MPI_UNSIGNED_CHAR;
    case flagcxInt:
      return MPI_INT;
    case flagcxUint32:
      return MPI_UNSIGNED;
    case flagcxInt64:
      return MPI_LONG_LONG;
    case flagcxUint64:
      return MPI_UNSIGNED_LONG_LONG;
    case flagcxFloat:
      return MPI_FLOAT;
    case flagcxDouble:
      return MPI_DOUBLE;
    case flagcxHalf:
      return MPI_UINT16_T;
    default:
      return MPI_DATATYPE_NULL;
  }
}

inline MPI_Op getFlagcxToMpiOp(flagcxRedOp_t op) {
  switch (op) {
    case flagcxSum:
      return MPI_SUM;
    case flagcxProd:
      return MPI_PROD;
    case flagcxMax:
      return MPI_MAX;
    case flagcxMin:
      return MPI_MIN;
    default:
      return MPI_OP_NULL;
  }
}

inline void callMpi(int func_type, flagcxDataType_t datatype,
                    const void *sendbuf, void *recvbuf, size_t count, MPI_Op op,
                    int root, MPI_Comm comm, int *result) {
  MPI_Datatype mpi_type = getFlagcxToMpiDataType(datatype);
  if (mpi_type == MPI_DATATYPE_NULL) {
    *result = MPI_ERR_TYPE;
    return;
  }
  switch (func_type) {
    case 0: // ALLREDUCE
      *result = MPI_Allreduce(sendbuf, recvbuf, count, mpi_type, op, comm);
      break;
    case 1: // REDUCE
      *result = MPI_Reduce(sendbuf, recvbuf, count, mpi_type, op, root, comm);
      break;
    case 2: // BCAST
      *result = MPI_Bcast(recvbuf, count, mpi_type, root, comm);
      break;
    case 3: // GATHER
      *result = MPI_Gather(sendbuf, count, mpi_type, recvbuf, count, mpi_type,
                           root, comm);
      break;
    case 4: // SCATTER
      *result = MPI_Scatter(sendbuf, count, mpi_type, recvbuf, count, mpi_type,
                            root, comm);
      break;
    case 5: // ALLGATHER
      *result = MPI_Allgather(sendbuf, count, mpi_type, recvbuf, count,
                              mpi_type, comm);
      break;
    case 6: // ALLTOALL
      *result = MPI_Alltoall(sendbuf, count, mpi_type, recvbuf, count, mpi_type,
                             comm);
      break;
    case 7: // REDUCE_SCATTER
    {
      int size;
      MPI_Comm_size(comm, &size);
      std::vector<int> recvcounts(size, count);
      *result = MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts.data(),
                                   mpi_type, op, comm);
    } break;
  }
}

#define CALL_MPI_ALLREDUCE(datatype, sendbuf, recvbuf, count, op, comm,        \
                           result)                                             \
  callMpi(0, datatype, sendbuf, recvbuf, count, op, 0, comm, result)

#define CALL_MPI_REDUCE(datatype, sendbuf, recvbuf, count, op, root, comm,     \
                        result)                                                \
  callMpi(1, datatype, sendbuf, recvbuf, count, op, root, comm, result)

#define CALL_MPI_BCAST(datatype, buffer, count, root, comm, result)            \
  callMpi(2, datatype, nullptr, buffer, count, MPI_OP_NULL, root, comm, result)

#define CALL_MPI_GATHER(datatype, sendbuf, sendcount, recvbuf, recvcount,      \
                        root, comm, result)                                    \
  callMpi(3, datatype, sendbuf, recvbuf, sendcount, MPI_OP_NULL, root, comm,   \
          result)

#define CALL_MPI_SCATTER(datatype, sendbuf, sendcount, recvbuf, recvcount,     \
                         root, comm, result)                                   \
  callMpi(4, datatype, sendbuf, recvbuf, sendcount, MPI_OP_NULL, root, comm,   \
          result)

#define CALL_MPI_ALLGATHER(datatype, sendbuf, sendcount, recvbuf, recvcount,   \
                           comm, result)                                       \
  callMpi(5, datatype, sendbuf, recvbuf, sendcount, MPI_OP_NULL, 0, comm,      \
          result)

#define CALL_MPI_ALLTOALL(datatype, sendbuf, sendcount, recvbuf, recvcount,    \
                          comm, result)                                        \
  callMpi(6, datatype, sendbuf, recvbuf, sendcount, MPI_OP_NULL, 0, comm,      \
          result)

#define CALL_MPI_REDUCE_SCATTER(datatype, sendbuf, recvbuf, recvcount, op,     \
                                comm, result)                                  \
  callMpi(7, datatype, sendbuf, recvbuf, recvcount, op, 0, comm, result)

class flagcxMpiContext {
public:
  flagcxMpiContext(int rank, int nranks, struct bootstrapState *bootstrap);
  ~flagcxMpiContext();

  // Getters
  MPI_Comm getMpiComm() const { return mpiComm_; }
  int getRank() const { return rank_; }
  int getSize() const { return size_; }
  struct bootstrapState *getBootstrap() const {
    return bootstrap_;
  }
  bool isValidContext() const { return isValid_; }
  const std::string &getLastError() const { return lastError_; }

  // Operations
  bool createCustomComm(MPI_Comm baseComm = MPI_COMM_WORLD);
  void setComm(MPI_Comm comm) { mpiComm_ = comm; }

private:
  MPI_Comm mpiComm_;
  int rank_;
  int size_;
  struct bootstrapState *bootstrap_;
  bool isValid_;
  bool ownsComm_;
  std::string lastError_;

  // Helper methods
  void setError(const std::string &error) {
    lastError_ = error;
    isValid_ = false;
  }
  bool validateMpiEnvironment();
};

inline flagcxMpiContext::flagcxMpiContext(int rank, int nranks,
                                          struct bootstrapState *bootstrap)
    : mpiComm_(MPI_COMM_NULL), rank_(rank), size_(nranks),
      bootstrap_(bootstrap), isValid_(false), ownsComm_(false) {

  if (!validateMpiEnvironment()) {
    return;
  }

  mpiComm_ = MPI_COMM_WORLD;
  ownsComm_ = false;

  int mpi_rank, mpi_size;
  MPI_Comm_rank(mpiComm_, &mpi_rank);
  MPI_Comm_size(mpiComm_, &mpi_size);

  if (rank_ != mpi_rank || size_ != mpi_size) {
    printf("Warning: Context parameters (%d/%d) differ from MPI (%d/%d), using "
           "MPI values\n",
           rank_, size_, mpi_rank, mpi_size);
    rank_ = mpi_rank;
    size_ = mpi_size;
  }
  isValid_ = true;
}

inline flagcxMpiContext::~flagcxMpiContext() {
  // if comm is not MPI_COMM_WORLD, free it
  if (ownsComm_ && mpiComm_ != MPI_COMM_WORLD && mpiComm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&mpiComm_);
  }
}

inline bool flagcxMpiContext::createCustomComm(MPI_Comm baseComm) {
  if (ownsComm_ && mpiComm_ != MPI_COMM_WORLD && mpiComm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&mpiComm_);
  }

  int result = MPI_Comm_dup(baseComm, &mpiComm_);
  if (result != MPI_SUCCESS) {
    setError("Failed to duplicate MPI communicator");
    return false;
  }
  ownsComm_ = true;
  // update rank and size
  MPI_Comm_rank(mpiComm_, &rank_);
  MPI_Comm_size(mpiComm_, &size_);

  return true;
}

inline bool flagcxMpiContext::validateMpiEnvironment() {
  int initialized;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
    setError("Failed to check MPI initialization status");
    return false;
  }

  if (!initialized) {
    setError("MPI is not initialized");
    return false;
  }

  return true;
}

#define MPI_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8 * 1024 * 1024) // 8MB
struct stagedBuffer {
  flagcxInnerComm_t owner;
  int offset;
  int size;
  void *buffer;
};
typedef struct stagedBuffer *stagedBuffer_t;

struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  std::shared_ptr<flagcxMpiContext> base;
};

#endif // USE_MPI_ADAPTOR
