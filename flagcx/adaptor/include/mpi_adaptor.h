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

#define GENERATE_MPI_TYPES(type, func, args...)                                \
  switch (type) {                                                              \
    case flagcxChar:                                                           \
      func<char, MPI_CHAR>(args);                                              \
      break;                                                                   \
    case flagcxUint8:                                                          \
      func<uint8_t, MPI_UNSIGNED_CHAR>(args);                                  \
      break;                                                                   \
    case flagcxInt:                                                            \
      func<int, MPI_INT>(args);                                                \
      break;                                                                   \
    case flagcxUint32:                                                         \
      func<uint32_t, MPI_UNSIGNED>(args);                                      \
      break;                                                                   \
    case flagcxInt64:                                                          \
      func<int64_t, MPI_LONG_LONG>(args);                                      \
      break;                                                                   \
    case flagcxUint64:                                                         \
      func<uint64_t, MPI_UNSIGNED_LONG_LONG>(args);                            \
      break;                                                                   \
    case flagcxFloat:                                                          \
      func<float, MPI_FLOAT>(args);                                            \
      break;                                                                   \
    case flagcxDouble:                                                         \
      func<double, MPI_DOUBLE>(args);                                          \
      break;                                                                   \
    case flagcxHalf:                                                           \
      func<uint16_t, MPI_UINT16_T>(args);                                      \
      break;                                                                   \
    case flagcxBfloat16:                                                       \
      printf("Invalid data type");                                             \
      break;                                                                   \
    default:                                                                   \
      printf("Invalid data type");                                             \
      break;                                                                   \
  }

#define GENERATE_MPI_REDUCTION_OPS(op, func, args...)                          \
  switch (op) {                                                                \
    case flagcxSum:                                                            \
      func<MPI_SUM>(args);                                                     \
      break;                                                                   \
    case flagcxProd:                                                           \
      func<MPI_PROD>(args);                                                    \
      break;                                                                   \
    case flagcxMax:                                                            \
      func<MPI_MAX>(args);                                                     \
      break;                                                                   \
    case flagcxMin:                                                            \
      func<MPI_MIN>(args);                                                     \
      break;                                                                   \
    case flagcxAvg:                                                            \
      printf("MPI backend does not support flagcxAvg\n");                      \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
  }

template <typename T, MPI_Datatype mpi_type>
void getMpiDataType(MPI_Datatype &result) {
  result = mpi_type;
}

template <MPI_Op mpi_op>
void getMpiOp(MPI_Op &result) {
  result = mpi_op;
}

inline MPI_Datatype getFlagcxToMpiDataType(flagcxDataType_t datatype) {
  MPI_Datatype result = MPI_DATATYPE_NULL;
  GENERATE_MPI_TYPES(datatype, getMpiDataType, result);
  return result;
}

inline MPI_Op getFlagcxToMpiOp(flagcxRedOp_t op) {
  MPI_Op result = MPI_OP_NULL;
  GENERATE_MPI_REDUCTION_OPS(op, getMpiOp, result);
  return result;
}

template <typename T, MPI_Datatype mpi_type>
void callMpiFunction(int func_type, const void *sendbuf, void *recvbuf,
                     size_t count, MPI_Op op, int root, MPI_Comm comm,
                     int *result) {
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

template <typename... Args>
void callMpi(int func_type, flagcxDataType_t datatype, Args... args) {
  GENERATE_MPI_TYPES(datatype, callMpiFunction, func_type, args...);
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
  flagcxMpiContext(int rank, int nranks, bootstrapState *bootstrap);
  ~flagcxMpiContext();

  // Getters
  MPI_Comm getMpiComm() const { return mpiComm_; }
  int getRank() const { return rank_; }
  int getSize() const { return size_; }
  bootstrapState *getBootstrap() const { return bootstrap_; }
  bool isValidContext() const { return isValid_; }
  const std::string &getLastError() const { return lastError_; }

  // Operations
  bool createCustomComm(MPI_Comm baseComm = MPI_COMM_WORLD);
  void setComm(MPI_Comm comm) { mpiComm_ = comm; }

private:
  MPI_Comm mpiComm_;
  int rank_;
  int size_;
  bootstrapState *bootstrap_;
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
                                          bootstrapState *bootstrap)
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
  int offset;
  int size;
  void *buffer;
};
typedef struct stagedBuffer *stagedBuffer_t;

struct flagcxInnerComm {
  std::shared_ptr<flagcxMpiContext> base;
};

#endif // USE_MPI_ADAPTOR
