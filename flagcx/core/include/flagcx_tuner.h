#ifndef FLAGCX_TUNER_H_
#define FLAGCX_TUNER_H_

#include "../adaptor/include/tuner.h"

// A category of collective operation. the minimal unit for tuning.
struct TunerCollCategory {
  flagcxCommOp_t collType;
  size_t nBytes;
};

bool operator<(const struct TunerCollCategory &lhs,
               const struct TunerCollCategory &rhs);

struct flagcxTuner {
  // Name of the tuner
  const char *name;

  void *bootstrap;

  int rank;
  int nranks;

  float *profilingResults;
  // Initializes tuner states.
  // Inputs:
  //   - nRanks: number of ranks in current communicator. Each communicator
  //   initialize its own tuner.
  //   - nNodes: number of nodes in current communicator.
  //   - logFunction: a logFunction can be useful to integrate logging together
  //   with FLAGCX core.
  // Outputs:
  //   - context: tuner context object
  flagcxResult_t (*init)(size_t nRanks, size_t nNodes,
                         flagcxDebugLogger_t logFunction, void **context);

  // Gets number of candidate communicator env settings available from this
  // tuner. Inputs:
  //   - context: tuner context object
  // Outputs:
  //   - nCandidates: number of candidate communicator
  flagcxResult_t (*getCandidateNumber)(void *context, uint32_t *nCandidates);

  // Set appropriate environment variables according to index, and return the
  // communicator tag. Note that all the env settings are set before returning
  // from this function. Only env of type FLAGCX_ENV_TYPE_CREATION will be set
  // in this function. Inputs:
  //   - context: tuner context object
  //   - index: index of candidate communicator, range [0, nCandidates)
  // Outputs:
  //   - commTag: communicator tag for this particular candidate
  flagcxResult_t (*setCandidate)(void *context, uint32_t index,
                                 struct flagcxCommTag *commTag);

  // Select the best communicator candidate for this collective.
  // All the env of type FLAGCX_ENV_TYPE_COLL and FLAGCX_ENV_TYPE_ONETIME if
  // necessary will be set before returning from this function. Inputs:
  //   - context: tuner context object
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - numPipeOps: number of operations in the group
  //   - regBuff: If non-zero, register user buffer is used.
  // Outputs:
  //   - commTag: communicator tag, used to select the underlying communicator.
  //
  // InOut:
  //   - collCostTable: collective cost table.  the caller is responsible for
  //   allocating and
  //                    deallocating the memory
  //
  flagcxResult_t (*getCollInfo)(void *context, flagcxCommOp_t collType,
                                size_t nBytes, int numPipeOps,
                                float **collCostTable, int regBuff,
                                struct flagcxCommTag *commTag,
                                flagcxComm_t *comm);

  // Start profiling for a specific collective with given parameters.
  // Inputs:
  //   - context: tuner context object
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - stream: the stream that the collective will be launched on
  //   - commTag: communicator tag
  // Outputs:
  //   - key: profiling key to pair with stopProfiling
  //
  flagcxResult_t (*startProfiling)(void *context, flagcxCommOp_t collType,
                                   size_t nBytes, flagcxStream_t stream,
                                   const struct flagcxCommTag *commTag,
                                   struct flagcxProfileKey *key);

  // Stop profiling for a specific collective with given key.
  // Inputs:
  //   - context: tuner context object
  //   - key: profiling key returned by startProfiling
  // Outputs:
  //   - None
  //
  flagcxResult_t (*stopProfiling)(void *context,
                                  const struct flagcxProfileKey *key);

  // Terminates the tuner and cleans up any resources that the tuner allocated.
  flagcxResult_t (*destroy)(void *context);

  // Create/destroy communicator
  flagcxResult_t (*createOrReplaceHomoComm)(
      flagcxComm_t *comm, struct flagcxTunerContext *ctx, uint32_t seqId,
      const struct TunerCollCategory &collCat, bool createBest);
};

typedef struct flagcxTuner flagcxTuner_t;

bool operator<(const struct flagcxCommTag &lhs,
               const struct flagcxCommTag &rhs);
bool operator==(const struct flagcxCommTag &lhs,
                const struct flagcxCommTag &rhs);

extern flagcxTuner_t internalTuner;

// On-demand communicator lifecycle helpers implemented in flagcx/flagcx.cc
flagcxResult_t flagcxCreateHomoCommForTag(flagcxComm_t comm, uint32_t idx);
flagcxResult_t flagcxDestroyHomoCommByTag(flagcxComm_t comm, uint32_t idx);

#define FLAGCXCALLWITHTUNER(call, comm, commOp, count, datatype, stream)       \
  do {                                                                         \
    comm->tunerInnerComm = nullptr;                                            \
    size_t nBytes = count * getFlagcxDataTypeSize(datatype);                   \
    struct flagcxCommTag tag = {""};                                           \
    FLAGCXCHECK(comm->tuner->getCollInfo(comm->tunerContext, commOp, nBytes,   \
                                         0, NULL, 0, &tag, &comm));            \
    flagcxProfileKey pkey;                                                     \
    FLAGCXCHECK(comm->tuner->startProfiling(comm->tunerContext, commOp,        \
                                            nBytes, stream, &tag, &pkey));     \
    FLAGCXCHECK(call);                                                         \
    FLAGCXCHECK(comm->tuner->stopProfiling(comm->tunerContext, &pkey));        \
    return flagcxSuccess;                                                      \
  } while (0);

#endif // end of FLAGCX_TUNER_H_
