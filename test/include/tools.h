#pragma once

#include "check.h"
#include "flagcx.h"
#include "mpi.h"
#include <cstddef>
#include <cstdint>

void initMpiEnv(int argc, char **argv, int &worldRank, int &worldSize,
                int &proc, int &totalProcs, int &color, MPI_Comm &splitComm,
                uint64_t splitMask);

// Datatype tables (aligned with nccl-tests style)
extern const flagcxDataType_t test_types[];
extern const char *test_typenames[];
extern int test_typenum;

// Reduction op tables
extern const flagcxRedOp_t test_ops[];
extern const char *test_opnames[];
extern int test_opnum;

// Returns index into test_types[], or -1 for "all"
int flagcxStringToType(const char *str);
// Returns index into test_ops[], or -1 for "all"
int flagcxStringToOp(const char *str);

class timer {
public:
  timer();
  double elapsed() const;
  double reset();

  std::uint64_t start;
};

class parser {
public:
  parser(int argc, char **argv);
  size_t getMinBytes() const { return minBytes; }
  size_t getMaxBytes() const { return maxBytes; }
  size_t getStepFactor() const { return stepFactor; }
  int getWarmupIters() const { return warmupIters; }
  int getTestIters() const { return testIters; }
  bool isPrintBuffer() const { return printBuffer == 1; }
  int getRootRank() const { return root; }
  uint64_t getSplitMask() const { return splitMask; }
  int getLocalRegister() const { return localRegister; }
  bool isVerificationEnabled() const { return verifyData == 1; }
  int getDataType() const { return datatype; }
  int getOp() const { return op; }

  size_t minBytes;
  size_t maxBytes;
  int stepFactor;
  int warmupIters;
  int testIters;
  int printBuffer;
  int root;
  uint64_t splitMask;
  int localRegister;
  int verifyData;
  int datatype = -1; // index into test_types[], or -1 for "all"
  int op = -1;       // index into test_ops[], or -1 for "all"
};
