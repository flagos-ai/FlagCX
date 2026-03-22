#include "mpi.h"
#include <cstddef>
#include <cstdint>

void initMpiEnv(int argc, char **argv, int &worldRank, int &worldSize,
                int &proc, int &totalProcs, int &color, MPI_Comm &splitComm,
                uint64_t splitMask);

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

  size_t minBytes;
  size_t maxBytes;
  int stepFactor;
  int warmupIters;
  int testIters;
  int printBuffer;
  int root;
  uint64_t splitMask;
  int localRegister;
};