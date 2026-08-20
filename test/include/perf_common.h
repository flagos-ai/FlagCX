#pragma once

// Common infrastructure for FlagCX performance tests.
// Extracts the duplicated setup/teardown/benchmark boilerplate
// shared across all perf test files.

#include "flagcx.h"
#include "tools.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Holds all state shared across perf tests.
struct PerfContext {
  // Parsed arguments
  parser *args;
  size_t minBytes;
  size_t maxBytes;
  int stepFactor;
  int numWarmupIters;
  int numIters;
  int printBuffer;
  int root;
  uint64_t splitMask;
  int localRegister;
  flagcxDataType_t datatype;
  flagcxRedOp_t redOp;
  size_t typeSize;
  bool useTypedReduction;
  bool accuracyFailed;

  // FlagCX handles
  flagcxDeviceHandle_t devHandle;
  flagcxComm_t comm;

  // MPI info
  int color;
  int worldSize, worldRank;
  int totalProcs, proc;
  MPI_Comm splitComm;

  // Buffers
  void *sendbuff;
  void *recvbuff;
  void *hello; // host staging buffer
  void *sendHandle;
  void *recvHandle;

  // Stream
  flagcxStream_t stream;
  timer tim;

  // Test-specific data accessible to callbacks
  void *userData;
};

// Function pointer types for callbacks.
using PerfBufSizeFn = void (*)(PerfContext &ctx, size_t &sendBufSize,
                               size_t &recvBufSize);
using PerfCollFn = void (*)(PerfContext &ctx, size_t count);
using PerfBwFactorFn = double (*)(int totalProcs);
using PerfDataInitFn = void (*)(PerfContext &ctx, size_t size, size_t count);
using PerfPostIterFn = void (*)(PerfContext &ctx, size_t size, size_t count);

// Root-iterated benchmark loop types.
using PerfRootCollFn = void (*)(PerfContext &ctx, size_t count, int root);
using PerfRootDataInitFn = void (*)(PerfContext &ctx, size_t size, size_t count,
                                    int root);
using PerfRootPostIterFn = void (*)(PerfContext &ctx, size_t size, size_t count,
                                    int root);

// Initialize everything: parse args, MPI init, GPU setup, comm init,
// buffer allocation. bufSizeFn is called after MPI init (when totalProcs
// is known) to determine send/recv buffer sizes; nullptr = both maxBytes.
void perfSetup(PerfContext &ctx, int argc, char **argv,
               PerfBufSizeFn bufSizeFn = nullptr);

// Enable the selected dtype for the reduction benchmark family. Other host
// API perf tests intentionally retain their historical float32 payloads.
void perfEnableTypedReduction(PerfContext &ctx);

// Free all buffers, destroy comm/stream, free devHandle, MPI_Finalize.
void perfTeardown(PerfContext &ctx);

// Run warmup iterations for large and small message sizes.
void perfWarmup(PerfContext &ctx, PerfCollFn fn);

// Run the benchmark size sweep with timing, MPI averaging, and
// bandwidth reporting.
void perfBenchmarkLoop(PerfContext &ctx, PerfCollFn collFn,
                       PerfBwFactorFn bwFactorFn = nullptr,
                       PerfDataInitFn dataInitFn = nullptr,
                       PerfPostIterFn postIterFn = nullptr);

// Run the benchmark size sweep with root iteration (for reduce, broadcast,
// scatter, gather). Iterates over roots per size, accumulating BW.
void perfRootBenchmarkLoop(PerfContext &ctx, PerfRootCollFn collFn,
                           PerfBwFactorFn bwFactorFn = nullptr,
                           PerfRootDataInitFn dataInitFn = nullptr,
                           PerfRootPostIterFn postIterFn = nullptr);

// Typed reduction-test helpers shared by the host API accuracy checks.
const char *perfDataTypeName(flagcxDataType_t datatype);
const char *perfRedOpName(flagcxRedOp_t op);
void perfInitReductionBuffers(PerfContext &ctx, size_t size, size_t count,
                              bool initializeRecvWithLocalInput);
long double perfReduceAcrossRanks(flagcxRedOp_t op, int nranks);
long double perfReduceBinary(flagcxRedOp_t op, long double value);
bool perfCheckUniformReduction(PerfContext &ctx, size_t count,
                               long double expected, const char *label);
bool perfCheckLocalReduction(PerfContext &ctx, size_t count,
                             const char *label);
bool perfCheckRingGather(PerfContext &ctx, size_t count,
                         const char *label);
void perfPrintBuffer(const PerfContext &ctx, size_t count, const char *label);
