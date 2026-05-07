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

  // FlagCX handles
  flagcxHandlerGroup_t handler;
  flagcxComm_t comm;              // alias for handler->comm
  flagcxDeviceHandle_t devHandle; // alias for handler->devHandle

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

// Free all buffers, destroy comm/stream, free handler, MPI_Finalize.
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
