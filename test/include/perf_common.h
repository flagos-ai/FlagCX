#pragma once

// Common infrastructure for FlagCX performance tests.
// Extracts the duplicated setup/teardown/benchmark boilerplate
// shared across all 13 perf test files.

#include "flagcx.h"
#include "tools.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>

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
  flagcxComm_t comm;           // alias for handler->comm
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
};

// Initialize everything: parse args, MPI init, GPU setup, comm init,
// buffer allocation. Call this at the start of main().
// sendBufSize/recvBufSize: override buffer sizes (0 = use maxBytes).
void perfSetup(PerfContext &ctx, int argc, char **argv,
               size_t sendBufSize = 0, size_t recvBufSize = 0);

// Free all buffers, destroy comm/stream, free handler, MPI_Finalize.
void perfTeardown(PerfContext &ctx);

// Callback type for collective operations in the benchmark loop.
// Parameters: (ctx, sendbuff, recvbuff, count, stream)
using PerfCollFn = std::function<void(PerfContext &ctx, size_t count)>;

// Run warmup iterations for large and small message sizes.
void perfWarmup(PerfContext &ctx, PerfCollFn fn);

// Run the benchmark size sweep with timing, MPI averaging, and
// bandwidth reporting. bwFactor converts base bandwidth to bus bandwidth.
// dataInitFn: optional data initialization per size (can be nullptr).
// bwFactorFn: returns bus_bw factor given totalProcs (if nullptr, uses 1.0).
using PerfBwFactorFn = std::function<double(int totalProcs)>;
using PerfDataInitFn = std::function<void(PerfContext &ctx, size_t size,
                                          size_t count)>;

void perfBenchmarkLoop(PerfContext &ctx, PerfCollFn collFn,
                       PerfBwFactorFn bwFactorFn = nullptr,
                       PerfDataInitFn dataInitFn = nullptr);
