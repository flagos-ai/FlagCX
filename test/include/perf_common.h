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
  bool verifyData; // -v 0 skips post-iteration data verification.

  // Datatype/op for current iteration
  flagcxDataType_t datatype;
  flagcxRedOp_t op;

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

// Turn every FlagCX failure into a non-zero MPI job result. Perf binaries are
// CI correctness gates as well as benchmarks, so silently ignoring an adaptor
// or collective error would create a false pass.
[[noreturn]] void perfAbort(const char *message, const char *file, int line);
void perfCheck(flagcxResult_t result, const char *expression, const char *file,
               int line);

#define PERF_CHECK(expression)                                                 \
  perfCheck((expression), #expression, __FILE__, __LINE__)

// Initialize everything: parse args, MPI init, GPU setup, comm init,
// buffer allocation. bufSizeFn is called after MPI init (when totalProcs
// is known) to determine send/recv buffer sizes; nullptr = both maxBytes.
void perfSetup(PerfContext &ctx, int argc, char **argv,
               PerfBufSizeFn bufSizeFn = nullptr);

// Free all buffers, destroy comm/stream, free devHandle, MPI_Finalize.
void perfTeardown(PerfContext &ctx);

// Run warmup iterations for large and small message sizes.
void perfWarmup(PerfContext &ctx, PerfCollFn fn);

// Run the benchmark size sweep with timing, MPI averaging, and
// bandwidth reporting. Set iterateOps=false for collectives that don't
// use ctx.op (e.g., allgather, broadcast, sendrecv).
void perfBenchmarkLoop(PerfContext &ctx, PerfCollFn collFn,
                       PerfBwFactorFn bwFactorFn = nullptr,
                       PerfDataInitFn dataInitFn = nullptr,
                       PerfPostIterFn postIterFn = nullptr,
                       bool iterateOps = true);

// Run the benchmark size sweep with root iteration (for reduce, broadcast,
// scatter, gather). Iterates over roots per size, accumulating BW.
// Set iterateOps=false for collectives that don't use ctx.op.
void perfRootBenchmarkLoop(PerfContext &ctx, PerfRootCollFn collFn,
                           PerfBwFactorFn bwFactorFn = nullptr,
                           PerfRootDataInitFn dataInitFn = nullptr,
                           PerfRootPostIterFn postIterFn = nullptr,
                           bool iterateOps = true);
