#include "perf_common.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

struct FusedGemmPerfState {
  size_t n;
  size_t kPerRank;
  void *weight;
  std::vector<float> hostInput;
};

static size_t readPositiveSizeEnv(const char *name, size_t defaultValue) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return defaultValue;
  }
  errno = 0;
  char *end = NULL;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
      parsed > UINT32_MAX) {
    fprintf(stderr, "Invalid %s value: %s\n", name, value);
    exit(1);
  }
  return static_cast<size_t>(parsed);
}

static size_t perfN() {
  static size_t value = readPositiveSizeEnv("FLAGCX_PERF_GEMM_N", 256);
  return value;
}

static size_t perfK() {
  static size_t value = readPositiveSizeEnv("FLAGCX_PERF_GEMM_K", 256);
  return value;
}

[[noreturn]] static void fail(PerfContext &ctx, const char *message) {
  fprintf(stderr, "rank %d: %s\n", ctx.proc, message);
  MPI_Abort(MPI_COMM_WORLD, 1);
  abort();
}

// For this benchmark, size is the byte size of D=[m,n] before scattering.
static size_t mForCount(PerfContext &ctx, size_t count) {
  if (count == 0 || count % perfN() != 0) {
    fail(ctx, "message size must be divisible by GEMM_N * 4");
  }
  size_t m = count / perfN();
  if (m % static_cast<size_t>(ctx.totalProcs) != 0) {
    fail(ctx, "GEMM_M derived from message size must be divisible by nranks");
  }
  return m;
}

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  size_t maxCount = ctx.maxBytes / sizeof(float);
  size_t m = mForCount(ctx, maxCount);
  if (m > SIZE_MAX / perfK() || m * perfK() > SIZE_MAX / sizeof(float)) {
    fail(ctx, "input buffer size overflow");
  }
  sBuf = m * perfK() * sizeof(float);
  rBuf = ctx.maxBytes / ctx.totalProcs;
}

static float inputValue(int rank, size_t row, size_t col) {
  return 0.008f * static_cast<float>(rank + 1) +
         0.0001f * static_cast<float>((row * 17 + col) % 97);
}

static float weightValue(int rank, size_t row, size_t col) {
  return 0.015f * static_cast<float>(rank + 1) +
         0.0002f * static_cast<float>((row * 13 + col) % 89);
}

static void collFn(PerfContext &ctx, size_t count) {
  FusedGemmPerfState *state =
      static_cast<FusedGemmPerfState *>(ctx.userData);
  size_t m = mForCount(ctx, count);
  flagcxResult_t result = flagcxGemmReduceScatter(
      ctx.sendbuff, state->weight, ctx.recvbuff, m, state->n, state->kPerRank,
      flagcxFloat32, flagcxSum, ctx.comm, ctx.stream);
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d: flagcxGemmReduceScatter failed: %d\n",
            ctx.proc, result);
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(result));
  }
}

static double bwFactorFn(int totalProcs) {
  return static_cast<double>(totalProcs - 1) / totalProcs;
}

static void dataInitFn(PerfContext &ctx, size_t, size_t count) {
  FusedGemmPerfState *state =
      static_cast<FusedGemmPerfState *>(ctx.userData);
  size_t m = mForCount(ctx, count);
  size_t inputCount = m * state->kPerRank;
  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < state->kPerRank; ++col) {
      state->hostInput[row * state->kPerRank + col] =
          inputValue(ctx.proc, row, col);
    }
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, state->hostInput.data(),
                              inputCount * sizeof(float),
                              flagcxMemcpyHostToDevice, ctx.stream);
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  if (!ctx.printBuffer) {
    return;
  }
  FusedGemmPerfState *state =
      static_cast<FusedGemmPerfState *>(ctx.userData);
  size_t m = mForCount(ctx, count);
  size_t rowsPerRank = m / ctx.totalProcs;
  size_t outputSize = size / ctx.totalProcs;
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, outputSize,
                              flagcxMemcpyDeviceToHost, ctx.stream);
  ctx.devHandle->streamSynchronize(ctx.stream);

  int correct = 1;
  size_t globalRowStart = static_cast<size_t>(ctx.proc) * rowsPerRank;
  for (size_t row = 0; row < rowsPerRank && correct; ++row) {
    size_t globalRow = globalRowStart + row;
    for (size_t col = 0; col < state->n; ++col) {
      float expected = 0.0f;
      for (int sourceRank = 0; sourceRank < ctx.totalProcs; ++sourceRank) {
        for (size_t p = 0; p < state->kPerRank; ++p) {
          expected += inputValue(sourceRank, globalRow, p) *
                      weightValue(sourceRank, p, col);
        }
      }
      float actual = static_cast<float *>(ctx.hello)[row * state->n + col];
      float tolerance = 1e-4f + 1e-4f * std::fabs(expected);
      if (std::fabs(actual - expected) > tolerance) {
        fprintf(stderr,
                "rank %d wrong output at (%zu,%zu), expected %f, got %f\n",
                ctx.proc, row, col, expected, actual);
        correct = 0;
        break;
      }
    }
  }
  printf("rank %d gemm-reducescatter correctness = %d\n", ctx.proc,
         correct);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);

  FusedGemmPerfState state = {perfN(), perfK(), NULL, {}};
  if (state.kPerRank > SIZE_MAX / state.n ||
      state.kPerRank * state.n > SIZE_MAX / sizeof(float)) {
    fail(ctx, "weight buffer size overflow");
  }
  size_t weightCount = state.kPerRank * state.n;
  size_t maxM = ctx.maxBytes / sizeof(float) / state.n;
  state.hostInput.resize(maxM * state.kPerRank);
  std::vector<float> hostWeight(weightCount);
  for (size_t row = 0; row < state.kPerRank; ++row) {
    for (size_t col = 0; col < state.n; ++col) {
      hostWeight[row * state.n + col] = weightValue(ctx.proc, row, col);
    }
  }
  if (ctx.devHandle->deviceMalloc(&state.weight, weightCount * sizeof(float),
                                  flagcxMemDevice, NULL) != flagcxSuccess) {
    fail(ctx, "failed to allocate weight buffer");
  }
  if (ctx.devHandle->deviceMemcpy(state.weight, hostWeight.data(),
                                  weightCount * sizeof(float),
                                  flagcxMemcpyHostToDevice,
                                  ctx.stream) != flagcxSuccess) {
    fail(ctx, "failed to initialize weight buffer");
  }
  ctx.userData = &state;

  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);

  ctx.devHandle->deviceFree(state.weight, flagcxMemDevice, NULL);
  perfTeardown(ctx);
  return 0;
}
