#include "perf_common.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

struct FusedGemmPerfState {
  size_t n;
  size_t k;
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

// For this benchmark, size is the byte size of C=[P*mPerRank,n].
static size_t mPerRankForCount(PerfContext &ctx, size_t count) {
  size_t rowWidth = static_cast<size_t>(ctx.totalProcs) * perfN();
  if (rowWidth == 0 || count == 0 || count % rowWidth != 0) {
    fail(ctx, "message size must be divisible by nranks * GEMM_N * 4");
  }
  return count / rowWidth;
}

static void bufSizeFn(PerfContext &ctx, size_t &sBuf, size_t &rBuf) {
  size_t maxCount = ctx.maxBytes / sizeof(float);
  size_t mPerRank = mPerRankForCount(ctx, maxCount);
  if (mPerRank > SIZE_MAX / perfK() ||
      mPerRank * perfK() > SIZE_MAX / sizeof(float)) {
    fail(ctx, "input buffer size overflow");
  }
  sBuf = mPerRank * perfK() * sizeof(float);
  rBuf = ctx.maxBytes;
}

static float inputValue(int sourceRank, size_t row, size_t col) {
  return 0.01f * static_cast<float>(sourceRank + 1) +
         0.0001f * static_cast<float>((row * 17 + col) % 97);
}

static float weightValue(int rank, size_t row, size_t col) {
  return 0.02f * static_cast<float>(rank + 1) +
         0.0002f * static_cast<float>((row * 13 + col) % 89);
}

static void collFn(PerfContext &ctx, size_t count) {
  FusedGemmPerfState *state =
      static_cast<FusedGemmPerfState *>(ctx.userData);
  size_t mPerRank = mPerRankForCount(ctx, count);
  flagcxResult_t result = flagcxAllGatherGemm(
      ctx.sendbuff, state->weight, ctx.recvbuff, mPerRank, state->n, state->k,
      flagcxFloat32, ctx.comm, ctx.stream);
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d: flagcxAllGatherGemm failed: %d\n", ctx.proc,
            result);
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(result));
  }
}

static double bwFactorFn(int totalProcs) {
  return static_cast<double>(totalProcs - 1) / totalProcs;
}

static void dataInitFn(PerfContext &ctx, size_t, size_t count) {
  FusedGemmPerfState *state =
      static_cast<FusedGemmPerfState *>(ctx.userData);
  size_t mPerRank = mPerRankForCount(ctx, count);
  size_t inputCount = mPerRank * state->k;
  for (size_t row = 0; row < mPerRank; ++row) {
    for (size_t col = 0; col < state->k; ++col) {
      state->hostInput[row * state->k + col] =
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
  size_t mPerRank = mPerRankForCount(ctx, count);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, ctx.stream);
  ctx.devHandle->streamSynchronize(ctx.stream);

  int correct = 1;
  size_t globalRows = static_cast<size_t>(ctx.totalProcs) * mPerRank;
  for (size_t row = 0; row < globalRows && correct; ++row) {
    int sourceRank = static_cast<int>(row / mPerRank);
    size_t localRow = row % mPerRank;
    for (size_t col = 0; col < state->n; ++col) {
      float expected = 0.0f;
      for (size_t p = 0; p < state->k; ++p) {
        expected += inputValue(sourceRank, localRow, p) *
                    weightValue(ctx.proc, p, col);
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
  printf("rank %d allgather-gemm correctness = %d\n", ctx.proc, correct);
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv, bufSizeFn);

  FusedGemmPerfState state = {perfN(), perfK(), NULL, {}};
  if (state.k > SIZE_MAX / state.n ||
      state.k * state.n > SIZE_MAX / sizeof(float)) {
    fail(ctx, "weight buffer size overflow");
  }
  size_t weightCount = state.k * state.n;
  size_t maxMPerRank =
      ctx.maxBytes / sizeof(float) / ctx.totalProcs / state.n;
  state.hostInput.resize(maxMPerRank * state.k);
  std::vector<float> hostWeight(weightCount);
  for (size_t row = 0; row < state.k; ++row) {
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
