#include "perf_common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

namespace {

static uint16_t floatToHalfBits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00u);
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa |= 0x800000u;
    const int shift = 14 - exponent;
    uint32_t halfMantissa = mantissa >> shift;
    if (((mantissa >> (shift - 1)) & 1u) != 0)
      ++halfMantissa;
    return static_cast<uint16_t>(sign | halfMantissa);
  }
  uint32_t halfMantissa = mantissa >> 13;
  if ((mantissa & 0x1000u) != 0)
    ++halfMantissa;
  if (halfMantissa == 0x400u)
    return static_cast<uint16_t>(sign | ((exponent + 1) << 10));
  return static_cast<uint16_t>(sign | (exponent << 10) | halfMantissa);
}

static float halfBitsToFloat(uint16_t value) {
  const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x3ffu;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int exp = -14;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exp;
      }
      mantissa &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(exp + 127) << 23) |
             (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static uint16_t floatToBfloat16Bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

static float bfloat16BitsToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename T> static void storeRaw(void *buffer, size_t index, T value) {
  memcpy(static_cast<char *>(buffer) + index * sizeof(T), &value, sizeof(T));
}

template <typename T> static T loadRaw(const void *buffer, size_t index) {
  T value{};
  memcpy(&value, static_cast<const char *>(buffer) + index * sizeof(T),
         sizeof(T));
  return value;
}

static void storeValue(void *buffer, flagcxDataType_t datatype, size_t index,
                       long double value) {
  switch (datatype) {
    case flagcxInt8:
      storeRaw<int8_t>(buffer, index, static_cast<int8_t>(value));
      break;
    case flagcxUint8:
      storeRaw<uint8_t>(buffer, index, static_cast<uint8_t>(value));
      break;
    case flagcxInt32:
      storeRaw<int32_t>(buffer, index, static_cast<int32_t>(value));
      break;
    case flagcxUint32:
      storeRaw<uint32_t>(buffer, index, static_cast<uint32_t>(value));
      break;
    case flagcxInt64:
      storeRaw<int64_t>(buffer, index, static_cast<int64_t>(value));
      break;
    case flagcxUint64:
      storeRaw<uint64_t>(buffer, index, static_cast<uint64_t>(value));
      break;
    case flagcxFloat16:
      storeRaw<uint16_t>(buffer, index, floatToHalfBits(static_cast<float>(value)));
      break;
    case flagcxFloat32:
      storeRaw<float>(buffer, index, static_cast<float>(value));
      break;
    case flagcxFloat64:
      storeRaw<double>(buffer, index, static_cast<double>(value));
      break;
    case flagcxBfloat16:
      storeRaw<uint16_t>(buffer, index,
                         floatToBfloat16Bits(static_cast<float>(value)));
      break;
    default:
      break;
  }
}

static long double loadValue(const void *buffer, flagcxDataType_t datatype,
                             size_t index) {
  switch (datatype) {
    case flagcxInt8:
      return loadRaw<int8_t>(buffer, index);
    case flagcxUint8:
      return loadRaw<uint8_t>(buffer, index);
    case flagcxInt32:
      return loadRaw<int32_t>(buffer, index);
    case flagcxUint32:
      return loadRaw<uint32_t>(buffer, index);
    case flagcxInt64:
      return static_cast<long double>(loadRaw<int64_t>(buffer, index));
    case flagcxUint64:
      return static_cast<long double>(loadRaw<uint64_t>(buffer, index));
    case flagcxFloat16:
      return halfBitsToFloat(loadRaw<uint16_t>(buffer, index));
    case flagcxFloat32:
      return loadRaw<float>(buffer, index);
    case flagcxFloat64:
      return loadRaw<double>(buffer, index);
    case flagcxBfloat16:
      return bfloat16BitsToFloat(loadRaw<uint16_t>(buffer, index));
    default:
      return 0.0L;
  }
}

static bool valueMatches(flagcxDataType_t datatype, long double actual,
                         long double expected) {
  const bool integral = datatype <= flagcxUint64;
  if (integral)
    return actual == expected;
  const long double diff = fabsl(actual - expected);
  const long double scale = std::max(1.0L, fabsl(expected));
  long double tolerance = 1e-5L;
  if (datatype == flagcxFloat16)
    tolerance = 2e-2L;
  else if (datatype == flagcxBfloat16)
    tolerance = 1e-1L;
  else if (datatype == flagcxFloat64)
    tolerance = 1e-12L;
  return diff <= tolerance * scale;
}

} // namespace

const char *perfDataTypeName(flagcxDataType_t datatype) {
  switch (datatype) {
    case flagcxInt8:
      return "int8";
    case flagcxUint8:
      return "uint8";
    case flagcxInt32:
      return "int32";
    case flagcxUint32:
      return "uint32";
    case flagcxInt64:
      return "int64";
    case flagcxUint64:
      return "uint64";
    case flagcxFloat16:
      return "float16";
    case flagcxFloat32:
      return "float32";
    case flagcxFloat64:
      return "float64";
    case flagcxBfloat16:
      return "bfloat16";
    default:
      return "unknown";
  }
}

const char *perfRedOpName(flagcxRedOp_t op) {
  switch (op) {
    case flagcxSum:
      return "sum";
    case flagcxProd:
      return "prod";
    case flagcxMax:
      return "max";
    case flagcxMin:
      return "min";
    case flagcxAvg:
      return "avg";
    default:
      return "unknown";
  }
}

void perfInitReductionBuffers(PerfContext &ctx, size_t size, size_t count,
                              bool initializeRecvWithLocalInput) {
  const long double localValue = static_cast<long double>(ctx.proc + 1);
  for (size_t i = 0; i < count; ++i)
    storeValue(ctx.hello, ctx.datatype, i, localValue);
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);

  if (initializeRecvWithLocalInput) {
    memset(ctx.hello, 0, size);
    for (size_t i = 0; i < count; ++i)
      storeValue(ctx.hello, ctx.datatype, i, localValue);
    ctx.devHandle->deviceMemcpy(ctx.recvbuff, ctx.hello, size,
                                flagcxMemcpyHostToDevice, NULL);
  }
}

long double perfReduceAcrossRanks(flagcxRedOp_t op, int nranks) {
  if (nranks <= 0)
    return 0.0L;
  long double result = static_cast<long double>(1);
  switch (op) {
    case flagcxSum:
    case flagcxAvg:
      result = 0.0L;
      for (int rank = 1; rank <= nranks; ++rank)
        result += static_cast<long double>(rank);
      if (op == flagcxAvg)
        result /= static_cast<long double>(nranks);
      return result;
    case flagcxProd:
      for (int rank = 1; rank <= nranks; ++rank)
        result *= static_cast<long double>(rank);
      return result;
    case flagcxMax:
      return static_cast<long double>(nranks);
    case flagcxMin:
      return 1.0L;
    default:
      return 0.0L;
  }
}

long double perfReduceBinary(flagcxRedOp_t op, long double value) {
  switch (op) {
    case flagcxSum:
      return value + value;
    case flagcxProd:
      return value * value;
    case flagcxMax:
    case flagcxMin:
    case flagcxAvg:
      return value;
    default:
      return 0.0L;
  }
}

bool perfCheckUniformReduction(PerfContext &ctx, size_t count,
                               long double expected, const char *label) {
  uint64_t expectedStorage[2] = {0, 0};
  storeValue(expectedStorage, ctx.datatype, 0, expected);
  const long double normalizedExpected =
      loadValue(expectedStorage, ctx.datatype, 0);
  const size_t size = count * ctx.typeSize;
  if (size != 0) {
    ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                flagcxMemcpyDeviceToHost, NULL);
    ctx.devHandle->streamSynchronize(ctx.stream);
  }
  for (size_t i = 0; i < count; ++i) {
    const long double actual = loadValue(ctx.hello, ctx.datatype, i);
    if (!valueMatches(ctx.datatype, actual, normalizedExpected)) {
      if (ctx.proc == 0 && ctx.color == 0) {
        fprintf(stderr,
                "%s failed (%s/%s) at element %zu: expected %.12Lg, got "
                "%.12Lg\n",
                label, perfDataTypeName(ctx.datatype),
                perfRedOpName(ctx.redOp), i, normalizedExpected, actual);
      }
      ctx.accuracyFailed = true;
      return false;
    }
  }
  return true;
}

bool perfCheckLocalReduction(PerfContext &ctx, size_t count,
                             const char *label) {
  const size_t size = count * ctx.typeSize;
  if (size != 0) {
    ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                flagcxMemcpyDeviceToHost, NULL);
    ctx.devHandle->streamSynchronize(ctx.stream);
  }
  const size_t begin = count * static_cast<size_t>(ctx.proc) /
                       static_cast<size_t>(ctx.totalProcs);
  const size_t end = count * static_cast<size_t>(ctx.proc + 1) /
                     static_cast<size_t>(ctx.totalProcs);
  for (size_t i = 0; i < count; ++i) {
    const long double local = static_cast<long double>(ctx.proc + 1);
    const long double expected =
        (i >= begin && i < end) ? perfReduceBinary(ctx.redOp, local) : local;
    uint64_t expectedStorage[2] = {0, 0};
    storeValue(expectedStorage, ctx.datatype, 0, expected);
    const long double normalizedExpected =
        loadValue(expectedStorage, ctx.datatype, 0);
    const long double actual = loadValue(ctx.hello, ctx.datatype, i);
    if (!valueMatches(ctx.datatype, actual, normalizedExpected)) {
      if (ctx.proc == 0 && ctx.color == 0) {
        fprintf(stderr,
                "%s failed (%s/%s) at element %zu: expected %.12Lg, got "
                "%.12Lg\n",
                label, perfDataTypeName(ctx.datatype),
                perfRedOpName(ctx.redOp), i, normalizedExpected, actual);
      }
      ctx.accuracyFailed = true;
      return false;
    }
  }
  return true;
}

bool perfCheckRingGather(PerfContext &ctx, size_t count,
                         const char *label) {
  const size_t size = count * ctx.typeSize;
  if (size != 0) {
    ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                flagcxMemcpyDeviceToHost, NULL);
    ctx.devHandle->streamSynchronize(ctx.stream);
  }
  for (size_t i = 0; i < count; ++i) {
    const int source = count == 0
                           ? 0
                           : std::min(ctx.totalProcs - 1,
                                      static_cast<int>(i * ctx.totalProcs /
                                                       count));
    const long double expected = static_cast<long double>(source + 1);
    const long double actual = loadValue(ctx.hello, ctx.datatype, i);
    if (!valueMatches(ctx.datatype, actual, expected)) {
      if (ctx.proc == 0 && ctx.color == 0) {
        fprintf(stderr,
                "%s failed (%s) at element %zu: expected %.12Lg, got "
                "%.12Lg\n",
                label, perfDataTypeName(ctx.datatype), i, expected, actual);
      }
      ctx.accuracyFailed = true;
      return false;
    }
  }
  return true;
}

void perfPrintBuffer(const PerfContext &ctx, size_t count, const char *label) {
  if (!ctx.printBuffer || ctx.proc != 0 || ctx.color != 0)
    return;
  const size_t size = count * ctx.typeSize;
  if (size != 0) {
    ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                flagcxMemcpyDeviceToHost, NULL);
    ctx.devHandle->streamSynchronize(ctx.stream);
  }
  printf("rank %d %s (%s/%s) = ", ctx.proc, label,
         perfDataTypeName(ctx.datatype), perfRedOpName(ctx.redOp));
  for (size_t i = 0; i < std::min<size_t>(count, 10); ++i)
    printf("%.6Lf ", loadValue(ctx.hello, ctx.datatype, i));
  printf("\n");
}

void perfSetup(PerfContext &ctx, int argc, char **argv,
               PerfBufSizeFn bufSizeFn) {
  // Parse arguments
  ctx.args = new parser(argc, argv);
  ctx.minBytes = ctx.args->getMinBytes();
  ctx.maxBytes = ctx.args->getMaxBytes();
  ctx.stepFactor = ctx.args->getStepFactor();
  ctx.numWarmupIters = ctx.args->getWarmupIters();
  ctx.numIters = ctx.args->getTestIters();
  ctx.printBuffer = ctx.args->isPrintBuffer();
  ctx.root = ctx.args->getRootRank();
  ctx.splitMask = ctx.args->getSplitMask();
  ctx.localRegister = ctx.args->getLocalRegister();
  ctx.datatype = ctx.args->getDataType();
  ctx.redOp = ctx.args->getRedOp();
  ctx.typeSize = sizeof(float);
  ctx.useTypedReduction = false;
  ctx.accuracyFailed = false;

  // Initialize FlagCX device handle
  flagcxDeviceHandleInit(&ctx.devHandle);

  // Initialize MPI environment
  ctx.color = 0;
  ctx.worldSize = 1;
  ctx.worldRank = 0;
  ctx.totalProcs = 1;
  ctx.proc = 0;
  initMpiEnv(argc, argv, ctx.worldRank, ctx.worldSize, ctx.proc, ctx.totalProcs,
             ctx.color, ctx.splitComm, ctx.splitMask);

  // Adjust root for totalProcs
  if (ctx.root >= 0)
    ctx.root = ctx.root % ctx.totalProcs;

  // GPU setup
  int nGpu;
  ctx.devHandle->getDeviceCount(&nGpu);
  ctx.devHandle->setDevice(ctx.worldRank % nGpu);

  // Create and broadcast uniqueId
  flagcxUniqueId uniqueId;
  if (ctx.proc == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)&uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            ctx.splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  // Initialize communicator
  flagcxCommInitRank(&ctx.comm, ctx.totalProcs, &uniqueId, ctx.proc);

  // Create stream
  ctx.devHandle->streamCreate(&ctx.stream);

  // Buffer sizes: call bufSizeFn if provided (totalProcs is now known)
  size_t sBufSize = ctx.maxBytes;
  size_t rBufSize = ctx.maxBytes;
  if (bufSizeFn) {
    bufSizeFn(ctx, sBufSize, rBufSize);
  }
  size_t hBufSize = ctx.maxBytes; // host buffer always maxBytes

  // Allocate buffers
  ctx.sendbuff = nullptr;
  ctx.recvbuff = nullptr;
  ctx.sendHandle = nullptr;
  ctx.recvHandle = nullptr;

  if (ctx.localRegister) {
    flagcxMemAlloc(&ctx.sendbuff, sBufSize);
    flagcxMemAlloc(&ctx.recvbuff, rBufSize);
    flagcxCommRegister(ctx.comm, ctx.sendbuff, sBufSize, &ctx.sendHandle);
    flagcxCommRegister(ctx.comm, ctx.recvbuff, rBufSize, &ctx.recvHandle);
  } else {
    ctx.devHandle->deviceMalloc(&ctx.sendbuff, sBufSize, flagcxMemDevice, NULL);
    ctx.devHandle->deviceMalloc(&ctx.recvbuff, rBufSize, flagcxMemDevice, NULL);
  }
  ctx.hello = malloc(hBufSize);
  memset(ctx.hello, 0, hBufSize);

  ctx.userData = nullptr;
}

void perfEnableTypedReduction(PerfContext &ctx) {
  ctx.typeSize = getFlagcxDataTypeSize(ctx.datatype);
  ctx.useTypedReduction = true;
}

void perfTeardown(PerfContext &ctx) {
  if (ctx.localRegister) {
    flagcxCommDeregister(ctx.comm, ctx.sendHandle);
    flagcxCommDeregister(ctx.comm, ctx.recvHandle);
    flagcxMemFree(ctx.sendbuff);
    flagcxMemFree(ctx.recvbuff);
  } else {
    ctx.devHandle->deviceFree(ctx.sendbuff, flagcxMemDevice, NULL);
    ctx.devHandle->deviceFree(ctx.recvbuff, flagcxMemDevice, NULL);
  }
  free(ctx.hello);
  ctx.devHandle->streamDestroy(ctx.stream);
  flagcxCommDestroy(ctx.comm);
  flagcxDeviceHandleFree(ctx.devHandle);
  delete ctx.args;

  MPI_Finalize();
}

void perfWarmup(PerfContext &ctx, PerfCollFn fn) {
  const size_t elementSize =
      ctx.useTypedReduction ? ctx.typeSize : sizeof(float);
  // Warmup for large size
  size_t largeCount = ctx.maxBytes / elementSize;
  for (int i = 0; i < ctx.numWarmupIters; i++) {
    fn(ctx, largeCount);
  }
  ctx.devHandle->streamSynchronize(ctx.stream);

  // Warmup for small size
  size_t smallCount = ctx.minBytes / elementSize;
  for (int i = 0; i < ctx.numWarmupIters; i++) {
    fn(ctx, smallCount);
  }
  ctx.devHandle->streamSynchronize(ctx.stream);
}

void perfBenchmarkLoop(PerfContext &ctx, PerfCollFn collFn,
                       PerfBwFactorFn bwFactorFn, PerfDataInitFn dataInitFn,
                       PerfPostIterFn postIterFn) {
  if (ctx.stepFactor <= 1) {
    fprintf(stderr, "Error: stepFactor must be > 1 (got %d)\n", ctx.stepFactor);
    return;
  }
  const size_t elementSize =
      ctx.useTypedReduction ? ctx.typeSize : sizeof(float);
  for (size_t size = ctx.minBytes; size <= ctx.maxBytes;
       size *= ctx.stepFactor) {
    size_t count = size / elementSize;
    size_t actualSize = count * elementSize;

    // Optional data initialization
    if (dataInitFn) {
      dataInitFn(ctx, actualSize, count);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Timed loop
    ctx.tim.reset();
    for (int i = 0; i < ctx.numIters; i++) {
      collFn(ctx, count);
    }
    ctx.devHandle->streamSynchronize(ctx.stream);

    // Compute average elapsed time across all ranks
    double elapsedTime = ctx.tim.elapsed() / ctx.numIters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= ctx.worldSize;

    // Bandwidth calculation
    double baseBw = (double)(actualSize) / 1.0E9 / elapsedTime;
    double algBw = baseBw;
    double factor = bwFactorFn ? bwFactorFn(ctx.totalProcs) : 1.0;
    double busBw = baseBw * factor;

    if (ctx.proc == 0 && ctx.color == 0) {
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: "
             "%lf GB/s; Bus bandwidth: %lf GB/s\n",
             actualSize, elapsedTime, algBw, busBw);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Optional post-iteration callback
    if (postIterFn) {
      postIterFn(ctx, actualSize, count);
    }
  }
}

void perfRootBenchmarkLoop(PerfContext &ctx, PerfRootCollFn collFn,
                           PerfBwFactorFn bwFactorFn,
                           PerfRootDataInitFn dataInitFn,
                           PerfRootPostIterFn postIterFn) {
  if (ctx.stepFactor <= 1) {
    fprintf(stderr, "Error: stepFactor must be > 1 (got %d)\n", ctx.stepFactor);
    return;
  }
  const size_t elementSize =
      ctx.useTypedReduction ? ctx.typeSize : sizeof(float);
  for (size_t size = ctx.minBytes; size <= ctx.maxBytes;
       size *= ctx.stepFactor) {
    const size_t reportedSize = (size / elementSize) * elementSize;
    int beginRoot, endRoot;
    double sumAlgBw = 0;
    double sumBusBw = 0;
    double sumTime = 0;
    int testCount = 0;

    if (ctx.root != -1) {
      beginRoot = endRoot = ctx.root;
    } else {
      beginRoot = 0;
      endRoot = ctx.totalProcs - 1;
    }

    for (int r = beginRoot; r <= endRoot; r++) {
      size_t count = size / elementSize;
      size_t actualSize = count * elementSize;

      if (dataInitFn) {
        dataInitFn(ctx, actualSize, count, r);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      ctx.tim.reset();
      for (int i = 0; i < ctx.numIters; i++) {
        collFn(ctx, count, r);
      }
      ctx.devHandle->streamSynchronize(ctx.stream);

      MPI_Barrier(MPI_COMM_WORLD);

      double elapsedTime = ctx.tim.elapsed() / ctx.numIters;
      MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsedTime /= ctx.worldSize;

      double baseBw = (double)(actualSize) / 1.0E9 / elapsedTime;
      double algBw = baseBw;
      double factor = bwFactorFn ? bwFactorFn(ctx.totalProcs) : 1.0;
      double busBw = baseBw * factor;
      sumAlgBw += algBw;
      sumBusBw += busBw;
      sumTime += elapsedTime;
      testCount++;

      if (postIterFn) {
        postIterFn(ctx, actualSize, count, r);
      }
    }

    if (ctx.proc == 0 && ctx.color == 0) {
      double algBw = sumAlgBw / testCount;
      double busBw = sumBusBw / testCount;
      double elapsedTime = sumTime / testCount;
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: "
             "%lf GB/s; Bus bandwidth: %lf GB/s\n",
             reportedSize, elapsedTime, algBw, busBw);
    }
  }
}
