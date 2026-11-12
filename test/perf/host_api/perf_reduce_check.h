#pragma once

#include "perf_common.h"

#include <algorithm>
#include <vector>

// Use small, exactly representable values so the reference is independent of
// reduction order, including for half and bfloat16.
static inline unsigned int perfReduceInputValue(int rank, size_t element,
                                                int ranks,
                                                flagcxDataType_t datatype,
                                                flagcxRedOp_t op) {
  const int selected = static_cast<int>((element / 2) % ranks);
  switch (op) {
    case flagcxSum:
    case flagcxMax: {
      if (rank == selected)
        return 1 + (element & 1);
      const int next = selected == ranks - 1 ? 0 : selected + 1;
      return ranks > 1 && rank == next ? 1 : 0;
    }
    case flagcxProd:
    case flagcxMin:
      return rank == selected ? ((element & 1) ? 2 : 0) : 1;
    case flagcxAvg:
      // Narrow integer accumulators cannot represent a sum of more than their
      // maximum value. Use zero in that rare case rather than testing overflow.
      if ((datatype == flagcxInt8 && ranks > 127) ||
          (datatype == flagcxUint8 && ranks > 255))
        return 0;
      if (ranks == 1)
        return 1;
      if (rank == selected)
        return 0;
      return rank == (selected == ranks - 1 ? 0 : selected + 1) ? 2 : 1;
    default:
      perfAbort("unsupported reduction operation", __FILE__, __LINE__);
  }
}

static inline unsigned int perfReduceExpectedValue(size_t element, int ranks,
                                                   flagcxDataType_t datatype,
                                                   flagcxRedOp_t op) {
  switch (op) {
    case flagcxSum:
      return (ranks > 1 ? 2 : 1) + (element & 1);
    case flagcxMax:
      return 1 + (element & 1);
    case flagcxProd:
      return (element & 1) ? 2 : 0;
    case flagcxMin:
      return (element & 1) ? (ranks == 1 ? 2 : 1) : 0;
    case flagcxAvg:
      return ((datatype == flagcxInt8 && ranks > 127) ||
              (datatype == flagcxUint8 && ranks > 255))
                 ? 0
                 : 1;
    default:
      perfAbort("unsupported reduction operation", __FILE__, __LINE__);
  }
}

static inline void perfReduceStore(unsigned char *dst,
                                   flagcxDataType_t datatype,
                                   unsigned int value) {
  switch (datatype) {
    case flagcxInt8: {
      const int8_t v = static_cast<int8_t>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxUint8: {
      const uint8_t v = static_cast<uint8_t>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxInt32: {
      const int32_t v = static_cast<int32_t>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxUint32: {
      const uint32_t v = static_cast<uint32_t>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxInt64: {
      const int64_t v = static_cast<int64_t>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxUint64: {
      const uint64_t v = static_cast<uint64_t>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxFloat16: {
      if (value > 3)
        perfAbort("half reduction value is out of range", __FILE__, __LINE__);
      const uint16_t bits[] = {0x0000, 0x3c00, 0x4000, 0x4200};
      const uint16_t encoded = bits[value];
      memcpy(dst, &encoded, sizeof(encoded));
      break;
    }
    case flagcxFloat32: {
      const float v = static_cast<float>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxFloat64: {
      const double v = static_cast<double>(value);
      memcpy(dst, &v, sizeof(v));
      break;
    }
    case flagcxBfloat16: {
      if (value > 3)
        perfAbort("bfloat16 reduction value is out of range", __FILE__,
                  __LINE__);
      const uint16_t bits[] = {0x0000, 0x3f80, 0x4000, 0x4040};
      const uint16_t encoded = bits[value];
      memcpy(dst, &encoded, sizeof(encoded));
      break;
    }
    default:
      perfAbort("unsupported reduction datatype", __FILE__, __LINE__);
  }
}

static inline void perfReduceFillInput(unsigned char *dst, size_t count,
                                       int rank, int ranks,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op) {
  if (ranks <= 0 || rank < 0 || rank >= ranks)
    perfAbort("invalid reduction rank", __FILE__, __LINE__);
  if (count == 0)
    return;
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  if (typeSize == 0)
    perfAbort("invalid reduction datatype", __FILE__, __LINE__);
  const size_t seedCount = std::min(count, 2 * static_cast<size_t>(ranks));
  const size_t seedBytes = seedCount * typeSize;
  std::vector<unsigned char> seed(seedBytes);
  for (size_t i = 0; i < seedCount; ++i)
    perfReduceStore(seed.data() + i * typeSize, datatype,
                    perfReduceInputValue(rank, i, ranks, datatype, op));
  memcpy(dst, seed.data(), seedBytes);
  size_t filled = seedBytes;
  const size_t totalBytes = count * typeSize;
  while (filled < totalBytes) {
    const size_t bytes = std::min(filled, totalBytes - filled);
    memcpy(dst + filled, dst, bytes);
    filled += bytes;
  }
}

static inline bool perfReduceMatches(const unsigned char *actual,
                                     const unsigned char *expected,
                                     flagcxDataType_t datatype,
                                     unsigned int expectedValue) {
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  if (memcmp(actual, expected, typeSize) == 0)
    return true;
  // Both signs of floating-point zero have the same numerical value.
  if (expectedValue == 0) {
    if (datatype == flagcxFloat16 || datatype == flagcxBfloat16) {
      uint16_t bits;
      memcpy(&bits, actual, sizeof(bits));
      return bits == 0x8000;
    }
    if (datatype == flagcxFloat32) {
      float value;
      memcpy(&value, actual, sizeof(value));
      return value == 0;
    }
    if (datatype == flagcxFloat64) {
      double value;
      memcpy(&value, actual, sizeof(value));
      return value == 0;
    }
  }
  return false;
}

static inline void perfReduceCheckOutput(const unsigned char *actual,
                                         size_t count, size_t firstElement,
                                         int rank, int ranks,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op,
                                         const char *operation) {
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  if (typeSize == 0 || typeSize > sizeof(uint64_t))
    perfAbort("invalid reduction datatype", __FILE__, __LINE__);
  const size_t blockCount = 4096 / typeSize;
  unsigned char pattern[4096];
  for (size_t i = 0; i < blockCount; ++i)
    perfReduceStore(
        pattern + i * typeSize, datatype,
        perfReduceExpectedValue(firstElement + i, ranks, datatype, op));
  for (size_t offset = 0; offset < count; offset += blockCount) {
    const size_t elements = std::min(blockCount, count - offset);
    if (memcmp(actual + offset * typeSize, pattern, elements * typeSize) == 0)
      continue;
    for (size_t i = 0; i < elements; ++i) {
      const size_t index = firstElement + offset + i;
      const unsigned int expectedValue =
          perfReduceExpectedValue(index, ranks, datatype, op);
      unsigned char expected[sizeof(uint64_t)] = {};
      perfReduceStore(expected, datatype, expectedValue);
      const unsigned char *item = actual + (offset + i) * typeSize;
      if (!perfReduceMatches(item, expected, datatype, expectedValue)) {
        uint64_t actualBits = 0;
        memcpy(&actualBits, item, typeSize);
        fprintf(stderr,
                "%s rank=%d datatype=%s op=%s element=%zu expected=%u "
                "actual_bits=0x%llx\n",
                operation, rank, test_typenames[static_cast<int>(datatype)],
                test_opnames[static_cast<int>(op)], index, expectedValue,
                static_cast<unsigned long long>(actualBits));
        perfAbort("reduction data mismatch", __FILE__, __LINE__);
      }
    }
  }
}
