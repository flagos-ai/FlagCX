#pragma once

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

// Verify that two buffers match element-by-element within a tolerance.
// Reports the first mismatch with index, expected, and actual values.
template <typename T>
::testing::AssertionResult verifyBuffer(const T *actual, const T *expected,
                                        size_t count, double tolerance = 1e-5) {
  for (size_t i = 0; i < count; i++) {
    double diff = std::fabs((double)actual[i] - (double)expected[i]);
    double magnitude = std::max(std::fabs((double)expected[i]), (double)1e-10);
    if (diff / magnitude > tolerance) {
      std::ostringstream ss;
      ss << "Mismatch at index " << i << ": expected " << expected[i]
         << ", got " << actual[i] << " (diff=" << diff
         << ", relErr=" << diff / magnitude << ")";
      return ::testing::AssertionFailure() << ss.str();
    }
  }
  return ::testing::AssertionSuccess();
}

// Fill a buffer with a deterministic pattern based on index and seed.
template <typename T>
void fillTestData(T *buffer, size_t count, T seed = 0) {
  for (size_t i = 0; i < count; i++) {
    buffer[i] = static_cast<T>((i % 10) + seed);
  }
}
