// Unit tests for environment parameter loading.
// Source: flagcx/service/include/param.h, flagcx/service/param.cc
// Links against libflagcx.

#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

#include "param.h"

TEST(Param, GetEnvReturnsNullForMissing) {
  // Ensure the variable does not exist
  unsetenv("FLAGCX_NONEXISTENT_VAR_12345");
  const char *val = flagcxGetEnv("FLAGCX_NONEXISTENT_VAR_12345");
  EXPECT_EQ(val, nullptr);
}

TEST(Param, GetEnvReturnsSetValue) {
  setenv("FLAGCX_TEST_VAR_UT", "42", 1);
  const char *val = flagcxGetEnv("FLAGCX_TEST_VAR_UT");
  ASSERT_NE(val, nullptr);
  EXPECT_STREQ(val, "42");
  unsetenv("FLAGCX_TEST_VAR_UT");
}

TEST(Param, LoadParamUsesDefault) {
  // Ensure env var is not set
  unsetenv("FLAGCX_UT_LOAD_PARAM_TEST");
  constexpr int64_t uninitialized = INT64_MIN;
  int64_t cache = uninitialized;
  flagcxLoadParam("FLAGCX_UT_LOAD_PARAM_TEST", /*deftVal=*/99, uninitialized,
                  &cache);
  EXPECT_EQ(cache, 99);
}

TEST(Param, LoadParamReadsEnv) {
  setenv("FLAGCX_UT_LOAD_PARAM_ENV", "777", 1);
  constexpr int64_t uninitialized = INT64_MIN;
  int64_t cache = uninitialized;
  flagcxLoadParam("FLAGCX_UT_LOAD_PARAM_ENV", /*deftVal=*/0, uninitialized,
                  &cache);
  EXPECT_EQ(cache, 777);
  unsetenv("FLAGCX_UT_LOAD_PARAM_ENV");
}
