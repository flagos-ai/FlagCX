/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_PARAM_H_
#define FLAGCX_PARAM_H_

#include "flagcx.h"
#include <stdint.h>

const char *userHomeDir();
void setEnvFile(const char *fileName);
void initEnv();
const char *flagcxGetEnv(const char *name);

void flagcxLoadParam(char const *env, int64_t deftVal, int64_t uninitialized,
                     int64_t *cache);

#define FLAGCX_PARAM(name, env, deftVal)                                       \
  int64_t flagcxParam##name() {                                                \
    constexpr int64_t uninitialized = INT64_MIN;                               \
    static_assert(deftVal != uninitialized,                                    \
                  "default value cannot be the uninitialized value.");         \
    static int64_t cache = uninitialized;                                      \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) ==          \
                             uninitialized,                                    \
                         false)) {                                             \
      flagcxLoadParam("FLAGCX_" env, deftVal, uninitialized, &cache);          \
    }                                                                          \
    return cache;                                                              \
  }

#endif
