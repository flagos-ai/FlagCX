/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_INT_DEBUG_H_
#define FLAGCX_INT_DEBUG_H_

#include "type.h"
#include <chrono>
#include <stdio.h>
#include <type_traits>

#include <limits.h>
#include <pthread.h>
#include <string.h>

typedef enum {
  FLAGCX_LOG_NONE = 0,
  FLAGCX_LOG_VERSION = 1,
  FLAGCX_LOG_WARN = 2,
  FLAGCX_LOG_INFO = 3,
  FLAGCX_LOG_ABORT = 4,
  FLAGCX_LOG_TRACE = 5
} flagcxDebugLogLevel;
typedef enum {
  FLAGCX_INIT = 1,
  FLAGCX_COLL = 2,
  FLAGCX_P2P = 4,
  FLAGCX_SHM = 8,
  FLAGCX_NET = 16,
  FLAGCX_GRAPH = 32,
  FLAGCX_TUNING = 64,
  FLAGCX_ENV = 128,
  FLAGCX_ALLOC = 256,
  FLAGCX_CALL = 512,
  FLAGCX_PROXY = 1024,
  FLAGCX_NVLS = 2048,
  FLAGCX_BOOTSTRAP = 4096,
  FLAGCX_REG = 8192,
  FLAGCX_KERNEL = 16384,
  FLAGCX_ALL = ~0
} flagcxDebugLogSubSys;

// Conform to pthread and NVTX standard
#define FLAGCX_THREAD_NAMELEN 16

extern int flagcxDebugLevel;
extern uint64_t flagcxDebugMask;
extern pthread_mutex_t flagcxDebugLock;
extern FILE *flagcxDebugFile;
extern flagcxResult_t getHostName(char *hostname, int maxlen, const char delim);

void flagcxDebugLog(flagcxDebugLogLevel level, unsigned long flags,
                    const char *filefunc, int line, const char *fmt, ...)
    __attribute__((format(printf, 5, 6)));

// Let code temporarily downgrade WARN into INFO
extern thread_local int flagcxDebugNoWarn;
extern char flagcxLastError[];

#define ENABLE_TRACE
#define WARN(...)                                                              \
  flagcxDebugLog(FLAGCX_LOG_WARN, FLAGCX_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...)                                                       \
  flagcxDebugLog(FLAGCX_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define TRACE_CALL(...)                                                        \
  flagcxDebugLog(FLAGCX_LOG_TRACE, FLAGCX_CALL, __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...)                                                      \
  flagcxDebugLog(FLAGCX_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::steady_clock::time_point flagcxEpoch;
#else
#define TRACE(...)
#endif

void flagcxSetThreadName(pthread_t thread, const char *fmt, ...);

typedef void (*flagcxDebugLogger_t)(flagcxDebugLogLevel level,
                                    unsigned long flags, const char *file,
                                    int line, const char *fmt, ...);

// time recorder
#define TIMER_COLL_TOTAL 0
#define TIMER_COLL_CALC 1
#define TIMER_COLL_COMM 2
#define TIMER_COLL_MEM 3
#define TIMER_COLL_MEM_D2H 4
#define TIMER_COLL_MEM_H2D 5
#define TIMER_COLL_ALLOC 6
#define TIMER_COLL_FREE 7
#define TIMERS_COLL_COUNT 8

#endif
