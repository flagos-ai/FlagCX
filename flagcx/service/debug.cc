/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "core.h"
#include "flagcx_net.h"
#include "param.h"
#include <stdarg.h>
#include <stdlib.h>
#include <sys/syscall.h>

int flagcxDebugLevel = -1;
static int pid = -1;
static char hostname[1024];
thread_local int flagcxDebugNoWarn = 0;
char flagcxLastError[1024] =
    ""; // Global string for the last error in human readable form
uint64_t flagcxDebugMask =
    FLAGCX_INIT | FLAGCX_ENV; // Default debug sub-system mask is INIT and ENV
FILE *flagcxDebugFile = stdout;
pthread_mutex_t flagcxDebugLock = PTHREAD_MUTEX_INITIALIZER;
std::chrono::steady_clock::time_point flagcxEpoch;

static __thread int tid = -1;

void flagcxDebugInit() {
  pthread_mutex_lock(&flagcxDebugLock);
  if (flagcxDebugLevel != -1) {
    pthread_mutex_unlock(&flagcxDebugLock);
    return;
  }
  const char *flagcx_debug = flagcxGetEnv("FLAGCX_DEBUG");
  int tempNcclDebugLevel = -1;
  if (flagcx_debug == NULL) {
    tempNcclDebugLevel = FLAGCX_LOG_NONE;
  } else if (strcasecmp(flagcx_debug, "VERSION") == 0) {
    tempNcclDebugLevel = FLAGCX_LOG_VERSION;
  } else if (strcasecmp(flagcx_debug, "WARN") == 0) {
    tempNcclDebugLevel = FLAGCX_LOG_WARN;
  } else if (strcasecmp(flagcx_debug, "INFO") == 0) {
    tempNcclDebugLevel = FLAGCX_LOG_INFO;
  } else if (strcasecmp(flagcx_debug, "ABORT") == 0) {
    tempNcclDebugLevel = FLAGCX_LOG_ABORT;
  } else if (strcasecmp(flagcx_debug, "TRACE") == 0) {
    tempNcclDebugLevel = FLAGCX_LOG_TRACE;
  }

  /* Parse the FLAGCX_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  const char *flagcxDebugSubsysEnv = flagcxGetEnv("FLAGCX_DEBUG_SUBSYS");
  if (flagcxDebugSubsysEnv != NULL) {
    int invert = 0;
    if (flagcxDebugSubsysEnv[0] == '^') {
      invert = 1;
      flagcxDebugSubsysEnv++;
    }
    flagcxDebugMask = invert ? ~0ULL : 0ULL;
    char *flagcxDebugSubsys = strdup(flagcxDebugSubsysEnv);
    char *subsys = strtok(flagcxDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = FLAGCX_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = FLAGCX_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = FLAGCX_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = FLAGCX_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = FLAGCX_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = FLAGCX_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = FLAGCX_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = FLAGCX_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = FLAGCX_ALLOC;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = FLAGCX_CALL;
      } else if (strcasecmp(subsys, "PROXY") == 0) {
        mask = FLAGCX_PROXY;
      } else if (strcasecmp(subsys, "NVLS") == 0) {
        mask = FLAGCX_NVLS;
      } else if (strcasecmp(subsys, "BOOTSTRAP") == 0) {
        mask = FLAGCX_BOOTSTRAP;
      } else if (strcasecmp(subsys, "REG") == 0) {
        mask = FLAGCX_REG;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = FLAGCX_ALL;
      }
      if (mask) {
        if (invert)
          flagcxDebugMask &= ~mask;
        else
          flagcxDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(flagcxDebugSubsys);
  }

  // Cache pid and hostname
  getHostName(hostname, 1024, '.');
  pid = getpid();

  /* Parse and expand the FLAGCX_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * FLAGCX_DEBUG level is > VERSION
   */
  const char *flagcxDebugFileEnv = flagcxGetEnv("FLAGCX_DEBUG_FILE");
  if (tempNcclDebugLevel > FLAGCX_LOG_VERSION && flagcxDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX + 1] = "";
    char *dfn = debugFn;
    while (flagcxDebugFileEnv[c] != '\0' && c < PATH_MAX) {
      if (flagcxDebugFileEnv[c++] != '%') {
        *dfn++ = flagcxDebugFileEnv[c - 1];
        continue;
      }
      switch (flagcxDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", pid);
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = flagcxDebugFileEnv[c - 1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != nullptr) {
        setbuf(file, nullptr); // disable buffering
        flagcxDebugFile = file;
      }
    }
  }

  flagcxEpoch = std::chrono::steady_clock::now();
  __atomic_store_n(&flagcxDebugLevel, tempNcclDebugLevel, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&flagcxDebugLock);
}

FLAGCX_PARAM(WarnSetDebugInfo, "WARN_ENABLE_DEBUG_INFO", 0);

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void flagcxDebugLog(flagcxDebugLogLevel level, unsigned long flags,
                    const char *filefunc, int line, const char *fmt, ...) {
  if (__atomic_load_n(&flagcxDebugLevel, __ATOMIC_ACQUIRE) == -1)
    flagcxDebugInit();
  if (flagcxDebugNoWarn != 0 && level == FLAGCX_LOG_WARN) {
    level = FLAGCX_LOG_INFO;
    flags = flagcxDebugNoWarn;
  }

  // Save the last error (WARN) as a human readable string
  if (level == FLAGCX_LOG_WARN) {
    pthread_mutex_lock(&flagcxDebugLock);
    va_list vargs;
    va_start(vargs, fmt);
    (void)vsnprintf(flagcxLastError, sizeof(flagcxLastError), fmt, vargs);
    va_end(vargs);
    pthread_mutex_unlock(&flagcxDebugLock);
  }
  if (flagcxDebugLevel < level || ((flags & flagcxDebugMask) == 0))
    return;

  if (tid == -1) {
    tid = syscall(SYS_gettid);
  }

  int cudaDev = 0;
  /**
   * TODO: How to get the GPU currently in use
   **/

  char buffer[1024];
  size_t len = 0;
  if (level == FLAGCX_LOG_WARN) {
    len = snprintf(buffer, sizeof(buffer), "\n%s:%d:%d [%d] %s:%d FLAGCX WARN ",
                   hostname, pid, tid, cudaDev, filefunc, line);
    if (flagcxParamWarnSetDebugInfo())
      flagcxDebugLevel = FLAGCX_LOG_INFO;
  } else if (level == FLAGCX_LOG_INFO) {
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] FLAGCX INFO ",
                   hostname, pid, tid, cudaDev);
  } else if (level == FLAGCX_LOG_TRACE && flags == FLAGCX_CALL) {
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d FLAGCX CALL ", hostname,
                   pid, tid);
  } else if (level == FLAGCX_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - flagcxEpoch;
    double timestamp =
        std::chrono::duration_cast<std::chrono::duration<double>>(delta)
            .count() *
        1000;
    len =
        snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] %f %s:%d FLAGCX TRACE ",
                 hostname, pid, tid, cudaDev, timestamp, filefunc, line);
  }

  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    len += vsnprintf(buffer + len, sizeof(buffer) - len, fmt, vargs);
    va_end(vargs);
    // vsnprintf may return len > sizeof(buffer) in the case of a truncated
    // output. Rewind len so that we can replace the final \0 by \n
    if (len > sizeof(buffer))
      len = sizeof(buffer) - 1;
    buffer[len++] = '\n';
    fwrite(buffer, 1, len, flagcxDebugFile);
  }
}

FLAGCX_PARAM(SetThreadName, "SET_THREAD_NAME", 0);

void flagcxSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (flagcxParamSetThreadName() != 1)
    return;
  char threadName[FLAGCX_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, FLAGCX_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
}
