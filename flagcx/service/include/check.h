/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_CHECKS_H_
#define FLAGCX_CHECKS_H_

#include "debug.h"
#include "type.h"
#include <errno.h>

// Check system calls
#define SYSCHECK(call, name)                                                   \
  do {                                                                         \
    int retval;                                                                \
    SYSCHECKVAL(call, name, retval);                                           \
  } while (false)

#define SYSCHECKVAL(call, name, retval)                                        \
  do {                                                                         \
    SYSCHECKSYNC(call, name, retval);                                          \
    if (retval == -1) {                                                        \
      WARN("Call to " name " failed : %s", strerror(errno));                   \
      return flagcxSystemError;                                                \
    }                                                                          \
  } while (false)

#define SYSCHECKSYNC(call, name, retval)                                       \
  do {                                                                         \
    retval = call;                                                             \
    if (retval == -1 &&                                                        \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {         \
      INFO(FLAGCX_ALL, "Call to " name " returned %s, retrying",               \
           strerror(errno));                                                   \
    } else {                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (true)

#define SYSCHECKGOTO(statement, RES, label)                                    \
  do {                                                                         \
    if ((statement) == -1) {                                                   \
      /* Print the back trace*/                                                \
      RES = flagcxSystemError;                                                 \
      INFO(FLAGCX_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__, RES,            \
           strerror(errno));                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#define NEQCHECK(statement, value)                                             \
  do {                                                                         \
    if ((statement) != value) {                                                \
      /* Print the back trace*/                                                \
      INFO(FLAGCX_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__,                 \
           flagcxSystemError, strerror(errno));                                \
      return flagcxSystemError;                                                \
    }                                                                          \
  } while (0);

#define NEQCHECKGOTO(statement, value, RES, label)                             \
  do {                                                                         \
    if ((statement) != value) {                                                \
      /* Print the back trace*/                                                \
      RES = flagcxSystemError;                                                 \
      INFO(FLAGCX_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__, RES,            \
           strerror(errno));                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#define EQCHECK(statement, value)                                              \
  do {                                                                         \
    if ((statement) == value) {                                                \
      /* Print the back trace*/                                                \
      INFO(FLAGCX_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__,                 \
           flagcxSystemError, strerror(errno));                                \
      return flagcxSystemError;                                                \
    }                                                                          \
  } while (0);

#define EQCHECKGOTO(statement, value, RES, label)                              \
  do {                                                                         \
    if ((statement) == value) {                                                \
      /* Print the back trace*/                                                \
      RES = flagcxSystemError;                                                 \
      INFO(FLAGCX_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__, RES,            \
           strerror(errno));                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0);

// Propagate errors up
#define FLAGCXCHECK(call)                                                      \
  do {                                                                         \
    flagcxResult_t RES = call;                                                 \
    if (RES != flagcxSuccess && RES != flagcxInProgress) {                     \
      /* Print the back trace*/                                                \
      if (flagcxDebugNoWarn == 0)                                              \
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      return RES;                                                              \
    }                                                                          \
  } while (0);

#define FLAGCXCHECKGOTO(call, RES, label)                                      \
  do {                                                                         \
    RES = call;                                                                \
    if (RES != flagcxSuccess && RES != flagcxInProgress) {                     \
      /* Print the back trace*/                                                \
      if (flagcxDebugNoWarn == 0)                                              \
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#define FLAGCXWAIT(call, cond, abortFlagPtr)                                   \
  do {                                                                         \
    volatile uint32_t *tmpAbortFlag = (abortFlagPtr);                          \
    flagcxResult_t RES = call;                                                 \
    if (RES != flagcxSuccess && RES != flagcxInProgress) {                     \
      if (flagcxDebugNoWarn == 0)                                              \
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      return flagcxInternalError;                                              \
    }                                                                          \
    if (tmpAbortFlag)                                                          \
      NEQCHECK(*tmpAbortFlag, 0);                                              \
  } while (!(cond));

#define FLAGCXWAITGOTO(call, cond, abortFlagPtr, RES, label)                   \
  do {                                                                         \
    volatile uint32_t *tmpAbortFlag = (abortFlagPtr);                          \
    RES = call;                                                                \
    if (RES != flagcxSuccess && RES != flagcxInProgress) {                     \
      if (flagcxDebugNoWarn == 0)                                              \
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      goto label;                                                              \
    }                                                                          \
    if (tmpAbortFlag)                                                          \
      NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label);                              \
  } while (!(cond));

#define FLAGCXCHECKTHREAD(a, args)                                             \
  do {                                                                         \
    if (((args)->ret = (a)) != flagcxSuccess &&                                \
        (args)->ret != flagcxInProgress) {                                     \
      INFO(FLAGCX_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__,      \
           (args)->ret);                                                       \
      return args;                                                             \
    }                                                                          \
  } while (0)

#define CUDACHECKTHREAD(a)                                                     \
  do {                                                                         \
    if ((a) != cudaSuccess) {                                                  \
      INFO(FLAGCX_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__,      \
           args->ret);                                                         \
      args->ret = flagcxUnhandledCudaError;                                    \
      return args;                                                             \
    }                                                                          \
  } while (0)

// Type definitions
#define PTHREADCHECKGOTO(statement, name, RES, label)                          \
  do {                                                                         \
    int retval = (statement);                                                  \
    if (retval != 0) {                                                         \
      WARN("Call to " name " failed: %s", strerror(retval));                   \
      RES = flagcxSystemError;                                                 \
      goto label;                                                              \
    }                                                                          \
  } while (0)
#endif
