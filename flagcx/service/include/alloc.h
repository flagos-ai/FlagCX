/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_ALLOC_H_
#define FLAGCX_ALLOC_H_

#include "align.h"
#include "check.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

uint64_t clockNano(); // from utils.h with which we have a circular dependency

template <typename T>
flagcxResult_t flagcxCallocDebug(T **ptr, size_t nelem, const char *filefunc,
                                 int line) {
  void *p = malloc(nelem * sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem * sizeof(T));
    return flagcxSystemError;
  }
  // INFO(FLAGCX_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line,
  // nelem*sizeof(T), p);
  memset(p, 0, nelem * sizeof(T));
  *ptr = (T *)p;
  return flagcxSuccess;
}
#define flagcxCalloc(...) flagcxCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
flagcxResult_t flagcxRealloc(T **ptr, size_t oldNelem, size_t nelem) {
  if (nelem < oldNelem)
    return flagcxInternalError;
  if (nelem == oldNelem)
    return flagcxSuccess;

  T *oldp = *ptr;
  T *p = (T *)malloc(nelem * sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem * sizeof(T));
    return flagcxSystemError;
  }
  memcpy(p, oldp, oldNelem * sizeof(T));
  free(oldp);
  memset(p + oldNelem, 0, (nelem - oldNelem) * sizeof(T));
  *ptr = (T *)p;
  INFO(FLAGCX_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p",
       oldNelem * sizeof(T), nelem * sizeof(T), *ptr);
  return flagcxSuccess;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline flagcxResult_t flagcxIbMallocDebug(void **ptr, size_t size,
                                          const char *filefunc, int line) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void *p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0)
    return flagcxSystemError;
  memset(p, 0, size);
  *ptr = p;
  INFO(FLAGCX_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size,
       *ptr);
  return flagcxSuccess;
}
#define flagcxIbMalloc(...) flagcxIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
