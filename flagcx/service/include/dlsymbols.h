/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_DLSYMBOLS_H_
#define FLAGCX_DLSYMBOLS_H_

#include "flagcx.h"
#include <dlfcn.h>

extern void *flagcxOpenLib(const char *path, int flags,
                           void (*error_handler)(const char *, int,
                                                 const char *));

// Function pointer types for custom operations
template <typename T, typename... Args>
using flagcxCustomOpFunc_t = T (*)(Args...);
using flagcxLaunchFunc_t = flagcxCustomOpFunc_t<void, flagcxStream_t, void *>;

// Load a custom operation symbol from a shared library
template <typename T>
inline flagcxResult_t loadCustomOpSymbol(const char *path, const char *name,
                                         T *fn) {
  void *handle = flagcxOpenLib(
      path, RTLD_LAZY, [](const char *p, int err, const char *msg) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
      });
  if (!handle)
    return flagcxSystemError;

  void *sym = dlsym(handle, name);
  if (!sym) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    dlclose(handle);
    return flagcxSystemError;
  }

  *fn = (T)sym;
  return flagcxSuccess;
}

inline flagcxResult_t loadKernelSymbol(const char *path, const char *name,
                                       flagcxLaunchFunc_t *fn) {
  return loadCustomOpSymbol<flagcxLaunchFunc_t>(path, name, fn);
}

#endif // FLAGCX_DLSYMBOLS_H_
