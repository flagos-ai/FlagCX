#pragma once

#include "flagcx.h"
#include <cstdlib>
#include <cstring>

// Test configuration chooses an explicit Window mode matching its allocation.
// The library itself never changes an explicitly requested mode.
inline int flagcxTestWindowFlags() {
  const char *mode = std::getenv("FLAGCX_TEST_WIN_MODE");
  if (mode != nullptr) {
    if (std::strcmp(mode, "asymmetric") == 0)
      return FLAGCX_WIN_ASYMMETRIC;
    if (std::strcmp(mode, "symmetric") == 0)
      return FLAGCX_WIN_SYMMETRIC;
    return -1;
  }
#if defined(FLAGCX_COMM_TRAITS_CCL) || defined(FLAGCX_TEST_ALLOCATOR_SHMEM)
  return FLAGCX_WIN_SYMMETRIC;
#else
  const char *vmm = std::getenv("FLAGCX_VMM_ENABLE");
  return vmm != nullptr && std::strcmp(vmm, "0") == 0 ? FLAGCX_WIN_ASYMMETRIC
                                                      : FLAGCX_WIN_SYMMETRIC;
#endif
}
