/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * SHMEM Adaptor Interface — struct of function pointers for SHMEM backends.
 * Implementations (e.g., nvshmem_adaptor.cc) fill these pointers.
 ************************************************************************/

#ifndef FLAGCX_SHMEM_ADAPTOR_H_
#define FLAGCX_SHMEM_ADAPTOR_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for SHMEM-backed device comm state
struct flagcxShmemCommInternal;
typedef struct flagcxShmemCommInternal *flagcxShmemComm_t;

// Forward declarations
struct flagcxDevCommRequirements;

struct flagcxShmemAdaptor {
  const char *name;

  // Lifecycle (reference-counted)
  flagcxResult_t (*init)(int rank, int nRanks);
  flagcxResult_t (*finalize)();

  // Symmetric memory management
  flagcxResult_t (*malloc)(void **ptr, size_t size);
  flagcxResult_t (*free)(void *ptr);

  // Device comm setup
  flagcxResult_t (*devCommCreate)(flagcxComm_t comm,
                                  const struct flagcxDevCommRequirements *reqs,
                                  flagcxShmemComm_t *shmemComm);
  flagcxResult_t (*devCommDestroy)(flagcxShmemComm_t shmemComm);
};

typedef struct flagcxShmemAdaptor flagcxShmemAdaptor_t;

// Global adaptor instance (set at load time by nvshmem_adaptor.cc).
// Defaults to nullptr when USE_SHMEM is not enabled.
extern flagcxShmemAdaptor_t *shmemAdaptor __attribute__((weak));

#ifdef __cplusplus
}
#endif

#endif // FLAGCX_SHMEM_ADAPTOR_H_
