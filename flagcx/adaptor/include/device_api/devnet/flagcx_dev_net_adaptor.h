/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-side GDA lifecycle for the DefaultBackend. Exactly one implementation
 * is linked into a GDA-enabled build; device operations remain platform traits.
 ************************************************************************/

#ifndef FLAGCX_DEV_NET_ADAPTOR_H_
#define FLAGCX_DEV_NET_ADAPTOR_H_

#include "flagcx_device_api.h"

struct flagcxDevNetAdaptor {
  // createComm performs capability checks and collective selection before
  // publishing a device context. There is no separate probe lifecycle.
  flagcxResult_t (*createComm)(flagcxComm_t, flagcxDevComm_t,
                               const flagcxDevCommRequirements *, int);
  flagcxResult_t (*destroyComm)(flagcxDevComm_t);
  flagcxResult_t (*createWindow)(flagcxComm_t, flagcxDevComm_t, flagcxDevMem_t,
                                 size_t, void *);
  flagcxResult_t (*destroyWindow)(flagcxDevMem_t);
};

// Supplied by the single GDA implementation selected at build time.
extern const flagcxDevNetAdaptor flagcxDevNetGda;

#endif // FLAGCX_DEV_NET_ADAPTOR_H_
