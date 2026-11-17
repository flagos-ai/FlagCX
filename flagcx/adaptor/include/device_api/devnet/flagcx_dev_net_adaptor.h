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
  // Registration is collective and belongs to the public Window lifecycle.
  flagcxResult_t (*registerWindow)(flagcxComm_t, flagcxDevComm_t,
                                   flagcxWindow_t);
  // Check local references and quiesce device work before any rank releases
  // its registration. This phase must not release resources.
  flagcxResult_t (*prepareWindowDeregister)(flagcxWindow_t);
  flagcxResult_t (*deregisterWindow)(flagcxWindow_t);
  // Binding a local DevMem view must never exchange peer metadata.
  flagcxResult_t (*bindDevMem)(flagcxWindow_t, flagcxDevMem_t, size_t, void *);
  flagcxResult_t (*unbindDevMem)(flagcxDevMem_t);
};

// Supplied by the single GDA implementation selected at build time.
extern const flagcxDevNetAdaptor flagcxDevNetGda;

#endif // FLAGCX_DEV_NET_ADAPTOR_H_
