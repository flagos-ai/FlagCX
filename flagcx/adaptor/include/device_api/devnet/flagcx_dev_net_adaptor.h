/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-side GDA lifecycle for the DefaultBackend. Exactly one implementation
 * is linked into every build; device operations remain platform traits.
 ************************************************************************/

#ifndef FLAGCX_DEV_NET_ADAPTOR_H_
#define FLAGCX_DEV_NET_ADAPTOR_H_

#include "flagcx_device_api.h"

struct flagcxDevNetAdaptor {
  const char *name;
  bool isGda; // Whether this build provides a GDA candidate.

  // createComm performs capability checks and collective validation before
  // publishing a device context. There is no separate probe lifecycle.
  flagcxResult_t (*createComm)(flagcxComm_t, flagcxDevComm_t,
                               const flagcxDevCommRequirements *);
  flagcxResult_t (*destroyComm)(flagcxDevComm_t);
  // Registration is collective and belongs to the public Window lifecycle.
  flagcxResult_t (*registerWindow)(flagcxComm_t, flagcxDevComm_t,
                                   flagcxWindow_t);
  // Check local references and quiesce device work before any rank releases
  // its registration. Error/abort requires collective NIC retirement first.
  // This phase must not release resources.
  flagcxResult_t (*prepareWindowDeregister)(flagcxWindow_t);
  flagcxResult_t (*deregisterWindow)(flagcxWindow_t);
  // Binding a local DevMem view must never exchange peer metadata.
  flagcxResult_t (*bindDevMem)(flagcxWindow_t, flagcxDevMem_t, size_t, void *);
  flagcxResult_t (*unbindDevMem)(flagcxDevMem_t);
  flagcxResult_t (*pollError)(flagcxDevComm_t);
  flagcxResult_t (*abortComm)(flagcxDevComm_t);
  // Collective GPU/NIC quiesce; retains MRs until every rank has retired QPs.
  flagcxResult_t (*prepareCommDestroy)(flagcxDevComm_t);
};

// Supplied by the single DevNet implementation selected at build time.
extern const flagcxDevNetAdaptor devNetAdaptor;

// Agree whether every rank can attempt the build-selected GDA candidate.
flagcxResult_t flagcxDevNetConfigure(flagcxComm_t comm, bool wantsNet,
                                     bool forceOneSidedNet, bool *attemptGda);

// Select GDA only on collective success; a clean unavailable candidate uses
// Proxy.
flagcxResult_t
flagcxDevNetCreateComm(flagcxComm_t comm, flagcxDevComm_t devComm,
                       const flagcxDevCommRequirements *requirements);

#endif // FLAGCX_DEV_NET_ADAPTOR_H_
