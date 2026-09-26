/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Transport-agnostic one-sided handle info and globals.
 * Moved from ib_common.h so that core layer files do not depend on
 * the IB adaptor header.
 ************************************************************************/

#ifndef FLAGCX_ONESIDED_H_
#define FLAGCX_ONESIDED_H_

#include <stdint.h>

#include "comm.h" // for flagcxHeteroComm_t
#include "onesided_types.h"

// Internal implementation used by sym_heap and flagcxCommRegister
flagcxResult_t flagcxOneSideRegisterInternal(flagcxHeteroComm_t comm,
                                             void *buff, size_t size);

// Build IPC peer pointer table for a user buffer (intra-node D2D bypass).
// Stores results in comm->ipcTable and returns the table index.
// Returns -1 on failure (IPC not available for this buffer).
struct flagcxComm;
// Resolve the allocation exported by an IPC handle and the offset of the user
// buffer within it. Backends without allocation-range introspection retain the
// legacy exact-pointer behavior.
flagcxResult_t flagcxGetIpcExportRange(const void *buff, size_t size,
                                       void **exportBase,
                                       size_t *allocationSize,
                                       size_t *userOffset);
flagcxResult_t flagcxResolveIpcPeerAddress(void *importedBase,
                                           size_t allocationSize,
                                           size_t userOffset, size_t userSize,
                                           void **peerPtr);
int buildIpcPeerPointers(struct flagcxComm *comm, void *buff, size_t size);

#endif // FLAGCX_ONESIDED_H_
