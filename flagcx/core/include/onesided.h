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

#include "comm.h"                  // for flagcxHeteroComm_t
#include "flagcx_device_adaptor.h" // for flagcxVmmMrRoute_t
#include "onesided_types.h"

flagcxVmmMrRoute_t flagcxOneSideSelectVmmMrRoute(uint32_t deviceCaps,
                                                 int netPtrSupport,
                                                 bool dmaBufExportSupported,
                                                 bool hasDmaBufRegistration,
                                                 bool hasVaRegistration,
                                                 bool dmaBufEnabled = false);

// Select the lowest slot that is free on every rank. occupancy is laid out as
// [nRanks][nSlots]. Slot zero has special connection-mesh ownership and must
// either exist everywhere or nowhere.
flagcxResult_t flagcxOneSideSelectCommonPublishSlot(const uint8_t *occupancy,
                                                    int nRanks, int nSlots,
                                                    int *slot);

// Return allocation-time VMM provenance for a range owned by flagcxMemAlloc.
// This is only an ownership-registry hint. Symmetric-window registration also
// probes the native allocation through symPhysAlloc and must not treat a
// registry miss as proof that a buffer is not VMM-backed. Other public RMA
// registration entry points retain their ordinary-memory behavior on a miss.
bool flagcxOneSideRegistryRangeIsVmm(const void *buff, size_t size);

// Register one local MR using the common VMM route policy. Kept internal but
// exposed here so transport-independent unit tests can inject adaptor vtables
// and verify DMA-BUF-to-VA fallback behavior without hardware.
flagcxResult_t flagcxOneSideRegisterMr(flagcxHeteroComm_t comm, void *regComm,
                                       void *buff, size_t size, int ptrType,
                                       bool isVmm, int mrFlags, void **mrHandle,
                                       flagcxVmmMrRoute_t *selectedRoute,
                                       int mrRangeSegmentCount = 0);

// Internal implementation used by sym_heap and flagcxCommRegister
flagcxResult_t flagcxOneSideRegisterInternal(
    flagcxHeteroComm_t comm, void *buff, size_t size, bool isVmm,
    bool acquireWindowRef = false, int *mrIndex = nullptr,
    bool *rollbackPending = nullptr, int mrRangeSegmentCount = 0);

// Retry MR objects retained after a failed registration rollback. Symmetric
// windows use this before dropping their allocation lease.
flagcxResult_t flagcxOneSideRetryPendingCleanup(flagcxHeteroComm_t heteroComm);

// Release one symmetric-window reference without renumbering the shared handle
// table. The physical MR remains while another window or the communicator owns
// it. A failed final deregMr preserves the reference for retry.
flagcxResult_t flagcxOneSideDeregisterInternal(flagcxHeteroComm_t comm,
                                               int index);

// Internal signal-registration entry point for buffers whose allocation owner
// knows the memory provenance even when the buffer is not in the public
// flagcxMemAlloc registry. The public API keeps its existing signature and
// treats untracked external CUDA allocations as the legacy ordinary-VA path.
flagcxResult_t flagcxOneSideSignalRegisterInternal(const flagcxComm_t comm,
                                                   void *buff, size_t size,
                                                   int ptrType,
                                                   bool isVmmAllocation);

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
