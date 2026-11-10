/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API — public host-side lifecycle and configuration API.
 *
 * This header deliberately contains no adaptor or backend implementation
 * details. Device-side value types and primitives live in
 * flagcx_kernel_core.h; internal host helpers live outside the installed
 * public-header surface.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_PUBLIC_H_
#define FLAGCX_DEVICE_API_PUBLIC_H_

#include "flagcx.h"

#include <stdbool.h>
#include <stddef.h>

// Opaque host-side Device API handles.
#ifndef FLAGCX_DEV_COMM_T_DEFINED
#define FLAGCX_DEV_COMM_T_DEFINED
typedef struct flagcxDevCommInternal *flagcxDevComm_t;
#endif

#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// Requirements for creating a device communicator. Existing fields form a
// frozen ABI prefix: do not reorder, resize, or change their meaning. New
// fields may only be appended, and are consumed only when the caller-provided
// structure size reaches that field.
typedef struct flagcxDevCommRequirements {
  bool intraMulticast;

  int barrierCount;
  // Barrier indices are channels: simultaneous Coop groups need distinct
  // indices, and corresponding groups on peer ranks use the same index.
  int intraBarrierCount;
  int interBarrierCount;

  int intraLLA2ABlockCount;
  int intraLLA2ASlotCount;

  bool interForceEnable;
  int interContextCount;
  int interSignalCount;
  int interCounterCount;

  // Symmetric scratch bytes per rank, reachable by every intra-node peer.
  size_t intraScratchBytes;
} flagcxDevCommRequirements;

#define FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER                               \
  {                                                                            \
    false,       /* intraMulticast */                                          \
        0, 0, 0, /* barrierCount, intraBarrierCount, interBarrierCount */      \
        0, 0,    /* intraLLA2ABlockCount, intraLLA2ASlotCount */               \
        false, 4, 0, 0, /* interForceEnable, interContextCount,                \
                           interSignalCount, interCounterCount */              \
        0               /* intraScratchBytes */                                \
  }

#ifdef __cplusplus
extern "C" {
#endif

// Binary-compatible entry point retained for existing applications.
flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const struct flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm);

// Size-aware ABI entry point used by the source-level flagcxDevCommCreate call
// below. The explicit size is not user-managed; sizeof(*reqs) is supplied by
// the header so future tail fields require no new public API version.
flagcxResult_t
flagcxDevCommCreateSized(flagcxComm_t comm,
                         const struct flagcxDevCommRequirements *reqs,
                         size_t reqsSize, flagcxDevComm_t *devComm);

flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm, flagcxDevComm_t devComm);

flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff, size_t size,
                                  flagcxWindow_t win, flagcxDevMem_t *devMem);
flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm, flagcxDevMem_t devMem);

flagcxResult_t flagcxDevCommGetDevicePtr(flagcxDevComm_t devComm,
                                         void **devPtr);
// FreeDevicePtr requires a valid handle; a null handle returns
// flagcxInvalidArgument on every backend.
flagcxResult_t flagcxDevCommFreeDevicePtr(flagcxDevComm_t devComm);
flagcxResult_t flagcxDevMemGetDevicePtr(flagcxDevMem_t devMem, void **devPtr);
// A null devMem returns flagcxInvalidArgument on every backend.
flagcxResult_t flagcxDevMemFreeDevicePtr(flagcxDevMem_t devMem);

// Comm-level cleanup called by flagcxCommDestroy.
flagcxResult_t flagcxCommCleanup(flagcxComm_t comm);

#ifdef __cplusplus
}
#endif

// Keep the public call spelling stable while communicating the caller's
// concrete requirements size to the library. Defining the opt-out macro is
// reserved for the library implementation and legacy-ABI compatibility tests.
#ifndef FLAGCX_DISABLE_DEV_COMM_CREATE_SIZE_DISPATCH
#define flagcxDevCommCreate(comm, reqs, devComm)                               \
  flagcxDevCommCreateSized((comm), (reqs), sizeof(*(reqs)), (devComm))
#endif

#endif // FLAGCX_DEVICE_API_PUBLIC_H_
