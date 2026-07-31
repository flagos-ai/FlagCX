/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Kernel Internal — Host-side functions and lifecycle management.
 *
 * This header contains host-only content that requires adaptor.h and
 * other host infrastructure. It includes:
 *   - flagcxFifo struct and methods
 *   - Host-side FIFO functions (dequeue, enqueue)
 *   - Device communicator lifecycle (flagcxDevCommCreate, etc.)
 *   - Device memory lifecycle (flagcxDevMemCreate, etc.)
 *   - Test kernels and AlltoAll implementations
 *
 * NOT safe for LLVM bitcode compilation.
 * For device-side types and constants, see flagcx_kernel_core.h.
 * For normal builds, include flagcx_kernel.h (umbrella header).
 ************************************************************************/

#ifndef FLAGCX_KERNEL_INTERNAL_H_
#define FLAGCX_KERNEL_INTERNAL_H_

#include "adaptor.h"
#include "flagcx_kernel_core.h"

struct flagcxFifo {
  // Unified fifo layout: [capacity][consumed][produced][terminate][data...]
  // flagcxDeviceTrigger fifo: terminate slot is reserved but unused
  // flagcxReduceTrigger fifo: terminate slot is used
  // See flagcxFifoIndex enumeration for index values
  uint64_t *buffer;

public:
  flagcxFifo() {}
  ~flagcxFifo() {}
  flagcxResult_t flagcxFifoInit();
  flagcxResult_t flagcxRedFifoInit();
  flagcxResult_t flagcxFifoDestroy();
  flagcxResult_t flagcxRedFifoDestroy();
};
typedef struct flagcxFifo *flagcxFifo_t;

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger);
FLAGCX_HOST_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr1,
                                             uint64_t addr2, uint64_t addr3,
                                             size_t count, size_t nthreads,
                                             flagcxDataType_t datatype,
                                             flagcxRedOp_t redop, int *idx);
#ifdef COMPILE_KERNEL
FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(volatile uint64_t *buffer,
                                                      int *idx);

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer);
#endif // COMPILE_KERNEL

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream);

// ==========================================================================
// Device Communicator — Host-side lifecycle management
// ==========================================================================

// Requirements for creating a device communicator.
// Named fields map to NCCL ncclDevCommRequirements (Vendor).
// Naming: NCCL "lsa" → FlagCX "intra", "gin" → "inter", "multimem" →
// "multicast".
struct flagcxDevCommRequirements {
  bool intraMulticast; // → ncclReqs.lsaMultimem

  int barrierCount;      // → ncclReqs.barrierCount (world barrier)
  int intraBarrierCount; // → ncclReqs.lsaBarrierCount
  int interBarrierCount; // → ncclReqs.railGinBarrierCount

  int intraLLA2ABlockCount; // → ncclReqs.lsaLLA2ABlockCount
  int intraLLA2ASlotCount;  // → ncclReqs.lsaLLA2ASlotCount

  bool interForceEnable; // → ncclReqs.ginForceEnable
  int interContextCount; // → ncclReqs.ginContextCount (hint, default 4)
  int interSignalCount;  // → ncclReqs.ginSignalCount (start at id=0)
  int interCounterCount; // → ncclReqs.ginCounterCount (start at id=0)
};

#define FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER                               \
  {                                                                            \
    false,       /* intraMulticast */                                          \
        0, 0, 0, /* barrierCount, intraBarrierCount, interBarrierCount */      \
        0, 0,    /* intraLLA2ABlockCount, intraLLA2ASlotCount */               \
        false, 4, 0, 0 /* interForceEnable, interContextCount,                 \
                          interSignalCount, interCounterCount */               \
  }

// Network type enumeration (maps to ncclGinType_t on NVIDIA backend).
typedef enum {
  flagcxNetTypeNone = 0,  // → NCCL_GIN_TYPE_NONE
  flagcxNetTypeProxy = 2, // → NCCL_GIN_TYPE_PROXY
  flagcxNetTypeGdaki = 3, // → NCCL_GIN_TYPE_GDAKI
} flagcxNetType_t;

// Communicator properties — host-side queryable attributes.
struct flagcxCommProperties {
  int rank;
  int nRanks;
  int deviceId; // → ncclCommProperties.cudaDev (platform-neutral)
  bool vendorDeviceApiSupport; // → ncclCommProperties.deviceApiSupport
  bool multicastSupport;       // → ncclCommProperties.multimemSupport
  flagcxNetType_t netType;     // → ncclCommProperties.ginType
};
typedef struct flagcxCommProperties flagcxCommProperties_t;

// Opaque handle to a device communicator (host-side lifetime management).
// Internally wraps ncclDevComm on NVIDIA backend (Vendor),
// or IPC barrier state on default path (Default).
typedef struct flagcxDevCommInternal *flagcxDevComm_t;

// Opaque handle to device memory (host-side lifetime management).
// Internally wraps ncclWindow_t on NVIDIA backend (Vendor),
// or IPC peer pointer table on default path (Default).
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Create a device communicator for custom kernel usage.
// On NVIDIA backend (Vendor), internally calls pncclDevCommCreate.
// On default path (Default), sets up IPC-based barrier across intra-node peers.
// The returned handle must be destroyed with flagcxDevCommDestroy(comm,
// devComm).
flagcxResult_t flagcxDevCommCreate(flagcxComm_t comm,
                                   const flagcxDevCommRequirements *reqs,
                                   flagcxDevComm_t *devComm);

// Destroy a device communicator created by flagcxDevCommCreate.
flagcxResult_t flagcxDevCommDestroy(flagcxComm_t comm, flagcxDevComm_t devComm);

// Create a device memory handle for a registered buffer.
// Registration is the caller's responsibility (Decision 7.16):
//   - IPC mode (win=NULL): caller calls flagcxCommRegister first.
//   - Window mode (win!=NULL): caller calls flagcxCommWindowRegister first.
// This function exchanges IPC handles to build peer pointer tables (both modes)
// and stores the window handle (window mode only).
flagcxResult_t flagcxDevMemCreate(flagcxComm_t comm, void *buff, size_t size,
                                  flagcxWindow_t win, flagcxDevMem_t *devMem);

// Destroy a device memory handle created by flagcxDevMemCreate.
flagcxResult_t flagcxDevMemDestroy(flagcxComm_t comm, flagcxDevMem_t devMem);

// ---- Device Pointer API (for Triton integration) ----
// Allocate device memory, copy the DevComm/DevMem struct, return device
// pointer. The returned pointer is immutable (epoch lives in persistent buffer)
// and cached. Calling multiple times returns the same pointer.
flagcxResult_t flagcxDevCommGetDevicePtr(flagcxDevComm_t devComm,
                                         void **devPtr);
flagcxResult_t flagcxDevCommFreeDevicePtr(flagcxDevComm_t devComm);
flagcxResult_t flagcxDevMemGetDevicePtr(flagcxDevMem_t devMem, void **devPtr);
flagcxResult_t flagcxDevMemFreeDevicePtr(flagcxDevMem_t devMem);

// Comm-level cleanup — delegates to devApiBackend->commCleanup.
// Called from flagcxCommDestroy before heteroComm teardown.
flagcxResult_t flagcxCommCleanup(flagcxComm_t comm);

#ifdef __cplusplus
}
#endif

// Clean up IPC peer pointer table on comm.
// Must be called after homoComm destroy.
// so that cudaFree does not deadlock on device synchronization.
flagcxResult_t flagcxCommCleanupIpcTable(flagcxComm_t comm);

// Release an IPC table slot at runtime — moves resources to a deferred queue
// so the slot can be reused by buildIpcPeerPointers. Actual cleanup at destroy.
void releaseIpcTableSlot(flagcxComm_t comm, int slot);

// Drain deferred IPC entries queued by releaseIpcTableSlot.
flagcxResult_t flagcxCommDrainDeferredIpc(flagcxComm_t comm);

// Deferred device/host-pinned memory free.
// Collects pointers during DevComm/DevMem cleanup.
void flagcxCommDeferFree(flagcxComm_t comm, void *ptr, int memType);
flagcxResult_t flagcxCommDrainDeferredFrees(flagcxComm_t comm);

// Drain deferred DevComm buffer queue (localBarrierFlags, epoch, signal, etc.).
// Called at flagcxCommDestroy time when all peers are guaranteed done.
flagcxResult_t flagcxCommDrainDeferredBuffers(flagcxComm_t comm);

// Release data buffer resources (MR, network connections, handle arrays).
flagcxResult_t flagcxOneSideDeregister(struct flagcxHeteroComm *heteroComm);

// Release signal buffer resources (MR, network connections, handle arrays).
// flagcxOneSideSignalRegister / flagcxOneSideStagingRegister /
// flagcxOneSideStagingDeregister are declared in flagcx.h (extern "C").
flagcxResult_t flagcxOneSideSignalDeregister(flagcxComm_t comm);

// One-sided barrier MR registration (host-pinned memory for inter-node
// barrier). Collective: ALL ranks must call. Leaders pass recvComm+buff,
// non-leaders pass NULL.
flagcxResult_t
flagcxOneSideBarrierRegister(const flagcxComm_t comm, void *recvComm,
                             void *buff, size_t size,
                             struct flagcxOneSideHandleInfo **outInfo);
// Release barrier MR and free handle info.
flagcxResult_t
flagcxOneSideBarrierDeregister(const flagcxComm_t comm,
                               struct flagcxOneSideHandleInfo *info);

#endif // FLAGCX_KERNEL_INTERNAL_H_
