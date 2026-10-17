/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Kernel Internal — Host-side functions and lifecycle management.
 *
 * This header contains host-only content that requires adaptor.h and
 * other host infrastructure. It includes:
 *   - flagcxFifo struct and methods
 *   - Host-side FIFO functions (dequeue, enqueue)
 *   - Test kernels and AlltoAll implementations
 *
 * NOT safe for LLVM bitcode compilation.
 * For device-side types and constants, see flagcx_kernel_core.h.
 * Internal host sources that need these helpers must include this header
 * explicitly; it is not part of the installed public-header set.
 ************************************************************************/

#ifndef FLAGCX_KERNEL_INTERNAL_H_
#define FLAGCX_KERNEL_INTERNAL_H_

#include "adaptor.h"
#include "flagcx_device_api.h"
#include "flagcx_kernel_core.h"

// The original public requirements layout ends immediately before the first
// appended field. Keep compatibility metadata private to the library.
#define FLAGCX_DEV_COMM_REQUIREMENTS_LEGACY_SIZE                               \
  offsetof(struct flagcxDevCommRequirements, intraScratchBytes)

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

// Clean up IPC peer pointer table on comm.
// Must be called after homoComm destroy.
// so that cudaFree does not deadlock on device synchronization.
flagcxResult_t flagcxCommCleanupIpcTable(flagcxComm_t comm);

// Release an IPC table slot at runtime — moves resources to a deferred queue
// so the slot can be reused by buildIpcPeerPointers. Actual cleanup at destroy.
void releaseIpcTableSlot(flagcxComm_t comm, int slot);

// Drain deferred IPC entries queued by releaseIpcTableSlot.
flagcxResult_t flagcxCommDrainDeferredIpc(flagcxComm_t comm);

// Drain cached device allocations retained by retired DevComm/DevMem owners.
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
