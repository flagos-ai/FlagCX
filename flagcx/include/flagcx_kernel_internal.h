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
struct flagcxDevMemInternal;
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

struct flagcxFifo {
  // Unified fifo layout: [capacity][consumed][produced][terminate][data...]
  // flagcxDeviceTrigger fifo: terminate slot is reserved but unused
  // flagcxReduceTrigger fifo: terminate slot is used
  // See flagcxFifoIndex enumeration for index values
  uint64_t *buffer;

public:
  flagcxFifo() : buffer(NULL) {}
  ~flagcxFifo() {}
  flagcxResult_t flagcxFifoInit();
  flagcxResult_t flagcxRedFifoInit();
  flagcxResult_t flagcxRedFifoInit(size_t numTriggers);
  flagcxResult_t flagcxIpcFifoInit();
  flagcxResult_t flagcxIpcFifoInit(size_t numTriggers);
  flagcxResult_t flagcxFifoDestroy();
  flagcxResult_t flagcxRedFifoDestroy();
  flagcxResult_t flagcxIpcFifoDestroy();
};
typedef struct flagcxFifo *flagcxFifo_t;

flagcxResult_t flagcxCheckedFifoAllocationSize(size_t capacity,
                                               size_t elementSize,
                                               size_t *bytes);
flagcxResult_t resolveUniRunnerStaticExecutorResidencyBudget(
    bool cooperativeLaunch, bool concurrentKernels, size_t smCount,
    size_t activeBlocksPerSm, size_t maxThreadsPerBlock, size_t nthreads,
    size_t *maxExecutorBlocks);

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger);
FLAGCX_HOST_DECORATOR flagcxResult_t
enqueue(void *fifoBuffer, uint64_t addr1, uint64_t addr2, uint64_t addr3,
        size_t count, size_t nthreads, flagcxDataType_t datatype,
        flagcxRedOp_t redop, uint64_t flagIn, uint64_t flagOut, int *idx);
FLAGCX_HOST_DECORATOR flagcxResult_t enqueueReduceWorker(
    void *fifoBuffer, uint64_t addr1, uint64_t addr2, uint64_t addr3,
    size_t count, size_t nthreads, flagcxDataType_t datatype,
    flagcxRedOp_t redop, uint32_t workerId, uint32_t workerCount,
    uint64_t completionCounter, uint64_t flagIn, uint64_t flagOut, int *idx);
FLAGCX_HOST_DECORATOR flagcxResult_t
enqueueGemm(void *fifoBuffer, uint64_t a, uint64_t b, uint64_t c, uint32_t m,
            uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc,
            size_t nthreads, flagcxDataType_t datatype, int accumulate,
            uint32_t workerId, uint32_t workerCount, uint64_t completionCounter,
            uint64_t flagIn, uint64_t flagOut, int *idx);
FLAGCX_HOST_DECORATOR flagcxResult_t
enqueueIpc(void *fifoBuffer, uint64_t srcOffsetBytes, uint64_t dstOffsetBytes,
           uint64_t bytes, uint64_t chunkSize,
           flagcxIpcBufferType srcBufferType, int peerLocalRank,
           uint32_t readySlot, uint64_t epoch, uint32_t parentFlagsOffset,
           uint32_t numParentFlags, uint64_t flagOut, int *idx);
#ifdef COMPILE_KERNEL
FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(volatile uint64_t *buffer,
                                                      int *idx);

struct flagcxDevMem;
FLAGCX_GLOBAL_DECORATOR void
flagcxCollectiveKernel(void *redFifoBuffer, void *ipcFifoBuffer,
                       flagcxDevMem inputMem, flagcxDevMem outputMem,
                       flagcxDevMem readyMem, const uint64_t *ipcParentFlags,
                       int nRedBlocks, int nIpcBlocks, uint64_t avgDivisor);
FLAGCX_GLOBAL_DECORATOR void
flagcxStaticReduceKernel(flagcxReduceTrigger *triggers, uint64_t numTriggers,
                         const uint64_t *abortFlag, uint64_t avgDivisor,
                         uint64_t numNetTriggers, uint64_t numNetBlocks);
FLAGCX_GLOBAL_DECORATOR void flagcxStaticCollectiveKernel(
    flagcxReduceTrigger *redTriggers, uint64_t numRedTriggers,
    flagcxIpcTrigger *ipcTriggers, uint64_t numIpcTriggers,
    flagcxDevMem inputMem, flagcxDevMem outputMem, flagcxDevMem readyMem,
    const uint64_t *ipcParentFlags, uint64_t numIpcParentFlags,
    uint64_t numRedBlocks, uint64_t numIpcBlocks, uint64_t *blocksDone,
    const uint64_t *abortFlag, flagcxStaticIpcControl ipcControl,
    uint64_t avgDivisor);
FLAGCX_GLOBAL_DECORATOR void
flagcxStaticIpcRecoveryKernel(flagcxDevMem readyMem,
                              flagcxStaticIpcControl ipcControl);
FLAGCX_GLOBAL_DECORATOR void flagcxStaticRingDagKernel(
    flagcxRingLaneDesc *lanes, uint64_t numLanes,
    flagcxRingStepTrigger *triggers, uint64_t numTriggers,
    const void *inputBase, void *outputBase, flagcxDevMem scratchMem,
    flagcxDevMem progressMem, flagcxDevMem readyMem, uint64_t simpleBufferBytes,
    uint32_t datatype, uint32_t redOp, uint64_t avgDivisor,
    uint64_t *blocksDone, const uint64_t *abortFlag,
    flagcxStaticIpcControl ipcControl);
#endif // COMPILE_KERNEL

flagcxResult_t flagcxLaunchCollectiveKernel(
    void *redFifoBuffer, void *ipcFifoBuffer, flagcxDevMem_t inputMem,
    flagcxDevMem_t outputMem, flagcxDevMem_t readyMem,
    const uint64_t *ipcParentFlags, size_t nthreads, size_t nRedBlocks,
    size_t nIpcBlocks, uint64_t avgDivisor, flagcxStream_t stream);
flagcxResult_t
flagcxGetStaticReduceKernelMaxExecutorBlocks(size_t nthreads,
                                             size_t *maxExecutorBlocks);
flagcxResult_t
flagcxGetStaticCollectiveKernelMaxExecutorBlocks(size_t nthreads,
                                                 size_t *maxExecutorBlocks);
flagcxResult_t
flagcxGetStaticRingDagKernelMaxExecutorBlocks(size_t nthreads,
                                              size_t *maxExecutorBlocks);
flagcxResult_t flagcxLaunchStaticReduceKernel(
    void *redFifoDeviceBuffer, size_t numTriggers, size_t nthreads,
    size_t nRedBlocks, size_t maxExecutorBlocks, uint64_t avgDivisor,
    flagcxStream_t stream, size_t numNetTriggers = 0, size_t numNetBlocks = 0);
flagcxResult_t flagcxLaunchStaticCollectiveKernel(
    void *redFifoDeviceBuffer, size_t numRedTriggers, void *ipcFifoDeviceBuffer,
    size_t numIpcTriggers, flagcxDevMem_t inputMem, flagcxDevMem_t outputMem,
    flagcxDevMem_t readyMem, const uint64_t *ipcParentFlags,
    size_t numIpcParentFlags, size_t nthreads, size_t nRedBlocks,
    size_t nIpcBlocks, size_t maxExecutorBlocks, uint64_t avgDivisor,
    const flagcxStaticIpcControl *ipcControl, flagcxStream_t stream);
flagcxResult_t
flagcxLaunchStaticIpcRecoveryKernel(flagcxDevMem_t readyMem,
                                    const flagcxStaticIpcControl *ipcControl,
                                    flagcxStream_t stream);
flagcxResult_t flagcxLaunchStaticRingDagKernel(
    flagcxRingLaneDesc *lanes, size_t numLanes, flagcxRingStepTrigger *triggers,
    size_t numTriggers, const void *input, void *output,
    flagcxDevMem_t scratchMem, flagcxDevMem_t progressMem,
    flagcxDevMem_t readyMem, size_t simpleBufferBytes,
    flagcxDataType_t datatype, flagcxRedOp_t redOp, uint64_t avgDivisor,
    uint64_t *blocksDone, const uint64_t *abortFlag,
    const flagcxStaticIpcControl *ipcControl, size_t nthreads,
    size_t maxExecutorBlocks, flagcxStream_t stream);

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

// Agree on rank-local errors before entering another bootstrap collective.
flagcxResult_t flagcxConvergeBootstrapStatus(flagcxComm_t comm,
                                             flagcxResult_t local,
                                             flagcxResult_t *common);

// Drain deferred IPC entries queued by releaseIpcTableSlot.
flagcxResult_t flagcxCommDrainDeferredIpc(flagcxComm_t comm);

// Drain cached device allocations retained by retired DevComm/DevMem owners.
flagcxResult_t flagcxCommDrainDeferredFrees(flagcxComm_t comm);
flagcxResult_t flagcxCommDeferFree(flagcxComm_t comm, void *ptr, int memType);
void flagcxCommDeferFreeReserved(flagcxComm_t comm,
                                 struct flagcxDeferredFree *entry, void *ptr,
                                 int memType);

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
