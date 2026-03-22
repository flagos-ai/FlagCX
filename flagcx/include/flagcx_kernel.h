#ifndef FLAGCX_KERNEL_H_
#define FLAGCX_KERNEL_H_

#include "adaptor.h"
#include "flagcx.h"

#define FLAGCX_FIFO_CAPACITY 128
#define flagcxTriggerMask(w) ((w == 64) ? ~0ull : ((1ull << w) - 1))

typedef enum {
  flagcxDevicePrimSend = 0,
  flagcxDevicePrimRecv = 1,
  flagcxDevicePrimTerm = 2,
  flagcxDevicePrimWait = 3,
  flagcxDevicePrimPut = 4,
  flagcxDevicePrimSignal = 5,
  flagcxDevicePrimBarrierSignal = 6,
  flagcxDevicePrimWaitSignal = 7,
  flagcxDevicePrimPutValue = 8,
  flagcxDevicePrimPutSignal = 9,
  flagcxDevicePrimGet = 10
} flagcxDevicePrim;

// Unified buffer index enumeration for fifo
// Layout: [capacity][consumed][produced][terminate][data...]
// Note: flagcxFifoIdxTerminate is only used by flagcxReduceTrigger fifo
typedef enum {
  flagcxFifoIdxCapacity = 0,
  flagcxFifoIdxConsumed = 1,
  flagcxFifoIdxProduced = 2,
  flagcxFifoIdxTerminate = 3,
  flagcxFifoIdxData = 4
} flagcxFifoIndex;

typedef enum {
  flagcxReduceTriggerAvailable = 0,
  flagcxReduceTriggerEnqueued = 1,
  flagcxReduceTriggerInprogress = 2,
  flagcxReduceTriggerComplete = 3
} flagcxReduceTriggerState;

// ==========================================================================
// flagcxDeviceTrigger bit layout (24 bytes = 3 × uint64_t: fst, snd, trd)
//
// trd (word2, control header — written last with valid bit):
//   [63]    valid
//   [62:59] prim (4 bits)
//   [58:39] peerRank (20 bits)
//   [38:36] slotIdx (3 bits, reserved for future multi-FIFO)
//   [35:0]  prim-specific (36 bits)
//
// fst (word0, payload — written first):
//   prim-specific (64 bits)
//
// snd (word1, payload — written second):
//   prim-specific (64 bits)
// ==========================================================================

// Valid bit (trd[63])
constexpr unsigned int flagcxDeviceTriggerOffValid = 63;
constexpr uint64_t flagcxDeviceTriggerValidMask = (1ULL << 63);

// Common header in trd
constexpr unsigned int flagcxDeviceTriggerOffPrim = 59;
constexpr unsigned int flagcxDeviceTriggerBitsPrim = 4;
constexpr unsigned int flagcxDeviceTriggerOffPeerRank = 39;
constexpr unsigned int flagcxDeviceTriggerBitsPeerRank = 20;
constexpr unsigned int flagcxDeviceTriggerOffSlotIdx = 36;
constexpr unsigned int flagcxDeviceTriggerBitsSlotIdx = 3;

// Two-sided Send/Recv: trd prim-specific
//   trd[35:32] = datatype(4), trd[31:0] = count(32)
//   fst = addr(64), snd = 0
constexpr unsigned int flagcxDeviceTriggerOffDatatype = 32;
constexpr unsigned int flagcxDeviceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxDeviceTriggerOffCount = 0;
constexpr unsigned int flagcxDeviceTriggerBitsCount = 32;

// One-sided Put/PutSignal: trd prim-specific
//   trd[35:29] = srcMrIdx(7), trd[28:22] = dstMrIdx(7)
//   PutSignal: trd[21:14] = signalIdx(8), trd[13:0] = unused
//   fst = srcOffset(32)|dstOffset(32), snd =
//   size(32)|signalValue(16)|reserved(16)
constexpr unsigned int flagcxDeviceTriggerOffSrcMrIdx = 29;
constexpr unsigned int flagcxDeviceTriggerBitsSrcMrIdx = 7;
constexpr unsigned int flagcxDeviceTriggerOffDstMrIdx = 22;
constexpr unsigned int flagcxDeviceTriggerBitsDstMrIdx = 7;
constexpr unsigned int flagcxDeviceTriggerOffSignalIdx = 14;
constexpr unsigned int flagcxDeviceTriggerBitsSignalIdx = 8;
// PutSignal signalValue in snd[15:0] (same max as PrimSignal: 16b)
constexpr unsigned int flagcxDeviceTriggerOffSignalValuePut = 0;
constexpr unsigned int flagcxDeviceTriggerBitsSignalValuePut = 16;
// fst offsets for srcOffset/dstOffset (shared with PutValue dstOffset accessor)
constexpr unsigned int flagcxDeviceTriggerOffSrcOffset = 32;
constexpr unsigned int flagcxDeviceTriggerBitsSrcOffset = 32;
constexpr unsigned int flagcxDeviceTriggerOffDstOffset = 0;
constexpr unsigned int flagcxDeviceTriggerBitsDstOffset = 32;
// snd offset for size
constexpr unsigned int flagcxDeviceTriggerOffSize = 32;
constexpr unsigned int flagcxDeviceTriggerBitsSize = 32;

// One-sided PutValue: trd prim-specific
//   trd[28:22] = dstMrIdx(7) (same position as Put/PutSignal dstMrIdx)
//   fst = 0|dstOffset(32) (fst[31:0], same position as Put/PutSignal)
//   snd = value(64)

// Signal/WaitSignal: all in trd prim-specific
//   trd[35:34] = bufferType(2), trd[33:26] = signalIdx(8),
//   trd[25:10] = signalValue/expectedValue(16), trd[9:0] = unused
//   fst = 0, snd = 0
constexpr unsigned int flagcxDeviceTriggerOffBufferType = 34;
constexpr unsigned int flagcxDeviceTriggerBitsBufferType = 2;
constexpr unsigned int flagcxDeviceTriggerOffSignalIdxSig = 26;
constexpr unsigned int flagcxDeviceTriggerBitsSignalIdxSig = 8;
constexpr unsigned int flagcxDeviceTriggerOffSignalValue = 10;
constexpr unsigned int flagcxDeviceTriggerBitsSignalValue =
    16; // max signal value: 2^16 (65535)

constexpr unsigned int flagcxReduceTriggerBitsAddr = 64;
constexpr unsigned int flagcxReduceTriggerOffCount = 0;
constexpr unsigned int flagcxReduceTriggerBitsCount = 32;
constexpr unsigned int flagcxReduceTriggerOffNThreads =
    flagcxReduceTriggerOffCount + flagcxReduceTriggerBitsCount;
constexpr unsigned int flagcxReduceTriggerBitsNThreads = 16;
constexpr unsigned int flagcxReduceTriggerOffDatatype =
    flagcxReduceTriggerOffNThreads + flagcxReduceTriggerBitsNThreads;
constexpr unsigned int flagcxReduceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxReduceTriggerOffRedop =
    flagcxReduceTriggerOffDatatype + flagcxReduceTriggerBitsDatatype;
constexpr unsigned int flagcxReduceTriggerBitsRedop = 4;
constexpr unsigned int flagcxReduceTriggerOffState =
    flagcxReduceTriggerOffRedop + flagcxReduceTriggerBitsRedop;
/* op state: 0 for available, 1 for enqueued, 2 for in-progress, 3 for done */
constexpr unsigned int flagcxReduceTriggerBitsState = 2;
constexpr unsigned int flagcxReduceTriggerBitsFifoReserved = 1;

struct flagcxDeviceTrigger {
  uint64_t fst; // word0 — payload, written first
  uint64_t snd; // word1 — payload, written second
  uint64_t trd; // word2 — control header (valid bit), written last

  // Common accessors (trd common header)
  FLAGCX_HOST_DECORATOR uint64_t getPrim();
  FLAGCX_HOST_DECORATOR uint64_t getPeerRank();
  FLAGCX_HOST_DECORATOR uint64_t getSlotIdx();

  // Two-sided accessors (Send/Recv)
  FLAGCX_HOST_DECORATOR uint64_t getAddr();     // fst
  FLAGCX_HOST_DECORATOR uint64_t getDatatype(); // trd[35:32]
  FLAGCX_HOST_DECORATOR uint64_t getCount();    // trd[31:0]

  // One-sided accessors (Put/PutSignal/PutValue)
  FLAGCX_HOST_DECORATOR uint64_t getSrcMrIdx();  // trd[35:29]
  FLAGCX_HOST_DECORATOR uint64_t getDstMrIdx();  // trd[28:22]
  FLAGCX_HOST_DECORATOR uint64_t getSize();      // snd[63:32]
  FLAGCX_HOST_DECORATOR uint64_t getSrcOffset(); // fst[63:32]
  FLAGCX_HOST_DECORATOR uint64_t getDstOffset(); // fst[31:0]
  FLAGCX_HOST_DECORATOR uint64_t getValue();     // snd (PutValue)
  FLAGCX_HOST_DECORATOR uint64_t
  getSignalIdx(); // trd (PutSignal/Signal/WaitSignal)
  FLAGCX_HOST_DECORATOR uint64_t getSignalValue();   // trd (Signal)
  FLAGCX_HOST_DECORATOR uint64_t getExpectedValue(); // trd (WaitSignal)
  FLAGCX_HOST_DECORATOR uint64_t getBufferType();    // trd (Signal/WaitSignal)

  // Backward compat alias
  FLAGCX_HOST_DECORATOR uint64_t getType(); // alias for getPrim()
};
typedef flagcxDeviceTrigger *flagcxDeviceTrigger_t;

struct alignas(16) flagcxReduceTrigger {
  uint64_t value[4];

#ifdef COMPILE_KERNEL
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getInput1();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getInput2();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getOutput();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getCount();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getNThreads();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getDatatype();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getRedop();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getState();
  FLAGCX_DEVICE_INLINE_DECORATOR void setComplete();
#endif
  FLAGCX_HOST_DECORATOR void setValue(uint64_t fst, uint64_t snd, uint64_t out,
                                      size_t count, size_t nthreads,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t redOp,
                                      flagcxReduceTriggerState state);
  FLAGCX_HOST_DECORATOR uint64_t pollState();
  FLAGCX_HOST_DECORATOR void setState(int state);
};
typedef flagcxReduceTrigger *flagcxReduceTrigger_t;

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

FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype);

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
  int deviceId;            // → ncclCommProperties.cudaDev (platform-neutral)
  bool deviceApiSupport;   // → ncclCommProperties.deviceApiSupport
  bool multicastSupport;   // → ncclCommProperties.multimemSupport
  flagcxNetType_t netType; // → ncclCommProperties.ginType
};
typedef struct flagcxCommProperties flagcxCommProperties_t;

// Query communicator properties.
// Currently returns placeholder defaults; will delegate to backend
// (e.g. ncclCommQueryProperties) when wired through the adaptor layer.
flagcxResult_t flagcxCommQueryProperties(flagcxComm_t comm,
                                         flagcxCommProperties_t *props);

// Forward declarations for types defined in flagcx_device.h.
struct flagcxTeam;
typedef struct flagcxTeam flagcxTeam_t;
struct flagcxDevCommRequirements;
struct flagcxIntraBarrierHandle;
typedef struct flagcxIntraBarrierHandle flagcxIntraBarrierHandle_t;
struct flagcxInterBarrierHandle;
typedef struct flagcxInterBarrierHandle flagcxInterBarrierHandle_t;

// Create barrier requirement handles (stub — returns flagcxNotSupported).
// FlagCX currently uses intraBarrierCount in DevCommCreate directly;
// the resource-handle model will be implemented when needed.
flagcxResult_t
flagcxIntraBarrierCreateRequirement(flagcxTeam_t team, int nBarriers,
                                    flagcxIntraBarrierHandle_t *outHandle,
                                    flagcxDevCommRequirements *outReq);

flagcxResult_t flagcxInterBarrierCreateRequirement(
    flagcxComm_t comm, flagcxTeam_t team, int nBarriers,
    flagcxInterBarrierHandle_t *outHandle, flagcxDevCommRequirements *outReq);

// Opaque handle to a device communicator (host-side lifetime management).
// Internally wraps ncclDevComm on NVIDIA backend (Vendor),
// or IPC barrier state on fallback (Fallback).
typedef struct flagcxDevCommInternal *flagcxDevComm_t;

// Opaque handle to device memory (host-side lifetime management).
// Internally wraps ncclWindow_t on NVIDIA backend (Vendor),
// or IPC peer pointer table on fallback (Fallback).
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// Inter-node one-sided AlltoAll (put + waitSignal + flush).
flagcxResult_t flagcxInterOneSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node two-sided AlltoAll (send/recv + term/wait via FIFO).
flagcxResult_t flagcxInterTwoSidedAlltoAll(flagcxDevMem_t sendMem,
                                           flagcxDevMem_t recvMem, size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

// Inter-node Device API test kernels.
// Each kernel tests one API facet; host verifies after streamSynchronize.
flagcxResult_t flagcxInterTestSignalInc(flagcxDevMem_t sendMem,
                                        flagcxDevMem_t recvMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream);

flagcxResult_t flagcxInterTestSignalAdd(flagcxDevMem_t sendMem,
                                        flagcxDevMem_t recvMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream);

flagcxResult_t
flagcxInterTestCounterPipeline(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                               size_t count, flagcxDataType_t datatype,
                               flagcxDevComm_t devComm, flagcxStream_t stream,
                               uint64_t *resultBuf);

flagcxResult_t flagcxInterTestPutValue(flagcxDevMem_t recvMem,
                                       flagcxDevComm_t devComm,
                                       flagcxStream_t stream,
                                       size_t putValBase);

flagcxResult_t flagcxInterTestSignalOnly(flagcxDevComm_t devComm,
                                         flagcxStream_t stream);

flagcxResult_t
flagcxInterTestFlushDecouple(flagcxDevMem_t sendMem, flagcxDevMem_t recvMem,
                             size_t count, flagcxDataType_t datatype,
                             flagcxDevComm_t devComm, flagcxStream_t stream);

flagcxResult_t flagcxInterTestFollowShadow(flagcxDevComm_t devComm,
                                           flagcxStream_t stream);

flagcxResult_t flagcxInterTestMeetShadow(flagcxDevComm_t devComm,
                                         flagcxStream_t stream);

flagcxResult_t flagcxInterTestReset(flagcxDevComm_t devComm,
                                    flagcxStream_t stream, uint64_t *resultBuf);

// Kernel launch configuration constants.
// Also defined in device_api/flagcx_device.h (with same include guard).
#ifndef FLAGCX_DEVICE_CTA_COUNT
#define FLAGCX_DEVICE_CTA_COUNT 36
#endif
#ifndef FLAGCX_DEVICE_THREADS_PER_CTA
#define FLAGCX_DEVICE_THREADS_PER_CTA 512
#endif

// Create a device communicator for custom kernel usage.
// On NVIDIA backend (Vendor), internally calls pncclDevCommCreate.
// On fallback (Fallback), sets up IPC-based barrier across intra-node peers.
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

// Clean up IPC peer pointer table on comm.
// Must be called after homoComm destroy.
// so that cudaFree does not deadlock on device synchronization.
flagcxResult_t flagcxCommCleanupIpcTable(flagcxComm_t comm);

// Deferred device/host-pinned memory free.
// Collects pointers during DevComm/DevMem cleanup.
void flagcxCommDeferFree(flagcxComm_t comm, void *ptr, int memType);
flagcxResult_t flagcxCommDrainDeferredFrees(flagcxComm_t comm);

// One-sided data buffer registration.
// Must be called after flagcxCommInitRank and before one-sided operations.
flagcxResult_t flagcxOneSideRegister(const flagcxComm_t comm, void *buff,
                                     size_t size);
// Release data buffer resources (MR, network connections, handle arrays).
flagcxResult_t flagcxOneSideDeregister(const flagcxComm_t comm);

// One-sided signal buffer registration (GPU memory with FORCE_SO).
// Must be called after flagcxCommInitRank and before one-sided operations.
flagcxResult_t flagcxOneSideSignalRegister(const flagcxComm_t comm, void *buff,
                                           size_t size);
// Release signal buffer resources (MR, network connections, handle arrays).
flagcxResult_t flagcxOneSideSignalDeregister(const flagcxComm_t comm);

// One-sided staging buffer registration (host-pinned memory for PutValue).
// Must be called after flagcxOneSideSignalRegister (requires full-mesh
// connections).
flagcxResult_t flagcxOneSideStagingRegister(const flagcxComm_t comm, void *buff,
                                            size_t size);
// Release staging buffer MR resources.
flagcxResult_t flagcxOneSideStagingDeregister(const flagcxComm_t comm);

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

// Intra-node AllReduce using FlagCX Device API.
// The caller provides a registered buffer (via flagcxDevMemCreate)
// already containing the input data.  The kernel runs an in-place
// AllReduce across all intra-node GPUs.
// devComm must be created via flagcxDevCommCreate beforehand.
flagcxResult_t flagcxIntraAllReduce(flagcxDevMem_t devMem, size_t count,
                                    flagcxDataType_t datatype,
                                    flagcxDevComm_t devComm,
                                    flagcxStream_t stream);

#endif
