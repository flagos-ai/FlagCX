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
  flagcxDevicePrimWait = 3
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

constexpr unsigned int flagcxDeviceTriggerBitsAddr = 64;
constexpr unsigned int flagcxDeviceTriggerOffCount = 0;
constexpr unsigned int flagcxDeviceTriggerBitsCount = 32;
constexpr unsigned int flagcxDeviceTriggerOffPeerRank =
    flagcxDeviceTriggerOffCount + flagcxDeviceTriggerBitsCount;
constexpr unsigned int flagcxDeviceTriggerBitsPeerRank = 20;
constexpr unsigned int flagcxDeviceTriggerOffDatatype =
    flagcxDeviceTriggerOffPeerRank + flagcxDeviceTriggerBitsPeerRank;
constexpr unsigned int flagcxDeviceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxDeviceTriggerOffPrim =
    flagcxDeviceTriggerOffDatatype + flagcxDeviceTriggerBitsDatatype;
constexpr unsigned int flagcxDeviceTriggerBitsPrim = 4;
constexpr unsigned int flagcxDeviceTriggerBitsFifoReserved = 1;
// Valid bit for lock-free MPSC FIFO (bit 63 of snd field)
constexpr unsigned int flagcxDeviceTriggerOffValid = 63;
constexpr uint64_t flagcxDeviceTriggerValidMask = (1ULL << 63);

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
  uint64_t fst;
  uint64_t snd;

  FLAGCX_HOST_DECORATOR uint64_t getAddr();
  FLAGCX_HOST_DECORATOR uint64_t getCount();
  FLAGCX_HOST_DECORATOR uint64_t getPeerRank();
  FLAGCX_HOST_DECORATOR uint64_t getDatatype();
  FLAGCX_HOST_DECORATOR uint64_t getType();
  FLAGCX_DEVICE_DECORATOR void setValue(uint64_t addr, uint64_t count,
                                        uint64_t peerRank, uint64_t datatype,
                                        uint64_t type);
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
  flagcxResult_t flagcxFifoReset();
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
// Fixed 16-byte opaque blob — field interpretation is platform-specific.
// On NVIDIA: fields[0]=lsaBarrierCount, fields[1]=lsaMultimem,
//            fields[2]=ginBarrierCount,  fields[3]=ginSignalCount
typedef struct {
  int fields[4];
} flagcxDevCommRequirements;

#define FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER                               \
  {                                                                            \
    { 0, 0, 0, 0 }                                                             \
  }

// Opaque handle to a device communicator (host-side lifetime management).
// Internally wraps ncclDevComm on NVIDIA backend (Tier 1),
// or IPC barrier state on fallback (Tier 2).
typedef struct flagcxDevCommInternal *flagcxDevComm_t;

// Opaque handle to device memory (host-side lifetime management).
// Internally wraps ncclWindow_t on NVIDIA backend (Tier 1),
// or IPC peer pointer table on fallback (Tier 2).
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// Unified inter-node AlltoAll demo.
// Runtime dispatch: one-sided (put + signal) if window available (Tier 1 -R 2),
// two-sided (send/recv + FIFO) otherwise (all tiers, all -R modes).
flagcxResult_t flagcxInterAlltoAllDemo(flagcxDevMem_t sendMem,
                                       flagcxDevMem_t recvMem, size_t count,
                                       flagcxDataType_t datatype,
                                       flagcxDevComm_t devComm,
                                       flagcxStream_t stream);

// Kernel launch configuration constants.
// Also defined in device_api/flagcx_device.h (with same include guard).
#ifndef FLAGCX_DEVICE_CTA_COUNT
#define FLAGCX_DEVICE_CTA_COUNT 36
#endif
#ifndef FLAGCX_DEVICE_THREADS_PER_CTA
#define FLAGCX_DEVICE_THREADS_PER_CTA 512
#endif

// Create a device communicator for custom kernel usage.
// On NVIDIA backend (Tier 1), internally calls pncclDevCommCreate.
// On fallback (Tier 2), sets up IPC-based barrier across intra-node peers.
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

// Intra-node AllReduce using FlagCX Device API.
// The caller provides a registered buffer (via flagcxDevMemCreate)
// already containing the input data.  The kernel runs an in-place
// AllReduce across all intra-node GPUs.
// devComm must be created via flagcxDevCommCreate beforehand.
flagcxResult_t flagcxIntraAllReduceDemo(flagcxDevMem_t devMem, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxDevComm_t devComm,
                                        flagcxStream_t stream);
#endif
