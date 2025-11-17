#ifndef FLAGCX_KERNEL_H_
#define FLAGCX_KERNEL_H_

#include "adaptor.h"
#include "flagcx.h"

#define FLAGCX_KERNEL_FIFO_CAPACITY 16
#define flagcxTriggerMask(w) ((w == 64) ? ~0ull : ((1ull << w) - 1))

#ifdef COMPILE_KERNEL
FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iter) {
  int delay = 1 << (iter < 15 ? iter : 15);
#if __CUDA_ARCH__ >= 700
  __nanosleep(delay);
#else
  uint64_t start = clock64();
  while (clock64() - start < (uint64_t)delay) { /* spin */
  }
#endif
}
#endif

typedef enum {
  flagcxDevicePrimSend = 0,
  flagcxDevicePrimRecv = 1,
  flagcxDevicePrimTerm = 2,
  flagcxDevicePrimWait = 3
} flagcxDevicePrim;

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

struct flagcxReduceTrigger {
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
  // flagcxDeviceTrigger fifo:
  // 0: capacity, 1: consumed, 2: produced, 3+: buffer
  // flagcxReduceTrigger fifo:
  // 0: capacity, 1: consumed, 2: produced, 3: terminate, 4+:buffer
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
                                             flagcxRedOp_t redop);
#ifdef COMPILE_KERNEL
FLAGCX_DEVICE_DECORATOR flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr,
                                               uint64_t count,
                                               uint64_t peerRank,
                                               uint64_t datatype,
                                               uint64_t type);
FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                                      int *idx);

FLAGCX_DEVICE_DECORATOR size_t
getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype);

FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceSend(const void *sendbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer);
FLAGCX_DEVICE_DECORATOR flagcxResult_t
flagcxDeviceRecv(void *sendbuff, size_t count, flagcxDataType_t datatype,
                 int peer, void *fifoBuffer);
FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceTerm(void *fifoBuffer);
FLAGCX_DEVICE_DECORATOR flagcxResult_t flagcxDeviceWait(void *fifoBuffer);
FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer); // TBD
void flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                   flagcxDataType_t datatype, int sendPeer, int recvPeer,
                   flagcxComm_t comm, flagcxStream_t stream);
#endif // COMPILE_KERNEL
#endif
