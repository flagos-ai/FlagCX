#ifndef FLAGCX_INT_P2P_H_
#define FLAGCX_INT_P2P_H_

#include "adaptor.h"
#include "check.h"
#include "comm.h"
#include "device.h"
#include "shmutils.h"
#include "transport.h"
#include <stddef.h>
#define FLAGCX_P2P_BUFFERSIZE                                                  \
  (64ULL * 1024 * 1024) // 64MB buffer for P2P transfers
#define FLAGCX_P2P_CHUNKSIZE (4ULL * 1024 * 1024) // 4MB chunk size
#define FLAGCX_P2P_STEPS                                                       \
  (FLAGCX_P2P_BUFFERSIZE / FLAGCX_P2P_CHUNKSIZE) // 16 steps
#define FLAGCX_P2P_MAX_OPS                                                     \
  32 // Maximum number of concurrent P2P operation pairs
#define FLAGCX_P2P_IPC_HANDLE_SIZE 64

#ifdef __cplusplus
extern "C" {
#endif

struct flagcxP2pRequest {
  size_t size;
  int refcount;
};

typedef union {
  char reserved[FLAGCX_P2P_IPC_HANDLE_SIZE]; // Generic 64-byte buffer, may
                                             // differ on different devices
} flagcxIpcHandleData;

struct flagcxP2pIpcDesc {
  flagcxIpcHandleData handleData; // Actual IPC handle data
  size_t size;
};

struct flagcxP2pBuff {
  void *directPtr;
  size_t size;
  flagcxP2pIpcDesc ipcDesc;
};

struct flagcxP2pConnectInfo {
  int rank;
  int read;
  flagcxP2pBuff p2pBuff;
  flagcxShmIpcDesc_t desc;
};

// Synchronization structure for a single P2P operation pair
struct flagcxP2pSyncSlot {
  uint64_t sendHead;
  uint64_t recvTail;
  int opHash;   // Hash identifying which operation owns this slot
  int done;     // 1 = slot is free, 0 = slot is in use
  int peerDone; // 1 = slot is free, 0 = slot is in use
};

struct flagcxP2pShm {
  // Array of synchronization slots for multiple concurrent operations
  struct flagcxP2pSyncSlot slots[FLAGCX_P2P_MAX_OPS];
};

// need to make sure this matches flagcxP2pShmProxyInfo in p2p.cc
struct flagcxP2pShmProxyInfo {
  // CPU side
  struct flagcxP2pShm *shm;
  flagcxShmIpcDesc_t desc;

  // Device side
  char *recvFifo;
  flagcxStream_t stream;
  flagcxEvent_t events[FLAGCX_P2P_STEPS];
};

struct flagcxP2pResources {
  // Shared memory for synchronization
  struct flagcxP2pShm *shm;
  flagcxShmIpcDesc_t desc;

  // Proxy info for async operations
  struct flagcxP2pShmProxyInfo proxyInfo;
};

flagcxResult_t flagcxP2pProxySend(struct flagcxP2pResources *resources,
                                  void *data, size_t size,
                                  struct flagcxProxyArgs *args);

flagcxResult_t flagcxP2pProxyRecv(struct flagcxP2pResources *resources,
                                  void *data, size_t size,
                                  struct flagcxProxyArgs *args);

flagcxResult_t flagcxP2pProxySelfCopy(struct flagcxP2pResources *resources,
                                      void *sendData, void *recvData,
                                      size_t size,
                                      struct flagcxProxyArgs *args);

flagcxResult_t flagcxP2pSendProxySetup(struct flagcxProxyConnection *connection,
                                       struct flagcxProxyState *proxyState,
                                       void *reqBuff, int reqSize,
                                       void *respBuff, int respSize, int *done);

flagcxResult_t flagcxP2pRecvProxySetup(struct flagcxProxyConnection *connection,
                                       struct flagcxProxyState *proxyState,
                                       void *reqBuff, int reqSize,
                                       void *respBuff, int respSize, int *done);

flagcxResult_t
flagcxP2pSendProxyConnect(struct flagcxProxyConnection *connection,
                          struct flagcxProxyState *proxyState, void *reqBuff,
                          int reqSize, void *respBuff, int respSize, int *done);

flagcxResult_t
flagcxP2pRecvProxyConnect(struct flagcxProxyConnection *connection,
                          struct flagcxProxyState *proxyState, void *reqBuff,
                          int reqSize, void *respBuff, int respSize, int *done);

flagcxResult_t
flagcxP2pAllocateShareableBuffer(size_t size, int directMap,
                                 struct flagcxP2pIpcDesc *ipcDesc, void **ptr);

flagcxResult_t flagcxP2pImportShareableBuffer(struct flagcxHeteroComm *comm,
                                              int peer, size_t size,
                                              struct flagcxP2pIpcDesc *ipcDesc,
                                              void **devMemPtr);

flagcxResult_t flagcxP2pSendProxyFree(struct flagcxP2pResources *resources);

flagcxResult_t flagcxP2pRecvProxyFree(struct flagcxP2pResources *resources);

void setP2pSlotInfo(int rank, int peerRank, size_t size, flagcxDataType_t dtype,
                    int isRecv, int *opHash, size_t *slotIdx);

#ifdef __cplusplus
}
#endif

#endif // FLAGCX_INT_P2P_H_