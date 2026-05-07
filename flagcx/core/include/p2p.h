#ifndef FLAGCX_INT_P2P_H_
#define FLAGCX_INT_P2P_H_

#include "adaptor.h"
#include "check.h"
#include "comm.h"
#include "device.h"
#include "register.h"
#include "shmutils.h"
#include "transport.h"
#include <stddef.h>

extern int64_t flagcxP2pBufferSize;
extern int64_t flagcxP2pChunkSize;
extern int64_t flagcxP2pChunks;
size_t computeP2pChunkSize(size_t nbytes);
#define FLAGCX_P2P_MAX_STEPS 16
#define FLAGCX_P2P_MAX_OPS                                                     \
  (FLAGCX_P2P_MAX_STEPS * 2) // Maximum number of concurrent P2P operation pairs
#define FLAGCX_P2P_IPC_HANDLE_SIZE FLAGCX_IPC_HANDLE_SIZE

#ifdef __cplusplus
extern "C" {
#endif

struct flagcxP2pRequest {
  size_t size;
  int refcount;
};

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
  uint64_t opHash; // Hash identifying which operation owns this slot
  int done;        // 1 = slot is free, 0 = slot is in use
  int peerDone;    // 1 = slot is free, 0 = slot is in use
};

struct p2pRegInfo {
  int copyDone;    // Indicates if the copy operation is complete
  int copyStarted; // Indicates if the copy operation has started
  // WRITE mode: recv publishes into sender's slot
  int ipcRecvRegReady;      // 1 = ipcRecvRmtAddr valid; recv sets
  uintptr_t ipcRecvRmtAddr; // Recv's buffer mapped in sender's address space
  // READ mode: sender publishes into receiver's slot
  int ipcSendRegReady; // 1 = ipcSendRmtAddr valid; sender sets
  uintptr_t
      ipcSendRmtAddr; // Sender's buffer mapped in receiver's address space
};

struct flagcxP2pShm {
  // Array of synchronization slots for multiple concurrent operations
  struct flagcxP2pSyncSlot slots[FLAGCX_P2P_MAX_OPS];
  // Array of registration info for multiple concurrent operations
  struct p2pRegInfo regInfos[FLAGCX_P2P_MAX_OPS];
};

// need to make sure this matches flagcxP2pShmProxyInfo in p2p.cc
struct flagcxP2pShmProxyInfo {
  // CPU side
  struct flagcxP2pShm *shm;
  flagcxShmIpcDesc_t desc;

  // Device side
  char *recvFifo;
  flagcxStream_t stream;
  flagcxEvent_t events[FLAGCX_P2P_MAX_STEPS];
};

struct flagcxP2pResources {
  // Shared memory for synchronization
  struct flagcxP2pShm *shm;
  flagcxShmIpcDesc_t desc;

  // Proxy info for async operations
  struct flagcxP2pShmProxyInfo proxyInfo;
};

// Bootstrap tag for one-sided IPC registration (first call only).
// Recv-side sends to peer with tag = P2P_IPC_TAG_BASE + recvRank.
// Send-side receives from peer with tag = P2P_IPC_TAG_BASE + peerRank.
// Subsequent calls use SHM for offset refresh (zero bootstrap overhead).
#define P2P_IPC_TAG_BASE 4000

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

flagcxResult_t flagcxP2pProxyRegister(struct flagcxProxyConnection *connection,
                                      struct flagcxProxyState *proxyState,
                                      void *reqBuff, int reqSize,
                                      void *respBuff, int respSize, int *done);

flagcxResult_t
flagcxP2pProxyDeregister(struct flagcxProxyConnection *connection,
                         struct flagcxProxyState *proxyState, void *reqBuff,
                         int reqSize, int *done);

flagcxResult_t
flagcxP2pAllocateShareableBuffer(size_t size, int directMap,
                                 struct flagcxP2pIpcDesc *ipcDesc, void **ptr);

flagcxResult_t flagcxP2pImportShareableBuffer(struct flagcxHeteroComm *comm,
                                              int peer, size_t size,
                                              struct flagcxP2pIpcDesc *ipcDesc,
                                              void **devMemPtr);

flagcxResult_t flagcxP2pRegisterBuffer(struct flagcxHeteroComm *comm,
                                       const void *userbuff, size_t buffSize,
                                       int *peerRanks, int nPeers,
                                       int *regBufFlag, uintptr_t *offsetOut,
                                       uintptr_t **peerRmtAddrsOut);

flagcxResult_t flagcxP2pDeregisterBuffer(struct flagcxHeteroComm *comm,
                                         struct flagcxIpcRegInfo *info);

flagcxResult_t flagcxP2pSendProxyFree(struct flagcxP2pResources *resources);

flagcxResult_t flagcxP2pRecvProxyFree(struct flagcxP2pResources *resources);

void setP2pSlotInfo(int rank, int peerRank, size_t size, flagcxDataType_t dtype,
                    int isRecv, uint64_t *opHash, size_t *slotIdx);

#ifdef __cplusplus
}
#endif

#endif // FLAGCX_INT_P2P_H_