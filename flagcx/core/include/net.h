/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_INT_NET_H_
#define FLAGCX_INT_NET_H_

#include "check.h"
#include "comm.h"
#include "device.h"
#include "flagcx_net.h"
#include "register.h"
#include <socket.h>

typedef char flagcxNetHandle_t[FLAGCX_NET_HANDLE_MAXSIZE];

extern int64_t flagcxNetBufferSize;
extern int64_t flagcxNetChunkSize;
extern int64_t flagcxNetChunks;

enum flagcxNetState {
  flagcxNetStateInit = 0,
  flagcxNetStateEnabled = 1,
  flagcxNetStateDisabled = 2
};
extern enum flagcxNetState flagcxNetStates[3];
#define FLAGCX_NET_MAX_STEPS 16
#define FLAGCX_MAX_NET_SIZE_BYTES (1 * 1024 * 1024 * 1024 * 1024L)

flagcxResult_t flagcxNetInit(struct flagcxHeteroComm *comm);
int flagcxNetVersion(struct flagcxHeteroComm *comm);

// Test whether the current GPU support GPU Direct RDMA.
flagcxResult_t flagcxGpuGdrSupport(struct flagcxHeteroComm *comm,
                                   int *gdrSupport);
bool flagcxNetCanUseDeviceMemory(struct flagcxNetAdaptor *netAdaptor,
                                 const flagcxNetProperties_t *properties);

// Network adaptor declarations
extern struct flagcxNetAdaptor flagcxNetSocket;
extern struct flagcxNetAdaptor flagcxNetIb;
#ifdef USE_IBUC
extern struct flagcxNetAdaptor flagcxNetIbuc;
#endif
#ifdef USE_ACCL_BAREX
extern struct flagcxNetAdaptor flagcxNetBarex;
#endif

struct flagcxNetRelayPool;

// One IPC mapping per exported pool on the source rank. Multiple NET
// connectors can refer to this mapping without opening the handle again.
struct flagcxNetRelayImport {
  struct flagcxNetRelayImport *next;
  int proxyRank;
  uint64_t poolId;
  void *base;
  size_t capacity;
  flagcxIpcHandleData handleData;
  size_t handleSize;
  bool closed;
};

struct sendNetResources {
  void *netSendComm;
  struct flagcxSendMem *sendMem;
  struct flagcxRecvMem *recvMem;

  struct flagcxHeteroComm *commPtr;
  struct flagcxNetAdaptor *netAdaptor;
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int ptrSupport;
  int maxRecvs;
  uint64_t *gdcSync;
  void *gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char *buffers[FLAGCX_NUM_PROTOCOLS];
  int buffSizes[FLAGCX_NUM_PROTOCOLS];
  void *mhandles[1]; /*just one for memory copy from device to gdr buffer*/
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  flagcxNetDeviceType netDeviceType;
  flagcxNetDeviceHandle_t *netDeviceHandle;
  flagcxStream_t cpStream;
  flagcxEvent_t cpEvents[FLAGCX_NET_MAX_STEPS];
  // PXN: the relay owns a shared exported buffer for this source rank.
  flagcxIpcHandleData relayHandleData;
  size_t relayHandleSize;
  char *relayExportBuffer;
  struct flagcxNetRelayPool *relayPool;
  uint32_t relaySlot;
  bool relayIpcBuffer;
  bool relaySourceReleased;
};

// Initialize NET send resources in the process that owns the send proxy.
flagcxResult_t flagcxNetInitSendResources(struct flagcxNetAdaptor *netAdaptor,
                                          int netDev,
                                          struct sendNetResources *resources,
                                          bool relay = false);
flagcxResult_t flagcxNetDevFromGuid(struct flagcxNetAdaptor *netAdaptor,
                                    uint64_t netGuid, int *netDev);
flagcxResult_t
flagcxNetAttachRelayPool(struct flagcxProxyState *proxyState, int sourceRank,
                         struct flagcxProxyConnection *connection,
                         struct sendNetResources *resources);
flagcxResult_t flagcxNetLeaseRelaySlot(struct sendNetResources *resources,
                                       struct flagcxProxyConnection *connection,
                                       uint64_t requestId, uint32_t *slot);
bool flagcxNetRelaySlotOwned(struct sendNetResources *resources,
                             struct flagcxProxyConnection *connection,
                             uint64_t requestId, uint32_t slot);
void flagcxNetReleaseRelaySlot(struct sendNetResources *resources,
                               struct flagcxProxyConnection *connection,
                               uint64_t requestId);
bool flagcxNetRelayConnectionHasSlot(struct sendNetResources *resources,
                                     struct flagcxProxyConnection *connection);
uint64_t flagcxNetRelayPoolId(struct sendNetResources *resources);
void flagcxNetPoisonRelayPool(struct sendNetResources *resources);
bool flagcxNetRelayPoolUnsafe(struct sendNetResources *resources);
size_t flagcxNetRelaySlotOffset(uint32_t slot);

struct flagcxNetSendSetupRequest {
  // NET device numbers are process-local; the relay resolves this identity.
  uint64_t netGuid;
};

struct flagcxNetRelayBufferInfo {
  flagcxIpcHandleData handleData;
  size_t handleSize;
  size_t capacity;
  size_t chunkSize;
  uint64_t poolId;
};

struct flagcxNetRelayLeaseRequest {
  uint64_t requestId;
};

// One bounded source-to-relay chunk. The source writes into the persistent
// relay-owned buffer before this request; only scalar metadata crosses RPC.
struct flagcxNetRelaySendRequest {
  size_t bytes;
  uint64_t requestId;
  uint32_t slot;
  uint64_t generation;
  uint64_t orderingKey;
  uint64_t sequence;
  uint32_t submitFlags;
};

struct flagcxNetRelayCancelRequest {
  uint64_t requestId;
};

struct recvNetResources {
  void *netListenComm;
  void *netRecvComm;
  struct flagcxSendMem *sendMem;
  struct flagcxRecvMem *recvMem;

  struct flagcxHeteroComm *commPtr;
  struct flagcxNetAdaptor *netAdaptor;
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int tpRemoteProxyRank;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int ptrSupport;
  int needFlush;
  int maxRecvs;
  uint64_t *gdcSync;
  uint64_t *gdcFlush;
  void *gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char *buffers[FLAGCX_NUM_PROTOCOLS];
  int buffSizes[FLAGCX_NUM_PROTOCOLS];
  void *mhandles[FLAGCX_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  flagcxNetDeviceType netDeviceType;
  flagcxNetDeviceHandle_t *netDeviceHandle;
  flagcxStream_t cpStream;
  flagcxEvent_t cpEvents[FLAGCX_NET_MAX_STEPS];
};

enum flagcxIbCommState {
  flagcxIbCommStateStart = 0,
  flagcxIbCommStateConnect = 1,
  flagcxIbCommStateAccept = 3,
  flagcxIbCommStateSend = 4,
  flagcxIbCommStateRecv = 5,
  flagcxIbCommStateConnecting = 6,
  flagcxIbCommStateConnected = 7,
  flagcxIbCommStatePendingReady = 8,
};

struct flagcxIbCommStage {
  enum flagcxIbCommState state;
  int offset;
  void *buffer;
  void *comm;
};

struct sendRecvDataInfo {
  void *data;
  size_t size;
};

struct flagcxIbHandle {
  union flagcxSocketAddress connectAddr; // Filled by the target
  uint64_t magic;                        // random number to help debugging
  struct flagcxIbCommStage stage; // Used by the other side when connecting
};

flagcxResult_t flagcxSendRegMr(flagcxHeteroComm_t comm, void *data, size_t size,
                               int peer, int channel);
flagcxResult_t flagcxRecvRegMr(flagcxHeteroComm_t comm, void *data, size_t size,
                               int peer, int channel);
flagcxResult_t flagcxProxySend(sendNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args);
flagcxResult_t flagcxProxyRecv(recvNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args);
flagcxResult_t flagcxNetPrepareProxyOp(struct flagcxHeteroComm *comm,
                                       struct flagcxProxyOp *op, void *buffer,
                                       size_t size, int peer,
                                       flagcxDataType_t dtype);
flagcxResult_t
flagcxNetProgressProxyOp(struct flagcxProxyConnection *connection,
                         struct flagcxProxyOp *op);
void flagcxNetCleanupRelaySendOp(struct flagcxProxyOp *op);
void flagcxNetAbandonRelaySendOp(struct flagcxProxyOp *op);
flagcxResult_t
flagcxNetCleanupProxyConnection(struct flagcxProxyConnection *connection,
                                int cleanupPhase);
flagcxResult_t flagcxSend(flagcxHeteroComm_t comm, void *data, size_t size,
                          int peer, int channel);
flagcxResult_t flagcxRecv(flagcxHeteroComm_t comm, void *data, size_t size,
                          int peer, int channel);
flagcxResult_t flagcxSendProxyFree(sendNetResources *resources);
flagcxResult_t flagcxRecvProxyFree(recvNetResources *resources);

flagcxResult_t flagcxNetRegisterBuffer(flagcxHeteroComm *comm,
                                       const void *userbuff, size_t buffSize,
                                       struct flagcxConnector **peerConns,
                                       int nPeers, int *outRegBufFlag,
                                       void **outHandle);
flagcxResult_t flagcxNetDeregisterBuffer(void *comm,
                                         struct flagcxProxyConnector *proxyConn,
                                         void *handle);

#endif
