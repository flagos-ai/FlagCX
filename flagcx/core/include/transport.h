/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_TRANSPORT_H_
#define FLAGCX_TRANSPORT_H_

#include "core.h"
#include "device.h"

#define NTRANSPORTS 4
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3

#include "proxy.h"

extern struct flagcxTransport p2pTransport;
extern struct flagcxTransport shmTransport;
extern struct flagcxTransport netTransport;
extern struct flagcxTransport collNetTransport;

extern struct flagcxTransport *flagcxTransports[];

// Forward declarations
struct flagcxRing;
struct flagcxConnector;
struct flagcxHeteroComm;

struct flagcxPeerInfo {
  int rank;
  int cudaDev;
  int nvmlDev;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;
  int64_t busId;
  struct flagcxHeteroComm *comm;
  int cudaCompCap;
};

#define CONNECT_SIZE 128
struct flagcxConnect {
  char data[CONNECT_SIZE];
};

#if CUDART_VERSION >= 12010
/*
#define NVLS_HANDLE_SIZE 64
struct flagcxNvlsSharedRes {
  int refCount;
  CUmulticastObjectProp properties;
  CUmemAccessDesc accessDesc;
  int dev;
  size_t size;
  size_t granularity;
  CUmemGenericAllocationHandle mcHandle; // Multicast handle for NVLS buffer
  char* mcBuff; // Multicast NVLS buffer address
  CUmemGenericAllocationHandle ucHandle; // Unicast Handle for NVLS buffer
  char* ucBuff; // Unicast NVLS buffer address
  char shareableHandle[NVLS_HANDLE_SIZE];
  size_t ucGran;
  int nChannels;
  struct flagcxShmemCollBuff nvlsShmem;
  void *nvlsShmemHandle;
};
*/
#endif /* CUDART_VERSION >= 12010 */

struct flagcxCollNetSharedRes {
  int refCount;
  int size;
  char *cudaBuff;
  char *hostBuff;
  struct flagcxProxyArgs *proxyAppend[2 * FLAGCX_MAX_NETDEVS];
  void *resources;
  int nChannels;
  size_t buffSize;
};

struct flagcxTransportComm {
  flagcxResult_t (*setup)(struct flagcxHeteroComm *comm,
                          struct flagcxTopoGraph *graph,
                          struct flagcxPeerInfo *, struct flagcxPeerInfo *,
                          struct flagcxConnect *, struct flagcxConnector *,
                          int channelId, int connIndex);
  flagcxResult_t (*connect)(struct flagcxHeteroComm *comm,
                            struct flagcxConnect *, int nranks, int rank,
                            struct flagcxConnector *);
  flagcxResult_t (*free)(struct flagcxConnector *);
  flagcxResult_t (*proxySharedInit)(struct flagcxProxyConnection *connection,
                                    struct flagcxProxyState *proxyState,
                                    int nChannels);
  flagcxResult_t (*proxySetup)(struct flagcxProxyConnection *connection,
                               struct flagcxProxyState *proxyState,
                               void *reqBuff, int reqSize, void *respBuff,
                               int respSize, int *done);
  flagcxResult_t (*proxyConnect)(struct flagcxProxyConnection *connection,
                                 struct flagcxProxyState *proxyState,
                                 void *reqBuff, int reqSize, void *respBuff,
                                 int respSize, int *done);
  flagcxResult_t (*proxyFree)(struct flagcxProxyConnection *connection,
                              struct flagcxProxyState *proxyState);
  flagcxResult_t (*proxyProgress)(struct flagcxProxyState *proxyState,
                                  struct flagcxProxyArgs *);
  flagcxResult_t (*proxyRegister)(struct flagcxProxyConnection *connection,
                                  struct flagcxProxyState *proxyState,
                                  void *reqBuff, int reqSize, void *respBuff,
                                  int respSize, int *done);
  flagcxResult_t (*proxyDeregister)(struct flagcxProxyConnection *connection,
                                    struct flagcxProxyState *proxyState,
                                    void *reqBuff, int reqSize, int *done);
};

struct flagcxTransport {
  const char name[8];
  flagcxResult_t (*canConnect)(int *, struct flagcxTopoServer *topoServer,
                               struct flagcxTopoGraph *graph,
                               struct flagcxPeerInfo *,
                               struct flagcxPeerInfo *);
  struct flagcxTransportComm send;
  struct flagcxTransportComm recv;
};

flagcxResult_t flagcxTransportP2pConnect(struct flagcxHeteroComm *comm,
                                         int channelId, int nrecv,
                                         int *peerRecv, int nsend,
                                         int *peerSend, int connIndex);
flagcxResult_t flagcxTransportP2pSetup(struct flagcxHeteroComm *comm,
                                       struct flagcxTopoGraph *graph,
                                       int connIndex,
                                       int *highestTransportType = NULL);

flagcxResult_t flagcxNvlsInit(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxNvlsSetup(struct flagcxHeteroComm *comm,
                               struct flagcxHeteroComm *parent);
flagcxResult_t flagcxNvlsGraphRegisterBuffer(
    struct flagcxHeteroComm *comm, struct flagcxKernelPlan *plan,
    const void *sendbuff, void *recvbuff, size_t sendbuffSize,
    size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend,
    void **outRegBufRecv);
flagcxResult_t flagcxNvlsLocalRegisterBuffer(
    struct flagcxHeteroComm *comm, const void *sendbuff, void *recvbuff,
    size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed,
    void **outRegBufSend, void **outRegBufRecv);
flagcxResult_t flagcxNvlsFree(struct flagcxHeteroComm *comm);

enum { collNetRecv = 0, collNetSend = 1 };

int flagcxTransportCollNetSetup(struct flagcxHeteroComm *comm,
                                struct flagcxTopoGraph *collNetGraph,
                                struct flagcxChannel *channel, int masterRank,
                                int masterPeer, int collNetGraphChannelId,
                                int type, flagcxConnect *connect);
flagcxResult_t flagcxTransportCollNetCheck(struct flagcxHeteroComm *comm,
                                           int collNetSetupFail);
flagcxResult_t flagcxTransportCollNetFree(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxCollnetLocalRegisterBuffer(struct flagcxHeteroComm *comm,
                                                const void *userbuff,
                                                size_t buffSize, int type,
                                                int *outRegBufUsed,
                                                void **outHandle);
flagcxResult_t flagcxCollnetGraphRegisterBuffer(struct flagcxHeteroComm *comm,
                                                struct flagcxKernelPlan *plan,
                                                const void *userbuff,
                                                size_t buffSize, int type,
                                                int *outRegBufFlag,
                                                void **outHandle);
flagcxResult_t flagcxCollnetDeregBuffer(struct flagcxHeteroComm *comm,
                                        struct flagcxProxyConnector *proxyconn,
                                        void *handle);

#endif
