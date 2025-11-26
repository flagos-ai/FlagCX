/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_PROXY_H_
#define FLAGCX_PROXY_H_

#include "device.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "group.h"
#include "info.h"
#include "ipcsocket.h"
#include "launch_kernel.h"
#include "net.h"
#include "reg_pool.h"
#include "socket.h"
#include <memory>
#include <pthread.h>

enum flagcxProxyOpState {
  flagcxProxyOpNone,
  flagcxProxyOpReady,
  flagcxProxyOpProgress
};

struct flagcxProxyKernelState {
  pthread_t thread;
  flagcxFifo_t fifo;
  flagcxStream_t stream;
  int stop = 0;
};

struct flagcxProxyArgs;
typedef flagcxResult_t (*proxyProgressFunc_t)(struct flagcxProxyState *,
                                              struct flagcxProxyArgs *);

#define FLAGCX_PROXY_MAX_SUBS MAXCHANNELS
static_assert(FLAGCX_MAX_WORK_ELEMENTS <= MAXCHANNELS,
              "Not enough sub space for max work elements");

union flagcxProxyOpSpecifics {
  struct {
    size_t sizePerRank;
    int nNodes, node;
  } collnetDirect;
};

struct flagcxProxySubArgs {
  struct flagcxProxyConnection *connection;
  int reg;
  // p2p mhandle
  void *mhandle;
  int stepSize;
  void *stepBuff;
  void *stream;
  // kernel copy
  void *copyArgs;

  // collnet handles
  void *sendMhandle;
  void *recvMhandle;
  uint8_t *sendbuff;
  uint8_t *recvbuff;
  size_t offset;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int peer;

  int groupSize; // Number of consecutive sub operations sharing the same
                 // recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  void *requests[FLAGCX_STEPS];
  void *profilingEvents[FLAGCX_STEPS];
  void *recvRequestsCache[FLAGCX_STEPS];
  int recvRequestsSubCount;
};

struct flagcxProxyArgs {
  struct flagcxProxySubArgs subs[MAXSTEPS];
  proxyProgressFunc_t progress;
  int nsubs;
  int done;
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  size_t chunkSize;
  size_t stepSize;
  void *stepBuff;
  int waitCopy = 0;
  int posted = 0;
  int copied = 0;
  int postFlush = 0;
  int flushed = 0;
  int transmitted = 0;
  int sendStepMask;
  size_t totalCopySize;
  size_t totalPostSize;
  size_t totalSendSize;
  size_t totalRecvSize;
  size_t sendSizePerRound;
  size_t recvSizePerRound;
  uint8_t /*flagcxDataType_t*/ dtype;
  uint8_t /*flagcxDevRedOp_t*/ redOp;
  uint8_t /*flagcxPattern_t*/ pattern;
  uint8_t /*flagcxFunc_t*/ coll;
  uint8_t protocol;
  int state;
  char *sharedBuff[FLAGCX_STEPS];
  int sharedSize[FLAGCX_STEPS];

  int idle;

  // Element linking
  struct flagcxProxyArgs *next;
  struct flagcxProxyArgs *nextPeer;
  struct flagcxProxyArgs **proxyAppendPtr;

  /*for launch*/
  int deviceFuncRelaxedOrdering = 0;
  std::shared_ptr<flagcxHostSemaphore> semaphore;
  // only for device func, to be deprecated
  volatile bool eventRecorded = false;
  volatile bool hlArgs = false;
  volatile bool hEventReady = false;
  bool *volatile dlArgs;
  bool *volatile dEventReady;

  // user buffer registration
  void *regHandle = nullptr;
  int regBufFlag = 0;

  // P2P operation slot management
  int p2pOpHash = -1;
  int p2pPeerOpHash = -1;
  size_t p2pSlotIdx = 0;
  size_t p2pPeerSlotIdx = 0;

  union flagcxProxyOpSpecifics specifics;
};

struct flagcxProxyOp {
  struct flagcxProxyConnection *connection;
  ssize_t nbytes;
  uint64_t opCount;
  int root;
  struct flagcxProxyOp *next;
  int nsteps;
  int chunkSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;
  uint8_t /*flagcxDataType_t*/ dtype;
  uint8_t /*flagcxDevRedOp_t*/ redOp;
  uint8_t /*flagcxFunc_t*/ coll;
  uint8_t /*flagcxPattern_t*/ pattern;
  void *kernelSyncPtr;
  uint8_t protocol;
  uint8_t reg;
  // collnet buffer reg handles
  void *sendMhandle;
  void *recvMhandle;
  uint8_t *sendbuff;
  uint8_t *recvbuff;

  union flagcxProxyOpSpecifics specifics;

  struct flagcxProxyOp *enqNext;
  unsigned long long flagcxFuncTimes;
  int peerRank;
  int rank;
  uint64_t groupHash;
  /**
   * TODO: just for test, we will delete the flagcxHeteroComm_t comm;
   **/
  flagcxHeteroComm_t comm;
  flagcxProxyArgs args;
  flagcxStream_t stream;
  flagcxEvent_t event; // used to record host/device func
  int selfCopy = 0;
};

#define FLAGCX_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all
// channels. Otherwise we'd be unable to post half of them to free new elements.
#define MAX_OPS_PER_PEER (2 * MAXCHANNELS * FLAGCX_MAX_WORK_ELEMENTS_P2P)

struct flagcxProxyOpsPool {
  struct flagcxProxyOp ops[MAX_OPS_PER_PEER * FLAGCX_MAX_LOCAL_RANKS];
  volatile int nextOps;
  volatile int nextOpsEnd;
  volatile int freeOps[FLAGCX_MAX_LOCAL_RANKS];
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct flagcxProxyOps {
  pthread_mutex_t mutex;
  struct consPeer {
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next>
        sendQueue;
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next>
        recvQueue;
    struct consPeer *nextPeer;
    struct consPeer *prevPeer;
  };
  struct prodPeer {
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next>
        sendQueue;
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next>
        recvQueue;
  };

  struct consPeer *consPeers;
  struct prodPeer prodPeers;
  struct consPeer *consProgPeerHead;
  struct flagcxProxyOps *prodNextChannel;
  struct flagcxProxyOps *prodPrevChannel;
  struct flagcxProxyOps *consNextChannel;
  struct flagcxProxyOps *consPrevChannel;
};

struct flagcxProxySharedP2p {
  int refcount;
  int size;
  char *cudaBuff;
  char *hostBuff;
  struct flagcxProxyArgs *proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct flagcxProxyPeer {
  struct flagcxProxySharedP2p send;
  struct flagcxProxySharedP2p recv;
};

struct flagcxSharedNetComms {
  void *sendComm[MAXCHANNELS];
  void *recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct flagcxProxyPool;
struct flagcxProxyProgressState {
  // Used by main threads to send work to progress thread
  struct flagcxProxyOpsPool *opsPool;
  char opsPoolShmSuffix[6];

  pthread_t thread;
  volatile int stop;
  struct flagcxProxyPeer **localPeers;
  struct flagcxSharedNetComms *netComms[FLAGCX_MAX_NETDEVS];
  struct flagcxProxyArgs *active;
  struct flagcxProxyArgs *pool;
  struct flagcxProxyPool *pools;
  int nextOps;
};

// Expected proxy response fifo
struct flagcxExpectedProxyResponse {
  void *opId;
  int respSize;
  bool done;
  void *respBuff;
  flagcxResult_t res;
  struct flagcxExpectedProxyResponse *next;
};

struct flagcxProxyAsyncOp {
  int type;
  bool done;
  flagcxProxyArgs args;
  struct flagcxProxyConnection *connection;
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  void *opId;
  flagcxProxyAsyncOp *prev;
  flagcxProxyAsyncOp *next;
};

// Common response header for all proxyOps
// We pack this into a struct to reduce the number of blocking send and recv
// calls
struct flagcxProxyRpcResponseHeader {
  void *opId;
  flagcxResult_t res;
  int respSize;
};

// UDS support
struct flagcxIpcHdr {
  int type;
  int rank;
  int reqSize;
  int respSize;
  void *opId;
  uint64_t data[16]; // 128-bytes
};

struct flagcxProxyState {
  int refCount;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int cudaDev;
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[FLAGCX_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;
  struct flagcxNetAdaptor *netAdaptor;
  volatile uint32_t *abortFlag;
  // Service threads
  pthread_t thread;
  pthread_t threadUDS;
  struct flagcxSocket listenSock;
  struct flagcxSocket ipcSock;
  int stop;
  flagcxResult_t asyncResult;
  int nRanks;

  // Used by main thread
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  union flagcxSocketAddress *peerAddresses;
  struct flagcxSocket peerSock;
  struct flagcxProxyOps proxyOps[MAXCHANNELS];

  struct flagcxProxyOps *prodProgChannelHead; /*producer*/
  struct flagcxProxyOps *consProgChannelHead; /*consumer*/

  void **sharedDevMems;
  struct flagcxIpcSocket peerIpcSock; // cuMEM API support (UDS)
  uint64_t *peerAddressesUDS;         // cuMem API support (UDS)

  // Progress thread
  struct flagcxProxyProgressState progressState;

  // Kernel thread
  bool enableProxyKernel = false;
  struct flagcxProxyKernelState kernelState;

  // Queue of expected responses from the proxy
  struct flagcxExpectedProxyResponse *expectedResponses;

  // flag indicating if the proxy is initialized.
  // This flag is used for lazy initialization of the proxy.
  // Cooperate with FLAGCX_RUNTIME_PROXY environment variable.
  int initialized;
};

enum proxyConnectState {
  connUninitialized = 0,
  connInitialized = 1,
  connSharedInitialized = 2,
  connSetupDone = 3,
  connConnected = 4,
  numConnStates = 5
};

struct flagcxProxyConnection {
  int send, transport, shared;
  int tpLocalRank, sameProcess;
  struct flagcxSocket *sock;
  struct flagcxTransportComm *tcomm;
  struct flagcxProxyArgs *proxyAppend;
  struct flagcxProxyArgs **proxyAppendPtr;
  void *transportResources;
  flagcxNetDeviceHandle_t *netDeviceHandle;
  void *mhandles[FLAGCX_NUM_PROTOCOLS];
  proxyConnectState state;
  struct flagcxCollNetSharedRes *collNet;
  int needsProxyProgress;
};

typedef flagcxResult_t (*threadFunc_t)(struct flagcxProxyArgs *);

enum proxyMode { proxyRing = 0, proxyFrom = 1, proxyTo = 2 };

void *flagcxProxyService(void *args);
flagcxResult_t flagcxProxySaveOp(struct flagcxHeteroComm *comm,
                                 struct flagcxProxyOp *proxyOp,
                                 bool *justInquire = NULL);
flagcxResult_t flagcxProxyComputeP2p(struct flagcxInfo *info,
                                     struct flagcxProxyOp *proxyOp, int reg);
flagcxResult_t flagcxProxyStart(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxProxyInit(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxProxyCreate(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxProxyConnect(struct flagcxHeteroComm *comm, int transport,
                                  int send, int proxyRank,
                                  struct flagcxProxyConnector *proxyConn);

void *flagcxProxyKernelService(void *args);

// Only flagcxProxyMsgConnect & flagcxProxyMsgStop types are used for now.
enum flagcxProxyMsgType {
  flagcxProxyMsgInit = 1,
  flagcxProxyMsgSharedInit = 2,
  flagcxProxyMsgSetup = 3,
  flagcxProxyMsgConnect = 4,
  flagcxProxyMsgStart = 5,
  flagcxProxyMsgClose = 6,
  flagcxProxyMsgAbort = 7,
  flagcxProxyMsgStop = 8,
  flagcxProxyMsgGetFd = 9, // cuMem API support (UDS)
  flagcxProxyMsgRegister = 10,
  flagcxProxyMsgDeregister = 11,
  flagcxProxyMsgRegMr = 12,
  flagcxProxyMsgDeregMr = 13,
  flagcxProxyMsgSendRecv = 14
};

// This function is called by a client of the proxy that needs to invoke any of
// the non-progress proxyOp types Call this function on the client, supplying a
// locally unique opId. Then, poll on the return value of
// flagcxPollProxyResponse(), supplying the same opId to confirm the operation
// has completed
flagcxResult_t flagcxProxyCallAsync(struct flagcxHeteroComm *comm,
                                    struct flagcxProxyConnector *proxyConn,
                                    int type, void *reqBuff, int reqSize,
                                    int respSize, void *opId);

// This function will internally call flagcxProxyCallAsync() and spin until
// flagcxPollProxyResponse() confirms the result is received
flagcxResult_t flagcxProxyCallBlocking(struct flagcxHeteroComm *comm,
                                       struct flagcxProxyConnector *proxyConn,
                                       int type, void *reqBuff, int reqSize,
                                       void *respBuff, int respSize);
flagcxResult_t flagcxPollProxyResponse(struct flagcxHeteroComm *comm,
                                       struct flagcxProxyConnector *proxyConn,
                                       void *respBuff, void *opId);

// UDS support
flagcxResult_t flagcxProxyClientGetFdBlocking(struct flagcxHeteroComm *comm,
                                              int rank, void *handle,
                                              int *convertedFd);

flagcxResult_t flagcxProxyStop(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxProxyShmUnlink(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxProxyDestroy(struct flagcxHeteroComm *comm);

#endif
