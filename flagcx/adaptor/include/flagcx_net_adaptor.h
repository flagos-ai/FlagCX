/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_NET_ADAPTOR_H_
#define FLAGCX_NET_ADAPTOR_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

// MR registration flags for one-sided strong ordering
typedef enum {
  FLAGCX_NET_MR_FLAG_NONE = 0,
  FLAGCX_NET_MR_FLAG_FORCE_SO =
      (1 << 0), // Force strong ordering (disable relaxed ordering)
} flagcxNetMrFlag_t;

// Version history:
//   v1 — 22 function pointers: name, init, devices, getProperties,
//         listen, connect, accept, closeSend, closeRecv, closeListen,
//         regMr, regMrDmaBuf, deregMr, isend, irecv, iflush, test,
//         iput, iget, iputSignal, getDevFromName
struct flagcxNetAdaptor_v1 {
  // Basic functions
  const char *name;
  flagcxResult_t (*init)();
  flagcxResult_t (*devices)(int *ndev);
  flagcxResult_t (*getProperties)(int dev, void *props);

  // Setup functions
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  flagcxResult_t (*connect)(int dev, void *handle, void **sendComm);
  flagcxResult_t (*accept)(void *listenComm, void **recvComm);
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Memory region functions
  flagcxResult_t (*regMr)(void *comm, void *data, size_t size, int type,
                          int mrFlags, void **mhandle);
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, int mrFlags,
                                void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);

  // Two-sided functions
  flagcxResult_t (*isend)(void *sendComm, void *data, size_t size, int tag,
                          void *mhandle, void *phandle, void **request);
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, size_t *sizes,
                          int *tags, void **mhandles, void **phandles,
                          void **request);
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  flagcxResult_t (*test)(void *request, int *done, int *sizes);

  // One-sided (per-window MR: separate src/dst handles for independent buffers)
  flagcxResult_t (*iput)(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                         size_t size, int srcRank, int dstRank,
                         void **srcHandles, void **dstHandles, void **request);
  // RDMA READ: pull data from remote srcRank into local dstRank buffer
  flagcxResult_t (*iget)(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                         size_t size, int srcRank, int dstRank,
                         void **srcHandles, void **dstHandles, void **request);
  // Data + signal combined (NCCL GIN-aligned: enables chained WRITE + ATOMIC)
  // When size == 0, only signal ATOMIC is posted (signal-only mode)
  flagcxResult_t (*iputSignal)(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                               size_t size, int srcRank, int dstRank,
                               void **srcHandles, void **dstHandles,
                               uint64_t signalOff, void **signalHandles,
                               uint64_t signalValue, void **request);

  // Device name lookup
  flagcxResult_t (*getDevFromName)(char *name, int *dev);

  // Optional one-sided batch WRITE. Implementations should post up to
  // `count` WRs and set `posted` to the number accepted by the QP. A partial
  // post is allowed when the send queue is temporarily full; requests[0:posted]
  // must then be valid and requests[posted:] must remain untouched/NULL.
  flagcxResult_t (*iputBatch)(void *sendComm, int count,
                              const uint64_t *srcOffs,
                              const uint64_t *dstOffs, const size_t *sizes,
                              int srcRank, int dstRank, void **srcHandles,
                              void **dstHandles, void **requests,
                              int *posted);
};
#define flagcxNetAdaptor flagcxNetAdaptor_v1

// Versioned export symbol name
#define FLAGCX_NET_ADAPTOR_PLUGIN_SYMBOL_V1 flagcxNetAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // FLAGCX_NET_ADAPTOR_H_
