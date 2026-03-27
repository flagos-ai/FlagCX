/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_CCL_ADAPTOR_H_
#define FLAGCX_CCL_ADAPTOR_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for opaque types not in flagcx.h
// (flagcxStream_t, flagcxWindow_t are already typedef'd in flagcx.h)
typedef struct flagcxInnerComm *flagcxInnerComm_t;
typedef struct flagcxInnerDevComm *flagcxInnerDevComm_t;
struct flagcxDevCommRequirements;
struct bootstrapState;

// Version history:
//   v1 — 34 function pointers: getVersion, getUniqueId, getErrorString,
//         getLastError, getStagedBuffer, commInitRank, commFinalize,
//         commDestroy, commAbort, commResume, commSuspend, commCount,
//         commGetDeviceNumber, commUserRank, commGetAsyncError, memAlloc,
//         memFree, commRegister, commDeregister, commWindowRegister,
//         commWindowDeregister, reduce, gather, scatter, broadcast,
//         allReduce, reduceScatter, allGather, alltoAll, alltoAllv,
//         send, recv, groupStart, groupEnd
struct flagcxCCLAdaptor_v1 {
  const char *name;
  // Basic functions
  flagcxResult_t (*getVersion)(int *version);
  flagcxResult_t (*getUniqueId)(flagcxUniqueId_t *uniqueId);
  const char *(*getErrorString)(flagcxResult_t result);
  const char *(*getLastError)(flagcxInnerComm_t comm);
  flagcxResult_t (*getStagedBuffer)(const flagcxInnerComm_t comm, void **buff,
                                    size_t size, int isRecv);

  // Communicator functions
  flagcxResult_t (*commInitRank)(flagcxInnerComm_t *comm, int nranks,
                                 flagcxUniqueId *commId, int rank,
                                 bootstrapState *bootstrap);
  flagcxResult_t (*commFinalize)(flagcxInnerComm_t comm);
  flagcxResult_t (*commDestroy)(flagcxInnerComm_t comm);
  flagcxResult_t (*commAbort)(flagcxInnerComm_t comm);
  flagcxResult_t (*commResume)(flagcxInnerComm_t comm);
  flagcxResult_t (*commSuspend)(flagcxInnerComm_t comm);
  flagcxResult_t (*commCount)(const flagcxInnerComm_t comm, int *count);
  flagcxResult_t (*commGetDeviceNumber)(const flagcxInnerComm_t comm,
                                        int *device);
  flagcxResult_t (*commUserRank)(const flagcxInnerComm_t comm, int *rank);
  flagcxResult_t (*commGetAsyncError)(flagcxInnerComm_t comm,
                                      flagcxResult_t *asyncError);
  flagcxResult_t (*memAlloc)(void **ptr, size_t size);
  flagcxResult_t (*memFree)(void *ptr);
  flagcxResult_t (*commRegister)(const flagcxInnerComm_t comm, void *buff,
                                 size_t size, void **handle);
  flagcxResult_t (*commDeregister)(const flagcxInnerComm_t comm, void *handle);
  // Symmetric functions
  flagcxResult_t (*commWindowRegister)(flagcxInnerComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags);
  flagcxResult_t (*commWindowDeregister)(flagcxInnerComm_t comm,
                                         flagcxWindow_t win);

  // Communication functions
  flagcxResult_t (*reduce)(const void *sendbuff, void *recvbuff, size_t count,
                           flagcxDataType_t datatype, flagcxRedOp_t op,
                           int root, flagcxInnerComm_t comm,
                           flagcxStream_t stream);
  flagcxResult_t (*gather)(const void *sendbuff, void *recvbuff, size_t count,
                           flagcxDataType_t datatype, int root,
                           flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*scatter)(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, int root,
                            flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*broadcast)(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype, int root,
                              flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*allReduce)(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype,
                              flagcxRedOp_t op, flagcxInnerComm_t comm,
                              flagcxStream_t stream);
  flagcxResult_t (*reduceScatter)(const void *sendbuff, void *recvbuff,
                                  size_t recvcount, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, flagcxInnerComm_t comm,
                                  flagcxStream_t stream);
  flagcxResult_t (*allGather)(const void *sendbuff, void *recvbuff,
                              size_t sendcount, flagcxDataType_t datatype,
                              flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*alltoAll)(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, flagcxInnerComm_t comm,
                             flagcxStream_t stream);
  flagcxResult_t (*alltoAllv)(const void *sendbuff, size_t *sendcounts,
                              size_t *sdispls, void *recvbuff,
                              size_t *recvcounts, size_t *rdispls,
                              flagcxDataType_t datatype, flagcxInnerComm_t comm,
                              flagcxStream_t stream);
  flagcxResult_t (*send)(const void *sendbuff, size_t count,
                         flagcxDataType_t datatype, int peer,
                         flagcxInnerComm_t comm, flagcxStream_t stream);
  flagcxResult_t (*recv)(void *recvbuff, size_t count,
                         flagcxDataType_t datatype, int peer,
                         flagcxInnerComm_t comm, flagcxStream_t stream);

  // Group semantics
  flagcxResult_t (*groupStart)();
  flagcxResult_t (*groupEnd)();

  // Device API - Host-side management (NCCL > 2.28, CNCL device API, etc.)
  flagcxResult_t (*devCommCreate)(flagcxInnerComm_t comm,
                                  const flagcxDevCommRequirements *reqs,
                                  flagcxInnerDevComm_t *devComm);
  flagcxResult_t (*devCommDestroy)(flagcxInnerComm_t comm,
                                   flagcxInnerDevComm_t devComm);
};
#define flagcxCCLAdaptor flagcxCCLAdaptor_v1

// Versioned export symbol name
#define FLAGCX_CCL_ADAPTOR_PLUGIN_SYMBOL_V1 flagcxCCLAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // FLAGCX_CCL_ADAPTOR_H_
