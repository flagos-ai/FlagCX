/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All
 *Rights Reserved. Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/

#ifndef FLAGCX_ADAPTOR_H_
#define FLAGCX_ADAPTOR_H_

#include "bootstrap.h"
#include "device_utils.h"
#include "flagcx.h"
#include "global_comm.h"
#include "topo.h"

typedef void (*flagcxLaunchFunc_t)(flagcxStream_t, void *);

#ifdef __cplusplus
extern "C" {
#endif

#define NCCLADAPTORS 2
#define flagcxCCLAdaptorHost 0
#define flagcxCCLAdaptorDevice 1

extern struct flagcxCCLAdaptor bootstrapAdaptor;
extern struct flagcxCCLAdaptor glooAdaptor;
extern struct flagcxCCLAdaptor mpiAdaptor;
extern struct flagcxCCLAdaptor ncclAdaptor;
extern struct flagcxCCLAdaptor hcclAdaptor;
extern struct flagcxCCLAdaptor ixncclAdaptor;
extern struct flagcxCCLAdaptor cnclAdaptor;
extern struct flagcxCCLAdaptor mcclAdaptor;
extern struct flagcxCCLAdaptor musa_mcclAdaptor;
extern struct flagcxCCLAdaptor xcclAdaptor;
extern struct flagcxCCLAdaptor duncclAdaptor;
extern struct flagcxCCLAdaptor rcclAdaptor;
extern struct flagcxCCLAdaptor tcclAdaptor;
extern struct flagcxCCLAdaptor *cclAdaptors[];

extern struct flagcxDeviceAdaptor cudaAdaptor;
extern struct flagcxDeviceAdaptor cannAdaptor;
extern struct flagcxDeviceAdaptor ixcudaAdaptor;
extern struct flagcxDeviceAdaptor mluAdaptor;
extern struct flagcxDeviceAdaptor macaAdaptor;
extern struct flagcxDeviceAdaptor musaAdaptor;
extern struct flagcxDeviceAdaptor kunlunAdaptor;
extern struct flagcxDeviceAdaptor ducudaAdaptor;
extern struct flagcxDeviceAdaptor hipAdaptor;
extern struct flagcxDeviceAdaptor tsmicroAdaptor;
extern struct flagcxDeviceAdaptor *deviceAdaptor;

extern struct flagcxNetAdaptor *netAdaptor;

// Network type enumeration
enum NetType {
  IBRC = 1,   // InfiniBand RC (or UCX when USE_UCX=1)
  SOCKET = 2, // Socket
#ifdef USE_IBUC
  IBUC = 3 // InfiniBand UC
#endif
};

// Unified network adaptor function declarations
struct flagcxNetAdaptor *getUnifiedNetAdaptor(int netType);

inline bool flagcxCCLAdaptorNeedSendrecv(size_t value) { return value != 0; }

struct flagcxCCLAdaptor {
  const char name[32];
  // Basic functions
  flagcxResult_t (*getVersion)(int *version);
  flagcxResult_t (*getUniqueId)(flagcxUniqueId_t *uniqueId);
  const char *(*getErrorString)(flagcxResult_t result);
  const char *(*getLastError)(flagcxInnerComm_t comm);

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
};

const int MAX_VENDOR_LEN = 128;
typedef struct {
  char internal[MAX_VENDOR_LEN];
} flagcxVendor;

struct flagcxDeviceAdaptor {
  char name[32];
  // Basic functions
  flagcxResult_t (*deviceSynchronize)();
  flagcxResult_t (*deviceMemcpy)(void *dst, void *src, size_t size,
                                 flagcxMemcpyType_t type, flagcxStream_t stream,
                                 void *args);
  flagcxResult_t (*deviceMemset)(void *ptr, int value, size_t size,
                                 flagcxMemType_t type, flagcxStream_t stream);
  flagcxResult_t (*deviceMalloc)(void **ptr, size_t size, flagcxMemType_t type,
                                 flagcxStream_t stream);
  flagcxResult_t (*deviceFree)(void *ptr, flagcxMemType_t type,
                               flagcxStream_t stream);
  flagcxResult_t (*setDevice)(int dev);
  flagcxResult_t (*getDevice)(int *dev);
  flagcxResult_t (*getDeviceCount)(int *count);
  flagcxResult_t (*getVendor)(char *vendor);
  flagcxResult_t (*hostGetDevicePointer)(void **pDevice, void *pHost);

  // GDR functions
  flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
  flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
  flagcxResult_t (*gdrMemAlloc)(void **ptr, size_t size, void *memHandle);
  flagcxResult_t (*gdrMemFree)(void *ptr, void *memHandle);
  flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void *memHandle);
  flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
  flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t sz);
  flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);

  // Stream functions
  flagcxResult_t (*streamCreate)(flagcxStream_t *stream);
  flagcxResult_t (*streamDestroy)(flagcxStream_t stream);
  flagcxResult_t (*streamCopy)(flagcxStream_t *newStream, void *oldStream);
  flagcxResult_t (*streamFree)(flagcxStream_t stream);
  flagcxResult_t (*streamSynchronize)(flagcxStream_t stream);
  flagcxResult_t (*streamQuery)(flagcxStream_t stream);
  flagcxResult_t (*streamWaitEvent)(flagcxStream_t stream, flagcxEvent_t event);

  // Event functions
  flagcxResult_t (*eventCreate)(flagcxEvent_t *event,
                                flagcxEventType_t eventType);
  flagcxResult_t (*eventDestroy)(flagcxEvent_t event);
  flagcxResult_t (*eventRecord)(flagcxEvent_t event, flagcxStream_t stream);
  flagcxResult_t (*eventSynchronize)(flagcxEvent_t event);
  flagcxResult_t (*eventQuery)(flagcxEvent_t event);

  // IpcMemHandle functions
  flagcxResult_t (*ipcMemHandleCreate)(flagcxIpcMemHandle_t *handle,
                                       size_t *size);
  flagcxResult_t (*ipcMemHandleGet)(flagcxIpcMemHandle_t handle, void *devPtr);
  flagcxResult_t (*ipcMemHandleOpen)(flagcxIpcMemHandle_t handle,
                                     void **devPtr);
  flagcxResult_t (*ipcMemHandleClose)(void *devPtr);
  flagcxResult_t (*ipcMemHandleFree)(flagcxIpcMemHandle_t handle);

  // Kernel launch
  // TODO: verify if we do need these funcs, if so, figure out a way to
  // eliminate overly fine-grained arguments such as block_xxx, grid_xxx, etc.
  // And define more generic kernel launch APIs
  flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
                                 unsigned int block_y, unsigned int block_z,
                                 unsigned int grid_x, unsigned int grid_y,
                                 unsigned int grid_z, void **args,
                                 size_t share_mem, void *stream,
                                 void *memHandle);
  flagcxResult_t (*copyArgsInit)(void **args);
  flagcxResult_t (*copyArgsFree)(void *args);
  flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream,
                                     flagcxLaunchFunc_t fn, void *args);

  // Others
  // TODO: this one shall be moved into Flagcx Core Topology APIs
  // Here we only define several low-level APIs required by topology detection
  flagcxResult_t (*getDeviceProperties)(struct flagcxDevProps *props, int dev);
  flagcxResult_t (*getDevicePciBusId)(char *pciBusId, int len, int dev);
  flagcxResult_t (*getDeviceByPciBusId)(int *dev, const char *pciBusId);

  // HostFunc launch
  flagcxResult_t (*launchHostFunc)(flagcxStream_t stream, void (*fn)(void *),
                                   void *args);
  // DMA buffer
  flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
  flagcxResult_t (*getHandleForAddressRange)(void *handleOut, void *buffer,
                                             size_t size,
                                             unsigned long long flags);
  // Event elapsed time
  flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t start,
                                     flagcxEvent_t end);
};

struct flagcxNetAdaptor {
  // Basic functions
  const char *name;
  flagcxResult_t (*init)();
  flagcxResult_t (*devices)(int *ndev);
  flagcxResult_t (*getProperties)(
      int dev, void *props); // TODO: add flagcxNetProperties_t* props
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  flagcxResult_t (*getDeviceMr)(void *comm, void *mhandle, void **dptr_mhandle);
  flagcxResult_t (*irecvConsumed)(void *recvComm, int n, void *request);
  // flagcxResult_t (*makeVDevice)(int* d, flagcxNetVDeviceProps_t* props);

  // Setup functions
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  flagcxResult_t (*connect)(
      int dev, void *handle,
      void **sendComm); // TODO: add flagcxNetDeviceHandle_t** sendDevComm
  // flagcxResult_t (*connect)(void* handles[], int nranks, int rank, void*
  // listenComm, void** collComm);
  flagcxResult_t (*accept)(
      void *listenComm,
      void **recvComm); // TODO: add flagcxNetDeviceHandle_t** recvDevComm
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Memory region functions
  flagcxResult_t (*regMr)(void *comm, void *data, size_t size, int type,
                          void **mhandle);
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, void **mhandle);
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

  // One-sided functions
  flagcxResult_t (*write)(void *sendComm, void *data, size_t size, int tag,
                          void *mhandle, void *phandle, void **request);
  flagcxResult_t (*read)(void *recvComm, void *data, size_t size, int tag,
                         void *mhandle, void *phandle, void **request);
  flagcxResult_t (*signal)(void *sendComm, void *data, size_t size, int tag,
                           void *mhandle, void *phandle, void **request);

  // Device name lookup
  flagcxResult_t (*getDevFromName)(char *name, int *dev);

  // TODO: add switch functions such as
  // iallreduce, iallgather, ireducescatter,
  // ireduce, ibroadcast, iflush, etc.
};

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard