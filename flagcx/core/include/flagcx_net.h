/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_NET_H_
#define FLAGCX_NET_H_

#include "core.h"
#include "flagcx_common.h"
#include "net_device.h"
#include <stdint.h>

#define FLAGCX_NET_HANDLE_MAXSIZE 128

#define FLAGCX_PTR_HOST 0x1
#define FLAGCX_PTR_CUDA 0x2
#define FLAGCX_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define FLAGCX_NET_MAX_REQUESTS 32

#define FLAGCX_NET_MAX_DEVS_PER_NIC_V10 4

typedef struct {
  int ndevs;
  int devs[FLAGCX_NET_MAX_DEVS_PER_NIC_V10];
} flagcxNetVDeviceProps_v10_t;

typedef struct {
  // Plugin-specific TC value
  int trafficClass;
} flagcxNetCommConfig_v10_t;

typedef struct {
  char *name;      // Used mostly for logging.
  char *pciPath;   // Path to the PCI device in /sys.
  uint64_t guid;   // Unique identifier for the NIC chip. Important for
                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;  // [FLAGCX_PTR_HOST|FLAGCX_PTR_CUDA|FLAGCX_PTR_DMABUF]
  int regIsGlobal; // regMr is not tied to a particular comm
  int forceFlush;  // Force a flush on receives
  int speed;       // Port speed in Mbps.
  int port;        // Port number.
  float latency;   // Network latency
  int maxComms;    // Maximum number of comms we can create
  int maxRecvs;    // Maximum number of grouped receives.
  flagcxNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;              // Version number for network offload
  flagcxNetVDeviceProps_v10_t vProps;
  size_t maxP2pBytes;  // Max transfer size for point-to-point operations
  size_t maxCollBytes; // Max transfer size for collective operations
} flagcxNetProperties_v10_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v10_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to flagcx_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then flagcx is requesting device
  // offload for this connection
  flagcxResult_t (*connect)(int dev, flagcxNetCommConfig_v10_t *config,
                            void *handle, void **sendComm,
                            flagcxNetDeviceHandle_v10_t **sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then flagcx is requesting device
  // offload for this connection
  flagcxResult_t (*accept)(void *listenComm, void **recvComm,
                           flagcxNetDeviceHandle_v10_t **recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either flagcx_PTR_HOST or flagcx_PTR_CUDA.
  flagcxResult_t (*regMr)(void *comm, void *data, size_t size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*isend)(void *sendComm, void *data, int size, int tag,
                          void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes,
                          int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with flagcx_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Copy the given mhandle to a dptr in a format usable by this plugin's device
  // code
  flagcxResult_t (*getDeviceMr)(void *comm, void *mhandle, void **dptr_mhandle);

  // Notify the plugin that a recv has completed by the device
  flagcxResult_t (*irecvConsumed)(void *recvComm, int n, void *request);

  // Virtual NIC APIs. makeVDevice will create a virtual NIC given the specified
  // properties, and tell the caller what index this new vNIC exists at
  flagcxResult_t (*makeVDevice)(int *d, flagcxNetVDeviceProps_v10_t *props);
} flagcxNet_v10_t;

typedef struct {
  void *mhandle;
  void *address;
  size_t size;
} flagcxNetSGE_v10_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v10_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to flagcx_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective
  // network.
  flagcxResult_t (*connect)(void *handles[], int nranks, int rank,
                            void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either flagcx_PTR_HOST or
  // flagcx_PTR_CUDA.
  flagcxResult_t (*regMr)(void *collComm, void *data, size_t size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *collComm, void *data, size_t size,
                                int type, uint64_t offset, int fd,
                                void **mhandle);
  flagcxResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would
  // block).
  flagcxResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData,
                               size_t count, flagcxDataType_t dataType,
                               flagcxRedOp_t redOp, void *sendMhandle,
                               void *recvMhandle, void **request);
  flagcxResult_t (*iallgather)(void *collComm, void *sendData, int nRecvParts,
                               flagcxNetSGE_v10_t *recvParts,
                               size_t bytesPerRank, size_t windowOffset,
                               size_t windowBytes, void *sendMhandle,
                               void **request);
  flagcxResult_t (*ireducescatter)(void *collComm, int nSendParts,
                                   flagcxNetSGE_v10_t *sendParts,
                                   void *recvData, size_t bytesPerRank,
                                   size_t windowOffset, size_t windowBytes,
                                   flagcxDataType_t dataType,
                                   flagcxRedOp_t redOp, void *recvMhandle,
                                   void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle,
                           void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  flagcxResult_t (*closeColl)(void *collComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Create a virtual NIC given the specified properties, which can be accessed
  // at device index d
  flagcxResult_t (*makeVDevice)(int *d, flagcxNetVDeviceProps_v10_t *props);
} flagcxCollNet_v10_t;

#define FLAGCX_NET_MAX_DEVS_PER_NIC_V9 4

typedef struct {
  int ndevs;
  int devs[FLAGCX_NET_MAX_DEVS_PER_NIC_V9];
} flagcxNetVDeviceProps_v9_t;

typedef struct {
  char *name;      // Used mostly for logging.
  char *pciPath;   // Path to the PCI device in /sys.
  uint64_t guid;   // Unique identifier for the NIC chip. Important for
                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;  // [FLAGCX_PTR_HOST|FLAGCX_PTR_CUDA|FLAGCX_PTR_DMABUF]
  int regIsGlobal; // regMr is not tied to a particular comm
  int forceFlush;  // Force a flush on receives
  int speed;       // Port speed in Mbps.
  int port;        // Port number.
  float latency;   // Network latency
  int maxComms;    // Maximum number of comms we can create
  int maxRecvs;    // Maximum number of grouped receives.
  flagcxNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;              // Version number for network offload
  flagcxNetVDeviceProps_v9_t vProps;
  size_t maxP2pBytes;  // Max transfer size for point-to-point operations
  size_t maxCollBytes; // Max transfer size for collective operations
} flagcxNetProperties_v9_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v9_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to flagcx_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then flagcx is requesting device
  // offload for this connection
  flagcxResult_t (*connect)(int dev, void *handle, void **sendComm,
                            flagcxNetDeviceHandle_v9_t **sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then flagcx is requesting device
  // offload for this connection
  flagcxResult_t (*accept)(void *listenComm, void **recvComm,
                           flagcxNetDeviceHandle_v9_t **recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either FLAGCX_PTR_HOST or FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *comm, void *data, size_t size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*isend)(void *sendComm, void *data, int size, int tag,
                          void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes,
                          int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Copy the given mhandle to a dptr in a format usable by this plugin's device
  // code
  flagcxResult_t (*getDeviceMr)(void *comm, void *mhandle, void **dptr_mhandle);

  // Notify the plugin that a recv has completed by the device
  flagcxResult_t (*irecvConsumed)(void *recvComm, int n, void *request);

  // Virtual NIC APIs. makeVDevice will create a virtual NIC given the specified
  // properties, and tell the caller what index this new vNIC exists at
  flagcxResult_t (*makeVDevice)(int *d, flagcxNetVDeviceProps_v9_t *props);
} flagcxNet_v9_t;

typedef struct {
  void *mhandle;
  void *address;
  size_t size;
} flagcxNetSGE_v9_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v9_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to flagcx_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective
  // network.
  flagcxResult_t (*connect)(void *handles[], int nranks, int rank,
                            void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either FLAGCX_PTR_HOST or
  // FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *collComm, void *data, size_t size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *collComm, void *data, size_t size,
                                int type, uint64_t offset, int fd,
                                void **mhandle);
  flagcxResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would
  // block).
  flagcxResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData,
                               size_t count, flagcxDataType_t dataType,
                               flagcxRedOp_t redOp, void *sendMhandle,
                               void *recvMhandle, void **request);
  flagcxResult_t (*iallgather)(void *collComm, void *sendData, int nRecvParts,
                               flagcxNetSGE_v9_t *recvParts,
                               size_t bytesPerRank, size_t windowOffset,
                               size_t windowBytes, void *sendMhandle,
                               void **request);
  flagcxResult_t (*ireducescatter)(void *collComm, int nSendParts,
                                   flagcxNetSGE_v9_t *sendParts, void *recvData,
                                   size_t bytesPerRank, size_t windowOffset,
                                   size_t windowBytes,
                                   flagcxDataType_t dataType,
                                   flagcxRedOp_t redOp, void *recvMhandle,
                                   void **request);
  // Perform a flush/fence to make sure all data received with flagcx_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle,
                           void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  flagcxResult_t (*closeColl)(void *collComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Create a virtual NIC given the specified properties, which can be accessed
  // at device index d
  flagcxResult_t (*makeVDevice)(int *d, flagcxNetVDeviceProps_v9_t *props);
} flagcxCollNet_v9_t;

#define FLAGCX_NET_MAX_DEVS_PER_NIC 4
typedef struct {
  int ndevs;
  int devs[FLAGCX_NET_MAX_DEVS_PER_NIC];
} flagcxNetVDeviceProps_t;
typedef struct {
  char *name;      // Used mostly for logging.
  char *pciPath;   // Path to the PCI device in /sys.
  uint64_t guid;   // Unique identifier for the NIC chip. Important for
                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;  // [FLAGCX_PTR_HOST|FLAGCX_PTR_CUDA|FLAGCX_PTR_DMABUF]
  int regIsGlobal; // regMr is not tied to a particular comm
  int speed;       // Port speed in Mbps.
  int port;        // Port number.
  float latency;   // Network latency
  int maxComms;    // Maximum number of comms we can create
  int maxRecvs;    // Maximum number of grouped receives.
  flagcxNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;              // Version number for network offload
} flagcxNetProperties_v8_t;

typedef flagcxNetProperties_v8_t flagcxNetProperties_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v8_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then FLAGCX is requesting device
  // offload for this connection
  flagcxResult_t (*connect)(int dev, void *handle, void **sendComm,
                            flagcxNetDeviceHandle_v8_t **sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then FLAGCX is requesting device
  // offload for this connection
  flagcxResult_t (*accept)(void *listenComm, void **recvComm,
                           flagcxNetDeviceHandle_v8_t **recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either FLAGCX_PTR_HOST or FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *comm, void *data, size_t size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*isend)(void *sendComm, void *data, int size, int tag,
                          void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes,
                          int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Copy the given mhandle to a dptr in a format usable by this plugin's device
  // code
  flagcxResult_t (*getDeviceMr)(void *comm, void *mhandle, void **dptr_mhandle);

  // Notify the plugin that a recv has completed by the device
  flagcxResult_t (*irecvConsumed)(void *recvComm, int n, void *request);
  flagcxResult_t (*getDevFromName)(char *name, int *dev);

} flagcxNet_v8_t;

typedef flagcxNet_v8_t flagcxNet_t;

#define FLAGCX_NET_PLUGIN_SYMBOL flagcxNetPlugin_v8

typedef struct {
  void *mhandle;
  void *address;
  uint32_t size;
} flagcxNetSGE_v8_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v8_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective
  // network.
  flagcxResult_t (*connect)(void *handles[], int nranks, int rank,
                            void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either FLAGCX_PTR_HOST or
  // FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *collComm, void *data, size_t size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *collComm, void *data, size_t size,
                                int type, uint64_t offset, int fd,
                                void **mhandle);
  flagcxResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would
  // block).
  flagcxResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData,
                               int count, flagcxDataType_t dataType,
                               flagcxRedOp_t redOp, void *sendMhandle,
                               void *recvMhandle, void **request);
  flagcxResult_t (*iallgather)(void *collComm, void *sendData, int nRecvParts,
                               flagcxNetSGE_v8_t *recvParts,
                               size_t bytesPerRank, size_t windowOffset,
                               size_t windowBytes, void *sendMhandle,
                               void **request);
  flagcxResult_t (*ireducescatter)(void *collComm, int nSendParts,
                                   flagcxNetSGE_v8_t *sendParts, void *recvData,
                                   size_t bytesPerRank, size_t windowOffset,
                                   size_t windowBytes,
                                   flagcxDataType_t dataType,
                                   flagcxRedOp_t redOp, void *recvMhandle,
                                   void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle,
                           void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  flagcxResult_t (*closeColl)(void *collComm);
  flagcxResult_t (*closeListen)(void *listenComm);
} flagcxCollNet_v8_t;

typedef flagcxCollNet_v8_t flagcxCollNet_t;

#define FLAGCX_COLLNET_PLUGIN_SYMBOL flagcxCollNetPlugin_v8

typedef struct {
  char *name;     // Used mostly for logging.
  char *pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // [FLAGCX_PTR_HOST|FLAGCX_PTR_CUDA|FLAGCX_PTR_DMABUF]
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  float latency;  // Network latency
  int maxComms;   // Maximum number of comms we can create
  int maxRecvs;   // Maximum number of grouped receives.
  flagcxNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;              // Version number for network offload
} flagcxNetProperties_v7_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v7_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then FLAGCX is requesting device
  // offload for this connection
  flagcxResult_t (*connect)(int dev, void *handle, void **sendComm,
                            flagcxNetDeviceHandle_v7_t **sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then FLAGCX is requesting device
  // offload for this connection
  flagcxResult_t (*accept)(void *listenComm, void **recvComm,
                           flagcxNetDeviceHandle_v7_t **recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either FLAGCX_PTR_HOST or FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *comm, void *data, int size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*isend)(void *sendComm, void *data, int size, int tag,
                          void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes,
                          int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);

  // Copy the given mhandle to a dptr in a format usable by this plugin's device
  // code
  flagcxResult_t (*getDeviceMr)(void *comm, void *mhandle, void **dptr_mhandle);

  // Notify the plugin that a recv has completed by the device
  flagcxResult_t (*irecvConsumed)(void *recvComm, int n, void *request);
} flagcxNet_v7_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v7_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective
  // network.
  flagcxResult_t (*connect)(void *handles[], int nranks, int rank,
                            void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either FLAGCX_PTR_HOST or
  // FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *collComm, void *data, int size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *collComm, void *data, size_t size,
                                int type, uint64_t offset, int fd,
                                void **mhandle);
  flagcxResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would
  // block).
  flagcxResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData,
                               int count, flagcxDataType_t dataType,
                               flagcxRedOp_t redOp, void *sendMhandle,
                               void *recvMhandle, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle,
                           void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  flagcxResult_t (*closeColl)(void *collComm);
  flagcxResult_t (*closeListen)(void *listenComm);
} flagcxCollNet_v7_t;

#define FLAGCX_NET_MAX_REQUESTS_V6 8

// v6 struct for backwards compatibility
typedef struct {
  char *name;     // Used mostly for logging.
  char *pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // [FLAGCX_PTR_HOST|FLAGCX_PTR_CUDA|FLAGCX_PTR_DMABUF]
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  float latency;  // Network latency
  int maxComms;   // Maximum number of comms we can create
  int maxRecvs;   // Maximum number of grouped receives.
} flagcxNetProperties_v6_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  flagcxResult_t (*connect)(int dev, void *handle, void **sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  flagcxResult_t (*accept)(void *listenComm, void **recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either FLAGCX_PTR_HOST or FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *comm, void *data, int size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*isend)(void *sendComm, void *data, int size, int tag,
                          void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes,
                          int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);
} flagcxNet_v6_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective
  // network.
  flagcxResult_t (*connect)(void *handles[], int nranks, int rank,
                            void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either FLAGCX_PTR_HOST or
  // FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *collComm, void *data, int size, int type,
                          void **mhandle);
  /* DMA-BUF support */
  flagcxResult_t (*regMrDmaBuf)(void *collComm, void *data, size_t size,
                                int type, uint64_t offset, int fd,
                                void **mhandle);
  flagcxResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would
  // block).
  flagcxResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData,
                               int count, flagcxDataType_t dataType,
                               flagcxRedOp_t redOp, void *sendMhandle,
                               void *recvMhandle, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle,
                           void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  flagcxResult_t (*closeColl)(void *collComm);
  flagcxResult_t (*closeListen)(void *listenComm);
} flagcxCollNet_v6_t;

// v5 struct for backwards compatibility
typedef struct {
  // Name of the network (mainly for logs)
  const char *name;
  // Initialize the network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  flagcxResult_t (*connect)(int dev, void *handle, void **sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  flagcxResult_t (*accept)(void *listenComm, void **recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either FLAGCX_PTR_HOST or FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *comm, void *data, int size, int type,
                          void **mhandle);
  flagcxResult_t (*deregMr)(void *comm, void *mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*isend)(void *sendComm, void *data, int size, int tag,
                          void *mhandle, void **request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  flagcxResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes,
                          int *tags, void **mhandles, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *sizes);
  // Close and free send/recv comm objects
  flagcxResult_t (*closeSend)(void *sendComm);
  flagcxResult_t (*closeRecv)(void *recvComm);
  flagcxResult_t (*closeListen)(void *listenComm);
} flagcxNet_v5_t;

// v5 struct for backwards compatibility
typedef struct {
  // Name of the collective network (mainly for logs)
  const char *name;
  // Initialize the collective network.
  flagcxResult_t (*init)(flagcxDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  flagcxResult_t (*devices)(int *ndev);
  // Get various device properties.
  flagcxResult_t (*getProperties)(int dev, flagcxNetProperties_v6_t *props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to FLAGCX_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  flagcxResult_t (*listen)(int dev, void *handle, void **listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective
  // network.
  flagcxResult_t (*connect)(void *handles[], int nranks, int rank,
                            void *listenComm, void **collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  flagcxResult_t (*reduceSupport)(flagcxDataType_t dataType,
                                  flagcxRedOp_t redOp, int *supported);
  // Register/Deregister memory. Type is either FLAGCX_PTR_HOST or
  // FLAGCX_PTR_CUDA.
  flagcxResult_t (*regMr)(void *collComm, void *data, int size, int type,
                          void **mhandle);
  flagcxResult_t (*deregMr)(void *collComm, void *mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would
  // block).
  flagcxResult_t (*iallreduce)(void *collComm, void *sendData, void *recvData,
                               int count, flagcxDataType_t dataType,
                               flagcxRedOp_t redOp, void *sendMhandle,
                               void *recvMhandle, void **request);
  // Perform a flush/fence to make sure all data received with FLAGCX_PTR_CUDA
  // is visible to the GPU
  flagcxResult_t (*iflush)(void *collComm, void *data, int size, void *mhandle,
                           void **request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  flagcxResult_t (*test)(void *request, int *done, int *size);
  // Close and free collective comm objects
  flagcxResult_t (*closeColl)(void *collComm);
  flagcxResult_t (*closeListen)(void *listenComm);
} flagcxCollNet_v5_t;

#endif // end include guard
