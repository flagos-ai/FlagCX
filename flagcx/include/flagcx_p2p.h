/*
 * FlagCX P2P Engine API
 *
 * Point-to-point engine interface for one-sided RDMA operations,
 * designed for integration with transfer frameworks such as NIXL.
 *
 * Mirrors the UCCL P2P engine API (uccl_engine.h) with FlagCX naming
 * conventions.  The RDMA descriptor struct (FlagcxP2pRdmaDesc)
 * replaces UCCL's "FifoItem" to clarify that it is a remote memory
 * descriptor, not a software FIFO slot.
 */

#ifndef FLAGCX_P2P_H_
#define FLAGCX_P2P_H_

#include <cstring>
#include <stddef.h>
#include <stdint.h>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */

#define FLAGCX_P2P_MSG_SIZE      256
#define FLAGCX_P2P_DESC_SIZE     64
#define FLAGCX_P2P_IPC_INFO_SIZE 128

/* ------------------------------------------------------------------ */
/*  Opaque handle types                                               */
/* ------------------------------------------------------------------ */

/* Handle for the FlagCX P2P engine instance. */
typedef struct FlagcxP2pEngine FlagcxP2pEngine;

/* Handle for a connection to a remote peer. */
typedef struct FlagcxP2pConn FlagcxP2pConn;

/* Handle for a registered memory region. */
typedef uint64_t FlagcxP2pMr;

/* ------------------------------------------------------------------ */
/*  RDMA descriptor (replaces UCCL "FifoItem")                        */
/*                                                                    */
/*  A 64-byte descriptor that carries the information the initiator   */
/*  of a one-sided RDMA READ/WRITE needs about the remote buffer:     */
/*    - addr  : remote virtual address                                */
/*    - size  : buffer length                                         */
/*    - rkey  : remote memory-region key                              */
/*  The remaining fields (nmsgs, rid, idx) are reserved for internal  */
/*  bookkeeping by the transport implementation.                      */
/* ------------------------------------------------------------------ */

struct FlagcxP2pRdmaDesc {
    uint64_t addr;        /* Remote buffer virtual address            */
    uint32_t size;        /* Buffer length in bytes                   */
    uint32_t rkey;        /* Remote memory-region key (rkey)          */
    uint32_t nmsgs;       /* Reserved: message count                  */
    uint32_t rid;         /* Reserved: request id                     */
    uint64_t idx;         /* Reserved: index                          */
    char     padding[32]; /* Pad to 64 bytes                          */
};
static_assert(sizeof(FlagcxP2pRdmaDesc) == 64,
              "FlagcxP2pRdmaDesc must be 64 bytes");

/* Serialization helpers (byte-level, endian-neutral on same arch). */

inline void
flagcxP2pSerializeRdmaDesc(const FlagcxP2pRdmaDesc &desc, char *buf) {
    std::memcpy(buf + 0,  &desc.addr,    sizeof(uint64_t));
    std::memcpy(buf + 8,  &desc.size,    sizeof(uint32_t));
    std::memcpy(buf + 12, &desc.rkey,    sizeof(uint32_t));
    std::memcpy(buf + 16, &desc.nmsgs,   sizeof(uint32_t));
    std::memcpy(buf + 20, &desc.rid,     sizeof(uint32_t));
    std::memcpy(buf + 24, &desc.idx,     sizeof(uint64_t));
    std::memcpy(buf + 32, &desc.padding, sizeof(desc.padding));
}

inline void
flagcxP2pDeserializeRdmaDesc(const char *buf, FlagcxP2pRdmaDesc *desc) {
    std::memcpy(&desc->addr,    buf + 0,  sizeof(uint64_t));
    std::memcpy(&desc->size,    buf + 8,  sizeof(uint32_t));
    std::memcpy(&desc->rkey,    buf + 12, sizeof(uint32_t));
    std::memcpy(&desc->nmsgs,   buf + 16, sizeof(uint32_t));
    std::memcpy(&desc->rid,     buf + 20, sizeof(uint32_t));
    std::memcpy(&desc->idx,     buf + 24, sizeof(uint64_t));
    std::memcpy(desc->padding,  buf + 32, sizeof(desc->padding));
}

/* ------------------------------------------------------------------ */
/*  Notification message                                              */
/* ------------------------------------------------------------------ */

struct FlagcxP2pNotifyMsg {
    char name[FLAGCX_P2P_MSG_SIZE]; /* Sender agent name              */
    char msg[FLAGCX_P2P_MSG_SIZE];  /* Payload (serialized)           */
};

struct FlagcxP2pMd {
    FlagcxP2pNotifyMsg notifyData;
};

/* ================================================================== */
/*  Engine lifecycle                                                   */
/* ================================================================== */

/**
 * Create and initialize a P2P engine instance.
 * @return              Pointer to the engine instance, or NULL on failure.
 */
FlagcxP2pEngine *flagcxP2pEngineCreate();

/**
 * Destroy the engine instance and free all resources.
 * @param engine        The engine instance to destroy.
 */
void flagcxP2pEngineDestroy(FlagcxP2pEngine *engine);

/**
 * Stop the accept thread for the engine.
 * @param engine        The engine instance.
 */
void flagcxP2pEngineStopAccept(FlagcxP2pEngine *engine);

/* ================================================================== */
/*  Connection management                                             */
/* ================================================================== */

/**
 * Connect to a remote peer.
 * @param engine        The engine instance.
 * @param ipAddr        IP address of the remote server.
 * @param remoteGpuIdx  CUDA device index of the remote GPU.
 * @param remotePort    Port of the remote server.
 * @param sameProcess   True if the remote peer is in the same process.
 * @return              Connection handle, or NULL on failure.
 */
FlagcxP2pConn *flagcxP2pEngineConnect(FlagcxP2pEngine *engine,
                                      const char *ipAddr,
                                      int remoteGpuIdx,
                                      int remotePort,
                                      bool sameProcess = false);

/**
 * Start the listener/notification thread for a connection.
 * @param conn          Connection handle.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineStartListener(FlagcxP2pConn *conn);

/**
 * Accept an incoming connection (blocking).
 * @param engine        The engine instance.
 * @param ipAddrBuf     Buffer to store the remote peer's IP address.
 * @param ipAddrBufLen  Length of the buffer.
 * @param remoteGpuIdx  Pointer to store the remote GPU index.
 * @return              Connection handle, or NULL on failure.
 */
FlagcxP2pConn *flagcxP2pEngineAccept(FlagcxP2pEngine *engine,
                                     char *ipAddrBuf,
                                     size_t ipAddrBufLen,
                                     int *remoteGpuIdx);

/**
 * Destroy a connection and release its resources.
 * @param conn          Connection handle to destroy.
 */
void flagcxP2pEngineConnDestroy(FlagcxP2pConn *conn);

/**
 * Check if a connection is to a peer on the same node (IPC-eligible).
 * @param conn          Connection handle.
 * @return              True if the connection is intra-node.
 */
bool flagcxP2pEngineConnIsLocal(FlagcxP2pConn *conn);

/* ================================================================== */
/*  Memory registration                                               */
/* ================================================================== */

/**
 * Register a memory region for RDMA operations.
 * @param engine        The engine instance.
 * @param data          Address of the buffer to register.
 * @param size          Size of the buffer in bytes.
 * @param mrId          [out] Memory region handle.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineReg(FlagcxP2pEngine *engine, uintptr_t data,
                       size_t size, FlagcxP2pMr &mrId);

/**
 * Deregister a memory region.
 * @param engine        The engine instance.
 * @param mr            Memory region handle to deregister.
 */
void flagcxP2pEngineMrDestroy(FlagcxP2pEngine *engine, FlagcxP2pMr mr);

/* ================================================================== */
/*  RDMA descriptor helpers                                           */
/* ================================================================== */

/**
 * Pre-compute an RDMA descriptor for a registered memory region.
 * Can be called at registration time (no connection required) to prepare
 * the descriptor that remote peers will use for one-sided operations.
 *
 * @param engine        The engine instance.
 * @param mr            Memory region handle.
 * @param data          Pointer to the data buffer.
 * @param size          Size of the buffer.
 * @param descBuf       [out] Buffer to store the serialized descriptor
 *                      (FLAGCX_P2P_DESC_SIZE bytes).
 * @return              0 on success, -1 on failure.
 */
int flagcxP2pEnginePrepareDesc(FlagcxP2pEngine *engine,
                               FlagcxP2pMr mr,
                               const void *data, size_t size,
                               char *descBuf);

/**
 * Update the remote address and size in an RDMA descriptor.
 * Used by the initiator to retarget a pre-computed descriptor at a
 * specific sub-range of the remote buffer before posting a READ/WRITE.
 *
 * @param desc          Reference to the descriptor to update.
 * @param remoteAddr    New remote address.
 * @param size          New transfer size.
 * @return              0 on success, -1 on failure.
 */
int flagcxP2pEngineUpdateDesc(FlagcxP2pRdmaDesc &desc,
                              uint64_t remoteAddr, uint32_t size);

/* ================================================================== */
/*  One-sided READ (RDMA GET)                                         */
/* ================================================================== */

/**
 * One-sided read of a single buffer (non-blocking).
 * @param conn          Connection handle.
 * @param mr            Local memory region handle.
 * @param data          Local destination address.
 * @param size          Number of bytes to read.
 * @param desc          Remote RDMA descriptor (addr + rkey).
 * @param transferId    [out] Transfer ID for status polling.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineRead(FlagcxP2pConn *conn, FlagcxP2pMr mr,
                        const void *data, size_t size,
                        FlagcxP2pRdmaDesc desc,
                        uint64_t *transferId);

/**
 * One-sided vectored read (non-blocking).
 * Dispatches to RDMA, same-process IPC, or cross-process IPC based on
 * connection type and whether ipcBufs is provided.
 *
 * @param conn          Connection handle.
 * @param mrIds         Vector of local memory region handles.
 * @param dstVec        Vector of local destination addresses.
 * @param sizeVec       Vector of transfer sizes.
 * @param descs         Vector of remote RDMA descriptors.
 * @param numIovs       Number of IO vectors.
 * @param transferId    [out] Transfer ID for status polling.
 * @param ipcBufs       Optional: serialized IPC info for local transfers.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineReadVector(
    FlagcxP2pConn *conn,
    std::vector<FlagcxP2pMr> mrIds,
    std::vector<void *> dstVec,
    std::vector<size_t> sizeVec,
    std::vector<FlagcxP2pRdmaDesc> descs,
    int numIovs,
    uint64_t *transferId,
    std::vector<char *> ipcBufs = {});

/* ================================================================== */
/*  One-sided WRITE (RDMA PUT)                                        */
/* ================================================================== */

/**
 * One-sided write of a single buffer (non-blocking).
 * @param conn          Connection handle.
 * @param mr            Local memory region handle.
 * @param data          Local source address.
 * @param size          Number of bytes to write.
 * @param desc          Remote RDMA descriptor (addr + rkey).
 * @param transferId    [out] Transfer ID for status polling.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineWrite(FlagcxP2pConn *conn, FlagcxP2pMr mr,
                         const void *data, size_t size,
                         FlagcxP2pRdmaDesc desc,
                         uint64_t *transferId);

/**
 * One-sided vectored write (non-blocking).
 * Dispatches to RDMA, same-process IPC, or cross-process IPC based on
 * connection type and whether ipcBufs is provided.
 *
 * @param conn          Connection handle.
 * @param mrIds         Vector of local memory region handles.
 * @param dstVec        Vector of local source addresses.
 * @param sizeVec       Vector of transfer sizes.
 * @param descs         Vector of remote RDMA descriptors.
 * @param numIovs       Number of IO vectors.
 * @param transferId    [out] Transfer ID for status polling.
 * @param ipcBufs       Optional: serialized IPC info for local transfers.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineWriteVector(
    FlagcxP2pConn *conn,
    std::vector<FlagcxP2pMr> mrIds,
    std::vector<void *> dstVec,
    std::vector<size_t> sizeVec,
    std::vector<FlagcxP2pRdmaDesc> descs,
    int numIovs,
    uint64_t *transferId,
    std::vector<char *> ipcBufs = {});

/* ================================================================== */
/*  Two-sided send / recv                                             */
/* ================================================================== */

/**
 * Send data to a peer (non-blocking).
 * @param conn          Connection handle.
 * @param mr            Local memory region handle.
 * @param data          Local source address.
 * @param size          Number of bytes to send.
 * @param transferId    [out] Transfer ID for status polling.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineSend(FlagcxP2pConn *conn, FlagcxP2pMr mr,
                        const void *data, size_t size,
                        uint64_t *transferId);

/**
 * Vectored send (non-blocking).
 * @param conn          Connection handle.
 * @param mrIds         Vector of local memory region handles.
 * @param srcVec        Vector of local source addresses.
 * @param sizeVec       Vector of sizes.
 * @param numIovs       Number of IO vectors.
 * @param transferId    [out] Transfer ID for status polling.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineSendVector(
    FlagcxP2pConn *conn,
    std::vector<FlagcxP2pMr> mrIds,
    std::vector<const void *> srcVec,
    std::vector<size_t> sizeVec,
    int numIovs,
    uint64_t *transferId);

/**
 * Receive data from a peer (blocking).
 * @param conn          Connection handle.
 * @param mr            Local memory region handle.
 * @param data          Local destination buffer.
 * @param maxSize       Maximum number of bytes to receive.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineRecv(FlagcxP2pConn *conn, FlagcxP2pMr mr,
                        void *data, size_t maxSize);

/* ================================================================== */
/*  Transfer status                                                   */
/* ================================================================== */

/**
 * Check the completion status of an asynchronous transfer.
 * @param conn          Connection handle.
 * @param transferId    Transfer ID returned by a read/write/send call.
 * @return              True if the transfer has completed.
 */
bool flagcxP2pEngineXferStatus(FlagcxP2pConn *conn, uint64_t transferId);

/* ================================================================== */
/*  Metadata exchange                                                 */
/* ================================================================== */

/**
 * Get engine metadata for connection establishment.
 * The returned string encodes the engine's listening address in the
 * format "ip:rdma_port?gpu_index?notif_port" and must be freed by the
 * caller.
 *
 * @param engine        The engine instance.
 * @param metadataStr   [out] Pointer to store the metadata string.
 * @return              0 on success, non-zero on failure.
 */
int flagcxP2pEngineGetMetadata(FlagcxP2pEngine *engine, char **metadataStr);

/* ================================================================== */
/*  Notifications (out-of-band completion signaling)                  */
/* ================================================================== */

/**
 * Get all pending notification messages and clear the queue.
 * @return              Vector of notification messages received.
 */
std::vector<FlagcxP2pNotifyMsg> flagcxP2pEngineGetNotifs();

/**
 * Send a notification message to a remote peer.
 * @param conn          Connection handle.
 * @param notifyMsg     Notification message to send.
 * @return              Number of bytes sent, or -1 on failure.
 */
int flagcxP2pEngineSendNotif(FlagcxP2pConn *conn,
                             FlagcxP2pNotifyMsg *notifyMsg);

/* ================================================================== */
/*  IPC helpers (intra-node GPU memory sharing)                       */
/* ================================================================== */

/**
 * Get serialized IPC info for a registered buffer.
 * @param engine        The engine instance.
 * @param addr          Base address of the registered buffer.
 * @param ipcBuf        [out] Buffer (FLAGCX_P2P_IPC_INFO_SIZE bytes).
 * @param hasIpc        [out] True if valid IPC info exists (GPU memory).
 * @return              0 on success, -1 on failure.
 */
int flagcxP2pEngineGetIpcInfo(FlagcxP2pEngine *engine,
                              uintptr_t addr, char *ipcBuf,
                              bool *hasIpc);

/**
 * Update offset and size in a serialized IPC info buffer to point at
 * a sub-range of the registered buffer.
 * @param ipcBuf        IPC info buffer (FLAGCX_P2P_IPC_INFO_SIZE bytes).
 * @param addr          Target address within the registered region.
 * @param baseAddr      Base address of the registered region.
 * @param size          Size of the sub-range.
 * @return              0 on success, -1 on failure.
 */
int flagcxP2pEngineUpdateIpcInfo(char *ipcBuf, uintptr_t addr,
                                 uintptr_t baseAddr, size_t size);

#endif /* FLAGCX_P2P_H_ */
