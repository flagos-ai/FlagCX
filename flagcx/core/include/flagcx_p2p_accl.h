/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX P2P engine — ACCL (accl::barex) transport (internal header).
 *
 * The public flagcx_p2p.h API stays unchanged; flagcx_p2p.cc routes to
 * this transport when the engine is created with FLAGCX_P2P_TRANSPORT=accl.
 * Every internal engine/conn struct starts with a uint32_t kind tag so the
 * shared entry points can tell the transports apart.
 *
 * Built only when USE_ACCL_BAREX=1 (links libaccl_barex, required for
 * memory registration and RDMA on PPU + vsolar). Otherwise the stubs
 * below keep the dispatch sites compiling and accl engine creation fails.
 ************************************************************************/

#ifndef FLAGCX_P2P_ACCL_H_
#define FLAGCX_P2P_ACCL_H_

#include "flagcx_p2p.h"

enum FlagcxP2pTransportKind : uint32_t {
  FLAGCX_P2P_KIND_IBRC = 0x1B2C0001u,
  FLAGCX_P2P_KIND_ACCL = 0xACC10001u,
};

/* Engine and conn handles are opaque to callers; both transports place
   the kind tag as the first member, so routing only needs this peek. */
static inline bool flagcxP2pIsAccl(const void *handle) {
  return handle != nullptr &&
         *reinterpret_cast<const uint32_t *>(handle) == FLAGCX_P2P_KIND_ACCL;
}

/* Appends one message to the process-wide notification list owned by
   flagcx_p2p.cc (flagcxP2pEngineGetNotifs is transport-agnostic). */
void flagcxP2pNotifyAppend(const FlagcxP2pNotifyMsg &msg);

#ifdef USE_ACCL_BAREX

FlagcxP2pEngine *flagcxAcclEngineCreate();
void flagcxAcclEngineDestroy(FlagcxP2pEngine *engine);
void flagcxAcclEngineStopAccept(FlagcxP2pEngine *engine);

FlagcxP2pConn *flagcxAcclEngineConnect(FlagcxP2pEngine *engine,
                                       const char *ipAddr, int remoteGpuIdx,
                                       int remotePort, bool sameProcess);
FlagcxP2pConn *flagcxAcclEngineAccept(FlagcxP2pEngine *engine, char *ipAddrBuf,
                                      size_t ipAddrBufLen, int *remoteGpuIdx);
void flagcxAcclEngineConnDestroy(FlagcxP2pConn *conn);
bool flagcxAcclEngineConnIsLocal(FlagcxP2pConn *conn);

int flagcxAcclEngineReg(FlagcxP2pEngine *engine, uintptr_t data, size_t size,
                        FlagcxP2pMr &mrId);
void flagcxAcclEngineMrDestroy(FlagcxP2pEngine *engine, FlagcxP2pMr mr);
int flagcxAcclEnginePrepareDesc(FlagcxP2pEngine *engine, FlagcxP2pMr mr,
                                const void *data, size_t size, char *descBuf);
int flagcxAcclEngineMakeDesc(FlagcxP2pConn *conn, uint64_t remoteVa,
                             uint32_t size, FlagcxP2pRdmaDesc *desc);

int flagcxAcclEngineRead(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                         size_t size, FlagcxP2pRdmaDesc desc,
                         uint64_t *transferId);
int flagcxAcclEngineWrite(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                          size_t size, FlagcxP2pRdmaDesc desc,
                          uint64_t *transferId);
int flagcxAcclEngineReadVector(FlagcxP2pConn *conn,
                               const std::vector<FlagcxP2pMr> &mrIds,
                               const std::vector<void *> &dstVec,
                               const std::vector<size_t> &sizeVec,
                               const std::vector<FlagcxP2pRdmaDesc> &descs,
                               int numIovs, uint64_t *transferId);
int flagcxAcclEngineWriteVector(FlagcxP2pConn *conn,
                                const std::vector<FlagcxP2pMr> &mrIds,
                                const std::vector<void *> &srcVec,
                                const std::vector<size_t> &sizeVec,
                                const std::vector<FlagcxP2pRdmaDesc> &descs,
                                int numIovs, uint64_t *transferId);
int flagcxAcclEngineWriteVectorSync(
    FlagcxP2pConn *conn, const std::vector<FlagcxP2pMr> &mrIds,
    const std::vector<void *> &srcVec, const std::vector<size_t> &sizeVec,
    const std::vector<FlagcxP2pRdmaDesc> &descs);
bool flagcxAcclEngineXferStatus(FlagcxP2pConn *conn, uint64_t transferId);

int flagcxAcclEngineGetMetadata(FlagcxP2pEngine *engine, char **metadataStr);
int flagcxAcclEngineGetRpcPort(FlagcxP2pEngine *engine);
int flagcxAcclEngineStartRpcServer(FlagcxP2pEngine *engine);
FlagcxP2pConn *flagcxAcclEngineGetConn(FlagcxP2pEngine *engine,
                                       const char *session);

int flagcxAcclEngineSendNotif(FlagcxP2pConn *conn,
                              FlagcxP2pNotifyMsg *notifyMsg);
int flagcxAcclEngineGetIpcInfo(FlagcxP2pEngine *engine, uintptr_t addr,
                               char *ipcBuf, bool *hasIpc);

#else /* !USE_ACCL_BAREX — stubs so dispatch sites compile unchanged. */

static inline FlagcxP2pEngine *flagcxAcclEngineCreate() { return nullptr; }
static inline void flagcxAcclEngineDestroy(FlagcxP2pEngine *) {}
static inline void flagcxAcclEngineStopAccept(FlagcxP2pEngine *) {}
static inline FlagcxP2pConn *
flagcxAcclEngineConnect(FlagcxP2pEngine *, const char *, int, int, bool) {
  return nullptr;
}
static inline FlagcxP2pConn *flagcxAcclEngineAccept(FlagcxP2pEngine *, char *,
                                                    size_t, int *) {
  return nullptr;
}
static inline void flagcxAcclEngineConnDestroy(FlagcxP2pConn *) {}
static inline bool flagcxAcclEngineConnIsLocal(FlagcxP2pConn *) {
  return false;
}
static inline int flagcxAcclEngineReg(FlagcxP2pEngine *, uintptr_t, size_t,
                                      FlagcxP2pMr &) {
  return -1;
}
static inline void flagcxAcclEngineMrDestroy(FlagcxP2pEngine *, FlagcxP2pMr) {}
static inline int flagcxAcclEnginePrepareDesc(FlagcxP2pEngine *, FlagcxP2pMr,
                                              const void *, size_t, char *) {
  return -1;
}
static inline int flagcxAcclEngineMakeDesc(FlagcxP2pConn *, uint64_t, uint32_t,
                                           FlagcxP2pRdmaDesc *) {
  return -1;
}
static inline int flagcxAcclEngineRead(FlagcxP2pConn *, FlagcxP2pMr,
                                       const void *, size_t, FlagcxP2pRdmaDesc,
                                       uint64_t *) {
  return -1;
}
static inline int flagcxAcclEngineWrite(FlagcxP2pConn *, FlagcxP2pMr,
                                        const void *, size_t, FlagcxP2pRdmaDesc,
                                        uint64_t *) {
  return -1;
}
static inline int flagcxAcclEngineReadVector(
    FlagcxP2pConn *, const std::vector<FlagcxP2pMr> &,
    const std::vector<void *> &, const std::vector<size_t> &,
    const std::vector<FlagcxP2pRdmaDesc> &, int, uint64_t *) {
  return -1;
}
static inline int flagcxAcclEngineWriteVector(
    FlagcxP2pConn *, const std::vector<FlagcxP2pMr> &,
    const std::vector<void *> &, const std::vector<size_t> &,
    const std::vector<FlagcxP2pRdmaDesc> &, int, uint64_t *) {
  return -1;
}
static inline int flagcxAcclEngineWriteVectorSync(
    FlagcxP2pConn *, const std::vector<FlagcxP2pMr> &,
    const std::vector<void *> &, const std::vector<size_t> &,
    const std::vector<FlagcxP2pRdmaDesc> &) {
  return -1;
}
static inline bool flagcxAcclEngineXferStatus(FlagcxP2pConn *, uint64_t) {
  return true;
}
static inline int flagcxAcclEngineGetMetadata(FlagcxP2pEngine *, char **) {
  return -1;
}
static inline int flagcxAcclEngineGetRpcPort(FlagcxP2pEngine *) { return -1; }
static inline int flagcxAcclEngineStartRpcServer(FlagcxP2pEngine *) {
  return -1;
}
static inline FlagcxP2pConn *flagcxAcclEngineGetConn(FlagcxP2pEngine *,
                                                     const char *) {
  return nullptr;
}
static inline int flagcxAcclEngineSendNotif(FlagcxP2pConn *,
                                            FlagcxP2pNotifyMsg *) {
  return -1;
}
static inline int flagcxAcclEngineGetIpcInfo(FlagcxP2pEngine *, uintptr_t,
                                             char *, bool *hasIpc) {
  if (hasIpc)
    *hasIpc = false;
  return 0;
}

#endif /* USE_ACCL_BAREX */

#endif /* FLAGCX_P2P_ACCL_H_ */
