#ifndef FLAGCX_REGISTER_H_
#define FLAGCX_REGISTER_H_

#include "core.h"
#include "device.h"
#include <cstring>
#include <unordered_map>
#include <vector>

#define FLAGCX_IPC_HANDLE_SIZE 128

typedef union {
  alignas(8) char reserved[FLAGCX_IPC_HANDLE_SIZE];
} flagcxIpcHandleData;

static_assert(sizeof(flagcxIpcHandleData) == FLAGCX_IPC_HANDLE_SIZE,
              "IPC handle storage must keep its fixed wire size");

// The adaptor reports the actual vendor handle length at runtime. Keep wire
// storage fixed-size for bootstrap/proxy messages, but never copy the storage
// capacity from a smaller vendor allocation.
static inline flagcxResult_t flagcxStoreIpcHandle(flagcxIpcHandleData *dst,
                                                  const void *src,
                                                  size_t handleSize) {
  if (dst == nullptr || src == nullptr)
    return flagcxInvalidArgument;
  if (handleSize == 0 || handleSize > sizeof(*dst))
    return flagcxNotSupported;

  memset(dst, 0, sizeof(*dst));
  memcpy(dst, src, handleSize);
  return flagcxSuccess;
}

enum {
  NET_REG_COMPLETE = 0x01,
  NVLS_REG_COMPLETE = 0x02,
  NVLS_REG_POSSIBLE = 0x04,
  NVLS_REG_NO_SUPPORT = 0x08,
  COLLNET_REG_COMPLETE = 0x10,
  IPC_REG_COMPLETE = 0x20
};

struct netRegInfo {
  uintptr_t buffer;
  size_t size;
};

struct flagcxRegNetHandle {
  void *handle = NULL;
  struct flagcxProxyConnector *proxyConn = NULL;
  void *ownerComm = NULL; // comm that registered this handle
};

struct flagcxRegP2pHandle {
  void *handle = NULL;
  struct flagcxProxyConnector *proxyConn = NULL;
  void *ownerComm = NULL; // comm that registered this handle
};

struct flagcxIpcImpInfo {
  void *importedBase;
  bool legacyIpcCap;
};

struct flagcxPeerRegIpcAddr {
  uintptr_t *devPeerRmtAddrs;
  uintptr_t *hostPeerRmtAddrs;
};

struct flagcxIpcRegInfo {
  int peerRank;
  void *allocationBase;
  size_t allocationSize;
  int refCount;
  struct flagcxProxyConnector *ipcProxyconn;
  struct flagcxIpcImpInfo impInfo;
  bool handleReady;
  bool sameProcess; // cached at registration time for safe deregister
};

struct flagcxRegItem {
  uintptr_t beginAddr = 0;
  uintptr_t endAddr = 0;
  int refCount = 1;
  std::vector<std::pair<flagcxRegNetHandle, flagcxRegP2pHandle>> handles;
  std::unordered_map<uintptr_t, void *>
      homoRegHandles; // commKey → backend CCL handle
};

struct flagcxReg {
  // common attributes
  size_t pages;
  int refs;
  uintptr_t addr;
  uint32_t state;
  // net reg
  int nDevs;
  int devs[MAXCHANNELS];
  void **handles;
  // nvls reg
  uintptr_t baseAddr;
  size_t baseSize;
  size_t regSize;
  int dev;
  // collnet reg
  void *collnetHandle;
  struct flagcxProxyConnector *proxyconn;
};

struct flagcxRegCache {
  struct flagcxReg **slots;
  int capacity, population;
  uintptr_t pageSize;
  void *sComms[MAXCHANNELS];
  void *rComms[MAXCHANNELS];
};

flagcxResult_t flagcxRegCleanup(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxRegFind(struct flagcxHeteroComm *comm, const void *data,
                             size_t size, struct flagcxReg **reg);

#endif
