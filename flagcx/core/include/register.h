#ifndef FLAGCX_REGISTER_H_
#define FLAGCX_REGISTER_H_

#include "core.h"
#include "device.h"
#include <list>

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
};

struct flagcxRegP2pHandle {
  void *handle = NULL;
  struct flagcxProxyConnector *proxyConn = NULL;
};

struct flagcxIpcImpInfo {
  void *rmtRegAddr;
  bool legacyIpcCap;
  uintptr_t offset;
};

struct flagcxPeerRegIpcAddr {
  uintptr_t *devPeerRmtAddrs;
  uintptr_t *hostPeerRmtAddrs;
};

struct flagcxIpcRegInfo {
  int peerRank;
  void *baseAddr;
  struct flagcxProxyConnector *ipcProxyconn;
  struct flagcxIpcImpInfo impInfo;
  bool handleReady;
};

struct flagcxRegItem {
  uintptr_t beginAddr = 0;
  uintptr_t endAddr = 0;
  int refCount = 1;
  std::list<std::pair<flagcxRegNetHandle, flagcxRegP2pHandle>> handles;
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
