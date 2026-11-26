#ifndef FLAGCX_REGPOOL_H
#define FLAGCX_REGPOOL_H

#include "check.h"
#include "device.h"
#include "flagcx.h"
#include "net.h"
#include "register.h"
#include <map>
#include <memory>
#include <tuple>
#include <unistd.h>
#include <unordered_map>

class flagcxRegPool {
public:
  static constexpr uintptr_t GLOBAL_POOL_KEY = 0; // nullptr comm maps here

  flagcxRegPool();
  ~flagcxRegPool();

  void getPagedAddr(void *data, size_t length, uintptr_t *beginAddr,
                    uintptr_t *endAddr);
  flagcxResult_t addNetHandle(void *comm, flagcxRegItem *reg, void *handle,
                              struct flagcxProxyConnector *proxyConn);
  flagcxResult_t removeRegItemNetHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t addP2pHandle(void *comm, flagcxRegItem *reg,
                              struct flagcxIpcRegInfo *handle,
                              struct flagcxProxyConnector *proxyConn);
  struct flagcxIpcRegInfo *findP2pHandle(void *comm, int peerRank,
                                         void *allocationBase);
  flagcxResult_t removeRegItemP2pHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t removeAllP2pHandles(void *comm);
  flagcxResult_t removeAllNetHandles(void *comm);
  flagcxResult_t registerBuffer(void *comm, void *data, size_t length);
  flagcxResult_t deregisterBuffer(void *comm, void *handle);
  std::unordered_map<uintptr_t, std::unordered_map<uintptr_t, flagcxRegItem *>>
      &getGlobalMap();
  flagcxRegItem *getItem(const void *comm, void *data);
  void dump();

private:
  void mapRegItemPages(uintptr_t commKey, flagcxRegItem *reg);
  std::unordered_map<uintptr_t, std::unordered_map<uintptr_t, flagcxRegItem *>>
      regMap; // <commPtr, <pageBasePtr, regItemPtr>>
  std::unordered_map<
      uintptr_t, std::unordered_map<uintptr_t, std::unique_ptr<flagcxRegItem>>>
      regPool; // <commPtr, <beginAddr, regItem>> (only GLOBAL_POOL_KEY owns
               // data)
  using P2pIpcCacheKey = std::tuple<uintptr_t, int, uintptr_t>;
  std::map<P2pIpcCacheKey, struct flagcxIpcRegInfo *> p2pIpcCache;
  uintptr_t pageSize;
};

extern flagcxRegPool globalRegPool;

#endif // FLAGCX_REGPOOL_H
