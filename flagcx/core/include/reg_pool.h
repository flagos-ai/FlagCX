#ifndef FLAGCX_REGPOOL_H
#define FLAGCX_REGPOOL_H

#include "check.h"
#include "device.h"
#include "flagcx.h"
#include "net.h"
#include "register.h"
#include <map>
#include <memory>
#include <mutex>
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
                              struct flagcxProxyConnector *proxyConn,
                              bool complete = true, bool usable = true);
  flagcxResult_t removeRegItemNetHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t addP2pHandle(void *comm, flagcxRegItem *reg,
                              struct flagcxIpcRegInfo *handle,
                              struct flagcxProxyConnector *proxyConn);
  struct flagcxIpcRegInfo *findP2pHandle(void *comm, int peerRank,
                                         void *allocationBase);
  flagcxResult_t reserveP2pRegistration(void *comm, flagcxRegItem *reg,
                                        struct flagcxIpcRegInfo *info);
  flagcxResult_t removeRegItemP2pHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t removeAllP2pHandles(void *comm);
  flagcxResult_t removeAllNetHandles(void *comm);
  flagcxResult_t registerBuffer(void *comm, void *data, size_t length,
                                flagcxRegItem **pinned = nullptr);
  flagcxResult_t deregisterBuffer(void *comm, void *handle,
                                  void *nativeComm = nullptr);
  flagcxResult_t releaseOwner(void *comm, void *nativeComm = nullptr);
  flagcxRegItem *acquireItem(const void *comm, void *data);
  void releaseItem(flagcxRegItem *item);
  bool isGlobalOnly(void *handle);
  flagcxResult_t reserveNetHandle(flagcxRegItem *item, void *comm,
                                  struct flagcxProxyConnector *connector);
  flagcxResult_t reserveHomoHandle(flagcxRegItem *item, void *comm,
                                   void *nativeComm = nullptr);
  void *getNetHandle(flagcxRegItem *item, void *comm,
                     struct flagcxProxyConnector *connector);
  void *getHomoHandle(flagcxRegItem *item, void *comm);
  flagcxResult_t setHomoHandle(flagcxRegItem *item, void *comm, void *handle);
  flagcxRegItem *getItem(const void *comm, void *data);
  void dump();

private:
  // Pins cover host metadata/RPC submission, not device operation completion.
  std::mutex mutex;
  unsigned int readers = 0;
  bool cleaning = false;
  flagcxRegItem *findItem(const void *comm, void *data);
  bool contains(void *handle);
  flagcxResult_t removeHandles(void *comm, flagcxRegItem *reg, bool p2p);
  flagcxResult_t removeAllHandles(void *comm, bool p2p);
  flagcxResult_t releaseRegistration(void *comm, flagcxRegItem *reg,
                                     void *nativeComm, bool allRefs);
  void eraseItem(flagcxRegItem *reg);
  std::unordered_map<uintptr_t, std::unordered_map<uintptr_t, flagcxRegItem *>>
      regMap; // <commPtr, <pageBasePtr, regItemPtr>>
  std::unordered_map<
      uintptr_t, std::unordered_map<uintptr_t, std::unique_ptr<flagcxRegItem>>>
      regPool; // <commPtr, <initialBeginAddr, regItem>> (only GLOBAL_POOL_KEY
               // owns data)
  using P2pIpcCacheKey = std::tuple<uintptr_t, int, uintptr_t>;
  std::map<P2pIpcCacheKey, struct flagcxIpcRegInfo *> p2pIpcCache;
  uintptr_t pageSize;
};

extern flagcxRegPool globalRegPool;

#endif // FLAGCX_REGPOOL_H
