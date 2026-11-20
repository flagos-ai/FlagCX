#ifndef FLAGCX_REGPOOL_H
#define FLAGCX_REGPOOL_H

#include "check.h"
#include "device.h"
#include "flagcx.h"
#include "net.h"
#include "register.h"
#include <map>
#include <memory>
#include <pthread.h>
#include <tuple>
#include <unistd.h>
#include <unordered_map>

struct flagcxRegSnapshot {
  // Owns one operation pin until releaseItem; the copied range is immutable.
  flagcxRegItem *item;
  uintptr_t beginAddr, endAddr;
};
struct flagcxRegPoolOps {
  flagcxResult_t (*closeP2p)(flagcxHeteroComm *, flagcxIpcRegInfo *);
  flagcxResult_t (*closeNet)(void *, flagcxProxyConnector *, void *);
  flagcxResult_t (*closeHomo)(void *, void *);
};

class flagcxRegPool {
public:
  static constexpr uintptr_t GLOBAL_POOL_KEY = 0;
  explicit flagcxRegPool(const flagcxRegPoolOps *operations = NULL);
  ~flagcxRegPool();
  void getPagedAddr(void *data, size_t length, uintptr_t *beginAddr,
                    uintptr_t *endAddr);
  flagcxResult_t registerBuffer(void *comm, void *data, size_t length);
  flagcxResult_t registerAndAcquire(void *comm, void *data, size_t length,
                                    flagcxRegSnapshot *snapshot);
  flagcxResult_t acquireItem(const void *comm, void *data,
                             flagcxRegSnapshot *snapshot);
  void releaseItem(flagcxRegSnapshot *snapshot);
  flagcxResult_t acquireHandle(flagcxRegItem *reg, flagcxRegSnapshot *snapshot);
  flagcxResult_t deregisterBuffer(void *comm, void *handle);
  flagcxResult_t canDeregisterWithoutComm(flagcxRegItem *reg);

  flagcxResult_t addNetHandle(void *comm, flagcxRegItem *reg, void *handle,
                              flagcxProxyConnector *proxy);
  flagcxResult_t beginNetHandle(void *comm, flagcxRegItem *reg,
                                flagcxProxyConnector *proxy, void **handle,
                                int *create);
  flagcxResult_t finishNetHandle(void *comm, flagcxRegItem *reg,
                                 flagcxProxyConnector *proxy, void *handle,
                                 flagcxResult_t status);
  flagcxResult_t beginHomoHandle(void *comm, flagcxRegItem *reg, void **handle,
                                 int *create);
  flagcxResult_t finishHomoHandle(void *comm, flagcxRegItem *reg, void *handle,
                                  flagcxResult_t status);
  flagcxResult_t addP2pHandle(void *comm, flagcxRegItem *reg,
                              flagcxIpcRegInfo *handle,
                              flagcxProxyConnector *proxy);
  flagcxResult_t acquireP2pHandle(void *comm, flagcxRegItem *reg, int peer,
                                  void *base, flagcxIpcRegInfo **handle);
  void releaseP2pHandle(flagcxIpcRegInfo *handle);
  flagcxResult_t removeRegItemNetHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t removeRegItemP2pHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t removeHomoHandles(void *comm, flagcxRegItem *reg);
  flagcxResult_t removeAllP2pHandles(void *comm);
  flagcxResult_t removeAllNetHandles(void *comm);
  flagcxResult_t removeAllHomoHandles(void *comm);
  size_t cachedP2pCount(void *comm);
  flagcxResult_t releaseOwnerReferences(void *comm);
  // Borrowed observations for serialized callers/tests. Production data paths
  // use snapshots/pins above and must not dereference these without a lease.
  flagcxRegItem *getItem(const void *comm, void *data);
  flagcxIpcRegInfo *findP2pHandle(void *comm, int peer, void *base);
  void dump();

private:
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  flagcxRegPoolOps ops;
  void mapRegItemPages(uintptr_t commKey, flagcxRegItem *reg);
  flagcxResult_t registerBufferLocked(void *comm, void *data, size_t length);
  flagcxRegItem *getItemLocked(const void *comm, void *data);
  bool knownLocked(flagcxRegItem *reg);
  void collectLocked(flagcxRegItem *reg);
  void removeOwnerPagesLocked(flagcxRegItem *reg, uintptr_t owner);
  flagcxResult_t addP2pHandleLocked(void *comm, flagcxRegItem *reg,
                                    flagcxIpcRegInfo *handle,
                                    flagcxProxyConnector *proxy);
  flagcxResult_t removeHandles(void *comm, flagcxRegItem *reg, int kind);
  flagcxResult_t removeAllHandles(void *comm, int kind);
  std::unordered_map<uintptr_t, std::unordered_map<uintptr_t, flagcxRegItem *>>
      regMap; // <comm, <page base, borrowed record>>; mutex protected
  std::unordered_map<
      uintptr_t, std::unordered_map<uintptr_t, std::unique_ptr<flagcxRegItem>>>
      regPool; // Only GLOBAL_POOL_KEY owns records; mutex protected
  using P2pIpcCacheKey = std::tuple<uintptr_t, int, uintptr_t>;
  std::map<P2pIpcCacheKey, flagcxIpcRegInfo *> p2pIpcCache;
  uintptr_t pageSize;
};

extern flagcxRegPool globalRegPool;

#endif // FLAGCX_REGPOOL_H
