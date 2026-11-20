#include "reg_pool.h"
#include "adaptor.h"
#include "p2p.h"
#include "proxy.h"
#include <cstdlib>

constexpr uintptr_t flagcxRegPool::GLOBAL_POOL_KEY;
static flagcxResult_t closeHomoRegistration(void *comm, void *handle) {
  if (cclAdaptors[flagcxCCLAdaptorDevice]->commDeregister == NULL)
    return flagcxNotSupported;
  return cclAdaptors[flagcxCCLAdaptorDevice]->commDeregister(
      (flagcxInnerComm_t)comm, handle);
}
flagcxRegPool::flagcxRegPool(const flagcxRegPoolOps *operations) {
  pageSize = sysconf(_SC_PAGESIZE);
  ops = operations != NULL ? *operations
                           : flagcxRegPoolOps{flagcxP2pDeregisterBuffer,
                                              flagcxNetDeregisterBuffer,
                                              closeHomoRegistration};
}
flagcxRegPool::~flagcxRegPool() {
  for (auto &pool : regPool)
    for (auto &item : pool.second) {
      auto *owner = item.second->owners;
      while (owner != NULL) {
        auto *next = owner->next;
        free(owner);
        owner = next;
      }
    }
  pthread_mutex_destroy(&mutex);
}
void flagcxRegPool::getPagedAddr(void *data, size_t length, uintptr_t *begin,
                                 uintptr_t *end) {
  *begin = reinterpret_cast<uintptr_t>(data) & -pageSize;
  *end =
      (reinterpret_cast<uintptr_t>(data) + length + pageSize - 1) & -pageSize;
}
bool flagcxRegPool::knownLocked(flagcxRegItem *reg) {
  auto pool = regPool.find(GLOBAL_POOL_KEY);
  if (pool != regPool.end())
    for (const auto &item : pool->second)
      if (item.second.get() == reg)
        return true;
  return false;
}
void flagcxRegPool::collectLocked(flagcxRegItem *reg) {
  if (reg->refCount != 0 || reg->opRefs != 0 || !reg->handles.empty() ||
      !reg->homoRegHandles.empty())
    return;
  for (auto map = regMap.begin(); map != regMap.end();) {
    for (auto page = map->second.begin(); page != map->second.end();)
      if (page->second == reg)
        page = map->second.erase(page);
      else
        ++page;
    if (map->second.empty())
      map = regMap.erase(map);
    else
      ++map;
  }
  while (reg->owners != NULL) {
    auto *owner = reg->owners;
    reg->owners = owner->next;
    free(owner);
  }
  regPool[GLOBAL_POOL_KEY].erase(reg->beginAddr);
}
flagcxResult_t flagcxRegPool::registerBuffer(void *comm, void *data,
                                             size_t size) {
  pthread_mutex_lock(&mutex);
  flagcxResult_t result = registerBufferLocked(comm, data, size);
  pthread_mutex_unlock(&mutex);
  return result;
}
flagcxResult_t flagcxRegPool::registerAndAcquire(void *comm, void *data,
                                                 size_t size,
                                                 flagcxRegSnapshot *snapshot) {
  if (snapshot == NULL)
    return flagcxInvalidArgument;
  *snapshot = {};
  pthread_mutex_lock(&mutex);
  flagcxResult_t result = registerBufferLocked(comm, data, size);
  flagcxRegItem *reg =
      result == flagcxSuccess ? getItemLocked(comm, data) : NULL;
  if (reg != NULL) {
    reg->opRefs++;
    *snapshot = {reg, reg->beginAddr, reg->endAddr};
  }
  pthread_mutex_unlock(&mutex);
  return result;
}
flagcxResult_t flagcxRegPool::acquireItem(const void *comm, void *data,
                                          flagcxRegSnapshot *snapshot) {
  if (snapshot == NULL)
    return flagcxInvalidArgument;
  *snapshot = {};
  pthread_mutex_lock(&mutex);
  flagcxRegItem *reg = getItemLocked(comm, data);
  if (reg != NULL && reg->refCount > 0 && !reg->deregistering) {
    reg->opRefs++;
    *snapshot = {reg, reg->beginAddr, reg->endAddr};
  }
  pthread_mutex_unlock(&mutex);
  return flagcxSuccess;
}
flagcxResult_t flagcxRegPool::acquireHandle(flagcxRegItem *reg,
                                            flagcxRegSnapshot *snapshot) {
  if (snapshot == NULL)
    return flagcxInvalidArgument;
  *snapshot = {};
  pthread_mutex_lock(&mutex);
  flagcxResult_t result = knownLocked(reg) && reg->refCount > 0
                              ? flagcxSuccess
                              : flagcxInvalidArgument;
  if (result == flagcxSuccess && reg->deregistering)
    result = flagcxInProgress;
  if (result == flagcxSuccess) {
    reg->opRefs++;
    *snapshot = {reg, reg->beginAddr, reg->endAddr};
  }
  pthread_mutex_unlock(&mutex);
  return result;
}
void flagcxRegPool::releaseItem(flagcxRegSnapshot *snapshot) {
  if (snapshot == NULL || snapshot->item == NULL)
    return;
  pthread_mutex_lock(&mutex);
  flagcxRegItem *reg = snapshot->item;
  reg->opRefs--;
  collectLocked(reg);
  pthread_mutex_unlock(&mutex);
  snapshot->item = NULL;
}
flagcxRegItem *flagcxRegPool::getItem(const void *comm, void *data) {
  pthread_mutex_lock(&mutex);
  auto *reg = getItemLocked(comm, data);
  pthread_mutex_unlock(&mutex);
  return reg;
}
flagcxResult_t flagcxRegPool::canDeregisterWithoutComm(flagcxRegItem *reg) {
  pthread_mutex_lock(&mutex);
  flagcxResult_t result =
      knownLocked(reg) ? flagcxSuccess : flagcxInvalidArgument;
  if (result == flagcxSuccess) {
    if (!reg->handles.empty() || !reg->homoRegHandles.empty())
      result = flagcxInvalidArgument;
    for (const auto &map : regMap)
      if (map.first != GLOBAL_POOL_KEY)
        for (const auto &entry : map.second)
          if (entry.second == reg)
            result = flagcxInvalidArgument;
  }
  pthread_mutex_unlock(&mutex);
  return result;
}
void flagcxRegPool::mapRegItemPages(uintptr_t commKey, flagcxRegItem *reg) {
  if (reg == nullptr) {
    return;
  }
  auto &regCommMap = regMap[commKey];
  for (uintptr_t addr = reg->beginAddr; addr < reg->endAddr; addr += pageSize) {
    regCommMap[addr] = reg;
  }
}

flagcxResult_t flagcxRegPool::registerBufferLocked(void *comm, void *data,
                                                   size_t length) {
  if (data == nullptr || length == 0)
    return flagcxSuccess;

  uintptr_t commKey =
      comm ? reinterpret_cast<uintptr_t>(comm) : GLOBAL_POOL_KEY;
  uintptr_t beginAddr, endAddr;
  getPagedAddr(data, length, &beginAddr, &endAddr);

  // Check if ANY page in [beginAddr, endAddr) already belongs to an existing
  // item via regMap. This handles partial overlaps where the new buffer starts
  // on an unmapped page but overlaps an existing registration.
  flagcxRegItem *existing = nullptr;
  auto globalMapIt = regMap.find(GLOBAL_POOL_KEY);
  if (globalMapIt != regMap.end()) {
    for (uintptr_t addr = beginAddr; addr < endAddr; addr += pageSize) {
      auto it = globalMapIt->second.find(addr);
      if (it != globalMapIt->second.end()) {
        existing = it->second;
        break;
      }
    }
  }

  if (existing) {
    if (existing->deregistering || existing->refCount == 0)
      return flagcxInProgress;
    flagcxRegOwnerRef *owner = existing->owners;
    while (owner != NULL && owner->owner != commKey)
      owner = owner->next;
    if (owner != NULL && owner->closing)
      return flagcxInProgress;
    if (owner == NULL) {
      owner = (flagcxRegOwnerRef *)calloc(1, sizeof(*owner));
      if (owner == NULL)
        return flagcxSystemError;
      owner->owner = commKey;
      owner->next = existing->owners;
      existing->owners = owner;
    }
    owner->refs++;
    existing->refCount++;
    // Extend backward if new buffer starts before existing range
    if (beginAddr < existing->beginAddr) {
      uintptr_t oldBegin = existing->beginAddr;
      existing->beginAddr = beginAddr;
      for (uintptr_t addr = beginAddr; addr < oldBegin; addr += pageSize) {
        regMap[GLOBAL_POOL_KEY][addr] = existing;
      }
      // Update regPool key to match new beginAddr
      auto &globalPool = regPool[GLOBAL_POOL_KEY];
      auto nodeIt = globalPool.find(oldBegin);
      if (nodeIt == globalPool.end()) {
        WARN("registerBuffer: regPool key mismatch for oldBegin");
        return flagcxInternalError;
      }
      std::unique_ptr<flagcxRegItem> tmp = std::move(nodeIt->second);
      globalPool.erase(nodeIt);
      if (globalPool.count(beginAddr)) {
        WARN("registerBuffer: unexpected duplicate key at beginAddr");
        return flagcxInternalError;
      }
      globalPool.emplace(beginAddr, std::move(tmp));
    }
    // Extend forward if new buffer goes beyond existing range
    if (endAddr > existing->endAddr) {
      uintptr_t oldEnd = existing->endAddr;
      existing->endAddr = endAddr;
      for (uintptr_t addr = oldEnd; addr < endAddr; addr += pageSize) {
        regMap[GLOBAL_POOL_KEY][addr] = existing;
      }
    }
    // Ensure comm-specific mapping covers full range
    if (comm != nullptr) {
      mapRegItemPages(commKey, existing);
    }
    return flagcxSuccess;
  }

  // Not found: create new item in global pool
  auto &globalPool = regPool[GLOBAL_POOL_KEY];
  std::unique_ptr<flagcxRegItem> reg(new flagcxRegItem());
  reg->beginAddr = beginAddr;
  reg->endAddr = endAddr;
  reg->refCount = 1;
  reg->owners = (flagcxRegOwnerRef *)calloc(1, sizeof(flagcxRegOwnerRef));
  if (reg->owners == NULL)
    return flagcxSystemError;
  reg->owners->owner = commKey;
  reg->owners->refs = 1;
  auto result = globalPool.emplace(beginAddr, std::move(reg));
  flagcxRegItem *regPtr = result.first->second.get();

  // Map pages in global regMap
  mapRegItemPages(GLOBAL_POOL_KEY, regPtr);

  // If comm is non-null, also map pages in comm-specific regMap
  if (comm != nullptr) {
    mapRegItemPages(commKey, regPtr);
  }

  return flagcxSuccess;
}

flagcxRegItem *flagcxRegPool::getItemLocked(const void *comm, void *data) {
  uintptr_t beginAddr, endAddr;
  getPagedAddr(data, 0, &beginAddr, &endAddr);

  // If comm is non-null, check comm-specific regMap first
  if (comm != nullptr) {
    uintptr_t commKey = reinterpret_cast<uintptr_t>(comm);
    auto mapIt = regMap.find(commKey);
    if (mapIt != regMap.end()) {
      auto it = mapIt->second.find(beginAddr);
      if (it != mapIt->second.end()) {
        return it->second;
      }
    }
  }

  // Fall through to global pool
  auto globalMapIt = regMap.find(GLOBAL_POOL_KEY);
  if (globalMapIt != regMap.end()) {
    auto it = globalMapIt->second.find(beginAddr);
    if (it != globalMapIt->second.end()) {
      return it->second;
    }
  }

  return nullptr;
}

flagcxResult_t flagcxRegPool::addNetHandle(void *comm, flagcxRegItem *reg,
                                           void *handle,
                                           flagcxProxyConnector *proxy) {
  if (comm == NULL || reg == NULL)
    return flagcxSuccess;
  pthread_mutex_lock(&mutex);
  flagcxResult_t result = flagcxSuccess;
  bool found = false;
  for (auto &entry : reg->handles)
    if (entry.first.ownerComm == comm && entry.first.proxyConn == proxy) {
      found = true;
      if (entry.first.closeState != flagcxRegLive)
        result = flagcxInProgress;
      else
        entry.first.handle = handle;
      break;
    }
  if (!found)
    reg->handles.push_back({{handle, proxy, comm}, {}});
  pthread_mutex_unlock(&mutex);
  return result;
}
flagcxResult_t flagcxRegPool::beginNetHandle(void *comm, flagcxRegItem *reg,
                                             flagcxProxyConnector *proxy,
                                             void **handle, int *create) {
  *handle = NULL;
  *create = 0;
  pthread_mutex_lock(&mutex);
  for (const auto &entry : reg->handles) {
    if (entry.first.ownerComm == comm && entry.first.proxyConn == proxy) {
      flagcxResult_t result = entry.first.closeState == flagcxRegLive
                                  ? flagcxSuccess
                                  : flagcxInProgress;
      if (result == flagcxSuccess)
        *handle = entry.first.handle;
      pthread_mutex_unlock(&mutex);
      return result;
    }
  }
  flagcxRegNetHandle pending = {NULL, proxy, comm};
  pending.closeState = flagcxRegClosing;
  reg->handles.push_back({pending, {}});
  *create = 1;
  pthread_mutex_unlock(&mutex);
  return flagcxSuccess;
}
flagcxResult_t flagcxRegPool::finishNetHandle(void *comm, flagcxRegItem *reg,
                                              flagcxProxyConnector *proxy,
                                              void *handle,
                                              flagcxResult_t status) {
  pthread_mutex_lock(&mutex);
  for (size_t i = 0; i < reg->handles.size(); i++) {
    auto &entry = reg->handles[i].first;
    if (entry.ownerComm != comm || entry.proxyConn != proxy)
      continue;
    if (handle == NULL) {
      entry = {};
      if (reg->handles[i].second.handle == NULL) {
        reg->handles[i] = reg->handles.back();
        reg->handles.pop_back();
      }
    } else {
      entry.handle = handle;
      entry.closeState =
          status == flagcxSuccess ? flagcxRegLive : flagcxRegCloseFailed;
    }
    pthread_mutex_unlock(&mutex);
    return status;
  }
  pthread_mutex_unlock(&mutex);
  return flagcxInvalidUsage;
}
flagcxResult_t flagcxRegPool::beginHomoHandle(void *comm, flagcxRegItem *reg,
                                              void **handle, int *create) {
  *handle = NULL;
  *create = 0;
  pthread_mutex_lock(&mutex);
  auto it = reg->homoRegHandles.find((uintptr_t)comm);
  if (it != reg->homoRegHandles.end()) {
    flagcxResult_t result = it->second.closeState == flagcxRegLive
                                ? flagcxSuccess
                                : flagcxInProgress;
    if (result == flagcxSuccess)
      *handle = it->second.handle;
    pthread_mutex_unlock(&mutex);
    return result;
  }
  flagcxRegNetHandle entry = {NULL, NULL, comm};
  entry.closeState = flagcxRegClosing;
  reg->homoRegHandles.emplace((uintptr_t)comm, entry);
  *create = 1;
  pthread_mutex_unlock(&mutex);
  return flagcxSuccess;
}
flagcxResult_t flagcxRegPool::finishHomoHandle(void *comm, flagcxRegItem *reg,
                                               void *handle,
                                               flagcxResult_t status) {
  pthread_mutex_lock(&mutex);
  auto it = reg->homoRegHandles.find((uintptr_t)comm);
  if (it == reg->homoRegHandles.end()) {
    pthread_mutex_unlock(&mutex);
    return flagcxInvalidUsage;
  }
  if (handle == NULL)
    reg->homoRegHandles.erase(it);
  else {
    it->second.handle = handle;
    it->second.closeState =
        status == flagcxSuccess ? flagcxRegLive : flagcxRegCloseFailed;
  }
  pthread_mutex_unlock(&mutex);
  return status;
}
flagcxResult_t flagcxRegPool::addP2pHandleLocked(void *comm, flagcxRegItem *reg,
                                                 flagcxIpcRegInfo *info,
                                                 flagcxProxyConnector *proxy) {
  if (!comm || !reg || !info)
    return flagcxSuccess;
  if (info->closeState != flagcxRegLive)
    return flagcxInProgress;
  for (const auto &entry : reg->handles)
    if (entry.second.handle == info && entry.second.ownerComm == comm)
      return flagcxSuccess;
  P2pIpcCacheKey key{(uintptr_t)comm, info->peerRank,
                     (uintptr_t)info->allocationBase};
  auto it = p2pIpcCache.find(key);
  if (it == p2pIpcCache.end()) {
    info->refCount = 1;
    p2pIpcCache.emplace(key, info);
  } else if (it->second == info)
    info->refCount++;
  else
    return flagcxInvalidUsage;
  reg->handles.push_back({{}, {info, proxy, comm}});
  return flagcxSuccess;
}
flagcxResult_t flagcxRegPool::addP2pHandle(void *comm, flagcxRegItem *reg,
                                           flagcxIpcRegInfo *info,
                                           flagcxProxyConnector *proxy) {
  pthread_mutex_lock(&mutex);
  flagcxResult_t result = addP2pHandleLocked(comm, reg, info, proxy);
  pthread_mutex_unlock(&mutex);
  return result;
}
flagcxIpcRegInfo *flagcxRegPool::findP2pHandle(void *comm, int peer,
                                               void *base) {
  pthread_mutex_lock(&mutex);
  auto it = p2pIpcCache.find({(uintptr_t)comm, peer, (uintptr_t)base});
  auto *info = it == p2pIpcCache.end() ? NULL : it->second;
  pthread_mutex_unlock(&mutex);
  return info;
}
flagcxResult_t flagcxRegPool::acquireP2pHandle(void *comm, flagcxRegItem *reg,
                                               int peer, void *base,
                                               flagcxIpcRegInfo **output) {
  *output = NULL;
  pthread_mutex_lock(&mutex);
  auto it = p2pIpcCache.find({(uintptr_t)comm, peer, (uintptr_t)base});
  flagcxResult_t result = flagcxSuccess;
  if (it != p2pIpcCache.end()) {
    auto *info = it->second;
    result = addP2pHandleLocked(comm, reg, info, info->ipcProxyconn);
    if (result == flagcxSuccess) {
      info->readers++;
      *output = info;
    }
  }
  pthread_mutex_unlock(&mutex);
  return result;
}
void flagcxRegPool::releaseP2pHandle(flagcxIpcRegInfo *info) {
  if (info == NULL)
    return;
  pthread_mutex_lock(&mutex);
  info->readers--;
  pthread_mutex_unlock(&mutex);
}

// kind: 0=NET, 1=P2P, 2=native CCL. Never retain a vector reference across
// an external close: another rank may compact the shared vector in that time.
flagcxResult_t flagcxRegPool::removeHandles(void *comm, flagcxRegItem *reg,
                                            int kind) {
  if (reg == NULL)
    return flagcxSuccess;
  pthread_mutex_lock(&mutex);
  if (!knownLocked(reg)) {
    pthread_mutex_unlock(&mutex);
    return flagcxInvalidUsage;
  }
  reg->opRefs++;
  pthread_mutex_unlock(&mutex);
  flagcxResult_t result = flagcxSuccess;
  for (;;) {
    void *owner = NULL, *handle = NULL;
    flagcxProxyConnector *proxy = NULL;
    bool found = false, closeNative = true;
    pthread_mutex_lock(&mutex);
    if (kind == 2) {
      for (auto &entry : reg->homoRegHandles) {
        if (comm != NULL && entry.first != (uintptr_t)comm)
          continue;
        found = true;
        owner = (void *)entry.first;
        handle = entry.second.handle;
        if (entry.second.closeState == flagcxRegClosing)
          result = flagcxInProgress;
        else
          entry.second.closeState = flagcxRegClosing;
        break;
      }
    } else {
      for (auto &pair : reg->handles) {
        if (kind == 0) {
          auto &entry = pair.first;
          if (entry.ownerComm == NULL ||
              (comm != NULL && entry.ownerComm != comm))
            continue;
          found = true;
          owner = entry.ownerComm;
          handle = entry.handle;
          proxy = entry.proxyConn;
          if (entry.closeState == flagcxRegClosing)
            result = flagcxInProgress;
          else
            entry.closeState = flagcxRegClosing;
        } else {
          auto &entry = pair.second;
          if (entry.handle == NULL || (comm != NULL && entry.ownerComm != comm))
            continue;
          found = true;
          owner = entry.ownerComm;
          handle = entry.handle;
          proxy = entry.proxyConn;
          auto *info = (flagcxIpcRegInfo *)handle;
          if (info->refCount <= 0)
            result = flagcxInternalError;
          else if (info->closeState == flagcxRegClosing || info->readers > 0)
            result = flagcxInProgress;
          else if (info->refCount > 1) {
            info->refCount--;
            entry = {};
            closeNative = false;
          } else
            info->closeState = flagcxRegClosing;
        }
        break;
      }
    }
    if (found && result == flagcxSuccess && !closeNative) {
      for (size_t i = 0; i < reg->handles.size();) {
        if (reg->handles[i].first.ownerComm == NULL &&
            reg->handles[i].second.handle == NULL) {
          reg->handles[i] = reg->handles.back();
          reg->handles.pop_back();
        } else
          i++;
      }
    }
    pthread_mutex_unlock(&mutex);
    if (!found || result != flagcxSuccess)
      break;
    if (!closeNative)
      continue;
    if (kind == 0)
      result = ops.closeNet != NULL ? ops.closeNet(owner, proxy, handle)
                                    : flagcxNotSupported;
    else if (kind == 1)
      result = ops.closeP2p != NULL ? ops.closeP2p((flagcxHeteroComm *)owner,
                                                   (flagcxIpcRegInfo *)handle)
                                    : flagcxNotSupported;
    else
      result = ops.closeHomo != NULL ? ops.closeHomo(owner, handle)
                                     : flagcxNotSupported;
    pthread_mutex_lock(&mutex);
    if (kind == 2) {
      auto it = reg->homoRegHandles.find((uintptr_t)owner);
      if (result == flagcxSuccess)
        reg->homoRegHandles.erase(it);
      else
        it->second.closeState = flagcxRegCloseFailed;
    } else {
      for (size_t i = 0; i < reg->handles.size(); i++) {
        auto &pair = reg->handles[i];
        bool match = kind == 0 ? pair.first.handle == handle &&
                                     pair.first.ownerComm == owner &&
                                     pair.first.proxyConn == proxy
                               : pair.second.handle == handle &&
                                     pair.second.ownerComm == owner;
        if (!match)
          continue;
        if (result == flagcxSuccess) {
          if (kind == 0)
            pair.first = {};
          else {
            auto *info = (flagcxIpcRegInfo *)handle;
            p2pIpcCache.erase({(uintptr_t)owner, info->peerRank,
                               (uintptr_t)info->allocationBase});
            free(info);
            pair.second = {};
          }
          if (pair.first.ownerComm == NULL && pair.second.handle == NULL) {
            pair = reg->handles.back();
            reg->handles.pop_back();
          }
        } else if (kind == 0)
          pair.first.closeState = flagcxRegCloseFailed;
        else
          ((flagcxIpcRegInfo *)handle)->closeState = flagcxRegCloseFailed;
        break;
      }
    }
    pthread_mutex_unlock(&mutex);
    if (result != flagcxSuccess)
      break;
  }
  flagcxRegSnapshot snapshot = {reg, 0, 0};
  releaseItem(&snapshot);
  return result;
}
flagcxResult_t flagcxRegPool::removeRegItemNetHandles(void *comm,
                                                      flagcxRegItem *reg) {
  return removeHandles(comm, reg, 0);
}
flagcxResult_t flagcxRegPool::removeRegItemP2pHandles(void *comm,
                                                      flagcxRegItem *reg) {
  return removeHandles(comm, reg, 1);
}
flagcxResult_t flagcxRegPool::removeHomoHandles(void *comm,
                                                flagcxRegItem *reg) {
  return removeHandles(comm, reg, 2);
}
flagcxResult_t flagcxRegPool::removeAllHandles(void *comm, int kind) {
  if (comm == NULL)
    return flagcxSuccess;
  pthread_mutex_lock(&mutex);
  auto &pool = regPool[GLOBAL_POOL_KEY];
  size_t count = pool.size();
  auto **items =
      (flagcxRegItem **)calloc(count ? count : 1, sizeof(flagcxRegItem *));
  if (items == NULL) {
    pthread_mutex_unlock(&mutex);
    return flagcxSystemError;
  }
  size_t i = 0;
  for (const auto &item : pool) {
    items[i++] = item.second.get();
    items[i - 1]->opRefs++;
  }
  pthread_mutex_unlock(&mutex);
  flagcxResult_t result = flagcxSuccess;
  for (i = 0; i < count; i++) {
    if (result == flagcxSuccess)
      result = removeHandles(comm, items[i], kind);
    flagcxRegSnapshot snapshot = {items[i], 0, 0};
    releaseItem(&snapshot);
  }
  free(items);
  return result;
}
flagcxResult_t flagcxRegPool::removeAllNetHandles(void *comm) {
  return removeAllHandles(comm, 0);
}
flagcxResult_t flagcxRegPool::removeAllP2pHandles(void *comm) {
  return removeAllHandles(comm, 1);
}
flagcxResult_t flagcxRegPool::removeAllHomoHandles(void *comm) {
  return removeAllHandles(comm, 2);
}
size_t flagcxRegPool::cachedP2pCount(void *comm) {
  pthread_mutex_lock(&mutex);
  size_t count = 0;
  for (const auto &entry : p2pIpcCache)
    if (comm == NULL || std::get<0>(entry.first) == (uintptr_t)comm)
      count++;
  pthread_mutex_unlock(&mutex);
  return count;
}
void flagcxRegPool::removeOwnerPagesLocked(flagcxRegItem *reg,
                                           uintptr_t owner) {
  if (owner == GLOBAL_POOL_KEY)
    return;
  auto map = regMap.find(owner);
  if (map == regMap.end())
    return;
  for (auto it = map->second.begin(); it != map->second.end();)
    if (it->second == reg)
      it = map->second.erase(it);
    else
      ++it;
  if (map->second.empty())
    regMap.erase(map);
}

flagcxResult_t flagcxRegPool::deregisterBuffer(void *comm, void *handle) {
  if (handle == NULL)
    return flagcxSuccess;
  auto *reg = (flagcxRegItem *)handle;
  pthread_mutex_lock(&mutex);
  if (!knownLocked(reg) || reg->refCount <= 0) {
    pthread_mutex_unlock(&mutex);
    return flagcxInvalidUsage;
  }
  auto *owner = reg->owners;
  while (owner != NULL && owner->owner != (uintptr_t)comm)
    owner = owner->next;
  // The pool-only internal interface historically accepts a final release
  // without a comm. The public API checks its stronger no-backend constraint.
  if (owner == NULL && comm == NULL && reg->refCount == 1)
    owner = reg->owners;
  if (owner == NULL) {
    pthread_mutex_unlock(&mutex);
    return flagcxInvalidUsage;
  }
  if (owner->closing || reg->deregistering) {
    pthread_mutex_unlock(&mutex);
    return flagcxInProgress;
  }
  if (owner->refs > 1) {
    owner->refs--;
    reg->refCount--;
    pthread_mutex_unlock(&mutex);
    return flagcxSuccess;
  }
  if (reg->refCount == 1 && reg->opRefs != 0) {
    pthread_mutex_unlock(&mutex);
    return flagcxInProgress;
  }
  owner->closing = 1;
  if (reg->refCount == 1)
    reg->deregistering = 1;
  reg->opRefs++;
  uintptr_t key = owner->owner;
  pthread_mutex_unlock(&mutex);
  flagcxResult_t result = removeHandles(comm, reg, 0);
  if (result == flagcxSuccess)
    result = removeHandles(comm, reg, 1);
  if (result == flagcxSuccess)
    result = removeHandles(comm, reg, 2);
  pthread_mutex_lock(&mutex);
  if (result == flagcxSuccess) {
    reg->refCount--;
    auto **link = &reg->owners;
    while (*link != owner)
      link = &(*link)->next;
    *link = owner->next;
    free(owner);
    removeOwnerPagesLocked(reg, key);
  } else
    owner->closing = 0;
  reg->deregistering = 0;
  reg->opRefs--;
  collectLocked(reg);
  pthread_mutex_unlock(&mutex);
  return result;
}

flagcxResult_t flagcxRegPool::releaseOwnerReferences(void *comm) {
  if (comm == NULL)
    return flagcxSuccess;
  pthread_mutex_lock(&mutex);
  auto &pool = regPool[GLOBAL_POOL_KEY];
  for (auto it = pool.begin(); it != pool.end();) {
    auto *reg = it->second.get();
    ++it;
    auto **link = &reg->owners;
    while (*link != NULL && (*link)->owner != (uintptr_t)comm)
      link = &(*link)->next;
    if (*link == NULL)
      continue;
    auto *owner = *link;
    if (owner->closing) {
      pthread_mutex_unlock(&mutex);
      return flagcxInProgress;
    }
    // No provider operation is allowed from local Destroy. Finalize must
    // already have removed this owner's native, NET and P2P handles.
    if (reg->homoRegHandles.count((uintptr_t)comm)) {
      pthread_mutex_unlock(&mutex);
      return flagcxInvalidUsage;
    }
    for (const auto &pair : reg->handles)
      if (pair.first.ownerComm == comm || pair.second.ownerComm == comm) {
        pthread_mutex_unlock(&mutex);
        return flagcxInvalidUsage;
      }
    reg->refCount -= owner->refs;
    *link = owner->next;
    free(owner);
    removeOwnerPagesLocked(reg, (uintptr_t)comm);
    collectLocked(reg);
  }
  pthread_mutex_unlock(&mutex);
  return flagcxSuccess;
}
void flagcxRegPool::dump() {
  pthread_mutex_lock(&mutex);
  INFO(FLAGCX_REG, "registration pool: records=%zu cachedP2p=%zu",
       regPool[GLOBAL_POOL_KEY].size(), p2pIpcCache.size());
  pthread_mutex_unlock(&mutex);
}
