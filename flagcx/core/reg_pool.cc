#include "reg_pool.h"
#include "p2p.h"
#include "proxy.h"
#include <cstdio>
#include <cstdlib>

flagcxRegPool::flagcxRegPool() { pageSize = sysconf(_SC_PAGESIZE); }

flagcxRegPool::~flagcxRegPool() {
  p2pIpcCache.clear();
  regMap.clear();
  regPool.clear();
}

inline void flagcxRegPool::getPagedAddr(void *data, size_t length,
                                        uintptr_t *beginAddr,
                                        uintptr_t *endAddr) {
  *beginAddr = reinterpret_cast<uintptr_t>(data) & -pageSize;
  *endAddr =
      (reinterpret_cast<uintptr_t>(data) + length + pageSize - 1) & -pageSize;
}

flagcxResult_t
flagcxRegPool::addNetHandle(void *comm, flagcxRegItem *reg, void *handle,
                            struct flagcxProxyConnector *proxyConn) {
  if (reg == nullptr || comm == nullptr) {
    return flagcxSuccess;
  }
  for (auto &handlePair : reg->handles) {
    if (handlePair.first.proxyConn == proxyConn) {
      handlePair.first.handle = handle;
      handlePair.first.ownerComm = comm;
      return flagcxSuccess;
    }
  }
  flagcxRegNetHandle netHandle{handle, proxyConn, comm};
  flagcxRegP2pHandle p2pHandle{nullptr, nullptr, nullptr};
  reg->handles.push_back(std::make_pair(netHandle, p2pHandle));

  return flagcxSuccess;
}

flagcxResult_t
flagcxRegPool::addP2pHandle(void *comm, flagcxRegItem *reg,
                            flagcxIpcRegInfo *handle,
                            struct flagcxProxyConnector *proxyConn) {
  if (reg == nullptr || comm == nullptr || handle == nullptr) {
    return flagcxSuccess;
  }
  for (auto &handlePair : reg->handles) {
    if (handlePair.second.handle == handle &&
        handlePair.second.ownerComm == comm)
      return flagcxSuccess;
  }

  P2pIpcCacheKey key{reinterpret_cast<uintptr_t>(comm), handle->peerRank,
                     reinterpret_cast<uintptr_t>(handle->allocationBase)};
  auto cached = p2pIpcCache.find(key);
  if (cached == p2pIpcCache.end()) {
    handle->refCount = 1;
    p2pIpcCache.emplace(key, handle);
  } else if (cached->second == handle) {
    handle->refCount++;
  } else {
    return flagcxInvalidUsage;
  }
  flagcxRegNetHandle netHandle{nullptr, nullptr, nullptr};
  flagcxRegP2pHandle p2pHandle{handle, proxyConn, comm};
  reg->handles.push_back(std::make_pair(netHandle, p2pHandle));

  return flagcxSuccess;
}

flagcxIpcRegInfo *flagcxRegPool::findP2pHandle(void *comm, int peerRank,
                                               void *allocationBase) {
  if (comm == nullptr || allocationBase == nullptr)
    return nullptr;
  P2pIpcCacheKey key{reinterpret_cast<uintptr_t>(comm), peerRank,
                     reinterpret_cast<uintptr_t>(allocationBase)};
  auto it = p2pIpcCache.find(key);
  return it == p2pIpcCache.end() ? nullptr : it->second;
}

flagcxResult_t flagcxRegPool::removeRegItemNetHandles(void *comm,
                                                      flagcxRegItem *reg) {
  if (reg == nullptr) {
    return flagcxSuccess;
  }

  for (size_t i = 0; i < reg->handles.size();) {
    auto &entry = reg->handles[i];
    // comm == nullptr: remove all; comm != nullptr: remove only this comm's
    if (entry.first.handle &&
        (comm == nullptr || entry.first.ownerComm == comm)) {
      FLAGCXCHECK(flagcxNetDeregisterBuffer(
          entry.first.ownerComm, entry.first.proxyConn, entry.first.handle));
      entry.first.handle = nullptr;
      entry.first.proxyConn = nullptr;
      entry.first.ownerComm = nullptr;
    }
    if (entry.first.handle == nullptr && entry.second.handle == nullptr) {
      reg->handles[i] = reg->handles.back();
      reg->handles.pop_back();
    } else {
      ++i;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::removeRegItemP2pHandles(void *comm,
                                                      flagcxRegItem *reg) {
  if (reg == nullptr) {
    return flagcxSuccess;
  }

  for (size_t i = 0; i < reg->handles.size();) {
    auto &entry = reg->handles[i];
    // comm == nullptr: remove all; comm != nullptr: remove only this comm's
    if (entry.second.handle &&
        (comm == nullptr || entry.second.ownerComm == comm)) {
      flagcxIpcRegInfo *ipcInfo = (flagcxIpcRegInfo *)entry.second.handle;
      if (ipcInfo->refCount <= 0) {
        WARN("removeRegItemP2pHandles: invalid IPC mapping refcount %d",
             ipcInfo->refCount);
        return flagcxInternalError;
      }
      if (ipcInfo->refCount == 1) {
        P2pIpcCacheKey key{
            reinterpret_cast<uintptr_t>(entry.second.ownerComm),
            ipcInfo->peerRank,
            reinterpret_cast<uintptr_t>(ipcInfo->allocationBase)};
        auto cached = p2pIpcCache.find(key);
        if (cached == p2pIpcCache.end() || cached->second != ipcInfo) {
          WARN("removeRegItemP2pHandles: IPC mapping cache mismatch");
          return flagcxInternalError;
        }
        FLAGCXCHECK(flagcxP2pDeregisterBuffer(
            reinterpret_cast<flagcxHeteroComm *>(entry.second.ownerComm),
            ipcInfo));
        p2pIpcCache.erase(cached);
        free(ipcInfo);
      } else {
        ipcInfo->refCount--;
      }
      entry.second.handle = nullptr;
      entry.second.proxyConn = nullptr;
      entry.second.ownerComm = nullptr;
    }
    if (entry.first.handle == nullptr && entry.second.handle == nullptr) {
      reg->handles[i] = reg->handles.back();
      reg->handles.pop_back();
    } else {
      ++i;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::removeAllP2pHandles(void *comm) {
  if (comm == nullptr) {
    return flagcxSuccess;
  }
  // Iterate over all items in the global pool and remove p2p handles
  // associated with this comm
  auto &globalPool = regPool[GLOBAL_POOL_KEY];
  for (auto &pair : globalPool) {
    FLAGCXCHECK(removeRegItemP2pHandles(comm, pair.second.get()));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::removeAllNetHandles(void *comm) {
  if (comm == nullptr) {
    return flagcxSuccess;
  }
  // Iterate over all items in the global pool and remove net handles
  // associated with this comm
  auto &globalPool = regPool[GLOBAL_POOL_KEY];
  for (auto &pair : globalPool) {
    FLAGCXCHECK(removeRegItemNetHandles(comm, pair.second.get()));
  }
  return flagcxSuccess;
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

flagcxResult_t flagcxRegPool::registerBuffer(void *comm, void *data,
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

flagcxResult_t flagcxRegPool::deregisterBuffer(void *comm, void *handle) {
  if (handle == nullptr) {
    return flagcxSuccess;
  }

  uintptr_t commKey =
      comm ? reinterpret_cast<uintptr_t>(comm) : GLOBAL_POOL_KEY;
  flagcxRegItem *reg = (flagcxRegItem *)handle;

  // Find the item in the global pool
  auto &globalPool = regPool[GLOBAL_POOL_KEY];
  auto poolIt = globalPool.find(reg->beginAddr);
  if (poolIt == globalPool.end() || poolIt->second.get() != reg) {
    WARN("Could not find the given handle in regPool");
    return flagcxInvalidUsage;
  }

  reg->refCount--;

  // Remove comm-specific page mappings
  if (comm != nullptr && commKey != GLOBAL_POOL_KEY) {
    auto mapIt = regMap.find(commKey);
    if (mapIt != regMap.end()) {
      auto &commMap = mapIt->second;
      for (uintptr_t addr = reg->beginAddr; addr < reg->endAddr;
           addr += pageSize) {
        commMap.erase(addr);
      }
      if (commMap.empty()) {
        regMap.erase(mapIt);
      }
    }
  }

  if (reg->refCount > 0) {
    return flagcxSuccess;
  }

  // refCount == 0: full cleanup (nullptr = remove all handles)
  FLAGCXCHECK(removeRegItemNetHandles(nullptr, reg));
  FLAGCXCHECK(removeRegItemP2pHandles(nullptr, reg));

  // Remove ALL regMap entries (global + comm-specific) that reference this item
  for (auto mapIt = regMap.begin(); mapIt != regMap.end();) {
    auto &pageMap = mapIt->second;
    for (uintptr_t addr = reg->beginAddr; addr < reg->endAddr;
         addr += pageSize) {
      auto it = pageMap.find(addr);
      if (it != pageMap.end() && it->second == reg) {
        pageMap.erase(it);
      }
    }
    if (pageMap.empty()) {
      mapIt = regMap.erase(mapIt);
    } else {
      ++mapIt;
    }
  }

  // Remove from global pool (this destroys the flagcxRegItem)
  globalPool.erase(poolIt);
  return flagcxSuccess;
}

std::unordered_map<uintptr_t, std::unordered_map<uintptr_t, flagcxRegItem *>> &
flagcxRegPool::getGlobalMap() {
  return regMap;
}

flagcxRegItem *flagcxRegPool::getItem(const void *comm, void *data) {
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

void flagcxRegPool::dump() {
  printf("========================\n");
  printf("RegPool(pageSize=%lu\n", pageSize);
  for (auto &c : regMap) {
    printf("==comm(%lu)==\n", c.first);
    for (auto &p : c.second) {
      printf("beginAddr(%lu) -> regItem[%lu,%lu,%d]\n", p.first,
             p.second->beginAddr, p.second->endAddr, p.second->refCount);
      for (auto &h : p.second->handles) {
        printf("handlePtr(%p) -> netHandle[%p,%p] p2pHandle[%p,%p]\n", &h,
               h.first.handle, h.first.proxyConn, h.second.handle,
               h.second.proxyConn);
      }
    }
    printf("==comm(%lu)==\n", c.first);
  }
  printf("========================\n");
}
