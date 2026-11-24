#include "reg_pool.h"
#include "adaptor.h"
#include "p2p.h"
#include "proxy.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>

// The main library may be built with a pre-C++17 default language mode, where
// an odr-used static constexpr data member still needs one out-of-class
// definition. Unit tests link the shared library into a C++17 executable and
// therefore expose the otherwise deferred undefined symbol.
constexpr uintptr_t flagcxRegPool::GLOBAL_POOL_KEY;

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

bool flagcxRegPool::contains(void *handle) {
  for (auto &entry : regPool[GLOBAL_POOL_KEY])
    if (entry.second.get() == handle)
      return true;
  return false;
}

flagcxResult_t
flagcxRegPool::addNetHandle(void *comm, flagcxRegItem *reg, void *handle,
                            struct flagcxProxyConnector *proxyConn,
                            bool complete, bool usable) {
  std::lock_guard<std::mutex> lock(mutex);
  if (reg == nullptr || comm == nullptr)
    return flagcxSuccess;
  if (cleaning || !contains(reg) || reg->closing ||
      reg->closingOwners.count((uintptr_t)comm))
    return flagcxInvalidUsage;
  for (auto &pair : reg->handles) {
    if (pair.first.proxyConn == proxyConn && pair.first.ownerComm == comm) {
      pair.first.handle = handle;
      pair.first.state = !complete           ? flagcxRegNetPending
                         : handle == nullptr ? flagcxRegNetEmpty
                         : usable            ? flagcxRegNetReady
                                             : flagcxRegNetCleanupOnly;
      return flagcxSuccess;
    }
  }
  try {
    reg->handles.push_back({{handle, proxyConn, comm,
                             !complete           ? flagcxRegNetPending
                             : handle == nullptr ? flagcxRegNetEmpty
                             : usable            ? flagcxRegNetReady
                                                 : flagcxRegNetCleanupOnly},
                            {nullptr, nullptr, nullptr}});
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t
flagcxRegPool::reserveNetHandle(flagcxRegItem *reg, void *comm,
                                struct flagcxProxyConnector *connector) {
  std::unique_lock<std::mutex> lock(mutex);
  if (cleaning || !contains(reg) || reg->closing ||
      reg->closingOwners.count((uintptr_t)comm))
    return flagcxInvalidUsage;
  for (auto &pair : reg->handles) {
    if (pair.first.ownerComm != comm || pair.first.proxyConn != connector)
      continue;
    if (pair.first.state == flagcxRegNetPending ||
        pair.first.state == flagcxRegNetReady)
      return flagcxInProgress;
    void *cleanupHandle = pair.first.handle;
    pair.first.state = flagcxRegNetPending;
    if (cleanupHandle != nullptr) {
      // Keep the metadata pinned while retrying a failed registration's
      // cleanup. Other submissions see Pending, never a usable handle.
      ++readers;
      lock.unlock();
      flagcxResult_t result =
          flagcxNetDeregisterBuffer(comm, connector, cleanupHandle);
      lock.lock();
      --readers;
      // The vector may have grown while the SDK/proxy call ran.
      for (auto &entry : reg->handles) {
        if (entry.first.ownerComm != comm || entry.first.proxyConn != connector)
          continue;
        if (result != flagcxSuccess) {
          entry.first.state = flagcxRegNetCleanupOnly;
          return result;
        }
        entry.first.handle = nullptr;
        return flagcxSuccess;
      }
      return flagcxInternalError;
    }
    return flagcxSuccess;
  }
  try {
    reg->handles.push_back({{nullptr, connector, comm, flagcxRegNetPending},
                            {nullptr, nullptr, nullptr}});
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::reserveHomoHandle(flagcxRegItem *reg, void *comm,
                                                void *nativeComm) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning || !contains(reg) || reg->closing)
    return flagcxInvalidUsage;
  try {
    if (!reg->homoRegHandles.emplace((uintptr_t)comm, nullptr).second)
      return flagcxInProgress;
    try {
      reg->homoRegComms.emplace((uintptr_t)comm,
                                nativeComm != nullptr ? nativeComm : comm);
    } catch (...) {
      reg->homoRegHandles.erase((uintptr_t)comm);
      throw;
    }
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

void *flagcxRegPool::getNetHandle(flagcxRegItem *reg, void *comm,
                                  struct flagcxProxyConnector *connector) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning || !contains(reg) || reg->closing ||
      reg->closingOwners.count((uintptr_t)comm))
    return nullptr;
  for (auto &pair : reg->handles)
    if (pair.first.ownerComm == comm && pair.first.proxyConn == connector &&
        pair.first.state == flagcxRegNetReady)
      return pair.first.handle;
  return nullptr;
}

flagcxResult_t
flagcxRegPool::addP2pHandle(void *comm, flagcxRegItem *reg,
                            flagcxIpcRegInfo *handle,
                            struct flagcxProxyConnector *proxyConn) {
  std::lock_guard<std::mutex> lock(mutex);
  if (reg == nullptr || comm == nullptr || handle == nullptr)
    return flagcxSuccess;
  if (cleaning || !contains(reg) || reg->closing ||
      reg->closingOwners.count((uintptr_t)comm))
    return flagcxInvalidUsage;
  for (auto &pair : reg->handles)
    if (pair.second.handle == handle && pair.second.ownerComm == comm)
      return flagcxSuccess;
  P2pIpcCacheKey key{(uintptr_t)comm, handle->peerRank,
                     (uintptr_t)handle->allocationBase};
  auto cached = p2pIpcCache.find(key);
  if (cached != p2pIpcCache.end() && cached->second != handle)
    return flagcxInvalidUsage;
  try {
    reg->handles.reserve(reg->handles.size() + 1);
    if (cached == p2pIpcCache.end()) {
      p2pIpcCache.emplace(key, handle);
      handle->refCount = 1;
    } else {
      ++handle->refCount;
    }
    reg->handles.push_back(
        {{nullptr, nullptr, nullptr}, {handle, proxyConn, comm}});
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxIpcRegInfo *flagcxRegPool::findP2pHandle(void *comm, int peerRank,
                                               void *allocationBase) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning || comm == nullptr || allocationBase == nullptr)
    return nullptr;
  auto it =
      p2pIpcCache.find({(uintptr_t)comm, peerRank, (uintptr_t)allocationBase});
  if (it != p2pIpcCache.end()) {
    // A shared IPC cache entry must not bypass a registration's close gate.
    for (const auto &item : regPool[GLOBAL_POOL_KEY]) {
      const auto *reg = item.second.get();
      if (!reg->closing && !reg->closingOwners.count((uintptr_t)comm))
        continue;
      for (const auto &entry : reg->handles)
        if (entry.second.ownerComm == comm && entry.second.handle == it->second)
          return nullptr;
    }
  }
  return it == p2pIpcCache.end() ? nullptr : it->second;
}

void *flagcxRegPool::getHomoHandle(flagcxRegItem *reg, void *comm) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning || !contains(reg) || reg->closing ||
      reg->closingOwners.count((uintptr_t)comm))
    return nullptr;
  auto it = reg->homoRegHandles.find((uintptr_t)comm);
  return it == reg->homoRegHandles.end() ? nullptr : it->second;
}

flagcxResult_t flagcxRegPool::reserveP2pRegistration(void *comm,
                                                     flagcxRegItem *reg,
                                                     flagcxIpcRegInfo *info) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning || !contains(reg) || reg->closing || info == nullptr ||
      reg->closingOwners.count((uintptr_t)comm))
    return flagcxInvalidUsage;
  auto cached = p2pIpcCache.find(
      {(uintptr_t)comm, info->peerRank, (uintptr_t)info->allocationBase});
  if (cached == p2pIpcCache.end() || cached->second != info)
    return flagcxInvalidUsage;
  bool pending = false;
  if (__atomic_load_n(&info->handleReady, __ATOMIC_ACQUIRE) ||
      __atomic_load_n(&info->rejected, __ATOMIC_ACQUIRE) ||
      !__atomic_compare_exchange_n(&info->registrationPending, &pending, true,
                                   false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return flagcxInProgress;
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::setHomoHandle(flagcxRegItem *reg, void *comm,
                                            void *handle) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning || !contains(reg) || reg->closing)
    return flagcxInvalidUsage;
  auto entry = reg->homoRegHandles.find((uintptr_t)comm);
  if (entry == reg->homoRegHandles.end())
    return flagcxInvalidUsage;
  entry->second = handle;
  return flagcxSuccess;
}

// cleaning excludes all metadata readers/writers, but no mutex is held across
// the SDK/proxy calls. A failed cleanup retains its fields and logical refs.
flagcxResult_t flagcxRegPool::removeHandles(void *comm, flagcxRegItem *reg,
                                            bool p2p) {
  for (size_t i = 0; i < reg->handles.size();) {
    auto &entry = reg->handles[i];
    if (!p2p && entry.first.state == flagcxRegNetPending &&
        (comm == nullptr || entry.first.ownerComm == comm))
      return flagcxRemoteError;
    if (!p2p && entry.first.handle != nullptr &&
        (comm == nullptr || entry.first.ownerComm == comm)) {
      flagcxResult_t result = flagcxNetDeregisterBuffer(
          entry.first.ownerComm, entry.first.proxyConn, entry.first.handle);
      if (result != flagcxSuccess)
        return result;
      entry.first = {nullptr, nullptr, nullptr};
    }
    if (p2p && entry.second.handle != nullptr &&
        (comm == nullptr || entry.second.ownerComm == comm)) {
      flagcxIpcRegInfo *info = (flagcxIpcRegInfo *)entry.second.handle;
      if (info->refCount <= 0)
        return flagcxInternalError;
      if (info->refCount == 1) {
        P2pIpcCacheKey key{(uintptr_t)entry.second.ownerComm, info->peerRank,
                           (uintptr_t)info->allocationBase};
        auto cached = p2pIpcCache.find(key);
        if (cached == p2pIpcCache.end() || cached->second != info)
          return flagcxInternalError;
        flagcxResult_t result = flagcxP2pDeregisterBuffer(
            (flagcxHeteroComm *)entry.second.ownerComm, info);
        if (result != flagcxSuccess)
          return result;
        p2pIpcCache.erase(cached);
        free(info);
      } else {
        --info->refCount;
      }
      entry.second = {nullptr, nullptr, nullptr};
    }
    if (entry.first.handle == nullptr && entry.second.handle == nullptr &&
        entry.first.state != flagcxRegNetPending) {
      reg->handles[i] = reg->handles.back();
      reg->handles.pop_back();
    } else {
      ++i;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::removeRegItemNetHandles(void *comm,
                                                      flagcxRegItem *reg) {
  std::unique_lock<std::mutex> lock(mutex);
  if (reg == nullptr)
    return flagcxSuccess;
  if (cleaning || readers != 0)
    return flagcxInProgress;
  if (!contains(reg))
    return flagcxInvalidUsage;
  cleaning = true;
  lock.unlock();
  flagcxResult_t result = removeHandles(comm, reg, false);
  lock.lock();
  cleaning = false;
  return result;
}

flagcxResult_t flagcxRegPool::removeRegItemP2pHandles(void *comm,
                                                      flagcxRegItem *reg) {
  std::unique_lock<std::mutex> lock(mutex);
  if (reg == nullptr)
    return flagcxSuccess;
  if (cleaning || readers != 0)
    return flagcxInProgress;
  if (!contains(reg))
    return flagcxInvalidUsage;
  cleaning = true;
  lock.unlock();
  flagcxResult_t result = removeHandles(comm, reg, true);
  lock.lock();
  cleaning = false;
  return result;
}

flagcxResult_t flagcxRegPool::removeAllHandles(void *comm, bool p2p) {
  if (comm == nullptr)
    return flagcxSuccess;
  std::unique_lock<std::mutex> lock(mutex);
  if (cleaning || readers != 0)
    return flagcxInProgress;
  cleaning = true;
  lock.unlock();
  flagcxResult_t result = flagcxSuccess;
  for (auto &entry : regPool[GLOBAL_POOL_KEY]) {
    result = removeHandles(comm, entry.second.get(), p2p);
    if (result != flagcxSuccess)
      break;
  }
  lock.lock();
  cleaning = false;
  return result;
}
flagcxResult_t flagcxRegPool::removeAllP2pHandles(void *comm) {
  return removeAllHandles(comm, true);
}
flagcxResult_t flagcxRegPool::removeAllNetHandles(void *comm) {
  return removeAllHandles(comm, false);
}

flagcxResult_t flagcxRegPool::registerBuffer(void *comm, void *data,
                                             size_t length,
                                             flagcxRegItem **pinned) {
  std::lock_guard<std::mutex> lock(mutex);
  if (pinned != nullptr)
    *pinned = nullptr;
  if (cleaning)
    return flagcxInProgress;
  if (data == nullptr || length == 0)
    return flagcxSuccess;
  uintptr_t key = (uintptr_t)comm;
  uintptr_t begin, end;
  getPagedAddr(data, length, &begin, &end);
  if (end <= begin)
    return flagcxInvalidArgument;
  flagcxRegItem *reg = nullptr;
  bool newItem = false;
  bool newOwner = false;
  std::vector<uintptr_t> globalPages, ownerPages;
  try {
    auto &pages = regMap[GLOBAL_POOL_KEY];
    // A range cannot silently merge two records with independently owned
    // backend registrations. Keep both records intact on that error.
    for (uintptr_t address = begin; address < end; address += pageSize) {
      auto found = pages.find(address);
      if (found == pages.end())
        continue;
      if (reg != nullptr && reg != found->second)
        return flagcxInvalidUsage;
      reg = found->second;
    }
    uintptr_t first = reg == nullptr ? begin : std::min(begin, reg->beginAddr);
    uintptr_t last = reg == nullptr ? end : std::max(end, reg->endAddr);
    if (reg != nullptr) {
      if (reg->closing || reg->closingOwners.count(key))
        return flagcxInvalidUsage;
      if (readers != 0 && (first != reg->beginAddr || last != reg->endAddr))
        return flagcxInProgress;
    }
    // Allocate rollback bookkeeping before publishing any new page/ref.
    globalPages.reserve((last - first) / pageSize);
    if (comm != nullptr)
      ownerPages.reserve((last - first) / pageSize);
    if (reg == nullptr) {
      std::unique_ptr<flagcxRegItem> item(new flagcxRegItem);
      item->beginAddr = begin;
      item->endAddr = end;
      item->refCount = 0;
      auto inserted = regPool[GLOBAL_POOL_KEY].emplace(begin, std::move(item));
      if (!inserted.second)
        return flagcxInternalError;
      reg = inserted.first->second.get();
      newItem = true;
    }
    auto owner = reg->ownerRefs.emplace(key, 0);
    newOwner = owner.second;
    for (uintptr_t address = first; address < last; address += pageSize)
      if (pages.emplace(address, reg).second)
        globalPages.push_back(address);
    if (comm != nullptr) {
      auto &ownedPages = regMap[key];
      for (uintptr_t address = first; address < last; address += pageSize)
        if (ownedPages.emplace(address, reg).second)
          ownerPages.push_back(address);
    }
    // The private pool key stays stable when the visible range grows. Never
    // move the only unique_ptr through a potentially failing map insertion.
    reg->beginAddr = first;
    reg->endAddr = last;
    ++owner.first->second;
    ++reg->refCount;
    if (pinned != nullptr) {
      ++readers;
      *pinned = reg;
    }
    return flagcxSuccess;
  } catch (const std::bad_alloc &) {
    for (uintptr_t address : globalPages)
      regMap[GLOBAL_POOL_KEY].erase(address);
    auto pages = regMap.find(key);
    if (comm != nullptr && pages != regMap.end()) {
      for (uintptr_t address : ownerPages)
        pages->second.erase(address);
      if (pages->second.empty())
        regMap.erase(pages);
    }
    if (newItem)
      regPool[GLOBAL_POOL_KEY].erase(begin);
    else if (newOwner)
      reg->ownerRefs.erase(key);
    return flagcxSystemError;
  }
}

flagcxRegItem *flagcxRegPool::findItem(const void *comm, void *data) {
  uintptr_t beginAddr, endAddr;
  getPagedAddr(data, 0, &beginAddr, &endAddr);

  // If comm is non-null, check comm-specific regMap first
  if (comm != nullptr) {
    uintptr_t commKey = reinterpret_cast<uintptr_t>(comm);
    auto mapIt = regMap.find(commKey);
    if (mapIt != regMap.end()) {
      auto it = mapIt->second.find(beginAddr);
      if (it != mapIt->second.end()) {
        return it->second->closing || it->second->closingOwners.count(commKey)
                   ? nullptr
                   : it->second;
      }
    }
  }

  // Fall through to global pool
  auto globalMapIt = regMap.find(GLOBAL_POOL_KEY);
  if (globalMapIt != regMap.end()) {
    auto it = globalMapIt->second.find(beginAddr);
    if (it != globalMapIt->second.end()) {
      return it->second->closing ||
                     it->second->closingOwners.count((uintptr_t)comm)
                 ? nullptr
                 : it->second;
    }
  }

  return nullptr;
}

flagcxRegItem *flagcxRegPool::getItem(const void *comm, void *data) {
  std::lock_guard<std::mutex> lock(mutex);
  return cleaning ? nullptr : findItem(comm, data);
}

flagcxRegItem *flagcxRegPool::acquireItem(const void *comm, void *data) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning)
    return nullptr;
  flagcxRegItem *reg = findItem(comm, data);
  if (reg != nullptr && reg->closingOwners.count((uintptr_t)comm))
    return nullptr;
  if (reg != nullptr)
    ++readers;
  return reg;
}

void flagcxRegPool::releaseItem(flagcxRegItem *reg) {
  if (reg == nullptr)
    return;
  std::lock_guard<std::mutex> lock(mutex);
  --readers;
}

bool flagcxRegPool::isGlobalOnly(void *handle) {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning || !contains(handle))
    return false;
  auto *reg = (flagcxRegItem *)handle;
  return reg->ownerRefs.size() == 1 && reg->ownerRefs.count(GLOBAL_POOL_KEY) &&
         reg->homoRegHandles.empty() && reg->handles.empty();
}

void flagcxRegPool::eraseItem(flagcxRegItem *reg) {
  for (auto map = regMap.begin(); map != regMap.end();) {
    for (auto page = map->second.begin(); page != map->second.end();) {
      if (page->second == reg)
        page = map->second.erase(page);
      else
        ++page;
    }
    if (map->second.empty())
      map = regMap.erase(map);
    else
      ++map;
  }
  auto &items = regPool[GLOBAL_POOL_KEY];
  for (auto item = items.begin(); item != items.end(); ++item) {
    if (item->second.get() == reg) {
      items.erase(item);
      break;
    }
  }
}

flagcxResult_t flagcxRegPool::releaseRegistration(void *comm,
                                                  flagcxRegItem *reg,
                                                  void *nativeComm,
                                                  bool allRefs) {
  uintptr_t key = (uintptr_t)comm;
  auto owner = reg->ownerRefs.find(key);
  if (!allRefs && owner == reg->ownerRefs.end())
    return flagcxInvalidUsage;
  if (allRefs && owner == reg->ownerRefs.end() &&
      reg->homoRegHandles.count(key) == 0) {
    bool found = false;
    for (const auto &pair : reg->handles)
      found |= pair.first.ownerComm == comm || pair.second.ownerComm == comm;
    if (!found)
      return flagcxSuccess;
  }
  if (!allRefs && owner->second > 1) {
    --owner->second;
    --reg->refCount;
    return flagcxSuccess;
  }
  reg->closingOwners.insert(key);
  const bool lastRegistration =
      owner != reg->ownerRefs.end() && reg->refCount == (int)owner->second;
  if (lastRegistration)
    reg->closing = true;
  for (auto homo = reg->homoRegHandles.begin();
       homo != reg->homoRegHandles.end();) {
    if (!lastRegistration && homo->first != key) {
      ++homo;
      continue;
    }
    if (homo->second != nullptr) {
      void *context = homo->first == key ? nativeComm : nullptr;
      if (context == nullptr) {
        auto saved = reg->homoRegComms.find(homo->first);
        if (saved != reg->homoRegComms.end())
          context = saved->second;
      }
      if (context == nullptr)
        return flagcxInvalidUsage;
      flagcxResult_t result =
          cclAdaptors[flagcxCCLAdaptorDevice]->commDeregister(
              (flagcxInnerComm_t)context, homo->second);
      if (result != flagcxSuccess)
        return result;
    }
    reg->homoRegComms.erase(homo->first);
    homo = reg->homoRegHandles.erase(homo);
  }
  if (comm != nullptr || lastRegistration) {
    void *handleOwner = lastRegistration ? nullptr : comm;
    flagcxResult_t result = removeHandles(handleOwner, reg, true);
    if (result != flagcxSuccess)
      return result;
    result = removeHandles(handleOwner, reg, false);
    if (result != flagcxSuccess)
      return result;
  }
  if (owner != reg->ownerRefs.end()) {
    reg->refCount -= owner->second;
    reg->ownerRefs.erase(owner);
  }
  if (comm != nullptr) {
    auto map = regMap.find(key);
    if (map != regMap.end()) {
      for (auto page = map->second.begin(); page != map->second.end();) {
        if (page->second == reg)
          page = map->second.erase(page);
        else
          ++page;
      }
      if (map->second.empty())
        regMap.erase(map);
    }
  }
  reg->closingOwners.erase(key);
  if (reg->refCount == 0 && reg->handles.empty() && reg->homoRegHandles.empty())
    eraseItem(reg);
  return flagcxSuccess;
}

flagcxResult_t flagcxRegPool::deregisterBuffer(void *comm, void *handle,
                                               void *nativeComm) {
  if (handle == nullptr)
    return flagcxSuccess;
  std::unique_lock<std::mutex> lock(mutex);
  if (cleaning)
    return flagcxInProgress;
  if (!contains(handle))
    return flagcxInvalidUsage;
  auto *reg = (flagcxRegItem *)handle;
  auto owner = reg->ownerRefs.find((uintptr_t)comm);
  if (owner == reg->ownerRefs.end())
    return flagcxInvalidUsage;
  if (owner->second > 1) {
    --owner->second;
    --reg->refCount;
    return flagcxSuccess;
  }
  if (readers != 0)
    return flagcxInProgress;
  cleaning = true;
  lock.unlock();
  flagcxResult_t result;
  try {
    result =
        releaseRegistration(comm, (flagcxRegItem *)handle, nativeComm, false);
  } catch (const std::bad_alloc &) {
    result = flagcxSystemError;
  }
  lock.lock();
  cleaning = false;
  return result;
}

flagcxResult_t flagcxRegPool::releaseOwner(void *comm, void *nativeComm) {
  if (comm == nullptr)
    return flagcxSuccess;
  std::unique_lock<std::mutex> lock(mutex);
  if (cleaning || readers != 0)
    return flagcxInProgress;
  cleaning = true;
  lock.unlock();
  flagcxResult_t result = flagcxSuccess;
  auto &items = regPool[GLOBAL_POOL_KEY];
  for (auto it = items.begin(); it != items.end();) {
    flagcxRegItem *reg = it->second.get();
    ++it;
    try {
      result = releaseRegistration(comm, reg, nativeComm, true);
    } catch (const std::bad_alloc &) {
      result = flagcxSystemError;
    }
    if (result != flagcxSuccess)
      break;
  }
  lock.lock();
  cleaning = false;
  return result;
}

void flagcxRegPool::dump() {
  std::lock_guard<std::mutex> lock(mutex);
  if (cleaning)
    return;
  for (auto &entry : regPool[GLOBAL_POOL_KEY])
    printf("reg [%lu,%lu) refs=%d owners=%zu\n", entry.second->beginAddr,
           entry.second->endAddr, entry.second->refCount,
           entry.second->ownerRefs.size());
}
