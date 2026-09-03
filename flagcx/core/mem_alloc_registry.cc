/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "mem_alloc_registry.h"

#include <iterator>
#include <limits>
#include <new>

flagcxMemAllocRegistry globalMemAllocRegistry;

namespace {

bool validRange(uintptr_t base, size_t size) {
  return base != 0 && size != 0 &&
         size <= std::numeric_limits<uintptr_t>::max() - base;
}

} // namespace

flagcxResult_t
flagcxMemAllocRegistry::insert(const flagcxMemAllocationInfo &info) {
  uintptr_t base = reinterpret_cast<uintptr_t>(info.base);
  if (!validRange(base, info.size))
    return flagcxInvalidArgument;

  uintptr_t end = base + info.size;
  std::lock_guard<std::mutex> lock(mutex_);
  auto next = allocations_.lower_bound(base);
  if (next != allocations_.end() && next->first < end)
    return flagcxInvalidUsage;
  if (next != allocations_.begin()) {
    auto prev = std::prev(next);
    uintptr_t prevEnd = prev->first + prev->second.size;
    if (prevEnd > base)
      return flagcxInvalidUsage;
  }
  try {
    allocations_.emplace(base, info);
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t
flagcxMemAllocRegistry::findExact(const void *basePtr,
                                  flagcxMemAllocationInfo *info) const {
  if (basePtr == nullptr || info == nullptr)
    return flagcxInvalidArgument;

  uintptr_t base = reinterpret_cast<uintptr_t>(basePtr);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(base);
  if (it == allocations_.end())
    return flagcxInvalidUsage;
  *info = it->second;
  return flagcxSuccess;
}

flagcxResult_t
flagcxMemAllocRegistry::findRange(const void *ptr, size_t size,
                                  flagcxMemAllocationInfo *info) const {
  if (ptr == nullptr || size == 0 || info == nullptr)
    return flagcxInvalidArgument;

  uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.upper_bound(address);
  if (it == allocations_.begin())
    return flagcxInvalidUsage;
  --it;

  uintptr_t base = it->first;
  size_t allocationSize = it->second.size;
  size_t offset = address - base;
  if (offset > allocationSize || size > allocationSize - offset)
    return flagcxInvalidUsage;

  *info = it->second;
  return flagcxSuccess;
}

flagcxResult_t flagcxMemAllocRegistry::erase(const void *basePtr) {
  if (basePtr == nullptr)
    return flagcxInvalidArgument;

  uintptr_t base = reinterpret_cast<uintptr_t>(basePtr);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(base);
  if (it == allocations_.end())
    return flagcxInvalidUsage;
  allocations_.erase(it);
  return flagcxSuccess;
}
