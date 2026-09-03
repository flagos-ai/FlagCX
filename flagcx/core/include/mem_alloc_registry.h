/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Tracks memory allocated through flagcxMemAlloc. This is intentionally
 * separate from globalRegPool: allocation ownership uses exact byte ranges,
 * while transport registration uses page-aligned, communicator-scoped ranges.
 ************************************************************************/

#ifndef FLAGCX_MEM_ALLOC_REGISTRY_H_
#define FLAGCX_MEM_ALLOC_REGISTRY_H_

#include "mem_alloc_provenance.h"

#include <map>
#include <mutex>
#include <stdint.h>

class flagcxMemAllocRegistry {
public:
  flagcxResult_t insert(const flagcxMemAllocationInfo &info);
  flagcxResult_t findExact(const void *base,
                           flagcxMemAllocationInfo *info) const;
  flagcxResult_t findRange(const void *ptr, size_t size,
                           flagcxMemAllocationInfo *info) const;
  flagcxResult_t erase(const void *base);

private:
  mutable std::mutex mutex_;
  std::map<uintptr_t, flagcxMemAllocationInfo> allocations_;
};

extern flagcxMemAllocRegistry globalMemAllocRegistry;

#endif // FLAGCX_MEM_ALLOC_REGISTRY_H_
