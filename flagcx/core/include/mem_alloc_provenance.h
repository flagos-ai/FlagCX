/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Plain allocation provenance types shared by host lifecycle code. Keep this
 * header free of STL types so device host compilers can parse it cheaply.
 ************************************************************************/

#ifndef FLAGCX_MEM_ALLOC_PROVENANCE_H_
#define FLAGCX_MEM_ALLOC_PROVENANCE_H_

#include "flagcx.h"

enum flagcxMemAllocBackend {
  flagcxMemAllocBackendGDR = 0,
  flagcxMemAllocBackendCCL = 1,
  flagcxMemAllocBackendSHMEM = 2,
};

struct flagcxMemAllocationInfo {
  void *base;
  size_t size;
  flagcxMemAllocator_t allocator;
  flagcxMemAllocBackend backend;
};

#endif // FLAGCX_MEM_ALLOC_PROVENANCE_H_
