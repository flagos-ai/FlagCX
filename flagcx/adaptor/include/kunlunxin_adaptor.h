/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All
 *Rights Reserved.
 ************************************************************************/

#ifdef USE_KUNLUNXIN_ADAPTOR

#include <map>

#include <bkcl.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "flagcx.h"

struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  BKCLContext_t base;
};

struct flagcxStream {
  cudaStream_t base;
};

struct flagcxEvent {
  cudaEvent_t base;
};

struct flagcxIpcMemHandle {
  cudaIpcMemHandle_t base;
};

// GDR memory handle for alloc-internal mmap.
// Allocated by memHandleInit, released by memHandleDestroy.
// gdrMemAlloc fills all fields; gdrMemFree clears them after cleanup.
struct KunlunXinGdrMemHandle {
  void  *devPtr;         // device pointer from xpu_malloc
  void  *hostMappedPtr;  // host VA from xccl_mmap, passed to ibv_reg_mr
  size_t size;           // allocation size
  bool   mapped;         // true between xccl_mmap and xccl_munmap
};

namespace baidu {
namespace xpu {
namespace bkcl {

// External declaration
extern int xccl_mmap(void **pcpuptr, void *devptr, size_t sz);
extern int xccl_munmap(void *cpuptr, size_t sz);

} // namespace bkcl
} // namespace xpu
} // namespace baidu

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_KUNLUNXIN_ADAPTOR