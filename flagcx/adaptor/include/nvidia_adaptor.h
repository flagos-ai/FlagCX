#ifdef USE_NVIDIA_ADAPTOR

#include "flagcx.h"
#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"

#define NCCL_ADAPTOR_DEVICE_CTA_COUNT 36
#define NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA 512

struct flagcxInnerDevComm {
  ncclDevComm base;
};

#else

typedef void ncclDevComm;
struct flagcxInnerDevComm {};

#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)

struct flagcxInnerComm {
  ncclComm_t base;
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

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
#define FLAGCX_SYM_WINDOW_DEFINED
struct flagcxSymWindow {
  ncclWindow_t base;
  int winFlags;

  // Non-homo symmetric path fields (unused on homo NCCL path)
  void *flatBase;     // flat VA base (NULL if IPC fallback)
  void *mcBase;       // multicast base (NULL if no NVLS)
  void **devPeerPtrs; // device-side peer pointer array
  int mrIndex;        // one-sided MR index (-1 if none)
  uintptr_t mrBase;   // MR base VA
  size_t heapSize;    // currently backed size per peer
  size_t maxHeapSize; // total reserved VA per peer (for growth)
  int localRanks;     // number of intra-node peers
  void *physHandle;   // for cleanup (symPhysFree)
  void *mcHandle;     // multicast handle (for growth + cleanup)

  // Growth tracking
  void **growthPhysHandles;
  int growthCount;
  int growthCapacity;

  // Cleanup state
  bool isSymmetricFallback; // true if non-homo symmetric path
  bool isVMM;               // true if VMM path (false = IPC fallback)
};
#else
#define FLAGCX_SYM_WINDOW_DEFINED
struct flagcxSymWindow {
  int winFlags;

  // Non-homo symmetric path fields
  void *flatBase;
  void *mcBase;
  void **devPeerPtrs;
  int mrIndex;
  uintptr_t mrBase;
  size_t heapSize;
  size_t maxHeapSize;
  int localRanks;
  void *physHandle;
  void *mcHandle;

  void **growthPhysHandles;
  int growthCount;
  int growthCapacity;

  bool isSymmetricFallback;
  bool isVMM;
};
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_NVIDIA_ADAPTOR