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
#define NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8 * 1024 * 1024)

struct stagedBuffer {
  void *buff;
  ncclWindow_t win;
};
typedef struct stagedBuffer *stagedBuffer_t;

#if defined(COMPILE_KERNEL_HOST)
extern "C" ncclResult_t
ncclAdaptorLocalAllReduce(const void *sendbuff, void *recvbuff,
                          ncclWindow_t sendwin, ncclWindow_t recvwin,
                          size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                          ncclDevComm &devComm, cudaStream_t stream);

extern "C" ncclResult_t ncclAdaptorInterleavedAllReduce(
    const void *sendbuff, void *recvbuff, ncclWindow_t sendwin,
    ncclWindow_t recvwin, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    ncclDevComm &devComm, cudaStream_t stream);
#endif // COMPILE_KERNEL_HOST

struct flagcxInnerDevComm {
  ncclDevComm base;
};

#else

typedef void *stagedBuffer_t;
typedef void ncclDevComm;
struct flagcxInnerDevComm {};

#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)

struct flagcxInnerComm {
  ncclComm_t base;
  ncclDevComm *devBase;
  stagedBuffer_t sendStagedBuff;
  stagedBuffer_t recvStagedBuff;
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
struct flagcxWindow {
  ncclWindow_t base;
  int winFlags;
};
#else
struct flagcxWindow {
  int winFlags;
};
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_NVIDIA_ADAPTOR