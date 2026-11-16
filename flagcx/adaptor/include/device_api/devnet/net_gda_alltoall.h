/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Work descriptor for the Device API Net GDA AlltoAll kernel. The caller
 * owns every device view and the stream until the launch has completed.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_NET_GDA_ALLTOALL_H_
#define FLAGCX_DEVICE_API_NET_GDA_ALLTOALL_H_

#include "device_api/flagcx_device.h"

#if defined(FLAGCX_COMM_TRAITS_DEFAULT)

struct flagcxNetGdaAlltoAllWork {
  const flagcxDevComm *comm;
  const flagcxDevMem *send;
  const flagcxDevMem *recv;
  // Host clears this device word before launch. The GDA provider stores its
  // first device error here with release order; the kernel checks it before
  // entering the World Barrier.
  FLAGCX_DEVICE_GLOBAL_PTR uint32_t *errorFlag;
  size_t bytesPerPeer;
  size_t chunkBytes;
  int contextIndex;
  unsigned barrierIndex;
  flagcxDevNetFenceLevel fence;
};

// Internal launch entry. Host setup must publish the chosen GDA Provider,
// register both Windows, and reserve the World Barrier channel first. A
// successful return means only that CUDA accepted the launch; the caller
// checks stream completion and errorFlag separately.
flagcxResult_t flagcxNetGdaAlltoAllLaunch(const flagcxNetGdaAlltoAllWork &work,
                                          flagcxStream_t stream);

#endif // FLAGCX_COMM_TRAITS_DEFAULT

#endif // FLAGCX_DEVICE_API_NET_GDA_ALLTOALL_H_
