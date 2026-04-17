/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Custom op runtime state — DevComm, staged buffers, registered custom ops.
 * Controlled by environment variable FLAGCX_CUSTOM_OP_ENABLE=1.
 ************************************************************************/

#ifndef DEV_COMM_STATE_H_
#define DEV_COMM_STATE_H_

#include "flagcx.h"

// Forward declarations for Device API types (defined in flagcx_device.h)
struct flagcxDevCommInternal;
typedef struct flagcxDevCommInternal *flagcxDevComm_t;
struct flagcxDevMemInternal;
typedef struct flagcxDevMemInternal *flagcxDevMem_t;

// Custom op function signature — matches flagcxAllReduce parameters
typedef flagcxResult_t (*flagcxCustomAllReduceFn_t)(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t op, flagcxComm_t comm,
    flagcxStream_t stream);

// Custom op runtime state — contains DevComm, staged buffers, registered ops.
// Lifecycle managed by flagcxDevCommStateInit / flagcxDevCommStateDestroy.
struct flagcxDevCommState {
  // Device API resources (host-side handles)
  flagcxDevComm_t devComm;
  flagcxDevMem_t sendStagedMem;
  flagcxDevMem_t recvStagedMem;
  flagcxWindow_t sendStagedWin;
  flagcxWindow_t recvStagedWin;
  void *sendStagedBuff;
  void *recvStagedBuff;
  size_t stagedBuffSize;

  // Registered custom ops
  flagcxCustomAllReduceFn_t customAllReduce;

  bool initialized;
};

#endif // DEV_COMM_STATE_H_
