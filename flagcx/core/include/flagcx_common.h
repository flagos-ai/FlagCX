/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_COMMON_H_
#define FLAGCX_COMMON_H_

#include "debug.h"

#define FLAGCX_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum {
  flagcxFuncBroadcast = 0,
  flagcxFuncReduce = 1,
  flagcxFuncAllGather = 2,
  flagcxFuncReduceScatter = 3,
  flagcxFuncAllReduce = 4,
  flagcxFuncSendRecv = 5,
  flagcxFuncSend = 6,
  flagcxFuncRecv = 7,
  flagcxNumFuncs = 8
} flagcxFunc_t;

#define FLAGCX_NUM_ALGORITHMS 6 // Tree/Ring/CollNet*
#define FLAGCX_ALGO_UNDEF -1
#define FLAGCX_ALGO_TREE 0
#define FLAGCX_ALGO_RING 1
#define FLAGCX_ALGO_COLLNET_DIRECT 2
#define FLAGCX_ALGO_COLLNET_CHAIN 3
#define FLAGCX_ALGO_NVLS 4
#define FLAGCX_ALGO_NVLS_TREE 5

#define FLAGCX_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define FLAGCX_PROTO_UNDEF -1
#define FLAGCX_PROTO_LL 0
#define FLAGCX_PROTO_LL128 1
#define FLAGCX_PROTO_SIMPLE 2

#define FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE 16

#endif
