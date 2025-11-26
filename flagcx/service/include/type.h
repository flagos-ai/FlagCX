/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "flagcx.h"
#include <cstddef>

#ifndef FLAGCX_TYPE_H_
#define FLAGCX_TYPE_H_

#define FLAGCX_VERSION(X, Y, Z) ((X)*10000 + (Y)*100 + (Z))

/* flagcxScalarResidence_t: Location and dereferencing logic for scalar
 * arguments. */
typedef enum {
  /* flagcxScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  flagcxScalarDevice = 0,

  /* flagcxScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the flagcxRedOpCreate***() function returns. */
  flagcxScalarHostImmediate = 1
} flagcxScalarResidence_t;

#define FLAGCX_CONFIG_UNDEF_INT INT_MIN
#define FLAGCX_CONFIG_UNDEF_PTR NULL
#define FLAGCX_SPLIT_NOCOLOR -1

typedef struct flagcxConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char *netName;
  int splitShare;
} flagcxConfig_t;

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (FLAGCX_STEPS / 4)
#define ALLREDUCE_CHUNKSTEPS (FLAGCX_STEPS / 2)
#define ALLGATHER_SLICESTEPS (FLAGCX_STEPS / 4)
#define ALLGATHER_CHUNKSTEPS (FLAGCX_STEPS / 2)
#define REDUCESCATTER_SLICESTEPS (FLAGCX_STEPS / 4)
#define REDUCESCATTER_CHUNKSTEPS (FLAGCX_STEPS / 2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define FLAGCX_MAX_SLICE_PER_CHUNK                                             \
  2 // max value for CHUNKSTEPS/SLICESTEPS, must accord with above

#include <sys/types.h>

#define FLAGCX_MODE_NORMAL 0
#define FLAGCX_MODE_OFFSET 1
#define FLAGCX_MODE_PTR 2
struct flagcxConnFifo {
  int mode;
  int offset;
  ssize_t size;
  void *ptr;
};

#endif