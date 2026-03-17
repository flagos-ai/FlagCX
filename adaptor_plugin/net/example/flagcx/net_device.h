/*************************************************************************
 * Copyright (c) 2023-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_NET_DEVICE_H_
#define FLAGCX_NET_DEVICE_H_

#include <cstddef>

#define FLAGCX_NET_DEVICE_INVALID_VERSION 0x0
#define FLAGCX_NET_MTU_SIZE 4096

#define FLAGCX_NET_DEVICE_UNPACK_VERSION 0x7

typedef enum {
  FLAGCX_NET_DEVICE_HOST = 0,
  FLAGCX_NET_DEVICE_UNPACK = 1
} flagcxNetDeviceType;

typedef struct {
  flagcxNetDeviceType netDeviceType;
  int netDeviceVersion;
  void *handle;
  size_t size;
  int needsProxyProgress;
} flagcxNetDeviceHandle_v7_t;

typedef flagcxNetDeviceHandle_v7_t flagcxNetDeviceHandle_v8_t;
typedef flagcxNetDeviceHandle_v8_t flagcxNetDeviceHandle_v9_t;
typedef flagcxNetDeviceHandle_v9_t flagcxNetDeviceHandle_v10_t;
typedef flagcxNetDeviceHandle_v10_t flagcxNetDeviceHandle_t;

#endif
