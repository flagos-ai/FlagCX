/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Default Net GDA views for XPU NetIO. Host owners publish these PODs only
 *after XPU NetIO QPs and memory regions are ready on every participating rank.
 ************************************************************************/

#ifndef FLAGCX_XPU_NETIO_GDA_VIEW_H_
#define FLAGCX_XPU_NETIO_GDA_VIEW_H_

#include <xpu_netio/device.h>

struct flagcxXpuNetioDeviceWindow {
  xpu_netio_local_region_t local;
  const xpu_netio_remote_region_t *remote;
  uint64_t bytes;
};

struct flagcxXpuNetioDeviceContext {
  const xpu_netio_device_context_t *contexts;
  const xpu_netio_remote_region_t *signalRemote;
  xpu_netio_local_region_t signalLocal;
  uint64_t *signalBuffer;
  uint32_t *errorFlag;
  uint64_t timeoutNs;
  uint32_t rank;
  uint32_t nranks;
  uint32_t signalCount;
};

#endif // FLAGCX_XPU_NETIO_GDA_VIEW_H_
