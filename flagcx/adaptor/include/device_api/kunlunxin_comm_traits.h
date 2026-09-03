/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * KunlunXin CommTraits selector — included from comm_traits.h when
 * USE_KUNLUNXIN_ADAPTOR is defined. Mirrors nvidia_comm_traits.h:
 *
 *   FLAGCX_COMM_TRAITS_SHMEM defined (USE_XSHMEM=1):  DeviceAPI =
 * CommTraits<XshmemBackend> (symmetric-heap one-sided backend)
 *   otherwise:                                          DeviceAPI =
 * CommTraits<DefaultBackend<DefaultPlatform>> (IPC fallback)
 ************************************************************************/

#ifndef FLAGCX_KUNLUNXIN_COMM_TRAITS_H_
#define FLAGCX_KUNLUNXIN_COMM_TRAITS_H_

#if defined(FLAGCX_COMM_TRAITS_SHMEM)
#include "xshmem_comm_traits.h"
#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = CommTraits<XshmemBackend>;

#else
#include "default_comm_traits.h"
using DeviceAPI = CommTraits<DefaultBackend<KunlunxinPlatform>>;

#endif

#endif
