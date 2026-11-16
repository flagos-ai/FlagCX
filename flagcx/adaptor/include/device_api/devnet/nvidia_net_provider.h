/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVIDIA resources for the shared DefaultBackend Net provider contract.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_NET_PROVIDER_H_
#define FLAGCX_NVIDIA_NET_PROVIDER_H_

#include "default_net_gda_policy.h"

struct NvidiaPlatform;

// The device-view layout is shared with future GDA-enabled platforms. The
// NVIDIA operation implementation below remains specific to its device ISA.
template <>
struct DefaultNetProviderPolicy<NvidiaPlatform> : DefaultNetGdaPolicy {};

#if defined(FLAGCX_USE_XPU_NETIO)
#include "nvidia_xpu_netio_gda.h"
#endif

#endif // FLAGCX_NVIDIA_NET_PROVIDER_H_
