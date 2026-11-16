/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * PPU uses the CUDA-compatible device intrinsics and atomics implemented by
 * NvidiaPlatform. Keep a separate tag so the DefaultBackend provider and
 * completion policy can evolve independently of NVIDIA.
 ************************************************************************/

#ifndef FLAGCX_PPU_PLATFORM_TRAITS_H_
#define FLAGCX_PPU_PLATFORM_TRAITS_H_

#include "nvidia_platform_traits.h"

struct PpuPlatform {};

template <>
struct PlatformTraits<PpuPlatform> : PlatformTraits<NvidiaPlatform> {};

#endif // FLAGCX_PPU_PLATFORM_TRAITS_H_
