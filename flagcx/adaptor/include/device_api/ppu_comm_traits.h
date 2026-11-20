/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * PPU Device API: CUDA-compatible device operations with FlagCX's common
 * IPC/FIFO DefaultBackend. No NCCL device or NVIDIA GDA provider is selected.
 ************************************************************************/

#ifndef FLAGCX_PPU_COMM_TRAITS_H_
#define FLAGCX_PPU_COMM_TRAITS_H_

#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<PpuPlatform>>;

#endif // FLAGCX_PPU_COMM_TRAITS_H_
