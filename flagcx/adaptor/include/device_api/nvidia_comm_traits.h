/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVIDIA Device Traits — macro-based backend dispatch.
 *
 * Selects the active CommTraits specialization based on build-time macros:
 *   FLAGCX_COMM_TRAITS_SHMEM   → NvshmemBackend (NVSHMEM)
 *   FLAGCX_COMM_TRAITS_CCL     → NcclBackend (NCCL device API)
 *   FLAGCX_COMM_TRAITS_DEFAULT → DefaultBackend<NvidiaPlatform> (IPC +
 *one-sided)
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_DEVICE_TRAITS_H_
#define FLAGCX_NVIDIA_DEVICE_TRAITS_H_

#if defined(FLAGCX_COMM_TRAITS_SHMEM)
#include "nvshmem_comm_traits.h"
#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = CommTraits<NvshmemBackend>;

#elif defined(FLAGCX_COMM_TRAITS_CCL)
#include "nccl_comm_traits.h"
// nccl_comm_traits.h defines FLAGCX_DEVICE_API_VENDOR and DeviceAPI
// if NCCL version is sufficient; otherwise falls back to default internally.

#else // FLAGCX_COMM_TRAITS_DEFAULT
#include "default_comm_traits.h"
using DeviceAPI = CommTraits<DefaultBackend<NvidiaPlatform>>;

#endif

#endif // FLAGCX_NVIDIA_DEVICE_TRAITS_H_
