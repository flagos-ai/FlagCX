/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Du Device Traits — macro-based backend dispatch.
 *
 * Selects the active CommTraits specialization based on build-time macros:
 *   FLAGCX_COMM_TRAITS_SHMEM   → DushmemBackend (DUSHMEM, USE_SHMEM=1)
 *   FLAGCX_COMM_TRAITS_DEFAULT → DefaultBackend<DuPlatform> (IPC + one-sided)
 ************************************************************************/

#ifndef FLAGCX_DU_COMM_TRAITS_H_
#define FLAGCX_DU_COMM_TRAITS_H_

#if defined(FLAGCX_COMM_TRAITS_SHMEM)
#include "dushmem_comm_traits.h"
#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = CommTraits<DushmemBackend>;

#else // FLAGCX_COMM_TRAITS_DEFAULT
// ============================================================
// DU Default Backend (IPC barriers + FIFO one-sided)
// Uses common DefaultBackend<> partial specialization with DU platform
// ============================================================
#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<DuPlatform>>;

#endif

#endif // FLAGCX_DU_COMM_TRAITS_H_
