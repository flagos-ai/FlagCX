/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Du Vendor Device Traits.
 ************************************************************************/

#ifndef FLAGCX_DU_DEVICE_TRAITS_H_
#define FLAGCX_DU_DEVICE_TRAITS_H_

// ============================================================
// DU Fallback Backend (IPC barriers + FIFO one-sided)
// Uses common Fallback<> partial specialization with DU platform
// ============================================================
#include "fallback_device_traits.h"

using DeviceAPI = DeviceTraits<Fallback<DuPlatform>>;

#endif // FLAGCX_DU_DEVICE_TRAITS_H_
