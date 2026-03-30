/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Du Vendor Comm Traits.
 ************************************************************************/

#ifndef FLAGCX_DU_COMM_TRAITS_H_
#define FLAGCX_DU_COMM_TRAITS_H_

// ============================================================
// DU Fallback Backend (IPC barriers + FIFO one-sided)
// Uses common Fallback<> partial specialization with DU platform
// ============================================================
#include "fallback_comm_traits.h"

using DeviceAPI = CommTraits<Fallback<DuPlatform>>;

#endif // FLAGCX_DU_COMM_TRAITS_H_
