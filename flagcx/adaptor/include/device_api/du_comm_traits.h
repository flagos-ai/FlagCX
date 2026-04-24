/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Du Vendor Comm Traits.
 ************************************************************************/

#ifndef FLAGCX_DU_COMM_TRAITS_H_
#define FLAGCX_DU_COMM_TRAITS_H_

// ============================================================
// DU Default Backend (IPC barriers + FIFO one-sided)
// Uses common Default<> partial specialization with DU platform
// ============================================================
#include "default_comm_traits.h"

using DeviceAPI = CommTraits<Default<DuPlatform>>;

#endif // FLAGCX_DU_COMM_TRAITS_H_
