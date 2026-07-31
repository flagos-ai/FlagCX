/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Sunrise Vendor Comm Traits.
 ************************************************************************/

#ifndef FLAGCX_SUNRISE_COMM_TRAITS_H_
#define FLAGCX_SUNRISE_COMM_TRAITS_H_

// Sunrise default backend: reuse DefaultBackend<DefaultPlatform> (IPC barriers
// + FIFO one-sided). PCCL/PTPU already provide collectives, so FlagCX needs no
// SIMT kernels of its own.
#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<DefaultPlatform>>;

#endif // FLAGCX_SUNRISE_COMM_TRAITS_H_
