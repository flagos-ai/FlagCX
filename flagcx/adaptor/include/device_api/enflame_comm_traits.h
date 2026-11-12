/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Enflame Device Traits — selects the ECCL backend.
 *
 *   FLAGCX_COMM_TRAITS_CCL → EcclBackend (ECCL device API)
 *
 * ECCL is the only backend Enflame supports, so this header has nothing to
 * dispatch: it exists to keep the per-platform selector shape that
 * comm_traits.h relies on (one header per vendor that ends up defining
 * DeviceAPI), and to hold the reasoning for why there is no choice to make.
 *
 * Enflame provides neither a SHMEM backend nor a usable default backend:
 *   - SHMEM: nothing on TOPS exposes the symmetric-heap / IPC peer-mapping
 *     layer the default backend is written against.
 *   - DEFAULT: the portable backend's one-sided transport and barrier path are
 *     untested on GCU and would have to be revalidated before it could be
 *     considered a bring-up fallback.
 * If either is ever needed, add the branch here plus the matching entry in
 * makefiles/enflame.mk — do not resurrect a silent fallback, because a Device
 * API build that quietly lands on an unvalidated backend fails far from the
 * cause.
 ************************************************************************/

#ifndef FLAGCX_ENFLAME_DEVICE_TRAITS_H_
#define FLAGCX_ENFLAME_DEVICE_TRAITS_H_

#include "eccl_comm_traits.h"
// eccl_comm_traits.h defines FLAGCX_DEVICE_API_VENDOR and DeviceAPI.

#endif // FLAGCX_ENFLAME_DEVICE_TRAITS_H_
