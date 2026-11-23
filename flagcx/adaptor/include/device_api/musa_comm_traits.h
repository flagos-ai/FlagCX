/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * MUSA platform binding for the Default backend selected by musa.mk.
 ************************************************************************/

#ifndef FLAGCX_MUSA_COMM_TRAITS_H_
#define FLAGCX_MUSA_COMM_TRAITS_H_

#include "default_comm_traits.h"

// Conservative Proxy completion for diagnostic acceptance runs: wraps every
// DefaultNetDispatch proxy operation (Put/Get/PutValue/Signal/Reset) with a
// synchronous FIFO flush so a call does not return before the host proxy has
// completed it. MUSA-only; other vendors are untouched.
// Default 1 (on). Override with -DFLAGCX_MUSA_PROXY_COMPLETION_SYNC=0 to
// restore the plain asynchronous enqueue path for throughput runs.
#ifndef FLAGCX_MUSA_PROXY_COMPLETION_SYNC
#define FLAGCX_MUSA_PROXY_COMPLETION_SYNC 1
#endif
#if FLAGCX_MUSA_PROXY_COMPLETION_SYNC
// Must be included before the DeviceAPI alias below: the header carries the
// DefaultNetDispatch<MusaPlatform> specialization, which has to be declared
// before the first instantiation of the MUSA Default backend.
#include "musa_proxy_sync.h"
#endif

using DeviceAPI = CommTraits<DefaultBackend<MusaPlatform>>;

#endif // FLAGCX_MUSA_COMM_TRAITS_H_
