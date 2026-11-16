/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * MUSA build of the FlagCX reduce collective kernel.
 *
 * The implementation is the shared, platform-neutral device source. It only
 * relies on DeviceAPI plus the threadIdx / __syncthreads / fence macros, all of
 * which the MUSA branch of device_utils.h provides.
 ************************************************************************/

#include "musa_adaptor.h"

#ifdef USE_MUSA_ADAPTOR

#include "../nvidia/flagcx_reduce_kernel_device.cu"

#endif // USE_MUSA_ADAPTOR
