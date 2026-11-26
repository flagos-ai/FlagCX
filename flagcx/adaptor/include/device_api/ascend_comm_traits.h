/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ASCEND_COMM_TRAITS_H_
#define FLAGCX_ASCEND_COMM_TRAITS_H_

#if defined(FLAGCX_COMM_TRAITS_SHMEM)
#include "aclshmem_comm_traits.h"
#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = CommTraits<AclshmemBackend>;

#else
#include "default_comm_traits.h"
using DeviceAPI = CommTraits<DefaultBackend<AscendPlatform>>;

#endif

#endif // FLAGCX_ASCEND_COMM_TRAITS_H_
