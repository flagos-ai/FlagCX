/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ASCEND_COMM_TRAITS_H_
#define FLAGCX_ASCEND_COMM_TRAITS_H_

#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<AscendPlatform>>;

#endif // FLAGCX_ASCEND_COMM_TRAITS_H_
