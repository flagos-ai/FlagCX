/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ILUVATAR_COMM_TRAITS_H_
#define FLAGCX_ILUVATAR_COMM_TRAITS_H_

#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<IluvatarPlatform>>;

#endif // FLAGCX_ILUVATAR_COMM_TRAITS_H_
