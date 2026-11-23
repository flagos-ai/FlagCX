/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_METAX_COMM_TRAITS_H_
#define FLAGCX_METAX_COMM_TRAITS_H_

#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<MetaxPlatform>>;

#endif // FLAGCX_METAX_COMM_TRAITS_H_
