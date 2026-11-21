/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * MUSA platform binding for the Default backend selected by musa.mk.
 ************************************************************************/

#ifndef FLAGCX_MUSA_COMM_TRAITS_H_
#define FLAGCX_MUSA_COMM_TRAITS_H_

#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<MusaPlatform>>;

#endif // FLAGCX_MUSA_COMM_TRAITS_H_
