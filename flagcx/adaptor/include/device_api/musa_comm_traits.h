/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * MUSA Comm Traits — binds the shared IPC DefaultBackend to MusaPlatform.
 ************************************************************************/

#ifndef FLAGCX_MUSA_COMM_TRAITS_H_
#define FLAGCX_MUSA_COMM_TRAITS_H_

#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<MusaPlatform>>;

#endif // FLAGCX_MUSA_COMM_TRAITS_H_
