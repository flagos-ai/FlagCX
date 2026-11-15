#ifndef FLAGCX_PPU_COMM_TRAITS_H_
#define FLAGCX_PPU_COMM_TRAITS_H_

#include "default_comm_traits.h"

using DeviceAPI = CommTraits<DefaultBackend<PpuPlatform>>;

#endif // FLAGCX_PPU_COMM_TRAITS_H_
