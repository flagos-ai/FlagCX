// MUSA uses the common UniRunner executor with the MUSA Device API.
#include "../common/flagcx_reduce_kernel_device_impl.h"

static_assert(sizeof(DeviceAPI::CompletionWord) == sizeof(uint64_t),
              "MUSA collective FIFO requires matching 64-bit host controls");
