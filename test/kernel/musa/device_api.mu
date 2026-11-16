/* MUSA build of the shared FlagCX Device API test kernels.
 *
 * The kernel bodies live in ../nvidia/device_api.cu and are platform-neutral.
 * Include the MUSA adaptor first so the device-side types that the shared
 * header expects (flagcxStream and friends) exist before it is parsed; the
 * NVIDIA adaptor header is empty when USE_NVIDIA_ADAPTOR is undefined.
 */
#include "musa_adaptor.h"
#include "cuda_runtime.h"

#include "../nvidia/device_api.cu"
