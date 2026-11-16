/* MUSA build of the shared FlagCX Unified IR device kernels.
 *
 * Same reuse strategy as device_api.mu: the implementation is the shared
 * ../nvidia/device_ir.cu, with the MUSA adaptor providing the device-side
 * wrapper types.
 */
#include "musa_adaptor.h"

#include "../nvidia/device_ir.cu"
