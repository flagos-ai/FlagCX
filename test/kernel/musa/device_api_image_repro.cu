// Device-only load reproducer. These images never replace device_api.mubin.
// 0: SDK only; 1: FlagCX headers; 2: typed arguments; 3: CommQueries body.
#ifndef FLAGCX_MUSA_IMAGE_REPRO_STAGE
#error "Select FLAGCX_MUSA_IMAGE_REPRO_STAGE=0..3"
#endif
#if FLAGCX_MUSA_IMAGE_REPRO_STAGE < 0 || FLAGCX_MUSA_IMAGE_REPRO_STAGE > 3
#error "Invalid image reproducer stage"
#endif

#if FLAGCX_MUSA_IMAGE_REPRO_STAGE >= 1
#include "device_api/flagcx_device.h"
#include "musa_adaptor.h"
#include "global_comm.h"
#include "flagcx_kernel.h"
#endif
#include <musa_runtime.h>

// Match the formal CommQueries kernel's 512-thread launch bound throughout.
// C linkage keeps the lookup symbol identical across all four images.
extern "C" __global__ void __launch_bounds__(512)
flagcxMusaImageRepro(
#if FLAGCX_MUSA_IMAGE_REPRO_STAGE >= 2
    flagcxDevMem devMem, flagcxDevComm devComm,
#else
    void *devMem, void *devComm,
#endif
    int *results) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
#if FLAGCX_MUSA_IMAGE_REPRO_STAGE == 3
    results[0] = devMem.hasWindow() ? 1 : 0;
    results[1] = devComm.getIntraRank();
    results[2] = devComm.getIntraSize();
    results[3] = devComm.getRank();
    results[4] = devComm.getSize();
    results[5] = devMem.getDevPeerPtrs() != nullptr ? 1 : 0;
#else
    results[0] = 1;
#endif
  }
}
