/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Syncs the NVSHMEM device state into the consumer binary's own
 * __constant__ nvshmemi_device_state_d symbol.
 *
 * Mechanism: calling nvshmem_init() from the consumer binary's device-link
 * scope triggers the NVSHMEM callback (nvshmemi_get_mem_handle) which
 * registers this binary's __constant__ address with the host library.
 * Since NVSHMEM is already initialized (by libflagcx.so), the re-entrant
 * path in nvshmemid_hostlib_init_attr just registers the new device state
 * and calls nvshmemi_update_device_state() to populate it.
 *
 * Must be called AFTER flagcxDevCommCreate (which triggers nvshmem_init
 * inside libflagcx.so) and BEFORE any kernel launch that uses NVSHMEM
 * device functions.
 ************************************************************************/

#include <nvshmem.h>

extern "C" void flagcxNvshmemSyncDeviceState() {
  // Re-entrant: NVSHMEM is already initialized by libflagcx.so.
  // This call registers the consumer binary's nvshmemi_device_state_d
  // and populates it via nvshmemi_update_device_state().
  nvshmem_init();
}

extern "C" void flagcxNvshmemFinalizeDeviceState() {
  // Decrement refcount. Actual teardown happens in libflagcx.so's finalize.
  nvshmem_finalize();
}
