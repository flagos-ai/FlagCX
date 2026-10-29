/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Syncs the DUSHMEM device state into the consumer binary's own
 * __constant__ dushmemi_device_state_d symbol.
 *
 * This intentionally mirrors nvshmem_state_sync.cc. Calling dushmem_init()
 * from the consumer binary should register and populate this binary's device
 * state through DUSHMEM's re-entrant initialization path.
 *
 * Must be called AFTER flagcxDevCommCreate and BEFORE any kernel launch that
 * uses DUSHMEM device functions.
 ************************************************************************/

#include <dushmem.h>

extern "C" void flagcxDushmemSyncDeviceState() {
  dushmem_init();
}

extern "C" void flagcxDushmemFinalizeDeviceState() {
  dushmem_finalize();
}
