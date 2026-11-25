/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#include "probe.h"

#include "adaptor.h"

// Keep the real allocator behind a verbs-free interface shared by both
// workers. Native verbs never consumes FlagCX's embedded verbs declarations.
flagcxResult_t probeGdrAlloc(void **ptr, size_t bytes) {
  return macaAdaptor.gdrMemAlloc(ptr, bytes, nullptr);
}

flagcxResult_t probeGdrFree(void *ptr) {
  return macaAdaptor.gdrMemFree(ptr, nullptr);
}
