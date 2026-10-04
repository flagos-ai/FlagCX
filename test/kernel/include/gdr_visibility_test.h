/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_GDR_VISIBILITY_TEST_H_
#define FLAGCX_GDR_VISIBILITY_TEST_H_

#include "flagcx.h"

#include <stddef.h>
#include <stdint.h>

// status[0] is set to one on any mismatch. status[1] counts participating
// threads so the host can distinguish a clean kernel run from no execution.
extern "C" flagcxResult_t
flagcxTestLaunchGdrVisibilityConsumer(const void *data, size_t size,
                                      uint8_t expected, int *status,
                                      flagcxStream_t stream);

#endif // FLAGCX_GDR_VISIBILITY_TEST_H_
