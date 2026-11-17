#ifndef FLAGCX_MUSA_FIFO_ERRORS_H_
#define FLAGCX_MUSA_FIFO_ERRORS_H_

#include "flagcx.h"
#include <cstdint>

#ifdef USE_MUSA_ADAPTOR
// CPU addresses of pinned FIFO terminal words and per-Comm device error words.
// Unregister before freeing the owner. Errors are sticky, never cleared by a
// query; synchronization conservatively reports any error on the current GPU.
flagcxResult_t flagcxMusaRegisterFifoError(uint64_t *word);
void flagcxMusaUnregisterFifoError(uint64_t *word);
flagcxResult_t flagcxMusaCheckFifoErrors();
#endif

#endif
