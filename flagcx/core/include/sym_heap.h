/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Symmetric memory helpers for the non-homo (fallback) path.
 * Called from flagcx.cc when flagcxCommWindowRegister/Deregister/Grow
 * is invoked on the non-homo path with FLAGCX_WIN_COLL_SYMMETRIC.
 ************************************************************************/

#ifndef FLAGCX_SYM_HEAP_H_
#define FLAGCX_SYM_HEAP_H_

#include "flagcx.h"

flagcxResult_t flagcxSymWindowRegister(flagcxComm_t comm, void *buff,
                                       size_t size, flagcxSymWindow_t *win,
                                       int winFlags);

flagcxResult_t flagcxSymWindowDeregister(flagcxComm_t comm,
                                         flagcxSymWindow_t win);

flagcxResult_t flagcxSymWindowGrow(flagcxComm_t comm, flagcxSymWindow_t win,
                                   void *newBuff, size_t newSize);

#endif // FLAGCX_SYM_HEAP_H_
