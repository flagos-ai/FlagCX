/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_P2P_POINTER_H_
#define FLAGCX_P2P_POINTER_H_

#include "flagcx.h"
#include "flagcx_net.h"

#include <stdint.h>

#define FLAGCX_P2P_IPC_HANDLE_BYTES 64

// Internal P2P registration helper. Pointer classification is authoritative
// for upgraded adaptors; only loader-upgraded v1 plugins and built-ins carrying
// the explicit transitional flag may retain legacy IPC-export inference. IPC
// data is optional sharing metadata and is cleared when it cannot be exported.
flagcxResult_t flagcxP2pDetectPointerType(void *ptr, int *ptrType,
                                          char *ipcHandleBuf,
                                          uint32_t *ipcHandleSize);

#endif // FLAGCX_P2P_POINTER_H_
