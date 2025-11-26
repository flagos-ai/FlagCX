/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_SHMUTILS_H_
#define FLAGCX_SHMUTILS_H_

#include "flagcx.h"

#define SHM_PATH_MAX 128

typedef void *flagcxShmHandle_t;
flagcxResult_t flagcxShmOpen(char *shmPath, size_t shmPathSize, size_t shmSize,
                             void **shmPtr, void **devShmPtr, int refcount,
                             flagcxShmHandle_t *handle);
flagcxResult_t flagcxShmClose(flagcxShmHandle_t handle);
flagcxResult_t flagcxShmUnlink(flagcxShmHandle_t handle);

struct shmIpcDesc {
  char shmSuffix[7];
  flagcxShmHandle_t handle;
  size_t shmSize;
};
typedef struct shmIpcDesc flagcxShmIpcDesc_t;

flagcxResult_t flagcxShmAllocateShareableBuffer(size_t size,
                                                flagcxShmIpcDesc_t *descOut,
                                                void **hptr, void **dptr);
flagcxResult_t flagcxShmImportShareableBuffer(flagcxShmIpcDesc_t *desc,
                                              void **hptr, void **dptr,
                                              flagcxShmIpcDesc_t *descOut);
flagcxResult_t flagcxShmIpcClose(flagcxShmIpcDesc_t *desc);

#endif
