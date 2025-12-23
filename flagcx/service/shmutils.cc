/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "shmutils.h"
#include "check.h"
#include "debug.h"
#include "utils.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

struct shmHandleInternal {
  int fd;
  char *shmPath;
  char *shmPtr;
  void *devShmPtr; // not used in FlagCX
  size_t shmSize;
  size_t realShmSize;
  int *refcount;
};

static void shmHandleInit(int fd, char *shmPath, size_t shmSize,
                          size_t realShmSize, char *hptr, void *dptr,
                          bool create, struct shmHandleInternal *handle) {
  handle->fd = fd;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int *)(hptr + shmSize) : NULL;
  if (create) {
    int slen = strlen(shmPath);
    handle->shmPath = (char *)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    if (hptr)
      memset(hptr, 0, shmSize);
  } else {
    handle->shmPath = NULL;
  }
  return;
}

flagcxResult_t flagcxShmOpen(char *shmPath, size_t shmPathSize, size_t shmSize,
                             void **shmPtr, void **devShmPtr, int refcount,
                             flagcxShmHandle_t *handle) {
  int fd = -1;
  char *hptr = NULL;
  void *dptr = NULL;
  flagcxResult_t ret = flagcxSuccess;
  struct shmHandleInternal *tmphandle;
  bool create = refcount > 0;
  const size_t refSize =
      sizeof(int); /* extra sizeof(int) bytes for reference count */
  const size_t realShmSize = shmSize + refSize;

  *handle = *shmPtr =
      NULL; /* assume shmPtr and handle always set correctly by users. */
  EQCHECKGOTO(tmphandle = (struct shmHandleInternal *)calloc(
                  1, sizeof(struct shmHandleInternal)),
              NULL, ret, fail);
  if (create) {
    /* refcount > 0 means the caller tries to allocate a shared memory. This
     * shared memory segment will have refcount references; when the peer
     * attaches, it should pass -1 to reduce one reference count. When it goes
     * down to 0, unlink should be called in order to delete shared memory file.
     */
    if (shmPath[0] == '\0') {
      snprintf(shmPath, shmPathSize, "/dev/shm/flagcx-XXXXXX");
      fd = mkstemp(shmPath);
    } else {
      SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), ret,
                   fail);
    }

    if (fallocate(fd, 0, 0, realShmSize) != 0) {
      WARN("Error: failed to extend %s to %ld bytes", shmPath, realShmSize);
      ret = flagcxSystemError;
      goto fail;
    }
    INFO(FLAGCX_ALLOC, "Allocated %ld bytes of shared memory in %s",
         realShmSize, shmPath);
  } else {
    SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), ret, fail);
  }

  hptr = (char *)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                      0);
  if (hptr == MAP_FAILED) {
    WARN("Could not map %s size %zi, error: %s", shmPath, realShmSize,
         strerror(errno));
    ret = flagcxSystemError;
    hptr = NULL;
    goto fail;
  }

  if (create) {
    *(int *)(hptr + shmSize) = refcount;
  } else {
    int remref = flagcxAtomicRefCountDecrement((int *)(hptr + shmSize));
    if (remref == 0) {
      /* the last peer has completed attachment, it should unlink the shm mem
       * file. */
      if (unlink(shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s", shmPath,
             strerror(errno));
      }
    }
  }

  // devShmPtr is not used in FlagCX
  shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create,
                tmphandle);
exit:
  *shmPtr = hptr;
  // if (devShmPtr) *devShmPtr = dptr;
  *handle = (flagcxShmHandle_t)tmphandle;
  return ret;
fail:
  WARN("Error while %s shared memory segment %s (size %ld)",
       create ? "creating" : "attaching to", shmPath, shmSize);
  if (tmphandle) {
    shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create,
                  tmphandle);
    flagcxShmClose((flagcxShmHandle_t)tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;
}

flagcxResult_t flagcxShmClose(flagcxShmHandle_t handle) {
  flagcxResult_t ret = flagcxSuccess;
  struct shmHandleInternal *tmphandle = (struct shmHandleInternal *)handle;
  if (tmphandle) {
    if (tmphandle->fd >= 0) {
      close(tmphandle->fd);
      if (tmphandle->shmPath != NULL &&
          (tmphandle->shmPtr == NULL ||
           (tmphandle->refcount != NULL && *tmphandle->refcount > 0))) {
        if (unlink(tmphandle->shmPath) != 0) {
          WARN("unlink shared memory %s failed, error: %s", tmphandle->shmPath,
               strerror(errno));
          ret = flagcxSystemError;
        }
      }
      free(tmphandle->shmPath);
    }

    if (tmphandle->shmPtr) {
      // if (tmphandle->devShmPtr)
      // CUDACHECK(cudaHostUnregister(tmphandle->shmPtr));
      if (munmap(tmphandle->shmPtr, tmphandle->realShmSize) != 0) {
        WARN("munmap of shared memory %p size %ld failed, error: %s",
             tmphandle->shmPtr, tmphandle->realShmSize, strerror(errno));
        ret = flagcxSystemError;
      }
    }
    free(tmphandle);
  }
  return ret;
}

flagcxResult_t flagcxShmUnlink(flagcxShmHandle_t handle) {
  flagcxResult_t ret = flagcxSuccess;
  struct shmHandleInternal *tmphandle = (struct shmHandleInternal *)handle;
  if (tmphandle) {
    if (tmphandle->shmPath != NULL) {
      if (unlink(tmphandle->shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s", tmphandle->shmPath,
             strerror(errno));
        ret = flagcxSystemError;
      }
      free(tmphandle->shmPath);
      tmphandle->shmPath = NULL;
    }
  }
  return ret;
}

flagcxResult_t flagcxShmAllocateShareableBuffer(size_t size,
                                                flagcxShmIpcDesc_t *desc,
                                                void **hptr, void **dptr) {
  if (desc == NULL || hptr == NULL) {
    WARN("Invalid argument desc %p, hptr %p", desc, hptr);
    return flagcxInvalidArgument;
  }
  char shmPath[SHM_PATH_MAX] = {'\0'};
  desc->shmSize = size;
  FLAGCXCHECK(flagcxShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1,
                            &desc->handle));
  memcpy(desc->shmSuffix, shmPath + sizeof("/dev/shm/flagcx-") - 1,
         sizeof(desc->shmSuffix));
  INFO(FLAGCX_ALLOC, "MMAP allocated shareable host buffer %s size %zi ptr %p",
       shmPath, size, *hptr);
  return flagcxSuccess;
}

flagcxResult_t flagcxShmImportShareableBuffer(flagcxShmIpcDesc_t *desc,
                                              void **hptr, void **dptr,
                                              flagcxShmIpcDesc_t *descOut) {
  if (desc == NULL || hptr == NULL || descOut == NULL) {
    WARN("Invalid argument desc %p, hptr %p, descOut %p", desc, hptr, descOut);
    return flagcxInvalidArgument;
  }
  char shmPath[SHM_PATH_MAX];
  snprintf(shmPath, sizeof(shmPath), "/dev/shm/flagcx-%s", desc->shmSuffix);
  FLAGCXCHECK(flagcxShmOpen(shmPath, sizeof(shmPath), desc->shmSize, hptr, dptr,
                            -1, &descOut->handle));
  INFO(FLAGCX_ALLOC, "MMAP imported shareable host buffer %s size %zi ptr %p",
       shmPath, desc->shmSize, *hptr);
  return flagcxSuccess;
}

flagcxResult_t flagcxShmIpcClose(flagcxShmIpcDesc_t *desc) {
  if (desc) {
    FLAGCXCHECK(flagcxShmClose(desc->handle));
  }
  return flagcxSuccess;
}
