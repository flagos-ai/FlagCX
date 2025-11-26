/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_IPCSOCKET_H
#define FLAGCX_IPCSOCKET_H

#include "core.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <memory.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#define FLAGCX_IPC_SOCKNAME_LEN 64

struct flagcxIpcSocket {
  int fd;
  char socketName[FLAGCX_IPC_SOCKNAME_LEN];
  volatile uint32_t *abortFlag;
};

flagcxResult_t flagcxIpcSocketInit(struct flagcxIpcSocket *handle, int rank,
                                   uint64_t hash, int block);
flagcxResult_t flagcxIpcSocketClose(struct flagcxIpcSocket *handle);
flagcxResult_t flagcxIpcSocketGetFd(struct flagcxIpcSocket *handle, int *fd);

flagcxResult_t flagcxIpcSocketRecvFd(struct flagcxIpcSocket *handle, int *fd);
flagcxResult_t flagcxIpcSocketSendFd(struct flagcxIpcSocket *handle,
                                     const int fd, int rank, uint64_t hash);

flagcxResult_t flagcxIpcSocketSendMsg(flagcxIpcSocket *handle, void *hdr,
                                      int hdrLen, const int sendFd, int rank,
                                      uint64_t hash);
flagcxResult_t flagcxIpcSocketRecvMsg(flagcxIpcSocket *handle, void *hdr,
                                      int hdrLen, int *recvFd);

#endif /* FLAGCX_IPCSOCKET_H */