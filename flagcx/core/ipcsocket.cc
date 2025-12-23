/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "ipcsocket.h"
#include "utils.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

// Enable Linux abstract socket naming
#define USE_ABSTRACT_SOCKET

#define FLAGCX_IPC_SOCKNAME_STR "/tmp/flagcx-socket-%d-%lx"

/*
 * Create a Unix Domain Socket
 */
flagcxResult_t flagcxIpcSocketInit(flagcxIpcSocket *handle, int rank,
                                   uint64_t hash, int block) {
  int fd = -1;
  struct sockaddr_un cliaddr;
  char temp[FLAGCX_IPC_SOCKNAME_LEN] = "";

  if (handle == NULL) {
    return flagcxInternalError;
  }

  handle->fd = -1;
  handle->socketName[0] = '\0';
  if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
    WARN("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
    return flagcxSystemError;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  int len = snprintf(temp, FLAGCX_IPC_SOCKNAME_LEN, FLAGCX_IPC_SOCKNAME_STR,
                     rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot bind provided name to socket. Name too large");
    return flagcxInternalError;
  }
#ifndef USE_ABSTRACT_SOCKET
  unlink(temp);
#endif

  INFO(FLAGCX_NET, "UDS: Creating socket %s", temp);

  strncpy(cliaddr.sun_path, temp, len);
#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif
  if (bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    WARN("UDS: Binding to socket %s failed : %s (%d)", temp, strerror(errno),
         errno);
    close(fd);
    return flagcxSystemError;
  }

  handle->fd = fd;
  strcpy(handle->socketName, temp);

  if (!block) {
    int flags;
    EQCHECK(flags = fcntl(fd, F_GETFL), -1);
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIpcSocketGetFd(struct flagcxIpcSocket *handle, int *fd) {
  if (handle == NULL) {
    WARN("flagcxSocketGetFd: pass NULL socket");
    return flagcxInvalidArgument;
  }
  if (fd)
    *fd = handle->fd;
  return flagcxSuccess;
}

flagcxResult_t flagcxIpcSocketClose(flagcxIpcSocket *handle) {
  if (handle == NULL) {
    return flagcxInternalError;
  }
  if (handle->fd <= 0) {
    return flagcxSuccess;
  }
#ifndef USE_ABSTRACT_SOCKET
  if (handle->socketName[0] != '\0') {
    unlink(handle->socketName);
  }
#endif
  close(handle->fd);

  return flagcxSuccess;
}

flagcxResult_t flagcxIpcSocketRecvMsg(flagcxIpcSocket *handle, void *hdr,
                                      int hdrLen, int *recvFd) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  if (hdr == NULL) {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Receiving data over socket failed : %d", errno);
      return flagcxSystemError;
    }
    if (handle->abortFlag &&
        __atomic_load_n(handle->abortFlag, __ATOMIC_RELAXED))
      return flagcxInternalError;
  }

  if (recvFd != NULL) {
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) ||
          (cmptr->cmsg_type != SCM_RIGHTS)) {
        WARN("UDS: Receiving data over socket failed");
        return flagcxSystemError;
      }

      memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
    } else {
      WARN("UDS: Receiving data over socket %s failed", handle->socketName);
      return flagcxSystemError;
    }
    TRACE(FLAGCX_INIT | FLAGCX_P2P, "UDS: Got recvFd %d from socket %s",
          *recvFd, handle->socketName);
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIpcSocketRecvFd(flagcxIpcSocket *handle, int *recvFd) {
  return flagcxIpcSocketRecvMsg(handle, NULL, 0, recvFd);
}

flagcxResult_t flagcxIpcSocketSendMsg(flagcxIpcSocket *handle, void *hdr,
                                      int hdrLen, const int sendFd, int rank,
                                      uint64_t hash) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];
  char temp[FLAGCX_IPC_SOCKNAME_LEN];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  int len = snprintf(temp, FLAGCX_IPC_SOCKNAME_LEN, FLAGCX_IPC_SOCKNAME_STR,
                     rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot connect to provided name for socket. Name too large");
    return flagcxInternalError;
  }
  (void)strncpy(cliaddr.sun_path, temp, len);

#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif

  TRACE(FLAGCX_INIT, "UDS: Sending hdr %p len %d to UDS socket %s", hdr, hdrLen,
        temp);

  if (sendFd != -1) {
    TRACE(FLAGCX_INIT, "UDS: Sending fd %d to UDS socket %s", sendFd, temp);

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
  }

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  if (hdr == NULL) {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Sending data over socket %s failed : %s (%d)", temp,
           strerror(errno), errno);
      return flagcxSystemError;
    }
    if (handle->abortFlag &&
        __atomic_load_n(handle->abortFlag, __ATOMIC_RELAXED))
      return flagcxInternalError;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIpcSocketSendFd(flagcxIpcSocket *handle, const int sendFd,
                                     int rank, uint64_t hash) {
  return flagcxIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}
