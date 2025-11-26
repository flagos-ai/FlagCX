/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_SOCKET_H_
#define FLAGCX_SOCKET_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "type.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT 1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES                                                    \
  2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES                                                   \
  3 // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define FLAGCX_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union flagcxSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum flagcxSocketState {
  flagcxSocketStateNone = 0,
  flagcxSocketStateInitialized = 1,
  flagcxSocketStateAccepting = 2,
  flagcxSocketStateAccepted = 3,
  flagcxSocketStateConnecting = 4,
  flagcxSocketStateConnectPolling = 5,
  flagcxSocketStateConnected = 6,
  flagcxSocketStateReady = 7,
  flagcxSocketStateClosed = 8,
  flagcxSocketStateError = 9,
  flagcxSocketStateNum = 10
};

enum flagcxSocketType {
  flagcxSocketTypeUnknown = 0,
  flagcxSocketTypeBootstrap = 1,
  flagcxSocketTypeProxy = 2,
  flagcxSocketTypeNetSocket = 3,
  flagcxSocketTypeNetIb = 4
};

struct flagcxSocket {
  int fd;
  int acceptFd;
  int timedOutRetries;
  int refusedRetries;
  union flagcxSocketAddress addr;
  volatile uint32_t *abortFlag;
  int asyncFlag;
  enum flagcxSocketState state;
  int salen;
  uint64_t magic;
  enum flagcxSocketType type;
};

const char *flagcxSocketToString(union flagcxSocketAddress *addr, char *buf,
                                 const int numericHostForm = 1);
flagcxResult_t flagcxSocketGetAddrFromString(union flagcxSocketAddress *ua,
                                             const char *ip_port_pair);
int flagcxFindInterfaceMatchSubnet(char *ifNames,
                                   union flagcxSocketAddress *localAddrs,
                                   union flagcxSocketAddress *remoteAddr,
                                   int ifNameMaxSize, int maxIfs);
int flagcxFindInterfaces(char *ifNames, union flagcxSocketAddress *ifAddrs,
                         int ifNameMaxSize, int maxIfs);

// Initialize a socket
flagcxResult_t
flagcxSocketInit(struct flagcxSocket *sock,
                 union flagcxSocketAddress *addr = NULL,
                 uint64_t magic = FLAGCX_SOCKET_MAGIC,
                 enum flagcxSocketType type = flagcxSocketTypeUnknown,
                 volatile uint32_t *abortFlag = NULL, int asyncFlag = 0);
// Create a listening socket. sock->addr can be pre-filled with IP & port info.
// sock->fd is set after a successful call
flagcxResult_t flagcxSocketListen(struct flagcxSocket *sock);
flagcxResult_t flagcxSocketGetAddr(struct flagcxSocket *sock,
                                   union flagcxSocketAddress *addr);
// Connect to sock->addr. sock->fd is set after a successful call.
flagcxResult_t flagcxSocketConnect(struct flagcxSocket *sock);
// Return socket connection state.
flagcxResult_t flagcxSocketReady(struct flagcxSocket *sock, int *running);
// Accept an incoming connection from listenSock->fd and keep the file
// descriptor in sock->fd, with the remote side IP/port in sock->addr.
flagcxResult_t flagcxSocketAccept(struct flagcxSocket *sock,
                                  struct flagcxSocket *ulistenSock);
flagcxResult_t flagcxSocketGetFd(struct flagcxSocket *sock, int *fd);
flagcxResult_t flagcxSocketSetFd(int fd, struct flagcxSocket *sock);

#define FLAGCX_SOCKET_SEND 0
#define FLAGCX_SOCKET_RECV 1

flagcxResult_t flagcxSocketProgress(int op, struct flagcxSocket *sock,
                                    void *ptr, int size, int *offset);
flagcxResult_t flagcxSocketWait(int op, struct flagcxSocket *sock, void *ptr,
                                int size, int *offset);
flagcxResult_t flagcxSocketSend(struct flagcxSocket *sock, void *ptr, int size);
flagcxResult_t flagcxSocketRecv(struct flagcxSocket *sock, void *ptr, int size);
flagcxResult_t flagcxSocketSendRecv(struct flagcxSocket *sendSock,
                                    void *sendPtr, int sendSize,
                                    struct flagcxSocket *recvSock,
                                    void *recvPtr, int recvSize);
flagcxResult_t flagcxSocketTryRecv(struct flagcxSocket *sock, void *ptr,
                                   int size, int *closed, bool blocking);
flagcxResult_t flagcxSocketClose(struct flagcxSocket *sock);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
