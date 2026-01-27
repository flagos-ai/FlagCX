/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "socket.h"
#include "check.h"
#include "debug.h"
#include "param.h"
#include "utils.h"
#include <cstddef>
#include <stdlib.h>

#include <cerrno>
#include <ifaddrs.h>
#include <net/if.h>
#include <string.h>
#include <unistd.h>

static flagcxResult_t socketProgressOpt(int op, struct flagcxSocket *sock,
                                        void *ptr, int size, int *offset,
                                        int block, int *closed) {
  int bytes = 0;
  *closed = 0;
  char *data = (char *)ptr;
  char line[SOCKET_NAME_MAXLEN + 1];
  do {
    if (op == FLAGCX_SOCKET_RECV)
      bytes = recv(sock->fd, data + (*offset), size - (*offset),
                   block ? 0 : MSG_DONTWAIT);
    if (op == FLAGCX_SOCKET_SEND)
      bytes = send(sock->fd, data + (*offset), size - (*offset),
                   block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    if (op == FLAGCX_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return flagcxSuccess;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("socketProgressOpt: Call to recv from %s failed : %s",
             flagcxSocketToString(&sock->addr, line), strerror(errno));
        return flagcxRemoteError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
    if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED)) {
      INFO(FLAGCX_NET, "socketProgressOpt: abort called");
      return flagcxInternalError;
    }
  } while (bytes > 0 && (*offset) < size);
  return flagcxSuccess;
}

static flagcxResult_t socketProgress(int op, struct flagcxSocket *sock,
                                     void *ptr, int size, int *offset) {
  int closed;
  FLAGCXCHECK(
      socketProgressOpt(op, sock, ptr, size, offset, 0 /*block*/, &closed));
  if (closed) {
    char line[SOCKET_NAME_MAXLEN + 1];
    WARN("socketProgress: Connection closed by remote peer %s",
         flagcxSocketToString(&sock->addr, line, 0));
    return flagcxRemoteError;
  }
  return flagcxSuccess;
}

static flagcxResult_t socketWait(int op, struct flagcxSocket *sock, void *ptr,
                                 int size, int *offset) {
  while (*offset < size)
    FLAGCXCHECK(socketProgress(op, sock, ptr, size, offset));
  return flagcxSuccess;
}

/* Format a string representation of a (union flagcxSocketAddress *) socket
 * address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char *flagcxSocketToString(union flagcxSocketAddress *addr, char *buf,
                                 const int numericHostForm /*= 1*/) {
  if (buf == NULL || addr == NULL)
    return NULL;
  struct sockaddr *saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be
   * determined.)
   */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  (void)getnameinfo(saddr, sizeof(union flagcxSocketAddress), host, NI_MAXHOST,
                    service, NI_MAXSERV, flag);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

static uint16_t socketToPort(union flagcxSocketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port
                                           : addr->sin6.sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  const char *env = flagcxGetEnv("FLAGCX_SOCKET_FAMILY");
  if (env == NULL)
    return family;

  INFO(FLAGCX_ENV, "FLAGCX_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET; // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static int findInterfaces(const char *prefixList, char *names,
                          union flagcxSocketAddress *addrs, int sock_family,
                          int maxIfNameSize, int maxIfs) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot)
    prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact)
    prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL)
      continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    TRACE(FLAGCX_INIT | FLAGCX_NET, "Found interface %s:%s",
          interface->ifa_name,
          flagcxSocketToString((union flagcxSocketAddress *)interface->ifa_addr,
                               line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6 *sa = (struct sockaddr_in6 *)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr))
        continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^
          searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names + i * maxIfNameSize) == 0) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names + found * maxIfNameSize, interface->ifa_name,
              maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                      : sizeof(struct sockaddr_in6);
      memcpy(addrs + found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static bool matchSubnet(struct ifaddrs local_if,
                        union flagcxSocketAddress *remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in *local_addr = (struct sockaddr_in *)(local_if.ifa_addr);
    struct sockaddr_in *mask = (struct sockaddr_in *)(local_if.ifa_netmask);
    struct sockaddr_in &remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6 *local_addr =
        (struct sockaddr_in6 *)(local_if.ifa_addr);
    struct sockaddr_in6 *mask = (struct sockaddr_in6 *)(local_if.ifa_netmask);
    struct sockaddr_in6 &remote_addr = remote->sin6;
    struct in6_addr &local_in6 = local_addr->sin6_addr;
    struct in6_addr &mask_in6 = mask->sin6_addr;
    struct in6_addr &remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;                   // IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) { // Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they
    // are not in the same scope For Global type, this field is 0, so a
    // comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    WARN("Net : Unsupported address family type");
    return false;
  }
}

int flagcxFindInterfaceMatchSubnet(char *ifNames,
                                   union flagcxSocketAddress *localAddrs,
                                   union flagcxSocketAddress *remoteAddr,
                                   int ifNameMaxSize, int maxIfs) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  char line_a[SOCKET_NAME_MAXLEN + 1];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL)
      continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                    : sizeof(struct sockaddr_in6);
    memcpy(localAddrs + found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames + found * ifNameMaxSize, interface->ifa_name,
            ifNameMaxSize);

    TRACE(FLAGCX_INIT | FLAGCX_NET,
          "NET : Found interface %s:%s in the same subnet as remote address %s",
          interface->ifa_name, flagcxSocketToString(localAddrs + found, line),
          flagcxSocketToString(remoteAddr, line_a));
    found++;
    if (found == maxIfs)
      break;
  }

  if (found == 0) {
    WARN("Net : No interface found in the same subnet as remote address %s",
         flagcxSocketToString(remoteAddr, line_a));
  }
  freeifaddrs(interfaces);
  return found;
}

flagcxResult_t flagcxSocketGetAddrFromString(union flagcxSocketAddress *ua,
                                             const char *ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return flagcxInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return flagcxInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s",
           gai_strerror(rv));
      return flagcxInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in &sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET; // IPv4
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port); // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6 &sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;     // IPv6
      sin6.sin6_port = htons(ni.port); // port
      sin6.sin6_flowinfo = 0;          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      return flagcxInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%')
        j = i;
      if (ip_port_pair[i] == ']')
        break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return flagcxInvalidArgument;
    }
    bool global_scope =
        (j == -1 ? true
                 : false); // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);
    int port = atoi(port_str);
    if (!global_scope)
      strncpy(if_name, ip_port_pair + j + 1,
              i - j - 1); // If not global scope, we need the intf name

    struct sockaddr_in6 &sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                    // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr)); // IP address
    sin6.sin6_port = htons(port);                   // port
    sin6.sin6_flowinfo = 0; // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id =
        global_scope
            ? 0
            : if_nametoindex(
                  if_name); // 0 if global scope; intf index if link scope
  }
  return flagcxSuccess;
}

int flagcxFindInterfaces(char *ifNames, union flagcxSocketAddress *ifAddrs,
                         int ifNameMaxSize, int maxIfs) {
  static int shownIfName = 0;
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  const char *env = flagcxGetEnv("FLAGCX_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    INFO(FLAGCX_ENV, "FLAGCX_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0)
      INFO(FLAGCX_NET, "FLAGCX_SOCKET_IFNAME set to %s", env);
    nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize,
                          maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                          maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0) {
      const char *commId = flagcxGetEnv("FLAGCX_COMM_ID");
      if (commId && strlen(commId) > 1) {
        INFO(FLAGCX_ENV, "FLAGCX_COMM_ID set by environment to %s", commId);
        // Try to find interface that is in the same subnet as the IP in comm id
        union flagcxSocketAddress idAddr;
        flagcxSocketGetAddrFromString(&idAddr, commId);
        nIfs = flagcxFindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr,
                                              ifNameMaxSize, maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0)
      nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0)
      nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    if (nIfs == 0)
      nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                            maxIfs);
  }
  return nIfs;
}

flagcxResult_t flagcxSocketListen(struct flagcxSocket *sock) {
  if (sock == NULL) {
    WARN("flagcxSocketListen: pass NULL socket");
    return flagcxInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("flagcxSocketListen: file descriptor is -1");
    return flagcxInvalidArgument;
  }

  if (socketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                        sizeof(opt)),
             "setsockopt");
#else
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)),
             "setsockopt");
#endif
  }

  // addr port should be 0 (Any port)
  SYSCHECK(bind(sock->fd, &sock->addr.sa, sock->salen), "bind");

  /* Get the assigned Port */
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->fd, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  TRACE(FLAGCX_INIT | FLAGCX_NET, "Listening on socket %s",
        flagcxSocketToString(&sock->addr, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in
   * /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(sock->fd, 16384), "listen");
  sock->state = flagcxSocketStateReady;
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketGetAddr(struct flagcxSocket *sock,
                                   union flagcxSocketAddress *addr) {
  if (sock == NULL) {
    WARN("flagcxSocketGetAddr: pass NULL socket");
    return flagcxInvalidArgument;
  }
  if (sock->state != flagcxSocketStateReady)
    return flagcxInternalError;
  memcpy(addr, &sock->addr, sizeof(union flagcxSocketAddress));
  return flagcxSuccess;
}

static flagcxResult_t socketTryAccept(struct flagcxSocket *sock) {
  socklen_t socklen = sizeof(union flagcxSocketAddress);
  sock->fd = accept(sock->acceptFd, &sock->addr.sa, &socklen);
  if (sock->fd != -1) {
    sock->state = flagcxSocketStateAccepted;
  } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
    WARN("socketTryAccept: Accept failed: %s", strerror(errno));
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

static flagcxResult_t socketFinalizeAccept(struct flagcxSocket *sock) {
  uint64_t magic;
  enum flagcxSocketType type;
  int received = 0;
  const int one = 1;
  SYSCHECK(
      setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(int)),
      "setsockopt");

  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, sock, &magic,
                                   sizeof(magic), &received));
  if (received == 0)
    return flagcxSuccess;
  FLAGCXCHECK(
      socketWait(FLAGCX_SOCKET_RECV, sock, &magic, sizeof(magic), &received));
  if (magic != sock->magic) {
    WARN("socketFinalizeAccept: wrong magic %lx != %lx", magic, sock->magic);
    close(sock->fd);
    sock->fd = -1;
    // Ignore spurious connection and accept again
    sock->state = flagcxSocketStateAccepting;
    return flagcxSuccess;
  } else {
    received = 0;
    FLAGCXCHECK(
        socketWait(FLAGCX_SOCKET_RECV, sock, &type, sizeof(type), &received));
    if (type != sock->type) {
      WARN("socketFinalizeAccept: wrong type %d != %d", type, sock->type);
      sock->state = flagcxSocketStateError;
      close(sock->fd);
      sock->fd = -1;
      return flagcxInternalError;
    } else {
      sock->state = flagcxSocketStateReady;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t socketStartConnect(struct flagcxSocket *sock) {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->fd, &sock->addr.sa, sock->salen);

  if (ret == 0) {
    sock->state = flagcxSocketStateConnected;
    return flagcxSuccess;
  } else if (errno == EINPROGRESS) {
    sock->state = flagcxSocketStateConnectPolling;
    return flagcxSuccess;
  } else if (errno == ECONNREFUSED) {
    if (++sock->refusedRetries == RETRY_REFUSED_TIMES) {
      sock->state = flagcxSocketStateError;
      WARN("socketStartConnect: exceeded retries (%d)", sock->refusedRetries);
      return flagcxRemoteError;
    }
    usleep(SLEEP_INT);
    if (sock->refusedRetries % 1000 == 0)
      INFO(FLAGCX_ALL, "Call to connect returned %s, retrying",
           strerror(errno));
    return flagcxSuccess;
  } else if (errno == ETIMEDOUT) {
    if (++sock->timedOutRetries == RETRY_TIMEDOUT_TIMES) {
      sock->state = flagcxSocketStateError;
      WARN("socketStartConnect: exceeded timeouts (%d)", sock->timedOutRetries);
      return flagcxRemoteError;
    }
    usleep(SLEEP_INT);
    return flagcxSuccess;
  } else {
    char line[SOCKET_NAME_MAXLEN + 1];
    sock->state = flagcxSocketStateError;
    WARN("socketStartConnect: Connect to %s failed : %s",
         flagcxSocketToString(&sock->addr, line), strerror(errno));
    return flagcxSystemError;
  }
}

static flagcxResult_t socketPollConnect(struct flagcxSocket *sock) {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = sock->fd;
  pfd.events = POLLOUT;
  ret = poll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && errno == EINTR)) {
    return flagcxSuccess;
  } else if (ret < 0) {
    WARN("socketPollConnect poll() failed with error %s", strerror(errno));
    return flagcxRemoteError;
  } else {
    EQCHECK(ret == 1 && (pfd.revents & POLLOUT), 0);
  }

  /* check socket status */
  SYSCHECK(getsockopt(sock->fd, SOL_SOCKET, SO_ERROR, (void *)&ret, &rlen),
           "getsockopt");

  if (ret == 0) {
    sock->state = flagcxSocketStateConnected;
  } else if (ret == ECONNREFUSED) {
    if (++sock->refusedRetries == RETRY_REFUSED_TIMES) {
      sock->state = flagcxSocketStateError;
      WARN("socketPollConnect: exceeded retries (%d)", sock->refusedRetries);
      return flagcxRemoteError;
    }
    if (sock->refusedRetries % 1000 == 0)
      INFO(FLAGCX_ALL, "Call to connect returned %s, retrying",
           strerror(errno));
    usleep(SLEEP_INT);
    sock->state = flagcxSocketStateConnecting;
  } else if (ret == ETIMEDOUT) {
    if (++sock->timedOutRetries == RETRY_TIMEDOUT_TIMES) {
      sock->state = flagcxSocketStateError;
      WARN("socketPollConnect: exceeded timeouts (%d)", sock->timedOutRetries);
      return flagcxRemoteError;
    }
    usleep(SLEEP_INT);
    sock->state = flagcxSocketStateConnecting;
  } else if (ret != EINPROGRESS) {
    sock->state = flagcxSocketStateError;
    char line[SOCKET_NAME_MAXLEN + 1];
    WARN("socketPollConnect: Connect to %s returned %d(%s) errno %d(%s)",
         flagcxSocketToString(&sock->addr, line), ret, strerror(ret), errno,
         strerror(errno));
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketPollConnect(struct flagcxSocket *sock) {
  if (sock == NULL) {
    WARN("flagcxSocketPollConnect: pass NULL socket");
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(socketPollConnect(sock));
  return flagcxSuccess;
}

static flagcxResult_t socketFinalizeConnect(struct flagcxSocket *sock) {
  int sent = 0;
  FLAGCXCHECK(socketProgress(FLAGCX_SOCKET_SEND, sock, &sock->magic,
                             sizeof(sock->magic), &sent));
  if (sent == 0)
    return flagcxSuccess;
  FLAGCXCHECK(socketWait(FLAGCX_SOCKET_SEND, sock, &sock->magic,
                         sizeof(sock->magic), &sent));
  sent = 0;
  FLAGCXCHECK(socketWait(FLAGCX_SOCKET_SEND, sock, &sock->type,
                         sizeof(sock->type), &sent));
  sock->state = flagcxSocketStateReady;
  return flagcxSuccess;
}

static flagcxResult_t socketProgressState(struct flagcxSocket *sock) {
  if (sock->state == flagcxSocketStateAccepting) {
    FLAGCXCHECK(socketTryAccept(sock));
  }
  if (sock->state == flagcxSocketStateAccepted) {
    FLAGCXCHECK(socketFinalizeAccept(sock));
  }
  if (sock->state == flagcxSocketStateConnecting) {
    FLAGCXCHECK(socketStartConnect(sock));
  }
  if (sock->state == flagcxSocketStateConnectPolling) {
    FLAGCXCHECK(socketPollConnect(sock));
  }
  if (sock->state == flagcxSocketStateConnected) {
    FLAGCXCHECK(socketFinalizeConnect(sock));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketReady(struct flagcxSocket *sock, int *running) {
  if (sock == NULL) {
    *running = 0;
    return flagcxSuccess;
  }
  if (sock->state == flagcxSocketStateError ||
      sock->state == flagcxSocketStateClosed) {
    WARN("flagcxSocketReady: unexpected socket state %d", sock->state);
    return flagcxRemoteError;
  }
  *running = (sock->state == flagcxSocketStateReady) ? 1 : 0;
  if (*running == 0) {
    FLAGCXCHECK(socketProgressState(sock));
    *running = (sock->state == flagcxSocketStateReady) ? 1 : 0;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketConnect(struct flagcxSocket *sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  const int one = 1;

  if (sock == NULL) {
    WARN("flagcxSocketConnect: pass NULL socket");
    return flagcxInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("flagcxSocketConnect: file descriptor is -1");
    return flagcxInvalidArgument;
  }

  if (sock->state != flagcxSocketStateInitialized) {
    WARN("flagcxSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == flagcxSocketStateError)
      return flagcxRemoteError;
    return flagcxInternalError;
  }
  TRACE(FLAGCX_INIT | FLAGCX_NET, "Connecting to socket %s",
        flagcxSocketToString(&sock->addr, line));

  SYSCHECK(
      setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(int)),
      "setsockopt");

  sock->state = flagcxSocketStateConnecting;
  do {
    FLAGCXCHECK(socketProgressState(sock));
  } while (sock->asyncFlag == 0 &&
           (sock->abortFlag == NULL ||
            __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED) == 0) &&
           (sock->state == flagcxSocketStateConnecting ||
            sock->state == flagcxSocketStateConnectPolling ||
            sock->state == flagcxSocketStateConnected));

  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED))
    return flagcxInternalError;

  switch (sock->state) {
    case flagcxSocketStateConnecting:
    case flagcxSocketStateConnectPolling:
    case flagcxSocketStateConnected:
    case flagcxSocketStateReady:
      return flagcxSuccess;
    case flagcxSocketStateError:
      return flagcxSystemError;
    default:
      WARN("flagcxSocketConnect: wrong socket state %d", sock->state);
      return flagcxInternalError;
  }
}

flagcxResult_t flagcxSocketAccept(struct flagcxSocket *sock,
                                  struct flagcxSocket *listenSock) {
  flagcxResult_t ret = flagcxSuccess;

  if (listenSock == NULL || sock == NULL) {
    WARN("flagcxSocketAccept: pass NULL socket");
    ret = flagcxInvalidArgument;
    goto exit;
  }
  if (listenSock->state != flagcxSocketStateReady) {
    WARN("flagcxSocketAccept: wrong socket state %d", listenSock->state);
    if (listenSock->state == flagcxSocketStateError)
      ret = flagcxSystemError;
    else
      ret = flagcxInternalError;
    goto exit;
  }

  if (sock->acceptFd == -1) {
    memcpy(sock, listenSock, sizeof(struct flagcxSocket));
    sock->acceptFd = listenSock->fd;
    sock->state = flagcxSocketStateAccepting;
  }

  do {
    FLAGCXCHECKGOTO(socketProgressState(sock), ret, exit);
  } while (sock->asyncFlag == 0 &&
           (sock->abortFlag == NULL ||
            __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED) == 0) &&
           (sock->state == flagcxSocketStateAccepting ||
            sock->state == flagcxSocketStateAccepted));

  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED))
    return flagcxInternalError;

  switch (sock->state) {
    case flagcxSocketStateAccepting:
    case flagcxSocketStateAccepted:
    case flagcxSocketStateReady:
      ret = flagcxSuccess;
      break;
    case flagcxSocketStateError:
      ret = flagcxSystemError;
      break;
    default:
      WARN("flagcxSocketAccept: wrong socket state %d", sock->state);
      ret = flagcxInternalError;
      break;
  }

exit:
  return ret;
}

flagcxResult_t flagcxSocketInit(struct flagcxSocket *sock,
                                union flagcxSocketAddress *addr, uint64_t magic,
                                enum flagcxSocketType type,
                                volatile uint32_t *abortFlag, int asyncFlag) {
  flagcxResult_t ret = flagcxSuccess;

  if (sock == NULL)
    goto exit;
  sock->timedOutRetries = 0;
  sock->refusedRetries = 0;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->state = flagcxSocketStateInitialized;
  sock->magic = magic;
  sock->type = type;
  sock->fd = -1;
  sock->acceptFd = -1;

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    memcpy(&sock->addr, addr, sizeof(union flagcxSocketAddress));
    family = sock->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("flagcxSocketInit: connecting to address %s with family %d is "
           "neither AF_INET(%d) nor AF_INET6(%d)",
           flagcxSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = flagcxInternalError;
      goto fail;
    }
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                      : sizeof(struct sockaddr_in6);

    /* Connect to a hostname / port */
    sock->fd = socket(family, SOCK_STREAM, 0);
    if (sock->fd == -1) {
      WARN("flagcxSocketInit: Socket creation failed : %s", strerror(errno));
      ret = flagcxSystemError;
      goto fail;
    }
  } else {
    memset(&sock->addr, 0, sizeof(union flagcxSocketAddress));
  }

  /* Set socket as non-blocking if async or if we need to be able to abort */
  if ((sock->asyncFlag || sock->abortFlag) && sock->fd >= 0) {
    int flags;
    EQCHECKGOTO(flags = fcntl(sock->fd, F_GETFL), -1, ret, fail);
    SYSCHECKGOTO(fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

flagcxResult_t flagcxSocketProgress(int op, struct flagcxSocket *sock,
                                    void *ptr, int size, int *offset) {
  if (sock == NULL) {
    WARN("flagcxSocketProgress: pass NULL socket");
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(socketProgress(op, sock, ptr, size, offset));
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketWait(int op, struct flagcxSocket *sock, void *ptr,
                                int size, int *offset) {
  if (sock == NULL) {
    WARN("flagcxSocketWait: pass NULL socket");
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(socketWait(op, sock, ptr, size, offset));
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketSend(struct flagcxSocket *sock, void *ptr,
                                int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("flagcxSocketSend: pass NULL socket");
    return flagcxInvalidArgument;
  }
  if (sock->state != flagcxSocketStateReady) {
    WARN("flagcxSocketSend: socket state (%d) is not ready", sock->state);
    return flagcxInternalError;
  }
  FLAGCXCHECK(socketWait(FLAGCX_SOCKET_SEND, sock, ptr, size, &offset));
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketRecv(struct flagcxSocket *sock, void *ptr,
                                int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("flagcxSocketRecv: pass NULL socket");
    return flagcxInvalidArgument;
  }
  if (sock->state != flagcxSocketStateReady) {
    WARN("flagcxSocketRecv: socket state (%d) is not ready", sock->state);
    return flagcxInternalError;
  }
  FLAGCXCHECK(socketWait(FLAGCX_SOCKET_RECV, sock, ptr, size, &offset));
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketSendRecv(struct flagcxSocket *sendSock,
                                    void *sendPtr, int sendSize,
                                    struct flagcxSocket *recvSock,
                                    void *recvPtr, int recvSize) {
  int sendOffset = 0, recvOffset = 0;
  if (sendSock == NULL || recvSock == NULL) {
    WARN("flagcxSocketSendRecv: invalid socket %p/%p", sendSock, recvSock);
    return flagcxInternalError;
  }
  if (sendSock->state != flagcxSocketStateReady ||
      recvSock->state != flagcxSocketStateReady) {
    WARN("flagcxSocketSendRecv: socket state (%d/%d) is not ready",
         sendSock->state, recvSock->state);
    return flagcxInternalError;
  }
  while (sendOffset < sendSize || recvOffset < recvSize) {
    if (sendOffset < sendSize)
      FLAGCXCHECK(socketProgress(FLAGCX_SOCKET_SEND, sendSock, sendPtr,
                                 sendSize, &sendOffset));
    if (recvOffset < recvSize)
      FLAGCXCHECK(socketProgress(FLAGCX_SOCKET_RECV, recvSock, recvPtr,
                                 recvSize, &recvOffset));
  }
  return flagcxSuccess;
}

// Receive or detect connection closed
flagcxResult_t flagcxSocketTryRecv(struct flagcxSocket *sock, void *ptr,
                                   int size, int *closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("flagcxSocketTryRecv: pass NULL socket");
    return flagcxInvalidArgument;
  }
  *closed = 0;
  // Block until connection closes or nbytes received
  if (blocking) {
    while (offset < size) {
      FLAGCXCHECK(socketProgressOpt(FLAGCX_SOCKET_RECV, sock, ptr, size,
                                    &offset, 0, closed));
      if (*closed)
        return flagcxSuccess;
    }
  } else {
    FLAGCXCHECK(socketProgressOpt(FLAGCX_SOCKET_RECV, sock, ptr, size, &offset,
                                  0, closed));
    if (*closed)
      return flagcxSuccess;

    // If any bytes were received, block waiting for the rest
    if (offset > 0) {
      while (offset < size) {
        FLAGCXCHECK(socketProgressOpt(FLAGCX_SOCKET_RECV, sock, ptr, size,
                                      &offset, 0, closed));
        if (*closed)
          return flagcxSuccess;
      }
      // No bytes were received, return flagcxInProgress
    } else {
      return flagcxInProgress;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketClose(struct flagcxSocket *sock) {
  if (sock != NULL) {
    if (sock->fd >= 0) {
      /* shutdown() is needed to send FIN packet to proxy thread; shutdown() is
       * not affected by refcount of fd, but close() is. close() won't close a
       * fd and send FIN packet if the fd is duplicated (e.g. fork()). So
       * shutdown() guarantees the correct and graceful connection close here.
       */
      shutdown(sock->fd, SHUT_RDWR);
      close(sock->fd);
    }
    sock->state = flagcxSocketStateClosed;
    sock->fd = -1;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketGetFd(struct flagcxSocket *sock, int *fd) {
  if (sock == NULL) {
    WARN("flagcxSocketGetFd: pass NULL socket");
    return flagcxInvalidArgument;
  }
  if (fd)
    *fd = sock->fd;
  return flagcxSuccess;
}

flagcxResult_t flagcxSocketSetFd(int fd, struct flagcxSocket *sock) {
  if (sock == NULL) {
    WARN("flagcxSocketGetFd: pass NULL socket");
    return flagcxInvalidArgument;
  }
  sock->fd = fd;
  return flagcxSuccess;
}
