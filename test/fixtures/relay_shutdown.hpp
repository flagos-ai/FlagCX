#ifndef FLAGCX_TEST_RELAY_SHUTDOWN_HPP_
#define FLAGCX_TEST_RELAY_SHUTDOWN_HPP_
#include "net.h"
#include "proxy.h"
#include <sys/socket.h>
#include <unistd.h>

// A real control RPC endpoint with no GPU/NET provider. Replies use the actual
// wire header; tests replace the remote peer, not the caller's shutdown logic.
inline flagcxProxyConnector *
attachRelayShutdownEndpoint(flagcxHeteroComm_t comm, int *peerFd) {
  int fds[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0)
    return nullptr;
  *peerFd = fds[1];
  auto *proxy = comm->proxyState;
  proxy->peerSocks = (flagcxSocket *)calloc(1, sizeof(flagcxSocket));
  proxy->peerSocks[0].fd = fds[0];
  proxy->peerSocks[0].state = flagcxSocketStateReady;
  proxy->nPeerSocks = 1;
  auto *import =
      (flagcxNetRelayImport *)calloc(1, sizeof(flagcxNetRelayImport));
  import->base = (void *)0x66000;
  proxy->relayImports = import;
  comm->channels[0].peers[0] =
      (flagcxChannelPeer *)calloc(1, sizeof(flagcxChannelPeer));
  auto *connector = &comm->channels[0].peers[0]->send[0].proxyConn;
  connector->connection =
      (flagcxProxyConnection *)calloc(1, sizeof(flagcxProxyConnection));
  connector->remoteConnection = (flagcxProxyConnection *)1;
  connector->connection->relayImport = import;
  connector->connection->relayBufferImport = import->base;
  return connector;
}

inline bool replyRelayShutdown(int peerFd, flagcxProxyConnector *connector,
                               flagcxResult_t result) {
  flagcxProxyRpcResponseHeader reply = {connector, result, 0};
  return send(peerFd, &reply, sizeof(reply), 0) == (ssize_t)sizeof(reply);
}
#endif
