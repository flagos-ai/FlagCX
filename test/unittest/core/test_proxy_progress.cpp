#include <gtest/gtest.h>

#include "proxy.h"
#include <cstdint>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

TEST(ProxyProgressState, SuccessAndInProgressDoNotAbort) {
  uint32_t abort = 0;
  flagcxProxyState state{};
  state.abortFlag = &abort;

  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxSuccess), flagcxSuccess);
  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxInProgress),
            flagcxInProgress);
  EXPECT_EQ(state.asyncResult, flagcxSuccess);
  EXPECT_EQ(abort, 0u);
}

TEST(ProxyProgressState, HardErrorRecordsFirstFailureAndAborts) {
  uint32_t abort = 0;
  flagcxProxyState state{};
  state.abortFlag = &abort;

  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxUnhandledDeviceError),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(state.asyncResult, flagcxUnhandledDeviceError);
  EXPECT_EQ(abort, 1u);

  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxSystemError),
            flagcxSystemError);
  EXPECT_EQ(state.asyncResult, flagcxUnhandledDeviceError);
}

TEST(ProxyProgressState, NullStateIsSafe) {
  EXPECT_EQ(flagcxProxyRecordAsyncError(nullptr, flagcxSystemError),
            flagcxSystemError);
}

TEST(ProxyConnectionState, RetryStatusesDoNotPoisonConnection) {
  flagcxProxyConnection connection{};
  connection.state = connInitialized;

  EXPECT_EQ(flagcxProxyRecordConnectionError(&connection, flagcxSuccess),
            flagcxSuccess);
  EXPECT_EQ(flagcxProxyRecordConnectionError(&connection, flagcxInProgress),
            flagcxInProgress);
  EXPECT_EQ(flagcxProxyGetConnectionError(&connection), flagcxSuccess);
  EXPECT_EQ(connection.state, connInitialized);
}

TEST(ProxyConnectionState, PermanentErrorRecordsFirstFailure) {
  flagcxProxyConnection connection{};
  connection.state = connInitialized;

  EXPECT_EQ(flagcxProxyRecordConnectionError(&connection, flagcxSystemError),
            flagcxSystemError);
  EXPECT_EQ(connection.state, connFailed);
  EXPECT_EQ(flagcxProxyGetConnectionError(&connection), flagcxSystemError);

  EXPECT_EQ(flagcxProxyRecordConnectionError(&connection, flagcxRemoteError),
            flagcxRemoteError);
  EXPECT_EQ(flagcxProxyGetConnectionError(&connection), flagcxSystemError);
}

TEST(ProxyConnectionState, InvalidConnectionIsRejected) {
  EXPECT_EQ(flagcxProxyGetConnectionError(nullptr), flagcxInvalidArgument);
}

TEST(ProxyControlRpc, BlockingCallTreatsRemoteConnectionAsOpaque) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxSocket clientSocket{};
  clientSocket.fd = fds[0];
  clientSocket.state = flagcxSocketStateReady;
  state.peerSocks = &clientSocket;
  state.nPeerSocks = 1;
  comm.proxyState = &state;
  connector.connection =
      reinterpret_cast<flagcxProxyConnection *>(static_cast<std::uintptr_t>(1));
  connector.tpRank = 0;

  bool serverOk = true;
  std::thread server([&]() {
    auto recvAll = [&](void *buffer, size_t size) {
      return recv(fds[1], buffer, size, MSG_WAITALL) == (ssize_t)size;
    };
    auto sendAll = [&](const void *buffer, size_t size) {
      const char *bytes = static_cast<const char *>(buffer);
      size_t sent = 0;
      while (sent < size) {
        ssize_t result = send(fds[1], bytes + sent, size - sent, 0);
        if (result <= 0)
          return false;
        sent += result;
      }
      return true;
    };

    int type = 0;
    void *remoteConnection = nullptr;
    int requestSize = -1;
    int responseSize = -1;
    void *opId = nullptr;
    serverOk = recvAll(&type, sizeof(type)) &&
               recvAll(&remoteConnection, sizeof(remoteConnection)) &&
               recvAll(&requestSize, sizeof(requestSize)) &&
               recvAll(&responseSize, sizeof(responseSize)) &&
               recvAll(&opId, sizeof(opId));
    if (!serverOk)
      return;

    flagcxProxyRpcResponseHeader response = {opId, flagcxRemoteError, 0};
    serverOk = type == flagcxProxyMsgConnect &&
               remoteConnection == connector.connection && requestSize == 0 &&
               responseSize == 0 && sendAll(&response, sizeof(response));
  });

  EXPECT_EQ(flagcxProxyCallBlocking(&comm, &connector, flagcxProxyMsgConnect,
                                    nullptr, 0, nullptr, 0),
            flagcxRemoteError);
  server.join();
  EXPECT_TRUE(serverOk);
  EXPECT_EQ(state.expectedResponses, nullptr);

  close(fds[0]);
  close(fds[1]);
}

TEST(ProxyControlRpc, CleanupCallTreatsRemoteConnectionAsOpaque) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxSocket clientSocket{};
  clientSocket.fd = fds[0];
  clientSocket.state = flagcxSocketStateReady;
  state.peerSocks = &clientSocket;
  state.nPeerSocks = 1;
  state.asyncResult = flagcxRemoteError;
  comm.proxyState = &state;
  connector.connection =
      reinterpret_cast<flagcxProxyConnection *>(static_cast<std::uintptr_t>(1));
  connector.tpRank = 0;

  bool serverOk = true;
  std::thread server([&]() {
    auto recvAll = [&](void *buffer, size_t size) {
      return recv(fds[1], buffer, size, MSG_WAITALL) == (ssize_t)size;
    };
    auto sendAll = [&](const void *buffer, size_t size) {
      const char *bytes = static_cast<const char *>(buffer);
      size_t sent = 0;
      while (sent < size) {
        ssize_t result = send(fds[1], bytes + sent, size - sent, 0);
        if (result <= 0)
          return false;
        sent += result;
      }
      return true;
    };

    int type = 0;
    void *remoteConnection = nullptr;
    int requestSize = -1;
    int responseSize = -1;
    void *opId = nullptr;
    serverOk = recvAll(&type, sizeof(type)) &&
               recvAll(&remoteConnection, sizeof(remoteConnection)) &&
               recvAll(&requestSize, sizeof(requestSize)) &&
               recvAll(&responseSize, sizeof(responseSize)) &&
               recvAll(&opId, sizeof(opId));
    if (!serverOk)
      return;

    flagcxProxyRpcResponseHeader response = {opId, flagcxSuccess, 0};
    serverOk = type == flagcxProxyMsgDeregister &&
               remoteConnection == connector.connection && requestSize == 0 &&
               responseSize == 0 && sendAll(&response, sizeof(response));
  });

  EXPECT_EQ(flagcxProxyCallBlocking(&comm, &connector, flagcxProxyMsgDeregister,
                                    nullptr, 0, nullptr, 0),
            flagcxSuccess);
  server.join();
  EXPECT_TRUE(serverOk);
  EXPECT_EQ(state.expectedResponses, nullptr);

  close(fds[0]);
  close(fds[1]);
}
