#include <gtest/gtest.h>

#include "proxy.h"
#include "transport.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

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
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
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
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}

TEST(ProxyControlRpc, RemoteSetupUsesOpaqueHandleAndUpdatesLocalState) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxProxyConnection localConnection{};
  flagcxSocket clientSocket{};
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
  clientSocket.fd = fds[0];
  clientSocket.state = flagcxSocketStateReady;
  state.peerSocks = &clientSocket;
  state.nPeerSocks = 1;
  comm.proxyState = &state;
  connector.connection = &localConnection;
  connector.remoteConnection =
      reinterpret_cast<flagcxProxyConnection *>(static_cast<std::uintptr_t>(1));
  connector.tpRank = 0;

  bool serverOk = true;
  std::thread server([&]() {
    auto sendAll = [&](const void *buffer, size_t size) {
      const char *bytes = static_cast<const char *>(buffer);
      while (size > 0) {
        ssize_t sent = send(fds[1], bytes, size, 0);
        if (sent <= 0)
          return false;
        bytes += sent;
        size -= sent;
      }
      return true;
    };
    int type = 0;
    void *remoteConnection = nullptr;
    int requestSize = -1;
    int responseSize = -1;
    void *opId = nullptr;
    serverOk = recv(fds[1], &type, sizeof(type), MSG_WAITALL) == sizeof(type) &&
               recv(fds[1], &remoteConnection, sizeof(remoteConnection),
                    MSG_WAITALL) == sizeof(remoteConnection) &&
               recv(fds[1], &requestSize, sizeof(requestSize), MSG_WAITALL) ==
                   sizeof(requestSize) &&
               recv(fds[1], &responseSize, sizeof(responseSize), MSG_WAITALL) ==
                   sizeof(responseSize) &&
               recv(fds[1], &opId, sizeof(opId), MSG_WAITALL) == sizeof(opId);
    if (!serverOk)
      return;
    flagcxProxyRpcResponseHeader response = {opId, flagcxSuccess, 0};
    serverOk = type == flagcxProxyMsgSetup &&
               remoteConnection == connector.remoteConnection &&
               remoteConnection != connector.connection && requestSize == 0 &&
               responseSize == 0;
    serverOk = sendAll(&response, sizeof(response)) && serverOk;
  });

  EXPECT_EQ(flagcxProxyCallBlocking(&comm, &connector, flagcxProxyMsgSetup,
                                    nullptr, 0, nullptr, 0),
            flagcxSuccess);
  server.join();
  EXPECT_TRUE(serverOk);
  EXPECT_EQ(localConnection.state, connSetupDone);
  EXPECT_EQ(state.expectedResponses, nullptr);

  close(fds[0]);
  close(fds[1]);
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}

TEST(ProxyControlRpc, NetRelayCreatesLocalShadowForAnotherRankInSameProcess) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxPeerInfo peers[2]{};
  flagcxProxyConnector connector{};
  flagcxSocket sockets[2]{};
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
  sockets[1].fd = fds[0];
  sockets[1].state = flagcxSocketStateReady;
  state.peerSocks = sockets;
  state.nPeerSocks = 2;
  comm.proxyState = &state;
  comm.peerInfo = peers;
  comm.rank = 0;
  comm.nRanks = 2;
  peers[0].hostHash = peers[1].hostHash = 1;
  peers[0].pidHash = 1;
  peers[1].pidHash = 1;
  auto *remoteHandle = reinterpret_cast<flagcxProxyConnection *>(
      static_cast<std::uintptr_t>(0x1234));

  bool serverOk = true;
  std::thread server([&]() {
    auto recvAll = [&](void *buffer, size_t size) {
      return recv(fds[1], buffer, size, MSG_WAITALL) == (ssize_t)size;
    };
    auto sendAll = [&](const void *buffer, size_t size) {
      const char *bytes = static_cast<const char *>(buffer);
      while (size > 0) {
        ssize_t sent = send(fds[1], bytes, size, 0);
        if (sent <= 0)
          return false;
        bytes += sent;
        size -= sent;
      }
      return true;
    };
    int type = 0;
    void *requestConnection = nullptr;
    int requestSize = 0;
    int responseSize = 0;
    serverOk = recvAll(&type, sizeof(type)) &&
               recvAll(&requestConnection, sizeof(requestConnection)) &&
               recvAll(&requestSize, sizeof(requestSize)) &&
               recvAll(&responseSize, sizeof(responseSize));
    if (!serverOk || requestSize < 0 || requestSize > 64) {
      shutdown(fds[1], SHUT_RDWR);
      return;
    }
    std::vector<char> request(requestSize);
    void *opId = nullptr;
    serverOk =
        recvAll(request.data(), request.size()) && recvAll(&opId, sizeof(opId));
    if (!serverOk) {
      shutdown(fds[1], SHUT_RDWR);
      return;
    }
    flagcxProxyRpcResponseHeader header = {opId, flagcxSuccess,
                                           sizeof(remoteHandle)};
    const bool validRequest = type == flagcxProxyMsgInit &&
                              requestConnection == nullptr &&
                              responseSize == sizeof(remoteHandle);
    serverOk = sendAll(&header, sizeof(header)) &&
               sendAll(&remoteHandle, sizeof(remoteHandle)) && validRequest;
  });

  EXPECT_EQ(flagcxProxyConnect(&comm, TRANSPORT_NET, 1, 1, &connector),
            flagcxSuccess);
  server.join();
  EXPECT_TRUE(serverOk);
  EXPECT_EQ(connector.remoteConnection, remoteHandle);
  ASSERT_NE(connector.connection, nullptr);
  EXPECT_NE(connector.connection, remoteHandle);
  EXPECT_EQ(connector.connection->transport, TRANSPORT_NET);
  EXPECT_EQ(connector.connection->state, connInitialized);
  EXPECT_EQ(connector.sameProcess, 0);
  free(connector.connection);

  close(fds[0]);
  close(fds[1]);
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}

TEST(ProxyControlRpc, CleanupCallTreatsRemoteConnectionAsOpaque) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxSocket clientSocket{};
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
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
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}

TEST(ProxyControlRpc, ConcurrentCallsKeepFramesAndResponsesTogether) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxSocket clientSocket{};
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
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
      while (size > 0) {
        ssize_t sent = send(fds[1], bytes, size, 0);
        if (sent <= 0)
          return false;
        bytes += sent;
        size -= sent;
      }
      return true;
    };
    struct Request {
      int type;
      void *connection;
      int requestSize;
      int responseSize;
      int payload;
      void *opId;
    } requests[2]{};
    for (auto &request : requests) {
      if (!recvAll(&request.type, sizeof(request.type)) ||
          !recvAll(&request.connection, sizeof(request.connection)) ||
          !recvAll(&request.requestSize, sizeof(request.requestSize)) ||
          !recvAll(&request.responseSize, sizeof(request.responseSize)) ||
          request.requestSize != sizeof(request.payload) ||
          !recvAll(&request.payload, sizeof(request.payload)) ||
          !recvAll(&request.opId, sizeof(request.opId))) {
        serverOk = false;
        shutdown(fds[1], SHUT_RDWR);
        return;
      }
      serverOk &= request.type == flagcxProxyMsgConnect &&
                  request.connection == connector.connection &&
                  request.responseSize == sizeof(int);
    }
    serverOk &= requests[0].payload != requests[1].payload;
    for (int index = 1; index >= 0; --index) {
      auto &request = requests[index];
      flagcxProxyRpcResponseHeader response = {request.opId, flagcxSuccess,
                                               sizeof(int)};
      serverOk = sendAll(&response, sizeof(response)) &&
                 sendAll(&request.payload, sizeof(request.payload)) && serverOk;
    }
  });

  int replies[2] = {-1, -1};
  flagcxResult_t results[2] = {flagcxInternalError, flagcxInternalError};
  std::thread callers[2];
  for (int index = 0; index < 2; ++index) {
    callers[index] = std::thread([&, index]() {
      int payload = index + 17;
      results[index] = flagcxProxyCallBlocking(
          &comm, &connector, flagcxProxyMsgConnect, &payload, sizeof(payload),
          &replies[index], sizeof(replies[index]));
    });
  }
  for (auto &caller : callers)
    caller.join();
  server.join();
  EXPECT_TRUE(serverOk);
  for (int index = 0; index < 2; ++index) {
    EXPECT_EQ(results[index], flagcxSuccess);
    EXPECT_EQ(replies[index], index + 17);
  }
  EXPECT_EQ(state.expectedResponses, nullptr);
  close(fds[0]);
  close(fds[1]);
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}

TEST(ProxyControlRpc, RelayReleaseReplyDiffersFromLostReply) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxSocket clientSocket{};
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
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
      const char *data = static_cast<const char *>(buffer);
      while (size > 0) {
        ssize_t sent = send(fds[1], data, size, 0);
        if (sent <= 0)
          return false;
        data += sent;
        size -= sent;
      }
      return true;
    };
    int type = 0;
    void *connection = nullptr;
    int requestSize = -1;
    int responseSize = -1;
    flagcxNetRelayCancelRequest cancel{};
    void *opId = nullptr;
    serverOk = recvAll(&type, sizeof(type)) &&
               recvAll(&connection, sizeof(connection)) &&
               recvAll(&requestSize, sizeof(requestSize)) &&
               recvAll(&responseSize, sizeof(responseSize)) &&
               recvAll(&cancel, sizeof(cancel)) && recvAll(&opId, sizeof(opId));
    serverOk &= type == flagcxProxyMsgCancelRelay &&
                connection == connector.connection &&
                requestSize == sizeof(cancel) && responseSize == 1 &&
                cancel.requestId == 71;
    flagcxProxyRpcResponseHeader reply = {opId, flagcxRemoteError, 1};
    uint8_t released = 1;
    serverOk &=
        sendAll(&reply, sizeof(reply)) && sendAll(&released, sizeof(released));
    shutdown(fds[1], SHUT_RDWR);
  });

  flagcxNetRelayCancelRequest cancel = {71};
  int operation = 0;
  ASSERT_EQ(flagcxProxyCallAsync(&comm, &connector, flagcxProxyMsgCancelRelay,
                                 &cancel, sizeof(cancel), 1, &operation),
            flagcxSuccess);
  uint8_t released = 0;
  bool received = false;
  flagcxResult_t result;
  do {
    result = flagcxPollProxyResponseWithStatus(&comm, &connector, &released,
                                               &operation, &received);
    if (result == flagcxInProgress)
      std::this_thread::yield();
  } while (result == flagcxInProgress);
  EXPECT_EQ(result, flagcxRemoteError);
  EXPECT_TRUE(received);
  EXPECT_EQ(released, 1);
  EXPECT_EQ(state.expectedResponses, nullptr);

  server.join();
  EXPECT_TRUE(serverOk);
  received = true;
  EXPECT_NE(flagcxPollProxyResponseWithStatus(&comm, &connector, nullptr,
                                              &operation, &received),
            flagcxSuccess);
  EXPECT_FALSE(received);
  close(fds[0]);
  close(fds[1]);
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}

TEST(ProxyControlRpc, StopInterruptsPartialResponseFrames) {
  for (bool partialHeader : {true, false}) {
    int fds[2] = {-1, -1};
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
    flagcxProxyState state{};
    flagcxHeteroComm comm{};
    flagcxProxyConnector connector{};
    flagcxSocket socket{};
    ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
    socket.fd = fds[0];
    socket.state = flagcxSocketStateReady;
    state.peerSocks = &socket;
    state.nPeerSocks = 1;
    state.initialized = 1;
    comm.proxyState = &state;
    connector.tpRank = 0;
    int operation = 0;
    int reply = 0;
    const int replySize = partialHeader ? 0 : sizeof(reply);
    ASSERT_EQ(flagcxProxyCallAsync(&comm, &connector, flagcxProxyMsgConnect,
                                   nullptr, 0, replySize, &operation),
              flagcxSuccess);

    flagcxProxyRpcResponseHeader response = {&operation, flagcxSuccess,
                                             replySize};
    if (partialHeader) {
      ASSERT_EQ(send(fds[1], &response, sizeof(response) - 1, 0),
                sizeof(response) - 1);
    } else {
      ASSERT_EQ(send(fds[1], &response, sizeof(response), 0), sizeof(response));
      ASSERT_EQ(send(fds[1], &reply, 1, 0), 1);
    }
    bool received = true;
    EXPECT_EQ(flagcxPollProxyResponseWithStatus(&comm, &connector, &reply,
                                                &operation, &received),
              flagcxInProgress);
    EXPECT_FALSE(received);

    auto started = std::chrono::steady_clock::now();
    EXPECT_EQ(flagcxProxyStop(&comm), flagcxSuccess);
    EXPECT_LT(std::chrono::steady_clock::now() - started,
              std::chrono::seconds(1));
    EXPECT_EQ(state.rpcReadStates, nullptr);
    EXPECT_EQ(socket.fd, -1);
    EXPECT_EQ(state.expectedResponses, nullptr);
    close(fds[1]);
    EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
  }
}

TEST(ProxyControlRpc, PollResumesAFragmentedResponse) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxSocket socket{};
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
  socket.fd = fds[0];
  socket.state = flagcxSocketStateReady;
  state.peerSocks = &socket;
  state.nPeerSocks = 1;
  state.initialized = 1;
  comm.proxyState = &state;
  connector.tpRank = 0;
  int operation = 0;
  int reply = 71;
  ASSERT_EQ(flagcxProxyCallAsync(&comm, &connector, flagcxProxyMsgConnect,
                                 nullptr, 0, sizeof(reply), &operation),
            flagcxSuccess);

  flagcxProxyRpcResponseHeader response = {&operation, flagcxSuccess,
                                           sizeof(reply)};
  auto *headerBytes = reinterpret_cast<const char *>(&response);
  ASSERT_EQ(send(fds[1], headerBytes, 2, 0), 2);
  int receivedValue = 0;
  EXPECT_EQ(
      flagcxPollProxyResponse(&comm, &connector, &receivedValue, &operation),
      flagcxInProgress);
  ASSERT_EQ(send(fds[1], headerBytes + 2, sizeof(response) - 2, 0),
            sizeof(response) - 2);
  auto *replyBytes = reinterpret_cast<const char *>(&reply);
  ASSERT_EQ(send(fds[1], replyBytes, 1, 0), 1);
  EXPECT_EQ(
      flagcxPollProxyResponse(&comm, &connector, &receivedValue, &operation),
      flagcxInProgress);
  ASSERT_EQ(send(fds[1], replyBytes + 1, sizeof(reply) - 1, 0),
            sizeof(reply) - 1);
  EXPECT_EQ(
      flagcxPollProxyResponse(&comm, &connector, &receivedValue, &operation),
      flagcxSuccess);
  EXPECT_EQ(receivedValue, reply);
  EXPECT_EQ(state.expectedResponses, nullptr);
  EXPECT_EQ(flagcxProxyStop(&comm), flagcxSuccess);
  close(fds[1]);
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}

TEST(ProxyControlRpc, StopInterruptsAStalledRpcWrite) {
  int fds[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  ASSERT_EQ(fcntl(fds[0], F_SETFL, O_NONBLOCK), 0);
  char fill[4096] = {};
  while (send(fds[0], fill, sizeof(fill), MSG_NOSIGNAL) > 0) {
  }
  ASSERT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);

  flagcxProxyState state{};
  flagcxHeteroComm comm{};
  flagcxProxyConnector connector{};
  flagcxSocket socket{};
  ASSERT_EQ(pthread_mutex_init(&state.rpcMutex, nullptr), 0);
  socket.fd = fds[0];
  socket.state = flagcxSocketStateReady;
  state.peerSocks = &socket;
  state.nPeerSocks = 1;
  state.initialized = 1;
  comm.proxyState = &state;
  connector.tpRank = 0;
  int operation = 0;
  std::atomic<bool> started{false};
  flagcxResult_t writeResult = flagcxSuccess;
  std::thread writer([&]() {
    started.store(true, std::memory_order_release);
    writeResult = flagcxProxyCallAsync(&comm, &connector, flagcxProxyMsgConnect,
                                       nullptr, 0, 0, &operation);
  });
  while (!started.load(std::memory_order_acquire))
    std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  auto stopStarted = std::chrono::steady_clock::now();
  EXPECT_EQ(flagcxProxyStop(&comm), flagcxSuccess);
  writer.join();
  EXPECT_LT(std::chrono::steady_clock::now() - stopStarted,
            std::chrono::seconds(1));
  EXPECT_NE(writeResult, flagcxSuccess);
  EXPECT_EQ(state.expectedResponses, nullptr);
  close(fds[1]);
  EXPECT_EQ(pthread_mutex_destroy(&state.rpcMutex), 0);
}
