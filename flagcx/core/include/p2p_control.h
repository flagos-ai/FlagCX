// Internal P2P runtime control protocol. Bootstrap framing uses native-endian
// int32 tag/length, as does the existing connection handshake.
#ifndef FLAGCX_P2P_CONTROL_H_
#define FLAGCX_P2P_CONTROL_H_

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <poll.h>
#include <string>
#include <sys/socket.h>

namespace flagcxP2pControl {
constexpr int kTag = 0x46585431; // FXT1: version 1, distinct from connect tag 4
constexpr int kMaxRequest = 256;
constexpr uint32_t kMaxSliceSize = 1u << 30;

// Every data connection exchanges this preface before any transport-specific
// listen handle. Legacy and shared Engine builds intentionally use different
// wire layouts; identifying both implementation and transport makes a mixed
// deployment fail before either side interprets the other's payload.
constexpr int kProtocolTag = 0x46585031; // FXP1
constexpr uint32_t kProtocolMagic = 0x46585031;
constexpr uint16_t kProtocolVersionLegacy = 3;
constexpr uint16_t kProtocolVersionShared = 5;

enum ProtocolImplementation : uint16_t {
  kProtocolLegacy = 1,
  kProtocolShared = 2,
};

enum ProtocolTransport : uint16_t {
  kProtocolIbrc = 1,
  kProtocolBarex = 2,
};

struct ProtocolHello {
  uint32_t magic;
  uint16_t version;
  uint16_t implementation;
  uint16_t transport;
  uint16_t reserved;
  uint32_t wireSize;
};
static_assert(sizeof(ProtocolHello) == 16,
              "P2P protocol preface must have a stable wire layout");

inline ProtocolHello protocolHello(ProtocolImplementation implementation,
                                   ProtocolTransport transport) {
  const uint16_t version = implementation == kProtocolShared
                               ? kProtocolVersionShared
                               : kProtocolVersionLegacy;
  return ProtocolHello{kProtocolMagic,
                       version,
                       static_cast<uint16_t>(implementation),
                       static_cast<uint16_t>(transport),
                       0,
                       static_cast<uint32_t>(sizeof(ProtocolHello))};
}

inline bool protocolCompatible(const ProtocolHello &local,
                               const ProtocolHello &remote) {
  return remote.magic == kProtocolMagic && remote.version == local.version &&
         remote.implementation == local.implementation &&
         remote.transport == local.transport && remote.reserved == 0 &&
         remote.wireSize == sizeof(ProtocolHello);
}

// flagcxSocketGetAddrFromString selects its IPv6 parser from the leading '['.
// Accept both raw IPv6 literals and already-bracketed hosts at Engine API
// boundaries while leaving IPv4 addresses and DNS names unchanged.
inline std::string hostPort(const std::string &host, int port) {
  const bool bracketed =
      host.size() >= 2 && host.front() == '[' && host.back() == ']';
  if (host.find(':') != std::string::npos && !bracketed)
    return "[" + host + "]:" + std::to_string(port);
  return host + ":" + std::to_string(port);
}

inline uint64_t pack(uint32_t slice, uint32_t fragment) {
  return (uint64_t(slice) << 32) | (fragment < slice ? fragment : slice);
}

// One GET or up to two newline-separated FLAGCX_P2P_*=decimal assignments.
// Parse everything before publication; invalid requests make no changes.
inline const char *update(std::atomic<uint64_t> &config,
                          const std::string &request) {
  if (request == "GET")
    return nullptr;
  const uint64_t old = config.load(std::memory_order_acquire);
  uint32_t slice = old >> 32, fragment = uint32_t(old);
  unsigned seen = 0;
  size_t pos = 0;
  while (pos < request.size()) {
    const size_t end = request.find('\n', pos);
    const std::string line = request.substr(pos, end - pos);
    const size_t eq = line.find('=');
    if (eq == std::string::npos)
      return "expected KEY=decimal";
    const std::string key = line.substr(0, eq);
    const unsigned bit = key == "FLAGCX_P2P_SLICE_SIZE"       ? 1
                         : key == "FLAGCX_P2P_FRAGMENT_LIMIT" ? 2
                                                              : 0;
    if (bit == 0)
      return "parameter is not runtime tunable";
    if (seen & bit)
      return "duplicate parameter";
    seen |= bit;
    if (eq + 1 == line.size())
      return "missing value";
    uint64_t value = 0;
    for (size_t i = eq + 1; i < line.size(); ++i) {
      if (line[i] < '0' || line[i] > '9')
        return "value must be an unsigned decimal integer";
      value = value * 10 + (line[i] - '0');
      if (value > kMaxSliceSize)
        return "value exceeds 1 GiB";
    }
    if (bit == 1)
      slice = uint32_t(value);
    else
      fragment = uint32_t(value);
    if (end == std::string::npos)
      break;
    pos = end + 1;
  }
  if (!seen)
    return "empty request";
  // The accept loop is the sole writer; readers see both fields together.
  config.store(pack(slice, fragment), std::memory_order_release);
  return nullptr;
}

// Bound partial control frames so they cannot leave the accept loop waiting
// forever. Poll in short intervals to respect engine shutdown.
inline bool receive(int fd, void *buffer, size_t size,
                    const std::atomic<bool> &stop, int timeoutMs = 5000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  char *out = static_cast<char *>(buffer);
  while (size) {
    if (stop.load(std::memory_order_acquire) ||
        std::chrono::steady_clock::now() >= deadline)
      return false;
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, 50);
    if (ready < 0 && errno != EINTR)
      return false;
    if (ready <= 0)
      continue;
    const ssize_t n = recv(fd, out, size, MSG_DONTWAIT);
    if (n == 0)
      return false;
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
      return false;
    }
    out += n;
    size -= n;
  }
  return true;
}

// Symmetric counterpart to receive(). Notification/control sockets may be
// nonblocking, and a peer that stops consuming data must not wedge engine
// teardown indefinitely.
inline bool send(int fd, const void *buffer, size_t size,
                 const std::atomic<bool> &stop, int timeoutMs = 5000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  const char *in = static_cast<const char *>(buffer);
  while (size) {
    if (stop.load(std::memory_order_acquire) ||
        std::chrono::steady_clock::now() >= deadline)
      return false;
    pollfd pfd{fd, POLLOUT, 0};
    const int ready = poll(&pfd, 1, 50);
    if (ready < 0 && errno != EINTR)
      return false;
    if (ready <= 0)
      continue;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
      return false;
    int sendFlags = MSG_DONTWAIT;
#ifdef MSG_NOSIGNAL
    sendFlags |= MSG_NOSIGNAL;
#endif
    const ssize_t n = ::send(fd, in, size, sendFlags);
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
      return false;
    }
    if (n == 0)
      return false;
    in += n;
    size -= static_cast<size_t>(n);
  }
  return true;
}
} // namespace flagcxP2pControl
#endif
