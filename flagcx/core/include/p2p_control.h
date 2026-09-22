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
} // namespace flagcxP2pControl
#endif
