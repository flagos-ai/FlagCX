/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#pragma once

#include <nlohmann/json.hpp>

#include <cerrno>
#include <chrono>
#include <fstream>
#include <set>
#include <string>
#include <unistd.h>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

inline void probeWrite(const char *data, size_t bytes) {
  while (bytes) {
    ssize_t written = write(STDOUT_FILENO, data, bytes);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      break;
    data += written;
    bytes -= static_cast<size_t>(written);
  }
}

// This must be the first statement in main, before JSON/SDK/MPI work. If it
// is absent, even entry into main has not been established by the evidence.
inline void probeBoot() {
  constexpr char marker[] = "PROBE_BOOT main_enter\n";
  probeWrite(marker, sizeof(marker) - 1);
}

inline void probeEmit(Json record, int rank = -1) {
  record["rank"] = rank;
  record["pid"] = getpid();
  record["monotonic_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               Clock::now().time_since_epoch())
                               .count();
  const std::string line = "PROBE_JSON " + record.dump() + "\n";
  probeWrite(line.data(), line.size());
}

template <typename Call>
inline auto probeTimed(const char *stage, Call call, int rank = -1)
    -> decltype(call()) {
  probeEmit({{"event", "stage"}, {"stage", stage}, {"phase", "enter"}}, rank);
  const auto started = Clock::now();
  const auto result = call();
  const double elapsed =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  probeEmit({{"event", "stage"},
             {"stage", stage},
             {"phase", "return"},
             {"code", int(result)},
             {"elapsed_ms", elapsed}},
            rank);
  return result;
}

inline void probeLibraries(int rank = -1) {
  probeEmit(
      {{"event", "stage"}, {"stage", "loaded_libraries"}, {"phase", "enter"}},
      rank);
  std::ifstream maps("/proc/self/maps");
  std::string line;
  std::set<std::string> paths;
  while (std::getline(maps, line)) {
    if (line.find("libmc") == std::string::npos &&
        line.find("libibverbs") == std::string::npos &&
        line.find("libbnxt") == std::string::npos &&
        line.find("libflagcx") == std::string::npos &&
        line.find("libmpi") == std::string::npos &&
        line.find("libstdc++") == std::string::npos)
      continue;
    const auto start = line.find('/');
    if (start != std::string::npos)
      paths.insert(line.substr(start));
  }
  probeEmit({{"event", "loaded_libraries"}, {"paths", paths}}, rank);
  probeEmit(
      {{"event", "stage"}, {"stage", "loaded_libraries"}, {"phase", "return"}},
      rank);
}
