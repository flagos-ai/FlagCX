/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#pragma once

#include <infiniband/verbs.h>
#include <mcr/mc_runtime.h>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "adaptor.h"
#include "allocation_modes.h"

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// The low half wraps during the default eight increments. A truncated 32-bit
// update or an endian mismatch must not accidentally pass the 64-bit probe.
constexpr uint64_t kInitial = UINT64_C(0x12345678fffffffc);
constexpr uint64_t kPayload = UINT64_C(0x1234567800000000);
constexpr int kMaxRanks = 4;
constexpr int kMaxQps = 4;

struct Options {
  std::string layer = "native", operation = "fetch_add", hca;
  std::string registration = "mr";
  int mode = 0, direction = 0, qps = 1, depth = 16, ro = 0;
  int iterations = 8, rank = 0, ranks = 1, gpu = 0;
  size_t bytes = 4096, offset = 0;
  double timeout = 4;

  int target() const {
    return ranks == 2 ? 1 - direction : ranks - 1 - direction;
  }
  bool sender() const {
    return ranks == 2 ? rank == direction : rank != target();
  }
};

inline void emit(const Options &o, Json record) {
  record["rank"] = o.rank;
  record["pid"] = getpid();
  record["monotonic_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               Clock::now().time_since_epoch())
                               .count();
  const std::string line = "PROBE_JSON " + record.dump() + "\n";
  // One record per write, so MPI's output collector can preserve each line.
  const char *p = line.data();
  size_t remaining = line.size();
  while (remaining) {
    ssize_t n = write(STDOUT_FILENO, p, remaining);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      break;
    p += n;
    remaining -= static_cast<size_t>(n);
  }
}

struct Failure : std::runtime_error {
  std::string status;
  Failure(const std::string &s, const std::string &detail)
      : std::runtime_error(detail), status(s) {}
};

inline void fail(const Options &o, const char *stage, const char *status,
                 int code, const std::string &detail) {
  emit(o, {{"event", "error"},
           {"stage", stage},
           {"status", status},
           {"code", code},
           {"detail", detail}});
  throw Failure(status, detail);
}

inline void checkMaca(const Options &o, const char *stage, mcError_t error,
                      const char *status = "SDK_ERROR") {
  emit(o, {{"event", "stage"},
           {"stage", stage},
           {"code", int(error)},
           {"detail", mcGetErrorString(error)}});
  if (error != mcSuccess)
    fail(o, stage, status, int(error), mcGetErrorString(error));
}

inline void checkFlagcx(const Options &o, const char *stage,
                        flagcxResult_t error,
                        const char *status = "ADAPTOR_ERROR") {
  emit(o, {{"event", "stage"}, {"stage", stage}, {"code", int(error)}});
  if (error != flagcxSuccess)
    fail(o, stage, status, int(error), "FlagCX adaptor returned an error");
}

inline void checkVerbs(const Options &o, const char *stage, int error,
                       const char *status = "VERBS_ERROR") {
  if (error)
    fail(o, stage, status, error, std::strerror(error > 0 ? error : errno));
}

inline void deadline(const Options &o, Clock::time_point until,
                     const char *stage) {
  if (Clock::now() > until)
    fail(o, stage, "TIMEOUT", 0, "internal deadline exceeded");
}

inline Clock::time_point until(const Options &o) {
  return Clock::now() + std::chrono::milliseconds(int(o.timeout * 1000));
}

inline void describeQp(const Options &o, ibv_qp *qp) {
  ibv_qp_attr attr = {};
  ibv_qp_init_attr init = {};
  const int query = ibv_query_qp(
      qp, &attr,
      IBV_QP_STATE | IBV_QP_ACCESS_FLAGS | IBV_QP_PORT | IBV_QP_AV |
          IBV_QP_DEST_QPN | IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_MAX_DEST_RD_ATOMIC,
      &init);
  ibv_device_attr device = {};
  const int capability = ibv_query_device(qp->context, &device);
  emit(o,
       {{"event", "qp_state"},
        {"qpn", qp->qp_num},
        {"hca", ibv_get_device_name(qp->context->device)},
        {"query_code", query},
        {"capability_code", capability},
        {"atomic_cap", int(device.atomic_cap)},
        {"firmware", device.fw_ver},
        {"state", int(attr.qp_state)},
        {"access", attr.qp_access_flags},
        {"port", attr.port_num},
        {"remote_qpn", attr.dest_qp_num},
        {"initiator_depth", attr.max_rd_atomic},
        {"responder_depth", attr.max_dest_rd_atomic},
        {"gid_index", attr.ah_attr.grh.sgid_index},
        {"remote_gid", std::vector<uint8_t>(attr.ah_attr.grh.dgid.raw,
                                            attr.ah_attr.grh.dgid.raw + 16)}});
}

inline void loadedLibraries(const Options &o) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  std::set<std::string> paths;
  while (std::getline(maps, line)) {
    if (line.find("libmc") == std::string::npos &&
        line.find("libibverbs") == std::string::npos &&
        line.find("libbnxt") == std::string::npos)
      continue;
    const auto start = line.find('/');
    if (start != std::string::npos)
      paths.insert(line.substr(start));
  }
  emit(o, {{"event", "loaded_libraries"}, {"paths", paths}});
}

inline uint64_t initial(const Options &o) {
  if (o.operation == "write" && o.sender())
    return kPayload + o.rank;
  if (o.operation == "read" && o.rank == o.target())
    return kPayload + o.rank;
  return kInitial;
}

struct Allocation {
  void *ptr = nullptr;
  bool gdr = false;
  void create(const Options &o) {
    if (!probeModeAvailable(o.mode))
      fail(o, "allocate", "SDK_UNAVAILABLE", 0,
           "allocation enum absent in SDK");
    gdr = o.mode == 9;
    emit(o, {{"event", "allocation_request"},
             {"mode", o.mode},
             {"flag_value", probeFlagValue(o.mode)},
             {"bytes", o.bytes},
             {"gdr_interface", gdr}});
    if (gdr) {
      // Exercise the actual allocator, including its SyncMemops attribute.
      const auto result = macaAdaptor.gdrMemAlloc(&ptr, o.bytes, nullptr);
      emit(o, {{"event", "gdr_alloc_result"},
               {"code", int(result)},
               {"address", reinterpret_cast<uintptr_t>(ptr)}});
      checkFlagcx(o, "gdrMemAlloc", result, "ALLOC_ERROR");
    } else {
      checkMaca(o, "allocate", probeNativeAlloc(o.mode, &ptr, o.bytes),
                "ALLOC_ERROR");
    }
    if (!ptr || reinterpret_cast<uintptr_t>(ptr) % 8)
      fail(o, "allocate", "ALLOC_ERROR", 0, "null or unaligned GPU pointer");
    mcPointerAttribute_t attr = {};
    mcError_t attrResult = mcPointerGetAttributes(&attr, ptr);
    emit(o,
         {{"event", "allocation"},
          {"address", reinterpret_cast<uintptr_t>(ptr)},
          {"attribute_code", int(attrResult)},
          {"device", attr.device},
          {"device_address", reinterpret_cast<uintptr_t>(attr.devicePointer)},
          {"host_address", reinterpret_cast<uintptr_t>(attr.hostPointer)}});
    if (attrResult == mcSuccess) {
#if CUDART_VERSION >= 10000
      emit(o, {{"event", "pointer_type"}, {"memory_type", int(attr.type)}});
#else
      emit(o, {{"event", "pointer_type"},
               {"memory_type", int(attr.memoryType)},
               {"managed", int(attr.isManaged)}});
#endif
    }
    uint64_t value = initial(o);
    checkMaca(o, "initialize",
              mcMemcpy(static_cast<char *>(ptr) + o.offset, &value,
                       sizeof(value), mcMemcpyHostToDevice),
              "INIT_ERROR");
    checkMaca(o, "initialize_sync", mcDeviceSynchronize(), "INIT_ERROR");
  }
  uint64_t read(const Options &o) {
    uint64_t value = 0;
    checkMaca(o, "readback",
              mcMemcpy(&value, static_cast<char *>(ptr) + o.offset,
                       sizeof(value), mcMemcpyDeviceToHost),
              "COPY_ERROR");
    return value;
  }
  void release(const Options &o) {
    if (gdr)
      checkFlagcx(o, "gdrMemFree", macaAdaptor.gdrMemFree(ptr, nullptr),
                  "CLEANUP_ERROR");
    else
      checkMaca(o, "free", mcFree(ptr), "CLEANUP_ERROR");
    ptr = nullptr;
  }
};

inline void verify(const Options &o, Allocation &allocation, int totalPosts) {
  uint64_t expected = initial(o);
  if (o.operation == "write" && o.rank == o.target())
    expected = kPayload + o.direction;
  if (o.operation == "read" && o.sender())
    expected = kPayload + o.target();
  if (o.operation == "fetch_add" && o.rank == o.target())
    expected = kInitial + totalPosts;
  const uint64_t actual = allocation.read(o);
  emit(o, {{"event", "verify"}, {"expected", expected}, {"actual", actual}});
  if (actual != expected)
    fail(o, "verify", "VERIFY_ERROR", 0,
         "GPU target or destination value mismatch");
}

Json inventory();
void runNative(const Options &, Allocation &);
void runAdaptor(const Options &, Allocation &);
