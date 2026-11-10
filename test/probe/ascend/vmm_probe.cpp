// Standalone CANN capability probe. Intentionally does not link FlagCX.
#include <acl/acl.h>
#include <mpi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

enum class Result : int { pass = 0, unsupported = 1, fail = 2 };

const char *resultName(Result result) {
  switch (result) {
    case Result::pass:
      return "PASS";
    case Result::unsupported:
      return "UNSUPPORTED";
    case Result::fail:
      return "FAIL";
  }
  return "FAIL";
}

Result aclCheck(aclError error, int rank, const char *operation) {
  if (error == ACL_SUCCESS)
    return Result::pass;
  Result result = error == ACL_ERROR_RT_FEATURE_NOT_SUPPORT
                      ? Result::unsupported
                      : Result::fail;
  std::fprintf(stderr, "PROBE rank=%d operation=%s status=%s acl_error=%d\n",
               rank, operation, resultName(result), static_cast<int>(error));
  return result;
}

Result checkpoint(int rank, const char *stage, Result local) {
  int localCode = static_cast<int>(local);
  int globalCode = 0;
  if (MPI_Allreduce(&localCode, &globalCode, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
    return Result::fail;
  Result global = static_cast<Result>(globalCode);
  if (rank == 0)
    std::printf("PROBE stage=%s status=%s\n", stage, resultName(global));
  return global;
}

void *offset(void *base, size_t bytes) {
  return static_cast<void *>(static_cast<unsigned char *>(base) + bytes);
}

struct Reservation {
  void *base = nullptr;
  std::vector<void *> mapped;
};

class VmmProbe {
public:
  VmmProbe(int rank, int device, int peerDevice)
      : rank_(rank), device_(device), peerDevice_(peerDevice) {}

  Result run() {
    Result result = checkpoint(rank_, "allocation_and_local_mapping",
                               allocateAndInitialize());
    if (result != Result::pass)
      return result;

    result = checkpoint(rank_, "export", exportHandles());
    if (result != Result::pass)
      return result;
    if (MPI_Sendrecv(exports_.data(), 2, MPI_UINT64_T, 1 - rank_, 1,
                     peerExports_.data(), 2, MPI_UINT64_T, 1 - rank_, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS)
      return checkpoint(rank_, "exchange", Result::fail);
    result = checkpoint(rank_, "exchange", Result::pass);
    if (result != Result::pass)
      return result;

    result = checkpoint(rank_, "cross_device_import", importPeer());
    if (result != Result::pass)
      return result;
    result = checkpoint(rank_, "independent_peer_mapping", mapPeerSeparately());
    if (result != Result::pass)
      return result;
    result = checkpoint(rank_, "flat_mixed_device_mapping", mapFlat());
    if (result != Result::pass)
      return result;
    return checkpoint(rank_, "flat_subrange_and_segment_data", verifyFlat());
  }

  Result cleanup() {
    Result result = Result::pass;
    // All ranks must stop using peer memory before either owner releases it.
    for (auto it = reservations_.rbegin(); it != reservations_.rend(); ++it) {
      bool allUnmapped = true;
      for (auto map = it->mapped.rbegin(); map != it->mapped.rend(); ++map) {
        Result unmap = aclCheck(aclrtUnmapMem(*map), rank_, "aclrtUnmapMem");
        combine(&result, unmap);
        allUnmapped &= unmap == Result::pass;
      }
      if (it->base != nullptr && allUnmapped)
        combine(&result, aclCheck(aclrtReleaseMemAddress(it->base), rank_,
                                  "aclrtReleaseMemAddress"));
    }
    // A failed unmap may still refer to an imported handle. Leave it alive
    // until process/device teardown rather than creating a use-after-free.
    if (result == Result::pass)
      for (auto handle : imported_)
        if (handle != nullptr)
          combine(&result, aclCheck(aclrtFreePhysical(handle), rank_,
                                    "aclrtFreePhysical(imported)"));
    Result preRelease = checkpoint(rank_, "unmap_and_import_release", result);
    if (preRelease != Result::pass)
      return Result::fail;
    if (peerAccessEnabled_)
      combine(&result, aclCheck(aclrtDeviceDisablePeerAccess(peerDevice_),
                                rank_, "aclrtDeviceDisablePeerAccess"));
    for (auto handle : owned_)
      if (handle != nullptr)
        combine(&result, aclCheck(aclrtFreePhysical(handle), rank_,
                                  "aclrtFreePhysical(owned)"));
    return checkpoint(rank_, "cleanup", result);
  }

private:
  static void combine(Result *current, Result next) {
    if (static_cast<int>(next) > static_cast<int>(*current))
      *current = next;
  }

  Result reserve(size_t bytes, size_t *index) {
    Reservation reservation;
    Result result = aclCheck(
        aclrtReserveMemAddress(&reservation.base, bytes, 0, nullptr, 0), rank_,
        "aclrtReserveMemAddress");
    if (result != Result::pass)
      return result;
    *index = reservations_.size();
    reservations_.push_back(reservation);
    return Result::pass;
  }

  Result map(size_t reservation, size_t slot, aclrtDrvMemHandle handle) {
    void *address =
        offset(reservations_[reservation].base, slot * granularity_);
    Result result = aclCheck(aclrtMapMem(address, granularity_, 0, handle, 0),
                             rank_, "aclrtMapMem");
    if (result != Result::pass)
      return result;
    reservations_[reservation].mapped.push_back(address);
    aclrtMemAccessDesc access = {};
    access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    return aclCheck(aclrtMemSetAccess(address, granularity_, &access, 1), rank_,
                    "aclrtMemSetAccess");
  }

  Result copyPattern(void *address, unsigned char value) {
    std::array<unsigned char, 256> pattern;
    pattern.fill(value);
    Result result =
        aclCheck(aclrtMemcpy(address, pattern.size(), pattern.data(),
                             pattern.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                 rank_, "aclrtMemcpy(H2D head)");
    if (result != Result::pass)
      return result;
    return aclCheck(aclrtMemcpy(offset(address, granularity_ - 16), 16,
                                pattern.data(), 16, ACL_MEMCPY_HOST_TO_DEVICE),
                    rank_, "aclrtMemcpy(H2D tail)");
  }

  Result checkPattern(void *address, unsigned char value) {
    std::array<unsigned char, 32> readback = {};
    Result result = aclCheck(aclrtMemcpy(readback.data(), readback.size(),
                                         offset(address, 128), readback.size(),
                                         ACL_MEMCPY_DEVICE_TO_HOST),
                             rank_, "aclrtMemcpy(D2H subrange)");
    if (result != Result::pass)
      return result;
    for (unsigned char byte : readback)
      if (byte != value) {
        std::fprintf(stderr,
                     "PROBE rank=%d operation=subrange_data status=FAIL "
                     "expected=%u actual=%u\n",
                     rank_, static_cast<unsigned>(value),
                     static_cast<unsigned>(byte));
        return Result::fail;
      }
    std::array<unsigned char, 16> tail = {};
    result = aclCheck(aclrtMemcpy(tail.data(), tail.size(),
                                  offset(address, granularity_ - tail.size()),
                                  tail.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                      rank_, "aclrtMemcpy(D2H segment tail)");
    if (result != Result::pass)
      return result;
    for (unsigned char byte : tail)
      if (byte != value) {
        std::fprintf(stderr,
                     "PROBE rank=%d operation=segment_tail_data status=FAIL "
                     "expected=%u actual=%u\n",
                     rank_, static_cast<unsigned>(value),
                     static_cast<unsigned>(byte));
        return Result::fail;
      }
    return Result::pass;
  }

  Result allocateAndInitialize() {
    aclrtPhysicalMemProp property = {};
    property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device_;
    property.memAttr = ACL_HBM_MEM_NORMAL;
    Result result = aclCheck(
        aclrtMemGetAllocationGranularity(
            &property, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity_),
        rank_, "aclrtMemGetAllocationGranularity");
    if (result != Result::pass)
      return result;
    if (granularity_ < 256 || granularity_ > SIZE_MAX / 4) {
      std::fprintf(stderr, "PROBE rank=%d invalid_granularity=%zu\n", rank_,
                   granularity_);
      return Result::fail;
    }
    if (rank_ == 0)
      std::printf("PROBE granularity_bytes=%zu\n", granularity_);
    for (int segment = 0; segment < 2; ++segment) {
      result = aclCheck(
          aclrtMallocPhysical(&owned_[segment], granularity_, &property, 0),
          rank_, "aclrtMallocPhysical");
      if (result != Result::pass)
        return result;
      size_t reservation = 0;
      result = reserve(granularity_, &reservation);
      if (result != Result::pass)
        return result;
      result = map(reservation, 0, owned_[segment]);
      if (result != Result::pass)
        return result;
      result = copyPattern(reservations_[reservation].base,
                           static_cast<unsigned char>(rank_ * 2 + segment + 1));
      if (result != Result::pass)
        return result;
    }
    return Result::pass;
  }

  Result exportHandles() {
    for (int segment = 0; segment < 2; ++segment) {
      Result result =
          aclCheck(aclrtMemExportToShareableHandle(
                       owned_[segment], ACL_MEM_HANDLE_TYPE_NONE,
                       1 /* disable PID validation for this isolated probe */,
                       &exports_[segment]),
                   rank_, "aclrtMemExportToShareableHandle");
      if (result != Result::pass)
        return result;
    }
    return Result::pass;
  }

  Result importPeer() {
    if (device_ != peerDevice_) {
      int32_t canAccess = 0;
      Result result =
          aclCheck(aclrtDeviceCanAccessPeer(&canAccess, device_, peerDevice_),
                   rank_, "aclrtDeviceCanAccessPeer");
      if (result != Result::pass)
        return result;
      if (canAccess == 0) {
        std::fprintf(stderr,
                     "PROBE rank=%d operation=aclrtDeviceCanAccessPeer "
                     "status=UNSUPPORTED source=%d peer=%d\n",
                     rank_, device_, peerDevice_);
        return Result::unsupported;
      }
      result = aclCheck(aclrtDeviceEnablePeerAccess(peerDevice_, 0), rank_,
                        "aclrtDeviceEnablePeerAccess");
      if (result != Result::pass)
        return result;
      peerAccessEnabled_ = true;
    }
    for (int segment = 0; segment < 2; ++segment) {
      Result result =
          aclCheck(aclrtMemImportFromShareableHandle(
                       peerExports_[segment], device_, &imported_[segment]),
                   rank_, "aclrtMemImportFromShareableHandle");
      if (result != Result::pass)
        return result;
    }
    return Result::pass;
  }

  Result mapPeerSeparately() {
    for (int segment = 0; segment < 2; ++segment) {
      size_t reservation = 0;
      Result result = reserve(granularity_, &reservation);
      if (result != Result::pass)
        return result;
      result = map(reservation, 0, imported_[segment]);
      if (result != Result::pass)
        return result;
      result = checkPattern(
          reservations_[reservation].base,
          static_cast<unsigned char>((1 - rank_) * 2 + segment + 1));
      if (result != Result::pass)
        return result;
    }
    return Result::pass;
  }

  Result mapFlat() {
    Result result = reserve(4 * granularity_, &flatReservation_);
    if (result != Result::pass)
      return result;
    const std::array<aclrtDrvMemHandle, 4> handles = {
        owned_[0], owned_[1], imported_[0], imported_[1]};
    for (size_t segment = 0; segment < handles.size(); ++segment) {
      result = map(flatReservation_, segment, handles[segment]);
      if (result != Result::pass)
        return result;
    }
    return Result::pass;
  }

  Result verifyFlat() {
    void *base = reservations_[flatReservation_].base;
    for (int segment = 0; segment < 4; ++segment) {
      int owner = segment < 2 ? rank_ : 1 - rank_;
      Result result = checkPattern(
          offset(base, static_cast<size_t>(segment) * granularity_),
          static_cast<unsigned char>(owner * 2 + segment % 2 + 1));
      if (result != Result::pass)
        return result;
    }
    // One copy crossing each mapping boundary checks that the slot VAs are
    // genuinely contiguous rather than only independently readable.
    for (int boundary = 1; boundary < 4; ++boundary) {
      std::array<unsigned char, 32> readback = {};
      Result result = aclCheck(
          aclrtMemcpy(
              readback.data(), readback.size(),
              offset(base, static_cast<size_t>(boundary) * granularity_ - 16),
              readback.size(), ACL_MEMCPY_DEVICE_TO_HOST),
          rank_, "aclrtMemcpy(D2H flat boundary)");
      if (result != Result::pass)
        return result;
      const auto expectedFor = [&](int segment) {
        int owner = segment < 2 ? rank_ : 1 - rank_;
        return static_cast<unsigned char>(owner * 2 + segment % 2 + 1);
      };
      for (size_t byte = 0; byte < readback.size(); ++byte) {
        unsigned char expected =
            expectedFor(byte < 16 ? boundary - 1 : boundary);
        if (readback[byte] != expected) {
          std::fprintf(stderr,
                       "PROBE rank=%d operation=flat_boundary_data "
                       "status=FAIL boundary=%d byte=%zu expected=%u "
                       "actual=%u\n",
                       rank_, boundary, byte, static_cast<unsigned>(expected),
                       static_cast<unsigned>(readback[byte]));
          return Result::fail;
        }
      }
    }
    return Result::pass;
  }

  int rank_;
  int device_;
  int peerDevice_;
  size_t granularity_ = 0;
  size_t flatReservation_ = 0;
  bool peerAccessEnabled_ = false;
  std::array<aclrtDrvMemHandle, 2> owned_ = {{nullptr, nullptr}};
  std::array<aclrtDrvMemHandle, 2> imported_ = {{nullptr, nullptr}};
  std::array<uint64_t, 2> exports_ = {{0, 0}};
  std::array<uint64_t, 2> peerExports_ = {{0, 0}};
  std::vector<Reservation> reservations_;
};

bool parseDevices(int argc, char **argv, int *first, int *second) {
  if (argc != 3 || std::strcmp(argv[1], "--devices") != 0)
    return false;
  int consumed = 0;
  return std::sscanf(argv[2], "%d,%d%n", first, second, &consumed) == 2 &&
         consumed == static_cast<int>(std::strlen(argv[2])) && *first >= 0 &&
         *second >= 0;
}

} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    return 2;
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int first = 0;
  int second = 1;
  if (ranks != 2 || !parseDevices(argc, argv, &first, &second)) {
    if (rank == 0)
      std::fprintf(stderr, "Usage: mpirun -np 2 vmm_probe --devices 0,1\n");
    MPI_Finalize();
    return 2;
  }

  Result result = aclCheck(aclInit(nullptr), rank, "aclInit");
  bool initialized = result == Result::pass;
  result = checkpoint(rank, "acl_init", result);
  if (result != Result::pass) {
    if (initialized)
      aclFinalize();
    MPI_Finalize();
    return 2;
  }

  uint32_t count = 0;
  result = aclCheck(aclrtGetDeviceCount(&count), rank, "aclrtGetDeviceCount");
  int device = rank == 0 ? first : second;
  if (result == Result::pass && static_cast<uint32_t>(device) >= count)
    result = Result::fail;
  result = checkpoint(rank, "device_inventory", result);
  bool selected = false;
  if (result == Result::pass) {
    Result local = aclCheck(aclrtSetDevice(device), rank, "aclrtSetDevice");
    selected = local == Result::pass;
    result = checkpoint(rank, "device_select", local);
  }
  if (result == Result::pass) {
    if (rank == 0)
      std::printf("PROBE devices=%d,%d visible_device_count=%u\n", first,
                  second, count);
    VmmProbe probe(rank, device, rank == 0 ? second : first);
    result = probe.run();
    Result cleanup = probe.cleanup();
    if (cleanup != Result::pass)
      result = Result::fail;
  }
  if (selected) {
    Result local = aclCheck(aclrtResetDevice(device), rank, "aclrtResetDevice");
    if (local != Result::pass)
      result = Result::fail;
  }
  if (aclCheck(aclFinalize(), rank, "aclFinalize") != Result::pass)
    result = Result::fail;
  result = checkpoint(rank, "final_result", result);
  MPI_Finalize();
  return result == Result::pass ? 0 : result == Result::unsupported ? 3 : 2;
}
