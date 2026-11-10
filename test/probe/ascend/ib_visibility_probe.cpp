// Direct NIC RDMA WRITE -> CANN stream wait -> immediate device consumer.
// This probe intentionally bypasses FlagCX's proxy and network adaptors.
#include "stream_consumer.h"

#include <acl/acl.h>
#include <infiniband/verbs.h>
#include <mpi.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr size_t kAllocationBytes = 4096;
constexpr size_t kPayloadOffset = 64;
constexpr size_t kPayloadBytes = 256;
constexpr int32_t kTimeoutMs = 10000;
constexpr uint8_t kExpectedByte = 0xA7;

enum class Result : int { pass = 0, unsupported = 1, fail = 2 };
enum class Mode { readback, wait };

const char *name(Result result) {
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
               rank, operation, name(result), static_cast<int>(error));
  const char *recent = aclGetRecentErrMsg();
  if (recent != nullptr && *recent != '\0')
    std::fprintf(stderr, "PROBE rank=%d operation=%s recent_error=%s\n", rank,
                 operation, recent);
  return result;
}

Result ibCheck(bool success, int rank, const char *operation) {
  if (success)
    return Result::pass;
  std::fprintf(stderr, "PROBE rank=%d operation=%s status=FAIL errno=%d\n",
               rank, operation, errno);
  return Result::fail;
}

Result checkpoint(int rank, const char *stage, Result local) {
  int localCode = static_cast<int>(local);
  int globalCode = 0;
  if (MPI_Allreduce(&localCode, &globalCode, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
    return Result::fail;
  Result global = static_cast<Result>(globalCode);
  if (rank == 0)
    std::printf("PROBE stage=%s status=%s\n", stage, name(global));
  return global;
}

void combine(Result *result, Result next) {
  if (static_cast<int>(next) > static_cast<int>(*result))
    *result = next;
}

bool parseArgs(int argc, char **argv, int *first, int *second, Mode *mode) {
  if ((argc != 3 && argc != 5) || std::strcmp(argv[1], "--devices") != 0)
    return false;
  int consumed = 0;
  if (std::sscanf(argv[2], "%d,%d%n", first, second, &consumed) != 2 ||
      consumed != static_cast<int>(std::strlen(argv[2])) || *first < 0 ||
      *second < 0)
    return false;
  if (argc == 3) {
    *mode = Mode::wait;
    return true;
  }
  if (std::strcmp(argv[3], "--mode") != 0)
    return false;
  if (std::strcmp(argv[4], "readback") == 0)
    *mode = Mode::readback;
  else if (std::strcmp(argv[4], "wait") == 0)
    *mode = Mode::wait;
  else
    return false;
  return true;
}

int envInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    return fallback;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  return end != value && *end == '\0' && parsed >= 0 && parsed <= 255
             ? static_cast<int>(parsed)
             : -1;
}

struct PeerInfo {
  uint64_t address = 0;
  uint32_t rkey = 0;
  uint32_t qpn = 0;
  uint32_t psn = 0;
  uint16_t lid = 0;
  uint8_t gid[16] = {};
  uint8_t mtu = 0;
  uint8_t linkLayer = 0;
};

class IbProbe {
public:
  IbProbe(int rank, Mode mode) : rank_(rank), mode_(mode) {}

  Result run() {
    Result result = checkpoint(rank_, "ib_resources_and_gpu_mr", prepare());
    if (result != Result::pass)
      return result;

    PeerInfo local = {};
    local.address = rank_ == 1 ? reinterpret_cast<uint64_t>(gpuBuffer_) : 0;
    local.rkey = rank_ == 1 ? mr_->rkey : 0;
    local.qpn = qp_->qp_num;
    local.psn = psn_;
    local.lid = portAttr_.lid;
    local.mtu = static_cast<uint8_t>(portAttr_.active_mtu);
    local.linkLayer = static_cast<uint8_t>(portAttr_.link_layer);
    std::memcpy(local.gid, gid_.raw, sizeof(local.gid));
    PeerInfo peer = {};
    if (MPI_Sendrecv(&local, sizeof(local), MPI_BYTE, 1 - rank_, 7, &peer,
                     sizeof(peer), MPI_BYTE, 1 - rank_, 7, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS)
      return checkpoint(rank_, "qp_address_exchange", Result::fail);
    result = checkpoint(rank_, "qp_address_exchange", Result::pass);
    if (result != Result::pass)
      return result;
    result = checkpoint(rank_, "qp_connected", connect(peer));
    if (result != Result::pass)
      return result;

    if (mode_ == Mode::wait) {
      result = checkpoint(rank_, "nic_wait_and_consumer_enqueued",
                          enqueueReceiver());
      if (result != Result::pass)
        return result;
    }
    // In wait mode the receiver queues ValueWait before either RDMA WRITE.
    result = checkpoint(rank_, "nic_write_cq_completed", transfer(peer));
    if (result != Result::pass)
      return result;
    if (mode_ == Mode::readback)
      return checkpoint(rank_, "nic_readback", verifyReadback());
    result = checkpoint(rank_, "nic_wait_completed", synchronizeReceiver());
    if (result != Result::pass)
      return result;
    return checkpoint(rank_, "nic_device_consumer_data", verify());
  }

  Result cleanup() {
    Result result = Result::pass;
    if (stream_ != nullptr)
      combine(&result,
              aclCheck(streamSynced_ ? aclrtDestroyStream(stream_)
                                     : aclrtDestroyStreamForce(stream_),
                       rank_, "aclrtDestroyStream(IB)"));
    if (qp_ != nullptr)
      combine(&result,
              ibCheck(ibv_destroy_qp(qp_) == 0, rank_, "ibv_destroy_qp"));
    // Stop both QPs before a receiver can deregister the GPU MR.
    Result stopped = checkpoint(rank_, "ib_qps_stopped", result);
    if (stopped != Result::pass)
      return Result::fail;
    if (mr_ != nullptr)
      combine(&result, ibCheck(ibv_dereg_mr(mr_) == 0, rank_, "ibv_dereg_mr"));
    if (cq_ != nullptr)
      combine(&result,
              ibCheck(ibv_destroy_cq(cq_) == 0, rank_, "ibv_destroy_cq"));
    if (pd_ != nullptr)
      combine(&result,
              ibCheck(ibv_dealloc_pd(pd_) == 0, rank_, "ibv_dealloc_pd"));
    if (context_ != nullptr)
      combine(&result, ibCheck(ibv_close_device(context_) == 0, rank_,
                               "ibv_close_device"));
    if (status_ != nullptr)
      combine(&result,
              aclCheck(aclrtFree(status_), rank_, "aclrtFree(IB status)"));
    if (gpuBuffer_ != nullptr)
      combine(&result,
              aclCheck(aclrtFree(gpuBuffer_), rank_, "aclrtFree(IB buffer)"));
    std::free(hostBuffer_);
    return checkpoint(rank_, "ib_cleanup", result);
  }

private:
  Result prepare() {
    const char *selected = std::getenv("IB_DEVICE");
    if (selected == nullptr || *selected == '\0') {
      if (rank_ == 0)
        std::fprintf(stderr, "Set IB_DEVICE to the RDMA device name\n");
      return Result::fail;
    }
    port_ = envInt("IB_PORT", 1);
    gidIndex_ = envInt("IB_GID_INDEX", 0);
    if (port_ <= 0 || gidIndex_ < 0)
      return Result::fail;

    int deviceCount = 0;
    ibv_device **devices = ibv_get_device_list(&deviceCount);
    if (devices == nullptr)
      return ibCheck(false, rank_, "ibv_get_device_list");
    for (int i = 0; i < deviceCount; ++i)
      if (std::strcmp(ibv_get_device_name(devices[i]), selected) == 0) {
        context_ = ibv_open_device(devices[i]);
        break;
      }
    ibv_free_device_list(devices);
    if (context_ == nullptr)
      return ibCheck(false, rank_, "ibv_open_device(selected)");
    Result result = ibCheck(
        ibv_query_port(context_, static_cast<uint8_t>(port_), &portAttr_) == 0,
        rank_, "ibv_query_port");
    if (result != Result::pass)
      return result;
    if (portAttr_.state != IBV_PORT_ACTIVE) {
      std::fprintf(stderr, "PROBE rank=%d IB port is not ACTIVE\n", rank_);
      return Result::fail;
    }
    result = ibCheck(ibv_query_gid(context_, static_cast<uint8_t>(port_),
                                   gidIndex_, &gid_) == 0,
                     rank_, "ibv_query_gid");
    if (result != Result::pass)
      return result;
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr)
      return ibCheck(false, rank_, "ibv_alloc_pd");
    cq_ = ibv_create_cq(context_, 8, nullptr, nullptr, 0);
    if (cq_ == nullptr)
      return ibCheck(false, rank_, "ibv_create_cq");
    ibv_qp_init_attr init = {};
    init.send_cq = cq_;
    init.recv_cq = cq_;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 4;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    qp_ = ibv_create_qp(pd_, &init);
    if (qp_ == nullptr)
      return ibCheck(false, rank_, "ibv_create_qp");
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = static_cast<uint8_t>(port_);
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    result = ibCheck(ibv_modify_qp(qp_, &attr,
                                   IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                       IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                     rank_, "ibv_modify_qp(INIT)");
    if (result != Result::pass)
      return result;

    if (rank_ == 1) {
      result = aclCheck(
          aclrtMalloc(&gpuBuffer_, kAllocationBytes, ACL_MEM_MALLOC_HUGE_FIRST),
          rank_, "aclrtMalloc(IB receiver)");
      if (result != Result::pass)
        return result;
      result = aclCheck(
          aclrtMemset(gpuBuffer_, kAllocationBytes, 0, kAllocationBytes), rank_,
          "aclrtMemset(IB receiver)");
      if (result != Result::pass)
        return result;
      mr_ = ibv_reg_mr(pd_, gpuBuffer_, kAllocationBytes,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (mr_ == nullptr)
        return ibCheck(false, rank_, "ibv_reg_mr(GPU)");
      result =
          aclCheck(aclrtMalloc(reinterpret_cast<void **>(&status_),
                               4 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                   rank_, "aclrtMalloc(IB status)");
      if (result != Result::pass)
        return result;
      result = aclCheck(
          aclrtMemset(status_, 4 * sizeof(uint32_t), 0, 4 * sizeof(uint32_t)),
          rank_, "aclrtMemset(IB status)");
      if (result != Result::pass)
        return result;
      return aclCheck(aclrtCreateStream(&stream_), rank_,
                      "aclrtCreateStream(IB receiver)");
    }

    if (posix_memalign(&hostBuffer_, 4096, kAllocationBytes) != 0)
      return ibCheck(false, rank_, "posix_memalign");
    std::memset(hostBuffer_, 0, kAllocationBytes);
    std::memset(static_cast<uint8_t *>(hostBuffer_) + kPayloadOffset,
                kExpectedByte, kPayloadBytes);
    *static_cast<uint64_t *>(hostBuffer_) = 2;
    mr_ =
        ibv_reg_mr(pd_, hostBuffer_, kAllocationBytes, IBV_ACCESS_LOCAL_WRITE);
    return ibCheck(mr_ != nullptr, rank_, "ibv_reg_mr(Host)");
  }

  Result connect(const PeerInfo &peer) {
    if (peer.qpn == 0 || peer.mtu == 0 ||
        peer.linkLayer != static_cast<uint8_t>(portAttr_.link_layer))
      return Result::fail;
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = static_cast<ibv_mtu>(
        peer.mtu < static_cast<uint8_t>(portAttr_.active_mtu)
            ? peer.mtu
            : static_cast<uint8_t>(portAttr_.active_mtu));
    attr.dest_qp_num = peer.qpn;
    attr.rq_psn = peer.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = peer.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = static_cast<uint8_t>(port_);
    attr.ah_attr.is_global =
        portAttr_.link_layer == IBV_LINK_LAYER_ETHERNET ? 1 : 0;
    if (attr.ah_attr.is_global) {
      std::memcpy(attr.ah_attr.grh.dgid.raw, peer.gid, sizeof(peer.gid));
      attr.ah_attr.grh.sgid_index = gidIndex_;
      attr.ah_attr.grh.hop_limit = 1;
    }
    Result result =
        ibCheck(ibv_modify_qp(qp_, &attr,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                  IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                  IBV_QP_MAX_DEST_RD_ATOMIC |
                                  IBV_QP_MIN_RNR_TIMER) == 0,
                rank_, "ibv_modify_qp(RTR)");
    if (result != Result::pass)
      return result;
    attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = psn_;
    attr.max_rd_atomic = 1;
    return ibCheck(ibv_modify_qp(qp_, &attr,
                                 IBV_QP_STATE | IBV_QP_TIMEOUT |
                                     IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                     IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) ==
                       0,
                   rank_, "ibv_modify_qp(RTS)");
  }

  Result enqueueReceiver() {
    if (rank_ != 1)
      return Result::pass;
    Result result = aclCheck(
        aclrtValueWait(gpuBuffer_, 1, ACL_STREAM_WAIT_VALUE_GEQ, stream_),
        rank_, "aclrtValueWait(NIC signal)");
    if (result != Result::pass)
      return result;
    launchStreamConsumer(static_cast<uint8_t *>(gpuBuffer_) + kPayloadOffset,
                         status_, kExpectedByte, stream_);
    return Result::pass;
  }

  Result transfer(const PeerInfo &peer) {
    if (rank_ == 1)
      return Result::pass;
    if (peer.address == 0 || peer.rkey == 0)
      return Result::fail;
    ibv_sge data = {};
    data.addr = reinterpret_cast<uint64_t>(hostBuffer_) + kPayloadOffset;
    data.length = kPayloadBytes;
    data.lkey = mr_->lkey;
    ibv_sge signal = {};
    signal.addr = reinterpret_cast<uint64_t>(hostBuffer_);
    signal.length = sizeof(uint64_t);
    signal.lkey = mr_->lkey;
    ibv_send_wr signalWr = {};
    signalWr.wr_id = 2;
    signalWr.sg_list = &signal;
    signalWr.num_sge = 1;
    signalWr.opcode = IBV_WR_RDMA_WRITE;
    signalWr.send_flags = IBV_SEND_SIGNALED;
    signalWr.wr.rdma.remote_addr = peer.address;
    signalWr.wr.rdma.rkey = peer.rkey;
    ibv_send_wr dataWr = {};
    dataWr.wr_id = 1;
    dataWr.next = &signalWr;
    dataWr.sg_list = &data;
    dataWr.num_sge = 1;
    dataWr.opcode = IBV_WR_RDMA_WRITE;
    dataWr.wr.rdma.remote_addr = peer.address + kPayloadOffset;
    dataWr.wr.rdma.rkey = peer.rkey;
    ibv_send_wr *bad = nullptr;
    Result result = ibCheck(ibv_post_send(qp_, &dataWr, &bad) == 0, rank_,
                            "ibv_post_send(payload,signal)");
    if (result != Result::pass)
      return result;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
      ibv_wc completion = {};
      int polled = ibv_poll_cq(cq_, 1, &completion);
      if (polled < 0)
        return ibCheck(false, rank_, "ibv_poll_cq");
      if (polled == 1) {
        std::printf("PROBE rank=%d operation=IB_completion status=%s "
                    "wc_status=%d vendor_err=%u wr_id=%llu\n",
                    rank_,
                    completion.status == IBV_WC_SUCCESS && completion.wr_id == 2
                        ? "PASS"
                        : "FAIL",
                    static_cast<int>(completion.status), completion.vendor_err,
                    static_cast<unsigned long long>(completion.wr_id));
        if (completion.status != IBV_WC_SUCCESS || completion.wr_id != 2) {
          return Result::fail;
        }
        return Result::pass;
      }
    }
    std::fprintf(stderr,
                 "PROBE rank=%d operation=IB_completion status=FAIL "
                 "reason=timeout\n",
                 rank_);
    return Result::fail;
  }

  Result synchronizeReceiver() {
    if (rank_ != 1)
      return Result::pass;
    Result result =
        aclCheck(aclrtSynchronizeStreamWithTimeout(stream_, kTimeoutMs), rank_,
                 "aclrtSynchronizeStreamWithTimeout(NIC wait)");
    streamSynced_ = result == Result::pass;
    return result;
  }

  Result verifyReadback() {
    if (rank_ != 1)
      return Result::pass;
    // Device read runs before diagnostic Host copies touch the signal/payload.
    launchStreamConsumer(static_cast<uint8_t *>(gpuBuffer_) + kPayloadOffset,
                         status_, kExpectedByte, stream_);
    launchStreamSignalReader(gpuBuffer_, status_, stream_);
    Result result =
        aclCheck(aclrtSynchronizeStreamWithTimeout(stream_, kTimeoutMs), rank_,
                 "aclrtSynchronizeStreamWithTimeout(NIC direct consumer)");
    streamSynced_ = result == Result::pass;
    if (result != Result::pass)
      return result;
    std::array<uint32_t, 4> hostStatus = {};
    result =
        aclCheck(aclrtMemcpy(hostStatus.data(), sizeof(hostStatus), status_,
                             sizeof(hostStatus), ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(NIC device status)");
    if (result != Result::pass)
      return result;
    const uint64_t deviceSignal = static_cast<uint64_t>(hostStatus[2]) |
                                  (static_cast<uint64_t>(hostStatus[3]) << 32);
    std::printf("PROBE rank=%d operation=nic_device_read status=%s "
                "signal=%llu mismatches=%u checked=%u\n",
                rank_,
                deviceSignal == 2 && hostStatus[0] == 0 &&
                        hostStatus[1] == kPayloadBytes
                    ? "PASS"
                    : "FAIL",
                static_cast<unsigned long long>(deviceSignal), hostStatus[0],
                hostStatus[1]);
    Result outcome = deviceSignal == 2 && hostStatus[0] == 0 &&
                             hostStatus[1] == kPayloadBytes
                         ? Result::pass
                         : Result::fail;
    uint64_t signal = 0;
    std::array<uint8_t, kPayloadBytes> payload = {};
    const Result signalCopy =
        aclCheck(aclrtMemcpy(&signal, sizeof(signal), gpuBuffer_,
                             sizeof(signal), ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(NIC signal)");
    const Result payloadCopy = aclCheck(
        aclrtMemcpy(payload.data(), payload.size(),
                    static_cast<uint8_t *>(gpuBuffer_) + kPayloadOffset,
                    payload.size(), ACL_MEMCPY_DEVICE_TO_HOST),
        rank_, "aclrtMemcpy(NIC payload)");
    combine(&outcome, signalCopy);
    combine(&outcome, payloadCopy);
    if (signalCopy != Result::pass || payloadCopy != Result::pass)
      return outcome;
    unsigned mismatches = 0;
    for (uint8_t byte : payload)
      mismatches += byte != kExpectedByte;
    std::printf("PROBE rank=%d operation=nic_host_readback status=%s "
                "signal=%llu mismatches=%u\n",
                rank_, signal == 2 && mismatches == 0 ? "PASS" : "FAIL",
                static_cast<unsigned long long>(signal), mismatches);
    if (signal != 2 || mismatches != 0)
      outcome = Result::fail;
    return outcome;
  }

  Result verify() {
    if (rank_ != 1)
      return Result::pass;
    std::array<uint32_t, 2> hostStatus = {};
    Result result =
        aclCheck(aclrtMemcpy(hostStatus.data(), sizeof(hostStatus), status_,
                             sizeof(hostStatus), ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(IB status)");
    if (result != Result::pass)
      return result;
    uint64_t signal = 0;
    result = aclCheck(aclrtMemcpy(&signal, sizeof(signal), gpuBuffer_,
                                  sizeof(signal), ACL_MEMCPY_DEVICE_TO_HOST),
                      rank_, "aclrtMemcpy(IB signal)");
    if (result != Result::pass)
      return result;
    if (signal != 2 || hostStatus[0] != 0 || hostStatus[1] != kPayloadBytes) {
      std::fprintf(stderr,
                   "PROBE rank=%d operation=NIC_consumer status=FAIL "
                   "signal=%llu mismatches=%u checked=%u\n",
                   rank_, static_cast<unsigned long long>(signal),
                   hostStatus[0], hostStatus[1]);
      return Result::fail;
    }
    return Result::pass;
  }

  int rank_;
  Mode mode_;
  int port_ = 1;
  int gidIndex_ = 0;
  uint32_t psn_ = static_cast<uint32_t>(0x123400 + rank_);
  bool streamSynced_ = false;
  ibv_context *context_ = nullptr;
  ibv_pd *pd_ = nullptr;
  ibv_cq *cq_ = nullptr;
  ibv_qp *qp_ = nullptr;
  ibv_mr *mr_ = nullptr;
  ibv_port_attr portAttr_ = {};
  ibv_gid gid_ = {};
  void *hostBuffer_ = nullptr;
  void *gpuBuffer_ = nullptr;
  uint32_t *status_ = nullptr;
  aclrtStream stream_ = nullptr;
};

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
  Mode mode = Mode::wait;
  if (ranks != 2 || !parseArgs(argc, argv, &first, &second, &mode)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "Usage: mpirun -np 2 ib_visibility_probe --devices 0,1 "
                   "[--mode readback|wait]\n");
    MPI_Finalize();
    return 2;
  }
  Result local = aclCheck(aclInit(nullptr), rank, "aclInit");
  bool initialized = local == Result::pass;
  Result result = checkpoint(rank, "acl_init", local);
  if (result != Result::pass) {
    if (initialized)
      aclFinalize();
    MPI_Finalize();
    return 2;
  }
  uint32_t count = 0;
  local = aclCheck(aclrtGetDeviceCount(&count), rank, "aclrtGetDeviceCount");
  int device = rank == 0 ? first : second;
  if (local == Result::pass && static_cast<uint32_t>(device) >= count)
    local = Result::fail;
  result = checkpoint(rank, "device_inventory", local);
  bool selected = false;
  if (result == Result::pass) {
    local = aclCheck(aclrtSetDevice(device), rank, "aclrtSetDevice");
    selected = local == Result::pass;
    result = checkpoint(rank, "device_select", local);
  }
  if (result == Result::pass) {
    if (rank == 0)
      std::printf("PROBE transport=IB devices=%d,%d "
                  "visible_device_count=%u ib_device=%s\n",
                  first, second, count,
                  std::getenv("IB_DEVICE") == nullptr
                      ? "(unset)"
                      : std::getenv("IB_DEVICE"));
    IbProbe probe(rank, mode);
    result = probe.run();
    Result cleanup = probe.cleanup();
    if (cleanup != Result::pass)
      result = Result::fail;
  }
  if (selected && aclCheck(aclrtResetDevice(device), rank,
                           "aclrtResetDevice") != Result::pass)
    result = Result::fail;
  if (aclCheck(aclFinalize(), rank, "aclFinalize") != Result::pass)
    result = Result::fail;
  result = checkpoint(rank, "final_result", result);
  MPI_Finalize();
  return result == Result::pass ? 0 : result == Result::unsupported ? 3 : 2;
}
