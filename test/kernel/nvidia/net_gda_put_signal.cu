// Real GDA adapter, synthetic MLX5 SQ/CQ. No NIC or remote DMA is exercised.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "device_api/flagcx_device.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <initializer_list>

using NativeQp = xpu_netio_mlx5_doca_gpu_dev_verbs_qp;
using NativeWqe = xpu_netio_mlx5_doca_gpu_dev_verbs_wqe;
using NativeCqe = xpu_netio_mlx5_doca_gpunetio_ib_mlx5_cqe64;
constexpr int capacity = 8;

__global__ void exercise(flagcxXpuNetioDeviceContext *state,
                         flagcxXpuNetioDeviceWindow *window, int phase,
                         uint64_t *observed) {
  DeviceAPI::Comm comm{};
  comm.rank = 0;
  comm.nRanks = 2;
  comm.intraSize = 1;
  comm.contextCount = 1;
  comm.signalCount = 3;
  comm.barrierSignalBase = 1;
  comm.signalBuffer = state->signalBuffer;
  comm.netProvider = DefaultNetProviderGda;
  comm.netProviderId = 1;
  comm.gdaDeviceContext = state;
  DeviceAPI::Net net(comm, 0);
  DeviceAPI::Window win{};
  win.netDeviceHandle = window;
  DeviceAPI::Team team{2, 0, 1};
  if (phase < 2) {
    // A valid data chunk emits Write followed by FAA; invalid/failing Writes
    // must not publish a data-completion signal.
    net.put(team, 1, win, 0, win, 0, 16, flagcxDevNet_SignalInc{0},
            flagcxDevNet_None{}, DeviceAPI::CoopThread{}, flagcxDevNet_None{},
            flagcxDeviceScopeThread, flagcxDeviceScopeDevice);
    if (phase == 0)
      net.put(team, 1, win, 16, win, 16, 16, flagcxDevNet_SignalInc{0},
              flagcxDevNet_None{}, DeviceAPI::CoopThread{}, flagcxDevNet_None{},
              flagcxDeviceScopeThread, flagcxDeviceScopeDevice);
  } else if (phase == 2 || phase == 3) {
    *observed = net.readSignal(0, 64, flagcxDeviceMemoryOrderAcquire);
    net.waitSignal(DeviceAPI::CoopThread{}, 0, phase == 2 ? 2 : 3, 64,
                   flagcxDeviceMemoryOrderAcquire);
  } else if (phase == 4) {
    net.put(team, 1, win, 0, win, 0, 16, flagcxDevNet_SignalInc{1},
            flagcxDevNet_None{}, DeviceAPI::CoopThread{}, flagcxDevNet_None{},
            flagcxDeviceScopeThread, flagcxDeviceScopeDevice);
  } else if (phase == 5) {
    net.signal(team, 1, flagcxDevNet_SignalAdd{0, 2}, DeviceAPI::CoopThread{},
               flagcxDevNet_None{}, flagcxDeviceScopeThread,
               flagcxDeviceScopeDevice);
  } else {
    net.putValue(team, 1, win, 0, uint64_t{42}, flagcxDevNet_SignalInc{0},
                 DeviceAPI::CoopThread{}, flagcxDevNet_None{},
                 flagcxDeviceScopeThread, flagcxDeviceScopeDevice);
  }
}

int main() {
  xpu_netio_device_context_t *ctx;
  xpu_netio_device_qp_t *qps;
  NativeQp *qp;
  NativeWqe *sq;
  NativeCqe *cq;
  uint32_t *fatal, *error, *dbr;
  uint64_t *db, *signals, *observed;
  flagcxXpuNetioDeviceContext *state;
  flagcxXpuNetioDeviceWindow *window;
  xpu_netio_remote_region_t *remote;
  auto alloc = [](void **p, size_t bytes) {
    assert(cudaMallocManaged(p, bytes) == cudaSuccess);
    std::memset(*p, 0, bytes);
  };
#define ALLOC(p, bytes) alloc(reinterpret_cast<void **>(&p), bytes)
  ALLOC(ctx, sizeof(*ctx));
  ALLOC(qps, 2 * sizeof(*qps));
  ALLOC(qp, sizeof(*qp));
  ALLOC(sq, capacity * sizeof(*sq));
  ALLOC(cq, capacity * sizeof(*cq));
  ALLOC(fatal, 4);
  ALLOC(error, 4);
  ALLOC(dbr, 4);
  ALLOC(db, 8);
  ALLOC(signals, 3 * sizeof(*signals));
  ALLOC(observed, 8);
  ALLOC(state, sizeof(*state));
  ALLOC(window, sizeof(*window));
  ALLOC(remote, 2 * sizeof(*remote));
#undef ALLOC
  xpu_netio_device_struct_header_init(&ctx->hdr, sizeof(*ctx));
  ctx->provider_id = XPU_NETIO_PROVIDER_MLX5;
  ctx->queue_pairs = qps;
  ctx->queue_pair_count = 2;
  ctx->operation_bits = XPU_NETIO_OP_WRITE | XPU_NETIO_OP_WRITE_ASYNC |
                        XPU_NETIO_OP_INLINE | XPU_NETIO_OP_ATOMIC;
  ctx->atomic_bits = XPU_NETIO_ATOMIC_FETCH_ADD_64;
  ctx->memory_bits =
      XPU_NETIO_MEMORY_REMOTE_WRITE | XPU_NETIO_MEMORY_REMOTE_ATOMIC;
  ctx->ordering_bits =
      XPU_NETIO_ORDERED_TRANSPORT | XPU_NETIO_ORDER_SYSTEM_FENCE;
  ctx->sharing_bits = XPU_NETIO_SHARING_GPU;
  ctx->posting_bits = XPU_NETIO_POST_WAIT_CONTROL;
  ctx->completion_bits = XPU_NETIO_COMPLETION_ASYNC_REQUEST |
                         XPU_NETIO_COMPLETION_ATOMIC_FETCH_RESULT |
                         XPU_NETIO_COMPLETION_TIMEOUT_WAIT;
  ctx->routing_bits = XPU_NETIO_ROUTING_LOGICAL_PEER;
  ctx->required_alignment = 1;
  ctx->max_transfer_bytes = 128;
  ctx->max_inline_bytes = 16;
  for (int i = 0; i < 2; ++i) {
    xpu_netio_device_struct_header_init(&qps[i].hdr, sizeof(qps[i]));
    qps[i].native_queue_pair = qp;
    qps[i].fatal_error = fatal;
    qps[i].send_queue_depth = qps[i].send_queue_capacity =
        qps[i].completion_queue_depth = capacity;
    remote[i].address = 0x200000000ULL;
    remote[i].length = 128;
    remote[i].provider_key[0] = __builtin_bswap32(0x55667788);
  }
  window->local.address = 0x100000000ULL;
  window->local.length = 128;
  window->local.provider_key[0] = __builtin_bswap32(0x11223344);
  window->remote = remote;
  window->bytes = 128;
  state->contexts = ctx;
  state->signalRemote = remote;
  state->signalBuffer = signals;
  state->atomicResults = window->local;
  state->errorFlag = error;
  state->timeoutNs = 5000000;
  state->nranks = 2;
  state->signalCount = 3;
  state->userSignalCount = 1;
  for (int phase = 0; phase < 8; ++phase) {
    *fatal = *error = 0;
    *observed = 0;
    signals[0] = 2;
    std::memset(qp, 0, sizeof(*qp));
    std::memset(sq, 0, capacity * sizeof(*sq));
    std::memset(cq, 0, capacity * sizeof(*cq));
    qp->sq_num_shift8_be_3ds = __builtin_bswap32((0x12345 << 8) | 3);
    qp->sq_num_shift8_be_4ds = __builtin_bswap32((0x12345 << 8) | 4);
    qp->sq_wqe_num = capacity;
    qp->sq_wqe_mask = capacity - 1;
    qp->sq_wqe_daddr = reinterpret_cast<uint8_t *>(sq);
    qp->sq_dbrec = dbr;
    qp->sq_db = db;
    qp->nic_handler =
        XPU_NETIO_MLX5_DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR;
    qp->cq_sq.cqe_daddr = reinterpret_cast<uint8_t *>(cq);
    qp->cq_sq.cqe_num = capacity;
    qp->cq_sq.cqe_mask = capacity - 1;
    qp->cq_sq.cqe_size = 64;
    for (int i = 0; i < capacity; ++i)
      cq[i].op_own = XPU_NETIO_MLX5_DOCA_GPUNETIO_IB_MLX5_CQE_REQ << 4;
    if (phase == 1 || phase == 7)
      cq[0].op_own = XPU_NETIO_MLX5_DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR << 4;
    exercise<<<1, 1>>>(state, window, phase, observed);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    if (phase == 0) {
      assert(*error == 0 && qp->sq_rsvd_index == 4);
      for (int i = 0; i < 4; ++i)
        assert((__builtin_bswap32(sq[i].rw_cseg.opmod_idx_opcode) & 0xff) ==
               (i % 2 ? 0x12 : 0x08));
      assert(__builtin_bswap64(sq[1].at_seg.swap_add) == 1);
      assert(__builtin_bswap64(sq[3].at_seg.swap_add) == 1);
      assert(__builtin_bswap64(sq[1].at_dseg.addr) ==
             window->local.address + 8);
    } else if (phase == 1) {
      assert(*error == XPU_NETIO_PROVIDER_ERROR && qp->sq_rsvd_index == 1);
    } else if (phase == 2) {
      assert(*observed == 2 && *error == 0);
    } else if (phase == 3) {
      assert(*observed == 2 && *error == XPU_NETIO_TIMEOUT);
    } else if (phase == 4) {
      assert(*error == XPU_NETIO_INVALID_ARGUMENT && qp->sq_rsvd_index == 0);
    } else if (phase == 6) {
      assert(*error == 0 && qp->sq_rsvd_index == 2);
      assert((__builtin_bswap32(sq[1].rw_cseg.opmod_idx_opcode) & 0xff) ==
             0x12);
      assert(__builtin_bswap64(sq[1].at_seg.swap_add) == 1);
    } else if (phase == 7) {
      assert(*error == XPU_NETIO_PROVIDER_ERROR && qp->sq_rsvd_index == 1);
    } else {
      assert(*error == 0 && qp->sq_rsvd_index == 1);
      assert(__builtin_bswap64(sq[0].at_seg.swap_add) == 2);
    }
  }
  for (void *p : std::initializer_list<void *>{ctx, qps, qp, sq, cq, fatal,
                                               error, dbr, db, signals,
                                               observed, state, window, remote})
    assert(cudaFree(p) == cudaSuccess);
  std::puts("MOCK PASS: GDA Put(SignalInc), same-QP Write/FAA, write failure "
            "without signal, Read/WaitSignal, timeout, reserved slot, "
            "SignalAdd, PutValue(SignalInc); no NIC DMA");
}
