/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#include "probe.h"

#include <fstream>
#include <infiniband/verbs.h>
#include <sstream>

namespace {
void describeQp(const Options &o, ibv_qp *qp) {
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

std::string gidString(const ibv_gid &gid) {
  std::ostringstream out;
  out << std::hex;
  for (int i = 0; i < 16; ++i) {
    if (i)
      out << ':';
    out << unsigned(gid.raw[i]);
  }
  return out.str();
}

int chooseGid(ibv_context *ctx, const ProbePort &port, int count,
              ibv_gid *gid) {
  int fallback = -1;
  for (int i = 0; i < count; ++i) {
    ibv_gid candidate = {};
    if (ibv_query_gid(ctx, port.port, i, &candidate))
      continue;
    bool nonzero = false;
    for (auto byte : candidate.raw)
      nonzero |= byte != 0;
    if (!nonzero)
      continue;
    std::ifstream file("/sys/class/infiniband/" + port.hca + "/ports/" +
                       std::to_string(port.port) + "/gid_attrs/types/" +
                       std::to_string(i));
    std::string type;
    std::getline(file, type);
    const bool ipv4 = candidate.raw[10] == 0xff && candidate.raw[11] == 0xff;
    if (fallback < 0 || (type == "RoCE v2" && ipv4)) {
      *gid = candidate;
      fallback = i;
    }
    if (type == "RoCE v2" && ipv4)
      break;
  }
  return fallback;
}

struct Endpoint {
  uint64_t address;
  uint32_t rkey;
  uint32_t qpn[kMaxRanks][kMaxQps];
  uint16_t lid;
  uint8_t mtu;
  uint8_t responderDepth;
  ibv_gid gid;
};
} // namespace

void runNative(const Options &o, Allocation &allocation) {
  const auto available = probePorts();
  if (available.empty())
    fail(o, "open_hca", "SETUP_ERROR", 0, "no active verbs port");
  ProbePort selected = available[o.rank % available.size()];
  if (o.hca != "auto") {
    bool found = false;
    for (const auto &p : available)
      if (p.hca + ":" + std::to_string(p.port) == o.hca) {
        selected = p;
        found = true;
      }
    if (!found)
      fail(o, "open_hca", "SETUP_ERROR", 0, "requested HCA/port is not active");
  }
  int count = 0;
  ibv_device **devices = ibv_get_device_list(&count);
  ibv_context *ctx = nullptr;
  for (int i = 0; devices && i < count; ++i)
    if (selected.hca == ibv_get_device_name(devices[i]))
      ctx = ibv_open_device(devices[i]);
  if (devices)
    ibv_free_device_list(devices);
  if (!ctx)
    fail(o, "open_hca", "SETUP_ERROR", errno, selected.hca);
  ibv_device_attr device = {};
  ibv_port_attr port = {};
  checkVerbs(o, "query_device", ibv_query_device(ctx, &device));
  checkVerbs(o, "query_port", ibv_query_port(ctx, selected.port, &port));
  ibv_gid gid = {};
  int gidIndex = chooseGid(ctx, selected, port.gid_tbl_len, &gid);
  emit(o, {{"event", "hca"},
           {"hca", selected.hca},
           {"port", selected.port},
           {"firmware", device.fw_ver},
           {"atomic_cap", int(device.atomic_cap)},
           {"max_qp_rd_atom", device.max_qp_rd_atom},
           {"max_qp_init_rd_atom", device.max_qp_init_rd_atom},
           {"gid_index", gidIndex},
           {"gid", gidString(gid)},
           {"registration", o.registration},
           {"relaxed_ordering", o.ro}});
  if (o.operation == "fetch_add" && device.atomic_cap == IBV_ATOMIC_NONE)
    fail(o, "atomic_cap", "CAPABILITY_ERROR", 0, "HCA reports IBV_ATOMIC_NONE");
  if (port.link_layer == IBV_LINK_LAYER_ETHERNET && gidIndex < 0)
    fail(o, "gid", "SETUP_ERROR", 0, "no usable RoCE GID");
  ibv_pd *pd = ibv_alloc_pd(ctx);
  ibv_cq *cq = ibv_create_cq(ctx, 128, nullptr, nullptr, 0);
  if (!pd || !cq)
    fail(o, "pd_cq", "SETUP_ERROR", errno, "PD/CQ creation failed");
  int access =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  if (o.operation == "fetch_add")
    access |= IBV_ACCESS_REMOTE_ATOMIC;
  if (o.ro)
    access |= IBV_ACCESS_RELAXED_ORDERING;
  ibv_mr *mr = o.registration == "iova2"
                   ? ibv_reg_mr_iova2(
                         pd, allocation.ptr, o.bytes,
                         reinterpret_cast<uintptr_t>(allocation.ptr), access)
                   : ibv_reg_mr(pd, allocation.ptr, o.bytes, access);
  if (!mr)
    fail(o, "reg_gpu_mr", "REG_ERROR", errno, std::strerror(errno));
  std::vector<uint64_t> oldValues(o.qps * o.iterations, UINT64_MAX);
  ibv_mr *resultMr =
      ibv_reg_mr(pd, oldValues.data(), oldValues.size() * sizeof(uint64_t),
                 IBV_ACCESS_LOCAL_WRITE);
  if (!resultMr)
    fail(o, "reg_result_mr", "REG_ERROR", errno, std::strerror(errno));
  emit(o, {{"event", "mr"},
           {"pd", reinterpret_cast<uintptr_t>(pd)},
           {"gpu_base", reinterpret_cast<uintptr_t>(mr->addr)},
           {"length", mr->length},
           {"access", access},
           {"lkey", mr->lkey},
           {"rkey", mr->rkey},
           {"return_base", reinterpret_cast<uintptr_t>(resultMr->addr)},
           {"return_lkey", resultMr->lkey}});
  std::vector<std::vector<ibv_qp *>> qps(o.ranks, std::vector<ibv_qp *>(o.qps));
  Endpoint local = {};
  local.address = reinterpret_cast<uintptr_t>(allocation.ptr);
  local.rkey = mr->rkey;
  local.lid = port.lid;
  local.mtu = uint8_t(port.active_mtu);
  local.responderDepth = uint8_t(std::min(o.depth, device.max_qp_rd_atom));
  local.gid = gid;
  for (int peer = 0; peer < o.ranks; ++peer) {
    if (peer == o.rank)
      continue;
    for (int q = 0; q < o.qps; ++q) {
      ibv_qp_init_attr init = {};
      init.send_cq = init.recv_cq = cq;
      init.qp_type = IBV_QPT_RC;
      init.cap.max_send_wr = 32;
      init.cap.max_recv_wr = 1;
      init.cap.max_send_sge = init.cap.max_recv_sge = 1;
      auto *qp = qps[peer][q] = ibv_create_qp(pd, &init);
      if (!qp)
        fail(o, "create_qp", "SETUP_ERROR", errno, std::strerror(errno));
      local.qpn[peer][q] = qp->qp_num;
      ibv_qp_attr attr = {};
      attr.qp_state = IBV_QPS_INIT;
      attr.port_num = selected.port;
      attr.qp_access_flags =
          access & (IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_ATOMIC);
      checkVerbs(o, "qp_init",
                 ibv_modify_qp(qp, &attr,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                   IBV_QP_ACCESS_FLAGS));
    }
  }
  std::vector<Endpoint> endpoints(o.ranks);
  MPI_Allgather(&local, sizeof(local), MPI_BYTE, endpoints.data(),
                sizeof(local), MPI_BYTE, MPI_COMM_WORLD);
  for (int peer = 0; peer < o.ranks; ++peer) {
    if (peer == o.rank)
      continue;
    for (int q = 0; q < o.qps; ++q) {
      ibv_qp_attr attr = {};
      attr.qp_state = IBV_QPS_RTR;
      attr.path_mtu = ibv_mtu(std::min(local.mtu, endpoints[peer].mtu));
      attr.dest_qp_num = endpoints[peer].qpn[o.rank][q];
      attr.max_dest_rd_atomic = std::min(o.depth, device.max_qp_rd_atom);
      attr.min_rnr_timer = 12;
      attr.ah_attr.dlid = endpoints[peer].lid;
      attr.ah_attr.port_num = selected.port;
      if (gidIndex >= 0) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = endpoints[peer].gid;
        attr.ah_attr.grh.sgid_index = gidIndex;
        attr.ah_attr.grh.hop_limit = 255;
      }
      checkVerbs(o, "qp_rtr",
                 ibv_modify_qp(qps[peer][q], &attr,
                               IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                   IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                   IBV_QP_MAX_DEST_RD_ATOMIC |
                                   IBV_QP_MIN_RNR_TIMER));
      attr = {};
      attr.qp_state = IBV_QPS_RTS;
      attr.timeout = 14;
      attr.retry_cnt = 1;
      attr.rnr_retry = 1;
      attr.max_rd_atomic =
          std::min(std::min(o.depth, device.max_qp_init_rd_atom),
                   int(endpoints[peer].responderDepth));
      checkVerbs(o, "qp_rts",
                 ibv_modify_qp(qps[peer][q], &attr,
                               IBV_QP_STATE | IBV_QP_TIMEOUT |
                                   IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                   IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
      emit(o, {{"event", "qp"},
               {"qpn", qps[peer][q]->qp_num},
               {"remote_qpn", endpoints[peer].qpn[o.rank][q]},
               {"peer", peer},
               {"remote_responder_depth", endpoints[peer].responderDepth},
               {"initiator_depth", attr.max_rd_atomic},
               {"requested_depth", o.depth}});
      describeQp(o, qps[peer][q]);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int posts = 0;
  if (o.sender()) {
    const int peer = o.target();
    for (int iteration = 0; iteration < o.iterations; ++iteration) {
      for (int q = 0; q < o.qps; ++q) {
        const int slot = iteration * o.qps + q;
        ibv_sge sge = {};
        sge.addr = o.operation == "fetch_add"
                       ? reinterpret_cast<uintptr_t>(&oldValues[slot])
                       : local.address + o.offset;
        sge.length = sizeof(uint64_t);
        sge.lkey = o.operation == "fetch_add" ? resultMr->lkey : mr->lkey;
        ibv_send_wr wr = {};
        wr.wr_id = slot;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.send_flags = IBV_SEND_SIGNALED;
        if (o.operation == "fetch_add") {
          wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
          wr.wr.atomic.remote_addr = endpoints[peer].address + o.offset;
          wr.wr.atomic.rkey = endpoints[peer].rkey;
          wr.wr.atomic.compare_add = 1;
        } else {
          wr.opcode =
              o.operation == "write" ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
          wr.wr.rdma.remote_addr = endpoints[peer].address + o.offset;
          wr.wr.rdma.rkey = endpoints[peer].rkey;
        }
        emit(o, {{"event", "post"},
                 {"operation", o.operation},
                 {"wr_id", slot},
                 {"qpn", qps[peer][q]->qp_num},
                 {"peer", peer},
                 {"remote_address", endpoints[peer].address + o.offset},
                 {"rkey", endpoints[peer].rkey},
                 {"local_address", sge.addr},
                 {"lkey", sge.lkey},
                 {"length", sge.length}});
        ibv_send_wr *bad = nullptr;
        const int error = ibv_post_send(qps[peer][q], &wr, &bad);
        emit(o, {{"event", "post_result"},
                 {"code", error},
                 {"bad_wr", bad ? int(bad->wr_id) : -1}});
        checkVerbs(o, "post", error, "POST_ERROR");
        ++posts;
      }
      int completed = 0;
      const auto end = until(o);
      while (completed < o.qps) {
        ibv_wc wc = {};
        const int n = ibv_poll_cq(cq, 1, &wc);
        if (n < 0)
          fail(o, "poll", "CQ_ERROR", n, "ibv_poll_cq failed");
        if (n) {
          // opcode/byte_len are not valid evidence on an error CQE.
          emit(o, {{"event", "completion"},
                   {"wr_id", wc.wr_id},
                   {"qpn", wc.qp_num},
                   {"status", int(wc.status)},
                   {"vendor_err", wc.vendor_err}});
          if (wc.status != IBV_WC_SUCCESS)
            fail(o, "completion", "CQ_ERROR", int(wc.status),
                 ibv_wc_status_str(wc.status));
          ++completed;
        }
        deadline(o, end, "completion");
      }
    }
  }
  int total = 0;
  MPI_Allreduce(&posts, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  verify(o, allocation, total);
  if (o.operation == "fetch_add") {
    std::vector<uint64_t> all(oldValues.size() * o.ranks);
    MPI_Allgather(oldValues.data(), oldValues.size() * sizeof(uint64_t),
                  MPI_BYTE, all.data(), oldValues.size() * sizeof(uint64_t),
                  MPI_BYTE, MPI_COMM_WORLD);
    if (o.rank == o.target()) {
      all.erase(std::remove(all.begin(), all.end(), UINT64_MAX), all.end());
      std::sort(all.begin(), all.end());
      emit(o, {{"event", "atomic_returns"}, {"values", all}});
      if (int(all.size()) != total)
        fail(o, "atomic_returns", "VERIFY_ERROR", 0, "missing atomic results");
      for (int i = 0; i < total; ++i)
        if (all[i] != kInitial + i)
          fail(o, "atomic_returns", "VERIFY_ERROR", 0,
               "nonunique or incorrect FetchAdd old values");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (auto &peer : qps)
    for (auto *qp : peer)
      if (qp)
        checkVerbs(o, "destroy_qp", ibv_destroy_qp(qp), "CLEANUP_ERROR");
  checkVerbs(o, "dereg_result", ibv_dereg_mr(resultMr), "CLEANUP_ERROR");
  checkVerbs(o, "dereg_gpu", ibv_dereg_mr(mr), "CLEANUP_ERROR");
  checkVerbs(o, "destroy_cq", ibv_destroy_cq(cq), "CLEANUP_ERROR");
  checkVerbs(o, "dealloc_pd", ibv_dealloc_pd(pd), "CLEANUP_ERROR");
  checkVerbs(o, "close_device", ibv_close_device(ctx), "CLEANUP_ERROR");
}
