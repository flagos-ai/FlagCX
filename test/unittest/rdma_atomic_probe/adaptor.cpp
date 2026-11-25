/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#include "probe.h"

#include "flagcx_net.h"
#include "ib_transport.h"
#include "onesided_types.h"

namespace {
struct WindowRecord {
  uintptr_t address;
  size_t size;
  flagcxNetMrInfo keys;
};

struct Window {
  uintptr_t addresses[kMaxRanks] = {};
  size_t sizes[kMaxRanks] = {};
  flagcxNetMrInfo keys[kMaxRanks] = {};
  flagcxOneSideHandleInfo info = {};
  void **handle() {
    info.baseVas = addresses;
    info.regionSizes = sizes;
    info.mrInfos = keys;
    return reinterpret_cast<void **>(&info);
  }
};

void wait(const Options &o, flagcxNetAdaptor *net, void *request) {
  if (!request)
    fail(o, "post", "POST_ERROR", 0, "successful post returned no request");
  const auto end = until(o);
  int done = 0;
  while (!done) {
    const auto result = net->test(request, &done, nullptr);
    if (result != flagcxSuccess)
      checkFlagcx(o, "completion", result, "CQ_ERROR");
    deadline(o, end, "completion");
    if (!done)
      std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  emit(o, {{"event", "completion"}, {"status", 0}});
}
} // namespace

void runAdaptor(const Options &o, Allocation &allocation) {
  auto *net = getNetAdaptor(RDMA);
  if (!net || std::strcmp(net->name, "IB") || !net->getMrInfo ||
      !net->iputSignal)
    fail(o, "net_init", "SETUP_ERROR", 0, "probe requires the IBRC adaptor");
  checkFlagcx(o, "net_init", net->init());
  int count = 0;
  checkFlagcx(o, "net_devices", net->devices(&count));
  if (count < 1)
    fail(o, "net_devices", "SETUP_ERROR", 0, "no FlagCX net device");
  const int dev = o.hca == "auto" ? o.rank % count : 0;
  flagcxNetProperties_t properties = {};
  checkFlagcx(o, "net_properties", net->getProperties(dev, &properties));
  emit(o, {{"event", "net_device"},
           {"index", dev},
           {"count", count},
           {"name", properties.name ? properties.name : ""},
           {"port", properties.port},
           {"pci_path", properties.pciPath ? properties.pciPath : ""}});
  void *listener = nullptr;
  char localHandle[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  std::vector<char> handles(o.ranks * sizeof(localHandle));
  checkFlagcx(o, "listen", net->listen(dev, localHandle, &listener));
  MPI_Allgather(localHandle, sizeof(localHandle), MPI_BYTE, handles.data(),
                sizeof(localHandle), MPI_BYTE, MPI_COMM_WORLD);
  std::vector<void *> sends(o.ranks), receives(o.ranks);
  // Serialize edge creation so each accepted connection has an unambiguous
  // MPI peer. connect/accept are both progressed while the other rank runs.
  for (int source = 0; source < o.ranks; ++source) {
    for (int target = 0; target < o.ranks; ++target) {
      if (source == target)
        continue;
      const auto end = until(o);
      if (o.rank == source) {
        while (!sends[target]) {
          const auto result =
              net->connect(dev, handles.data() + target * sizeof(localHandle),
                           &sends[target]);
          if (result != flagcxSuccess)
            checkFlagcx(o, "connect", result, "CONNECT_ERROR");
          deadline(o, end, "connect");
        }
      } else if (o.rank == target) {
        while (!receives[source]) {
          const auto result = net->accept(listener, &receives[source]);
          if (result != flagcxSuccess)
            checkFlagcx(o, "accept", result, "CONNECT_ERROR");
          deadline(o, end, "accept");
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  std::vector<void *> sendMrs(o.ranks), recvMrs(o.ranks);
  std::vector<Window> windows(o.ranks);
  for (int peer = 0; peer < o.ranks; ++peer) {
    if (peer == o.rank)
      continue;
    const int flags =
        o.ro ? FLAGCX_NET_MR_FLAG_NONE : FLAGCX_NET_MR_FLAG_FORCE_SO;
    checkFlagcx(o, "reg_send_gpu_mr",
                net->regMr(sends[peer], allocation.ptr, o.bytes,
                           FLAGCX_PTR_CUDA, flags, &sendMrs[peer]),
                "REG_ERROR");
    checkFlagcx(o, "reg_recv_gpu_mr",
                net->regMr(receives[peer], allocation.ptr, o.bytes,
                           FLAGCX_PTR_CUDA, flags, &recvMrs[peer]),
                "REG_ERROR");
    auto &w = windows[peer];
    w.info.nRanks = o.ranks;
    w.addresses[o.rank] = reinterpret_cast<uintptr_t>(allocation.ptr);
    w.sizes[o.rank] = o.bytes;
    checkFlagcx(o, "send_mr_keys",
                net->getMrInfo(sendMrs[peer], &w.keys[o.rank]));
    WindowRecord local = {};
    local.address = w.addresses[o.rank];
    local.size = o.bytes;
    checkFlagcx(o, "recv_mr_keys", net->getMrInfo(recvMrs[peer], &local.keys));
    emit(o, {{"event", "mr"},
             {"peer", peer},
             {"base", local.address},
             {"length", local.size},
             {"flags", flags},
             {"send_lkeys", w.keys[o.rank].lkeys},
             {"recv_rkeys", local.keys.rkeys}});
  }
  for (int a = 0; a < o.ranks; ++a) {
    for (int b = a + 1; b < o.ranks; ++b) {
      if (o.rank != a && o.rank != b)
        continue;
      const int peer = o.rank == a ? b : a;
      WindowRecord local = {}, remote = {};
      local.address = reinterpret_cast<uintptr_t>(allocation.ptr);
      local.size = o.bytes;
      checkFlagcx(o, "recv_mr_keys",
                  net->getMrInfo(recvMrs[peer], &local.keys));
      MPI_Sendrecv(&local, sizeof(local), MPI_BYTE, peer, 17, &remote,
                   sizeof(remote), MPI_BYTE, peer, 17, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      windows[peer].addresses[peer] = remote.address;
      windows[peer].sizes[peer] = remote.size;
      windows[peer].keys[peer] = remote.keys;
      windows[peer].handle();
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int posts = 0;
  if (o.sender()) {
    const int peer = o.target();
    auto &w = windows[peer];
    auto *comm = static_cast<flagcxIbSendComm *>(sends[peer]);
    std::vector<uint64_t> orderingKeys(comm->base.nqps);
    for (int q = 0; q < comm->base.nqps; ++q) {
      bool found = false;
      for (uint64_t key = 0; key < 65536; ++key) {
        flagcxIbLane lane = {};
        const auto result = flagcxIbSelectLane(
            &comm->base, FLAGCX_NET_LANE_ORDERED, key, &lane);
        if (result != flagcxSuccess)
          checkFlagcx(o, "select_lane", result);
        if (int(lane.base.index) == q) {
          orderingKeys[q] = key;
          found = true;
          break;
        }
      }
      if (!found)
        fail(o, "select_lane", "SETUP_ERROR", q,
             "cannot address configured QP");
    }
    for (int i = 0; i < o.iterations; ++i) {
      std::vector<void *> requests(comm->base.nqps);
      for (int q = 0; q < comm->base.nqps; ++q) {
        flagcxNetSubmitContext submit = {};
        submit.flags = FLAGCX_NET_SUBMIT_INDEPENDENT;
        submit.orderingKey = orderingKeys[q];
        checkFlagcx(o, "submit_context", flagcxNetSetSubmitContext(&submit));
        flagcxIbLane lane = {};
        checkFlagcx(o, "select_lane",
                    flagcxIbSelectLane(&comm->base, FLAGCX_NET_LANE_ORDERED,
                                       submit.orderingKey, &lane));
        flagcxNetResolvedRange remote = {}, local = {};
        uint32_t rkey = 0, lkey = 0;
        checkFlagcx(o, "resolve_remote",
                    flagcxIbResolveOneSideRange(&w.info, peer, o.offset, 8,
                                                &lane, false, &remote, &rkey));
        checkFlagcx(o, "resolve_local",
                    flagcxIbResolveOneSideRange(&w.info, o.rank, o.offset, 8,
                                                &lane, true, &local, &lkey));
        const int devIndex = lane.ibQp->devIndex;
        if (i == 0)
          describeQp(o, lane.ibQp->qp);
        emit(o, {{"event", "post"},
                 {"operation", o.operation},
                 {"peer", peer},
                 {"qpn", lane.ibQp->qp->qp_num},
                 {"lane", lane.base.index},
                 {"local_dev_index", devIndex},
                 {"remote_dev_index", lane.ibQp->remDevIdx},
                 {"pd", reinterpret_cast<uintptr_t>(lane.ibQp->qp->pd)},
                 {"remote_address", remote.address},
                 {"rkey", rkey},
                 {"local_address", local.address},
                 {"lkey", lkey},
                 {"return_address",
                  reinterpret_cast<uintptr_t>(&comm->putSignalScratchpad)},
                 {"return_lkey",
                  comm->devs[devIndex].putSignalScratchpadMr->lkey}});
        const auto end = until(o);
        flagcxResult_t result;
        do {
          if (o.operation == "fetch_add")
            result =
                net->iputSignal(sends[peer], 0, 0, 0, o.rank, peer, nullptr,
                                nullptr, o.offset, w.handle(), 1, &requests[q]);
          else if (o.operation == "write")
            result = net->iput(sends[peer], o.offset, o.offset, 8, o.rank, peer,
                               w.handle(), w.handle(), &requests[q]);
          else
            result = net->iget(sends[peer], o.offset, o.offset, 8, peer, o.rank,
                               w.handle(), w.handle(), &requests[q]);
          deadline(o, end, "post");
          if (result == flagcxInProgress)
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        } while (result == flagcxInProgress);
        flagcxNetClearSubmitContext();
        checkFlagcx(o, "post", result, "POST_ERROR");
        ++posts;
      }
      for (auto *request : requests)
        wait(o, net, request);
    }
  }
  int total = 0;
  MPI_Allreduce(&posts, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  verify(o, allocation, total);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int peer = 0; peer < o.ranks; ++peer) {
    if (peer == o.rank)
      continue;
    checkFlagcx(o, "dereg_send", net->deregMr(sends[peer], sendMrs[peer]),
                "CLEANUP_ERROR");
    checkFlagcx(o, "dereg_recv", net->deregMr(receives[peer], recvMrs[peer]),
                "CLEANUP_ERROR");
  }
  for (int peer = 0; peer < o.ranks; ++peer) {
    if (peer == o.rank)
      continue;
    checkFlagcx(o, "close_send", net->closeSend(sends[peer]), "CLEANUP_ERROR");
    checkFlagcx(o, "close_recv", net->closeRecv(receives[peer]),
                "CLEANUP_ERROR");
  }
  checkFlagcx(o, "close_listen", net->closeListen(listener), "CLEANUP_ERROR");
}
