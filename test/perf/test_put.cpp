#include "flagcx.h"
#include "tools.h"

#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "comm.h"
#include "device.h"
#include "flagcx/adaptor/include/adaptor.h"
#include "flagcx/adaptor/include/ib_common.h"
#include "flagcx_net.h"
#include "global_comm.h"
#include "net.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>

namespace {

void fatal(flagcxResult_t res, const char *msg, int rank) {
  if (res != flagcxSuccess) {
    fprintf(stderr, "[rank %d] %s (err=%d)\n", rank, msg, int(res));
    MPI_Abort(MPI_COMM_WORLD, res);
  }
}
} // namespace

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t min_bytes = args.getMinBytes();
  size_t max_bytes = args.getMaxBytes();
  int step_factor = args.getStepFactor();
  int num_warmup_iters = args.getWarmupIters();
  int num_iters = args.getTestIters();
  int print_buffer = args.isPrintBuffer();
  uint64_t split_mask = args.getSplitMask();

  flagcxHandlerGroup_t handler;
  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxComm_t &comm = handler->comm;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, split_mask);

  int nGpu;
  devHandle->getDeviceCount(&nGpu);
  devHandle->setDevice(worldRank % nGpu);

  if (proc == 0)
    flagcxGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, totalProcs, uniqueId, proc);

  struct flagcxComm *innerComm = comm;
  struct flagcxHeteroComm *hetero = innerComm->hetero_comm;
  if (hetero == nullptr) {
    int isHomo = 0;
    flagcxIsHomoComm(comm, &isHomo);
    if (proc == 0) {
      printf("Skipping put benchmark: hetero communicator not initialised "
             "(isHomo=%d).\n",
             isHomo);
    }
    flagcxCommDestroy(comm);
    flagcxHandleFree(handler);
    MPI_Finalize();
    return 0;
  }

  struct flagcxNetAdaptor *netAdaptor = hetero->netAdaptor;
  if (netAdaptor == nullptr || netAdaptor->put == nullptr) {
    if (proc == 0)
      fprintf(stderr, "Current network adaptor does not support put\n");
    MPI_Finalize();
    return 0;
  }

  if (totalProcs < 2) {
    if (proc == 0)
      printf("test_put requires at least 2 MPI processes\n");
    MPI_Finalize();
    return 0;
  }

  const int senderRank = 0;
  const int receiverRank = 1;
  if (totalProcs != 2) {
    if (proc == 0)
      printf("test_put requires exactly 2 ranks (sender=0, receiver=1).\n");
    MPI_Finalize();
    return 0;
  }

  bool isSender = (proc == senderRank);
  bool isReceiver = (proc == receiverRank);

  int sendRank = (proc + 1) % totalProcs;
  int recvRank = (proc - 1 + totalProcs) % totalProcs;

  // Setup network connections
  flagcxNetHandle_t listenHandle = {};
  void *listenComm = nullptr;
  flagcxResult_t res =
      netAdaptor->listen(hetero->netDev, &listenHandle, &listenComm);
  fatal(res, "listen failed", proc);

  flagcxNetHandle_t peerHandle = {};
  MPI_Sendrecv(&listenHandle, sizeof(listenHandle), MPI_BYTE, recvRank, 100,
               &peerHandle, sizeof(peerHandle), MPI_BYTE, sendRank, 100,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  void *sendComm = nullptr;
  void *recvComm = nullptr;
  while (sendComm == nullptr || recvComm == nullptr) {
    if (sendComm == nullptr) {
      res = netAdaptor->connect(hetero->netDev, &peerHandle, &sendComm);
      fatal(res, "connect failed", proc);
    }

    if (recvComm == nullptr) {
      res = netAdaptor->accept(listenComm, &recvComm);
      fatal(res, "accept failed", proc);
    }

    if (sendComm == nullptr || recvComm == nullptr) {
      sched_yield();
    }
  }

  res = netAdaptor->closeListen(listenComm);
  fatal(res, "closeListen failed", proc);

  // Check one-sided extensions support
  if (netAdaptor->put == nullptr || netAdaptor->putSignal == nullptr ||
      netAdaptor->waitValue == nullptr || netAdaptor->test == nullptr ||
      netAdaptor->deregMr == nullptr) {
    fprintf(stderr,
            "[rank %d] Net adaptor does not support one-sided extensions\n",
            proc);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Prepare registration comm
  void *regComm = isSender ? sendComm : recvComm;
  if (netAdaptor == &flagcxNetIb) {
    if (isSender && sendComm != nullptr) {
      auto *ibSendComm = reinterpret_cast<struct flagcxIbSendComm *>(sendComm);
      regComm = static_cast<void *>(&ibSendComm->base);
    } else if (isReceiver && recvComm != nullptr) {
      auto *ibRecvComm = reinterpret_cast<struct flagcxIbRecvComm *>(recvComm);
      regComm = static_cast<void *>(&ibRecvComm->base);
    }
  }

  size_t signalBytes = sizeof(uint64_t);
  size_t max_iterations = std::max(num_warmup_iters, num_iters);
  size_t window_bytes =
      max_bytes * max_iterations + signalBytes * max_iterations;

  void *window = nullptr;
  if (posix_memalign(&window, 64, window_bytes) != 0 || window == nullptr) {
    fprintf(stderr,
            "[rank %d] posix_memalign failed for host window (size=%zu)\n",
            proc, window_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::memset(window, 0, window_bytes);

  struct bootstrapState *state = innerComm->bootstrap;
  void *mrHandle = nullptr;
  res = netAdaptor->regMr(regComm, window, window_bytes, FLAGCX_PTR_HOST,
                          &mrHandle);
  fatal(res, "netAdaptor->regMr failed", proc);

  struct flagcxIbMrHandle *localMrHandle = (struct flagcxIbMrHandle *)mrHandle;
  struct ibv_mr *mr = localMrHandle->mrs[0];

  int nranks = state->nranks;
  struct flagcxIbGlobalHandleInfo *info = nullptr;
  res = flagcxCalloc(&info, 1);
  fatal(res, "flagcxCalloc failed for info", proc);
  res = flagcxCalloc(&info->base_vas, nranks);
  fatal(res, "flagcxCalloc failed for base_vas", proc);
  res = flagcxCalloc(&info->rkeys, nranks);
  fatal(res, "flagcxCalloc failed for rkeys", proc);
  res = flagcxCalloc(&info->lkeys, nranks);
  fatal(res, "flagcxCalloc failed for lkeys", proc);

  info->base_vas[state->rank] = (uintptr_t)window;
  info->rkeys[state->rank] = mr->rkey;
  info->lkeys[state->rank] = mr->lkey;

  res = bootstrapAllGather(innerComm->bootstrap, (void *)info->base_vas,
                           sizeof(uintptr_t));
  fatal(res, "bootstrapAllGather failed for base_vas", proc);
  res = bootstrapAllGather(innerComm->bootstrap, (void *)info->rkeys,
                           sizeof(uint32_t));
  fatal(res, "bootstrapAllGather failed for rkeys", proc);
  res = bootstrapAllGather(innerComm->bootstrap, (void *)info->lkeys,
                           sizeof(uint32_t));
  fatal(res, "bootstrapAllGather failed for lkeys", proc);

  void *globalHandles = (void *)info;

  // Benchmark loop
  timer tim;
  size_t baseSignalOffset = max_bytes * max_iterations;

  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    if (size == 0)
      break;
    // Warmup iterations
    for (int i = 0; i < num_warmup_iters; ++i) {
      size_t signalOffset = baseSignalOffset + i * signalBytes;
      // Use different offset for each iteration to avoid address conflicts
      size_t current_send_offset = i * size;
      size_t current_recv_offset = i * size;

      if (isSender) {

        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset((char *)window + current_send_offset, value, size);

        void *putReq = nullptr;
        res = netAdaptor->put(sendComm, current_send_offset,
                              current_recv_offset, size, senderRank,
                              receiverRank, (void **)globalHandles, &putReq);
        fatal(res, "netAdaptor->put warmup failed", proc);

        void *sigReq = nullptr;
        res = netAdaptor->putSignal(sendComm, signalOffset, senderRank,
                                    receiverRank, (void **)globalHandles, &sigReq);
        fatal(res, "netAdaptor->putSignal warmup failed", proc);
      } else if (isReceiver) {
        res = netAdaptor->waitValue((void **)globalHandles, receiverRank,
                                    signalOffset, 1);
        fatal(res, "netAdaptor->waitValue warmup failed", proc);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tim.reset();

    // Benchmark iterations
    for (int i = 0; i < num_iters; ++i) {
      size_t signalOffset = baseSignalOffset + i * signalBytes;
      // Use different offset for each iteration to avoid address conflicts
      size_t current_send_offset = i * size;
      size_t current_recv_offset = i * size;

      if (isSender) {

        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset((char *)window + current_send_offset, value, size);

        void *putReq = nullptr;
        res = netAdaptor->put(sendComm, current_send_offset,
                              current_recv_offset, size, senderRank,
                              receiverRank, (void **)globalHandles, &putReq);
        fatal(res, "netAdaptor->put failed", proc);

        void *sigReq = nullptr;
        res = netAdaptor->putSignal(sendComm, signalOffset, senderRank,
                                    receiverRank, (void **)globalHandles, &sigReq);
        fatal(res, "netAdaptor->putSignal failed", proc);
      } else if (isReceiver) {
        res = netAdaptor->waitValue((void **)globalHandles, receiverRank,
                                    signalOffset, 1);
        fatal(res, "netAdaptor->waitValue failed", proc);

        if (print_buffer) {
          printf("[rank %d] Received data at offset %zu, size %zu:\n", proc,
                 current_recv_offset, size);
          for (size_t j = 0; j < size && j < 64; ++j) {
            printf("%02x ", ((unsigned char *)window)[current_recv_offset + j]);
            if ((j + 1) % 16 == 0)
              printf("\n");
          }
          if (size > 64)
            printf("... (truncated)\n");
          else
            printf("\n");
        }
      }
    }

    if (num_iters > 0) {
      double elapsed_time = tim.elapsed() / num_iters;
      MPI_Allreduce(MPI_IN_PLACE, &elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsed_time /= worldSize;

      double bandwidth = (double)size / 1.0e9 / elapsed_time;
      if (proc == 0 && color == 0) {
        printf("Size: %zu bytes; Avg time: %lf sec; Bandwidth: %lf GB/s\n",
               size, elapsed_time, bandwidth);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
  // Cleanup: Wait for all operations to complete before closing connections
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);
  res = netAdaptor->deregMr(regComm, mrHandle);
  fatal(res, "netAdaptor->deregMr failed", proc);
  // Free global handles first
  if (globalHandles != nullptr) {
    struct flagcxIbGlobalHandleInfo *info =
        (struct flagcxIbGlobalHandleInfo *)globalHandles;
    free(info->base_vas);
    free(info->rkeys);
    free(info->lkeys);
    free(info);
  }

  // Close connections
  if (sendComm != nullptr) {
    res = netAdaptor->closeSend(sendComm);
    if (res != flagcxSuccess) {
      // Ignore error if already closed or not needed
    }
  }
  if (recvComm != nullptr) {
    res = netAdaptor->closeRecv(recvComm);
    if (res != flagcxSuccess) {
      // Ignore error if already closed or not needed
    }
  }
  free(window);

  fatal(flagcxCommDestroy(comm), "flagcxCommDestroy failed", proc);
  fatal(flagcxHandleFree(handler), "flagcxHandleFree failed", proc);

  MPI_Finalize();
  return 0;
}
