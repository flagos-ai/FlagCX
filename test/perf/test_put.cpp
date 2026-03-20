#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <algorithm>
#include <cstring>
#include <unistd.h>

#define DATATYPE flagcxFloat

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t min_bytes = args.getMinBytes();
  size_t max_bytes = args.getMaxBytes();
  int step_factor = args.getStepFactor();
  int num_warmup_iters = args.getWarmupIters();
  int num_iters = args.getTestIters();
  int print_buffer = args.isPrintBuffer();
  uint64_t split_mask = args.getSplitMask();
  int local_register = args.getLocalRegister();

  // RMA requires flagcxMemAlloc (GDR memory with SYNC_MEMOPS)
  if (local_register < 1) {
    fprintf(stderr,
            "test_put requires -R 1 or -R 2 for GDR buffer allocation.\n");
    return 1;
  }

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

  size_t signalBytes = sizeof(uint64_t);
  size_t total_iters_per_size = num_warmup_iters + num_iters;
  size_t max_data_iters = std::max(num_warmup_iters, num_iters);
  size_t data_bytes = max_bytes * max_data_iters;
  size_t signal_total_bytes = signalBytes * total_iters_per_size;

  // Data buffer: GDR memory (SYNC_MEMOPS ensures NIC visibility via GDR BAR)
  void *dataWindow = nullptr;
  flagcxResult_t res = flagcxMemAlloc(&dataWindow, data_bytes, comm);
  if (res != flagcxSuccess || dataWindow == nullptr) {
    fprintf(stderr, "[rank %d] flagcxMemAlloc failed for data (size=%zu)\n",
            proc, data_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(dataWindow, 0, data_bytes, flagcxMemDevice, NULL);

  // Signal buffer: GDR memory (SYNC_MEMOPS for RDMA ATOMIC visibility)
  void *signalWindow = nullptr;
  res = flagcxMemAlloc(&signalWindow, signal_total_bytes, comm);
  if (res != flagcxSuccess || signalWindow == nullptr) {
    fprintf(stderr, "[rank %d] flagcxMemAlloc failed for signal (size=%zu)\n",
            proc, signal_total_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(signalWindow, 0, signal_total_bytes, flagcxMemDevice,
                          NULL);

  // Register data buffer
  void *dataHandle = nullptr;
  res = flagcxCommRegister(comm, dataWindow, data_bytes, &dataHandle);
  if (res != flagcxSuccess) {
    fprintf(stderr, "[rank %d] flagcxCommRegister (data) failed\n", proc);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  FLAGCXCHECK(flagcxOneSideRegister(comm, dataWindow, data_bytes));
  FLAGCXCHECK(flagcxOneSideSignalRegister(comm, signalWindow, signal_total_bytes));

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  // Establish connection
  void *dummyBuff = nullptr;
  devHandle->deviceMalloc(&dummyBuff, 1, flagcxMemDevice, NULL);
  flagcxGroupStart(comm);
  if (isSender) {
    flagcxSend(dummyBuff, 1, flagcxChar, receiverRank, comm, stream);
  } else if (isReceiver) {
    flagcxRecv(dummyBuff, 1, flagcxChar, senderRank, comm, stream);
  }
  flagcxGroupEnd(comm);
  devHandle->streamSynchronize(stream);
  devHandle->deviceFree(dummyBuff, flagcxMemDevice, NULL);
  MPI_Barrier(MPI_COMM_WORLD);

  void *hostStaging = nullptr;
  if (posix_memalign(&hostStaging, 64, max_bytes) != 0 ||
      hostStaging == nullptr) {
    fprintf(stderr, "[rank %d] posix_memalign failed (size=%zu)\n", proc,
            max_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Benchmark loop
  timer tim;
  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    if (size == 0)
      break;

    devHandle->deviceMemset(signalWindow, 0, signal_total_bytes,
                            flagcxMemDevice, NULL);
    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup
    for (int i = 0; i < num_warmup_iters; ++i) {
      size_t signalOffset = i * signalBytes;
      size_t current_send_offset = i * size;
      size_t current_recv_offset = i * size;

      if (isSender) {
        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + current_send_offset,
                                hostStaging, size, flagcxMemcpyHostToDevice,
                                NULL);
        FLAGCXCHECK(flagcxPutSignal(comm, receiverRank, current_send_offset,
                                    current_recv_offset, size / sizeof(float),
                                    DATATYPE, signalOffset, stream));
      } else if (isReceiver) {
        FLAGCXCHECK(
            flagcxWaitSignal(comm, senderRank, signalOffset, 1, stream));
        devHandle->streamSynchronize(stream);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tim.reset();

    // Benchmark
    for (int i = 0; i < num_iters; ++i) {
      size_t signalOffset = (num_warmup_iters + i) * signalBytes;
      size_t current_send_offset = i * size;
      size_t current_recv_offset = i * size;

      if (isSender) {
        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + current_send_offset,
                                hostStaging, size, flagcxMemcpyHostToDevice,
                                NULL);
        FLAGCXCHECK(flagcxPutSignal(comm, receiverRank, current_send_offset,
                                    current_recv_offset, size / sizeof(float),
                                    DATATYPE, signalOffset, stream));
      } else if (isReceiver) {
        FLAGCXCHECK(
            flagcxWaitSignal(comm, senderRank, signalOffset, 1, stream));
        devHandle->streamSynchronize(stream);

        if (print_buffer) {
          devHandle->deviceMemcpy(
              hostStaging, (char *)dataWindow + current_recv_offset,
              std::min(size, (size_t)64), flagcxMemcpyDeviceToHost, NULL);
          printf("[rank %d] Received data at offset %zu, size %zu:\n", proc,
                 current_recv_offset, size);
          for (size_t j = 0; j < size && j < 64; ++j) {
            printf("%02x ", ((unsigned char *)hostStaging)[j]);
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

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  flagcxCommDeregister(comm, dataHandle);
  flagcxOneSideDeregister(comm);
  flagcxOneSideSignalDeregister(comm);
  flagcxMemFree(dataWindow, comm);
  flagcxMemFree(signalWindow, comm);
  free(hostStaging);

  devHandle->streamDestroy(stream);
  flagcxCommDestroy(comm);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}
