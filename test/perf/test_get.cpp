#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>

// test_get: RDMA READ benchmark (flagcxGet)
//
// Protocol:
//   rank 0 (producer): fills local data buffer, then calls flagcxSignal to
//                       notify rank 1 that data is ready.
//   rank 1 (getter):   waits for the signal via flagcxWaitSignal, then
//                       calls flagcxGet to RDMA-READ from rank 0's buffer
//                       into its own local buffer.
//
// Both ranks register their data window (flagcxOneSideRegister) and signal
// buffer (flagcxOneSideSignalRegister) before the benchmark loop.

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
  int local_register = args.getLocalRegister();

  // RMA requires flagcxMemAlloc (GDR memory with SYNC_MEMOPS)
  if (local_register < 1) {
    fprintf(stderr,
            "test_get requires -R 1 or -R 2 for GDR buffer allocation.\n");
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

  int isHomo = 0;
  flagcxIsHomoComm(comm, &isHomo);
  if (isHomo) {
    if (proc == 0)
      printf("Skipping get benchmark: hetero communicator not initialised "
             "(isHomo=%d).\n",
             isHomo);
    flagcxCommDestroy(comm);
    flagcxHandleFree(handler);
    MPI_Finalize();
    return 0;
  }

  if (totalProcs != 2) {
    if (proc == 0)
      printf("test_get requires exactly 2 ranks (producer=0, getter=1).\n");
    MPI_Finalize();
    return 0;
  }

  const int producerRank = 0;
  const int getterRank = 1;

  bool isProducer = (proc == producerRank);
  bool isGetter = (proc == getterRank);

  flagcxResult_t res;

  size_t signalBytes = sizeof(uint64_t);
  size_t total_iters_per_size = num_warmup_iters + num_iters;
  size_t max_data_iters = std::max(num_warmup_iters, num_iters);
  size_t data_bytes = max_bytes * max_data_iters;
  size_t signal_total_bytes = signalBytes * total_iters_per_size;

  // Data buffer: GDR memory (SYNC_MEMOPS ensures NIC visibility via GDR BAR)
  // Producer: stores data to be read.  Getter: receives data via RDMA READ.
  void *dataWindow = nullptr;
  res = flagcxMemAlloc(&dataWindow, data_bytes);
  if (res != flagcxSuccess || dataWindow == nullptr) {
    fprintf(stderr, "[rank %d] flagcxMemAlloc failed for data (size=%zu)\n",
            proc, data_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(dataWindow, 0, data_bytes, flagcxMemDevice, NULL);

  // Signal buffer: producer signals getter when each chunk is ready.
  void *signalWindow = nullptr;
  res = flagcxMemAlloc(&signalWindow, signal_total_bytes);
  if (res != flagcxSuccess || signalWindow == nullptr) {
    fprintf(stderr, "[rank %d] flagcxMemAlloc failed for signal (size=%zu)\n",
            proc, signal_total_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(signalWindow, 0, signal_total_bytes, flagcxMemDevice,
                          NULL);

  // Register data buffer in global reg pool
  void *dataHandle = nullptr;
  res = flagcxCommRegister(comm, dataWindow, data_bytes, &dataHandle);
  fatal(res, "flagcxCommRegister (data) failed", proc);

  // Register data buffer for one-sided operations (MR index 0)
  res = flagcxOneSideRegister(comm, dataWindow, data_bytes);
  if (res == flagcxNotSupported) {
    if (proc == 0)
      printf("Skipping get benchmark: net adaptor does not support iget.\n");
    flagcxCommDeregister(comm, dataHandle);
    flagcxMemFree(dataWindow);
    flagcxMemFree(signalWindow);
    flagcxCommDestroy(comm);
    flagcxHandleFree(handler);
    MPI_Finalize();
    return 0;
  }
  fatal(res, "flagcxOneSideRegister (data) failed", proc);

  // Register signal buffer for one-sided operations
  res = flagcxOneSideSignalRegister(comm, signalWindow, signal_total_bytes);
  fatal(res, "flagcxOneSideSignalRegister failed", proc);

  // Dummy send/recv to establish full-mesh connections used by one-sided ops
  flagcxStream_t stream;
  devHandle->streamCreate(&stream);
  void *dummyBuff = nullptr;
  devHandle->deviceMalloc(&dummyBuff, 1, flagcxMemDevice, NULL);

  flagcxGroupStart(comm);
  if (isProducer) {
    flagcxSend(dummyBuff, 1, flagcxChar, getterRank, comm, stream);
  } else if (isGetter) {
    flagcxRecv(dummyBuff, 1, flagcxChar, producerRank, comm, stream);
  }
  flagcxGroupEnd(comm);

  devHandle->streamSynchronize(stream);
  devHandle->deviceFree(dummyBuff, flagcxMemDevice, NULL);
  devHandle->streamDestroy(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Host staging buffer for producer data fill and getter verification
  void *hostStaging = nullptr;
  if (posix_memalign(&hostStaging, 64, max_bytes) != 0 ||
      hostStaging == nullptr) {
    fprintf(stderr, "[rank %d] posix_memalign failed for staging (size=%zu)\n",
            proc, max_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Stream used by getter for streamWaitValue64 (flagcxWaitSignal)
  flagcxStream_t waitStream = nullptr;
  if (isGetter) {
    devHandle->streamCreate(&waitStream);
  }

  // Benchmark loop
  timer tim;
  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    if (size == 0)
      break;

    // Reset signal buffer before each size iteration
    devHandle->deviceMemset(signalWindow, 0, signal_total_bytes,
                            flagcxMemDevice, NULL);
    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup iterations (signal slots [0 .. num_warmup_iters-1])
    for (int i = 0; i < num_warmup_iters; ++i) {
      size_t signalOffset = i * signalBytes;
      size_t dataOffset = i * size; // same logical offset for both sides

      if (isProducer) {
        // Fill data then signal getter that chunk is ready
        uint8_t value = static_cast<uint8_t>((producerRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + dataOffset, hostStaging,
                                size, flagcxMemcpyHostToDevice, NULL);

        res = flagcxSignal(comm, getterRank, signalOffset, 1);
        fatal(res, "flagcxSignal warmup failed", proc);
      } else if (isGetter) {
        // Wait for producer's signal then RDMA READ from producer's buffer
        res = flagcxWaitSignal(comm, producerRank, signalOffset, 1, waitStream);
        fatal(res, "flagcxWaitSignal warmup failed", proc);
        devHandle->streamSynchronize(waitStream);

        res = flagcxGet(comm, producerRank, dataOffset, dataOffset, size, 0, 0);
        fatal(res, "flagcxGet warmup failed", proc);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tim.reset();

    // Benchmark iterations (signal slots [num_warmup_iters .. total_iters-1])
    for (int i = 0; i < num_iters; ++i) {
      size_t signalOffset = (num_warmup_iters + i) * signalBytes;
      size_t dataOffset = i * size;

      if (isProducer) {
        uint8_t value = static_cast<uint8_t>((producerRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + dataOffset, hostStaging,
                                size, flagcxMemcpyHostToDevice, NULL);

        res = flagcxSignal(comm, getterRank, signalOffset, 1);
        fatal(res, "flagcxSignal failed", proc);
      } else if (isGetter) {
        res = flagcxWaitSignal(comm, producerRank, signalOffset, 1, waitStream);
        fatal(res, "flagcxWaitSignal failed", proc);
        devHandle->streamSynchronize(waitStream);

        res = flagcxGet(comm, producerRank, dataOffset, dataOffset, size, 0, 0);
        fatal(res, "flagcxGet failed", proc);

        if (print_buffer) {
          devHandle->deviceMemcpy(hostStaging, (char *)dataWindow + dataOffset,
                                  std::min(size, (size_t)64),
                                  flagcxMemcpyDeviceToHost, NULL);
          printf("[rank %d] Got data at offset %zu, size %zu:\n", proc,
                 dataOffset, size);
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
  res = flagcxCommDeregister(comm, dataHandle);
  fatal(res, "flagcxCommDeregister failed", proc);

  flagcxOneSideDeregister(comm);
  flagcxOneSideSignalDeregister(comm);
  flagcxMemFree(dataWindow);
  flagcxMemFree(signalWindow);
  free(hostStaging);

  if (waitStream != nullptr) {
    devHandle->streamDestroy(waitStream);
  }

  fatal(flagcxCommDestroy(comm), "flagcxCommDestroy failed", proc);
  fatal(flagcxHandleFree(handler), "flagcxHandleFree failed", proc);

  MPI_Finalize();
  return 0;
}
