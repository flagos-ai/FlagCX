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
      printf("test_get requires at least 2 MPI processes\n");
    MPI_Finalize();
    return 0;
  }

  const int providerRank = 0;
  const int initiatorRank = 1;
  if (totalProcs != 2) {
    if (proc == 0)
      printf(
          "test_get requires exactly 2 ranks (provider=0, initiator=1).\n");
    MPI_Finalize();
    return 0;
  }

  bool isProvider = (proc == providerRank);
  bool isInitiator = (proc == initiatorRank);

  // Allocate data buffer for one-sided GET operations
  size_t max_iterations = std::max(num_warmup_iters, num_iters);
  size_t window_bytes = max_bytes * max_iterations;

  void *window = nullptr;
  if (posix_memalign(&window, 64, window_bytes) != 0 || window == nullptr) {
    fprintf(stderr,
            "[rank %d] posix_memalign failed for host window (size=%zu)\n",
            proc, window_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::memset(window, 0, window_bytes);

  void *windowHandle = nullptr;
  flagcxCommRegister(comm, window, window_bytes, &windowHandle);
  FLAGCXCHECK(flagcxOneSideRegister(comm, window, window_bytes));

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  void *hello = malloc(max_bytes);
  memset(hello, 0, max_bytes);

  // Warm-up iterations
  for (int i = 0; i < num_warmup_iters; ++i) {
    size_t current_offset = i * max_bytes;

    if (isProvider) {
      uint8_t value = static_cast<uint8_t>((providerRank + i) & 0xff);
      std::memset((char *)window + current_offset, value, max_bytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (isInitiator) {
      FLAGCXCHECK(flagcxGet(comm, providerRank, current_offset, current_offset,
                            max_bytes / sizeof(float), DATATYPE, stream));
    }
  }

  // Benchmark loop
  timer tim;
  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    if (size == 0)
      break;

    size_t count = size / sizeof(float);

    if (isProvider) {
      strcpy((char *)hello, "_0x1234");
      strcpy((char *)hello + size / 3, "_0x5678");
      strcpy((char *)hello + size / 3 * 2, "_0x9abc");

      if (proc == 0 && color == 0 && print_buffer) {
        printf("sendbuff = ");
        printf("%s", (const char *)((char *)hello));
        printf("%s", (const char *)((char *)hello + size / 3));
        printf("%s\n", (const char *)((char *)hello + size / 3 * 2));
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; ++i) {
      size_t current_src_offset = i * size;
      size_t current_dst_offset = i * size;

      if (isProvider) {
        uint8_t value = static_cast<uint8_t>((providerRank + i) & 0xff);
        std::memset((char *)window + current_src_offset, value, size);
        memcpy(hello, (char *)window + current_src_offset, size);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      if (isInitiator) {
        FLAGCXCHECK(flagcxGet(comm, providerRank, current_src_offset,
                              current_dst_offset, count, DATATYPE, stream));
      }
    }
    devHandle->streamSynchronize(stream);

    double elapsed_time = tim.elapsed() / num_iters;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed_time /= worldSize;

    double bandwidth = (double)size / 1.0e9 / elapsed_time;
    if (proc == 0 && color == 0) {
      printf("Size: %zu bytes; Avg time: %lf sec; Bandwidth: %lf GB/s\n", size,
             elapsed_time, bandwidth);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (isInitiator && num_iters > 0) {
      memset(hello, 0, size);
      memcpy(hello, (char *)window + 0, size);
      if (proc == 1 && color == 0 && print_buffer) {
        printf("recvbuff = ");
        printf("%s", (const char *)((char *)hello));
        printf("%s", (const char *)((char *)hello + size / 3));
        printf("%s\n", (const char *)((char *)hello + size / 3 * 2));
      }
    }
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  free(hello);
  flagcxOneSideDeregister(comm);

  if (windowHandle != nullptr) {
    flagcxCommDeregister(comm, windowHandle);
  }
  free(window);

  devHandle->streamDestroy(stream);
  flagcxCommDestroy(comm);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}
