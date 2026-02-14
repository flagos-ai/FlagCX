#include "flagcx.h"
#include "flagcx_kernel.h"
#include "tools.h"
#include <cstring>
#include <iostream>

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

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  void *sendbuff, *recvbuff, *hello;
  void *sendHandle, *recvHandle;
  size_t count;
  timer tim;

  if (local_register) {
    // allocate buffer
    flagcxMemAlloc(&sendbuff, max_bytes);
    flagcxMemAlloc(&recvbuff, max_bytes);
    // register buffer
    flagcxCommRegister(comm, sendbuff, max_bytes, &sendHandle);
    flagcxCommRegister(comm, recvbuff, max_bytes, &recvHandle);
  } else {
    devHandle->deviceMalloc(&sendbuff, max_bytes, flagcxMemDevice, NULL);
    devHandle->deviceMalloc(&recvbuff, max_bytes, flagcxMemDevice, NULL);
  }
  hello = malloc(max_bytes);
  memset(hello, 0, max_bytes);

  // Warm-up for large size
  // count is per-peer, total buffer = nRanks * count elements
  for (int i = 0; i < num_warmup_iters; i++) {
    // launch p2p kernel
    flagcxP2pDemo(sendbuff, recvbuff, max_bytes / sizeof(float) / totalProcs,
                  DATATYPE, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < num_warmup_iters; i++) {
    // launch p2p kernel
    flagcxP2pDemo(sendbuff, recvbuff, min_bytes / sizeof(float) / totalProcs,
                  DATATYPE, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    // count is per-peer elements; total buffer = nRanks * count
    count = size / sizeof(float) / totalProcs;

    // Initialize sendbuff: all elements = proc (my rank)
    // After alltoall, recvbuff[p] should contain p's rank value
    float *helloFloat = (float *)hello;
    size_t totalElements = size / sizeof(float);
    for (size_t i = 0; i < totalElements; i++) {
      helloFloat[i] = (float)proc;
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);

    // Print sendbuff from rank 0 and last rank
    if (color == 0 && print_buffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d sendbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; i++) {
      // launch p2p kernel
      flagcxP2pDemo(sendbuff, recvbuff, count, DATATYPE, comm, stream);
    }
    devHandle->streamSynchronize(stream);

    double elapsed_time = tim.elapsed() / num_iters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed_time /= worldSize;

    double base_bw = (double)(size) / 1.0E9 / elapsed_time;
    double alg_bw = base_bw;
    double factor = 1;
    double bus_bw = base_bw * factor;
    if (proc == 0 && color == 0) {
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: %lf "
             "GB/s; Bus bandwidth: %lf GB/s\n",
             size, elapsed_time, alg_bw, bus_bw);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    memset(hello, 0, size);
    devHandle->deviceMemcpy(hello, recvbuff, size, flagcxMemcpyDeviceToHost,
                            NULL);
    // Print recvbuff from rank 0 and last rank
    // Expected: 0 1 2 3 ... nRanks-1 (one value per peer)
    if (color == 0 && print_buffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d recvbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }
  }

  // Destroy stream first (sync any pending work)
  devHandle->streamDestroy(stream);

  if (local_register) {
    // deregister buffer (must be done before comm destroy)
    flagcxCommDeregister(comm, sendHandle);
    flagcxCommDeregister(comm, recvHandle);
  }

  // Destroy comm to stop kernel proxy thread BEFORE freeing device memory
  // The kernel proxy thread holds a CUDA stream that can interfere with
  // deviceFree
  flagcxCommDestroy(comm);

  if (local_register) {
    // deallocate buffer
    flagcxMemFree(sendbuff);
    flagcxMemFree(recvbuff);
  } else {
    devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
    devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  }
  free(hello);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}
