#include "flagcx.h"
#include "flagcx_hetero.h"
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

  flagcxHandlerGroup_t handler;
  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
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
    flagcxHeteroGetUniqueId(uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxHeteroComm_t comm;
  flagcxHeteroCommInitRank(&comm, totalProcs, *uniqueId, proc);

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  void *sendbuff = nullptr;
  void *recvbuff = nullptr;
  void *hello = nullptr;
  void *selfsendbuff1 = nullptr;
  void *selfsendbuff2 = nullptr;
  void *selfrecvbuff1 = nullptr;
  void *selfrecvbuff2 = nullptr;
  timer tim;
  int peerSend = (proc + 1) % totalProcs;
  int peerRecv = (proc - 1 + totalProcs) % totalProcs;
  int selfPeer = proc;

  devHandle->deviceMalloc(&sendbuff, max_bytes, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, max_bytes, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfsendbuff1, 100, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfsendbuff2, 200, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfrecvbuff1, 100, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfrecvbuff2, 200, flagcxMemDevice, NULL);
  hello = malloc(max_bytes);
  memset(hello, 0, max_bytes);

  // Warm-up for large size
  for (int i = 0; i < num_warmup_iters; i++) {
    flagcxHeteroGroupStart();
    flagcxHeteroSend(sendbuff, max_bytes, flagcxChar, peerSend, comm, stream);
    flagcxHeteroRecv(recvbuff, max_bytes, flagcxChar, peerRecv, comm, stream);
    flagcxHeteroGroupEnd();
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < num_warmup_iters; i++) {
    flagcxHeteroGroupStart();
    flagcxHeteroSend(sendbuff, min_bytes, flagcxChar, peerSend, comm, stream);
    flagcxHeteroRecv(recvbuff, min_bytes, flagcxChar, peerRecv, comm, stream);
    flagcxHeteroGroupEnd();
  }
  devHandle->streamSynchronize(stream);
  void *testdata1 = malloc(100);
  void *testdata2 = malloc(200);
  memset(testdata1, 0xAA, 100);
  memset(testdata2, 0xBB, 200);
  devHandle->deviceMemcpy(selfsendbuff1, testdata1, 100,
                          flagcxMemcpyHostToDevice, NULL);
  devHandle->deviceMemcpy(selfsendbuff2, testdata2, 200,
                          flagcxMemcpyHostToDevice, NULL);
  memset(testdata1, 0, 100);
  memset(testdata2, 0, 200);
  devHandle->deviceMemcpy(selfrecvbuff1, testdata1, 100,
                          flagcxMemcpyHostToDevice, NULL);
  devHandle->deviceMemcpy(selfrecvbuff2, testdata2, 200,
                          flagcxMemcpyHostToDevice, NULL);

  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {

    for (size_t i = 0; i + 13 <= size; i += 13) {
      strcpy((char *)hello + i, std::to_string(i / (13)).c_str());
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);

    if (proc == 0 && color == 0 && print_buffer) {
      printf("sendbuff = ");
      for (size_t i = 0; i + 13 <= 50; i += 13) {
        printf("%c", ((char *)hello)[i]);
      }
      printf("\n");
      memset(testdata1, 0, 100);
      devHandle->deviceMemcpy(testdata1, selfsendbuff1, 100,
                              flagcxMemcpyDeviceToHost, NULL);
      printf("selfsendbuff1 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata1)[i]);
      }
      printf("\n");
      memset(testdata2, 0, 200);
      devHandle->deviceMemcpy(testdata2, selfsendbuff2, 200,
                              flagcxMemcpyDeviceToHost, NULL);
      printf("selfsendbuff2 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata2)[i]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; i++) {
      flagcxHeteroGroupStart();
      flagcxHeteroSend(sendbuff, size, flagcxChar, peerSend, comm, stream);
      flagcxHeteroRecv(recvbuff, size, flagcxChar, peerRecv, comm, stream);
      flagcxHeteroSend(selfsendbuff1, 100, flagcxChar, selfPeer, comm, stream);
      flagcxHeteroSend(selfsendbuff2, 200, flagcxChar, selfPeer, comm, stream);
      flagcxHeteroRecv(selfrecvbuff2, 200, flagcxChar, selfPeer, comm, stream);
      flagcxHeteroRecv(selfrecvbuff1, 100, flagcxChar, selfPeer, comm, stream);
      flagcxHeteroGroupEnd();
    }
    devHandle->streamSynchronize(stream);

    devHandle->deviceMemcpy(testdata1, selfrecvbuff1, 100,
                            flagcxMemcpyDeviceToHost, NULL);
    devHandle->deviceMemcpy(testdata2, selfrecvbuff2, 200,
                            flagcxMemcpyDeviceToHost, NULL);
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

    if (proc == 0 && color == 0 && print_buffer) {
      memset(hello, 0, size);
      devHandle->deviceMemcpy(hello, recvbuff, size, flagcxMemcpyDeviceToHost,
                              NULL);
      printf("recvbuff = ");
      for (size_t i = 0; i + 13 <= 50; i += 13) {
        printf("%c", ((char *)hello)[i]);
      }
      printf("\n");
      memset(testdata1, 0, 100);
      devHandle->deviceMemcpy(testdata1, selfrecvbuff1, 100,
                              flagcxMemcpyDeviceToHost, NULL);
      printf("selfrecvbuff1 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata1)[i]);
      }
      printf("\n");
      memset(testdata2, 0, 200);
      devHandle->deviceMemcpy(testdata2, selfrecvbuff2, 200,
                              flagcxMemcpyDeviceToHost, NULL);
      printf("selfrecvbuff2 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata2)[i]);
      }
      printf("\n");
    }
  }

  // if (local_register) {
  //   // deregister buffer
  //   flagcxCommDeregister(comm, sendHandle);
  //   flagcxCommDeregister(comm, recvHandle);
  //   // deallocate buffer
  //   flagcxMemFree(sendbuff);
  //   flagcxMemFree(recvbuff);
  // } else {
  devHandle->deviceFree(sendbuff, flagcxMemDevice, NULL);
  devHandle->deviceFree(recvbuff, flagcxMemDevice, NULL);
  devHandle->deviceFree(selfsendbuff1, flagcxMemDevice, NULL);
  devHandle->deviceFree(selfsendbuff2, flagcxMemDevice, NULL);
  devHandle->deviceFree(selfrecvbuff1, flagcxMemDevice, NULL);
  devHandle->deviceFree(selfrecvbuff2, flagcxMemDevice, NULL);
  // }
  free(hello);
  free(testdata1);
  free(testdata2);
  flagcxHeteroCommDestroy(comm);
  devHandle->streamDestroy(stream);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}