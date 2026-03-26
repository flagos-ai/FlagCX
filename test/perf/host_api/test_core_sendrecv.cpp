#include "flagcx.h"
#include "flagcx_hetero.h"
#include "tools.h"
#include <cstring>
#include <iostream>

#define DATATYPE flagcxFloat

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();
  int printBuffer = args.isPrintBuffer();
  uint64_t splitMask = args.getSplitMask();

  flagcxHandlerGroup_t handler;
  flagcxHandleInit(&handler);
  flagcxUniqueId_t &uniqueId = handler->uniqueId;
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

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

  devHandle->deviceMalloc(&sendbuff, maxBytes, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, maxBytes, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfsendbuff1, 100, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfsendbuff2, 200, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfrecvbuff1, 100, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&selfrecvbuff2, 200, flagcxMemDevice, NULL);
  hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);

  // Warm-up for large size
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxHeteroGroupStart();
    flagcxHeteroSend(sendbuff, maxBytes, flagcxChar, peerSend, comm, stream);
    flagcxHeteroRecv(recvbuff, maxBytes, flagcxChar, peerRecv, comm, stream);
    flagcxHeteroGroupEnd();
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < numWarmupIters; i++) {
    flagcxHeteroGroupStart();
    flagcxHeteroSend(sendbuff, minBytes, flagcxChar, peerSend, comm, stream);
    flagcxHeteroRecv(recvbuff, minBytes, flagcxChar, peerRecv, comm, stream);
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

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {

    for (size_t i = 0; i + 13 <= size; i += 13) {
      strcpy((char *)hello + i, std::to_string(i / (13)).c_str());
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);

    if (proc == 0 && color == 0 && printBuffer) {
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
    for (int i = 0; i < numIters; i++) {
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
    double elapsedTime = tim.elapsed() / numIters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;

    double baseBw = (double)(size) / 1.0E9 / elapsedTime;
    double algBw = baseBw;
    double factor = 1;
    double busBw = baseBw * factor;
    if (proc == 0 && color == 0) {
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: %lf "
             "GB/s; Bus bandwidth: %lf GB/s\n",
             size, elapsedTime, algBw, busBw);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (proc == 0 && color == 0 && printBuffer) {
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

  // if (localRegister) {
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