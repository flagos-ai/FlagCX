#include "alloc.h"
#include "flagcx.h"
#include "shmutils.h"
#include "tools.h"
#include "utils.h"
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
  // int localRegister = args.getLocalRegister();

  flagcxHandlerGroup_t handler;
  flagcxHandleInit(&handler);
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

  flagcxStream_t stream;
  devHandle->streamCreate(&stream);

  void *sendbuff = nullptr;
  void *recvbuff = nullptr;
  void *hello = nullptr;
  // void *sendHandle = nullptr;
  // void *recvHandle = nullptr;
  timer tim;
  int peerSend = (proc + 1) % totalProcs;

  // if (localRegister) {
  //   // allocate buffer
  //   flagcxMemAlloc(&sendbuff, maxBytes);
  //   flagcxMemAlloc(&recvbuff, maxBytes);
  //   // register buffer
  //   flagcxCommRegister(comm, sendbuff, maxBytes, &sendHandle);
  //   flagcxCommRegister(comm, recvbuff, maxBytes, &recvHandle);
  // } else {
  devHandle->deviceMalloc(&sendbuff, maxBytes, flagcxMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, maxBytes, flagcxMemDevice, NULL);
  // }
  hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);

  // get myIpcHandle from recvbuff
  size_t handleSize;
  flagcxIpcMemHandle_t myIpcHandle;
  devHandle->ipcMemHandleCreate(&myIpcHandle, &handleSize);
  devHandle->ipcMemHandleGet(myIpcHandle, recvbuff);

  // init myShmDesc and myShmPtr
  // copy myIpcHandle to myShmPtr
  flagcxShmIpcDesc_t myShmDesc;
  void *myShmPtr;
  flagcxShmAllocateShareableBuffer(handleSize, &myShmDesc, &myShmPtr, NULL);
  memcpy(myShmPtr, (void *)myIpcHandle, handleSize);
  MPI_Barrier(MPI_COMM_WORLD);

  // use MPI_Allgather to collect all shmDescs
  void *allHandles = malloc(sizeof(flagcxShmIpcDesc_t) * totalProcs);
  MPI_Allgather(&myShmDesc, sizeof(flagcxShmIpcDesc_t), MPI_BYTE, allHandles,
                sizeof(flagcxShmIpcDesc_t), MPI_BYTE, MPI_COMM_WORLD);

  // import to peerShmDesc and peerShmPtr
  flagcxShmIpcDesc_t peerShmDesc;
  void *peerShmPtr;
  flagcxShmImportShareableBuffer((flagcxShmIpcDesc_t *)allHandles + peerSend,
                                 &peerShmPtr, NULL, &peerShmDesc);
  MPI_Barrier(MPI_COMM_WORLD);

  // create peerIpcHandle
  flagcxIpcMemHandle_t peerIpcHandle;
  devHandle->ipcMemHandleCreate(&peerIpcHandle, NULL);
  // copy peerShmPtr to peerIpcHandle
  memcpy((void *)peerIpcHandle, peerShmPtr, handleSize);
  MPI_Barrier(MPI_COMM_WORLD);

  // open peerIpcHandle
  void *peerbuff;
  devHandle->ipcMemHandleOpen(peerIpcHandle, &peerbuff);

  // Warm-up for large size
  for (int i = 0; i < numWarmupIters; i++) {
    devHandle->deviceMemcpy(sendbuff, sendbuff, maxBytes,
                            flagcxMemcpyDeviceToDevice, stream);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < numWarmupIters; i++) {
    devHandle->deviceMemcpy(sendbuff, sendbuff, minBytes,
                            flagcxMemcpyDeviceToDevice, stream);
  }
  devHandle->streamSynchronize(stream);

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {

    strcpy((char *)hello, "_0x1234");
    strcpy((char *)hello + size / 3, "_0x5678");
    strcpy((char *)hello + size / 3 * 2, "_0x9abc");

    devHandle->deviceMemcpy(sendbuff, hello, size, flagcxMemcpyHostToDevice,
                            NULL);

    if (proc == 0 && color == 0 && printBuffer) {
      printf("sendbuff = ");
      printf("%s", (const char *)((char *)hello));
      printf("%s", (const char *)((char *)hello + size / 3));
      printf("%s\n", (const char *)((char *)hello + size / 3 * 2));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      devHandle->deviceMemcpy(peerbuff, sendbuff, size,
                              flagcxMemcpyDeviceToDevice, stream);
    }
    devHandle->streamSynchronize(stream);

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

    memset(hello, 0, size);
    devHandle->deviceMemcpy(hello, recvbuff, size, flagcxMemcpyDeviceToHost,
                            NULL);
    if (proc == 0 && color == 0 && printBuffer) {
      printf("recvbuff = ");
      printf("%s", (const char *)((char *)hello));
      printf("%s", (const char *)((char *)hello + size / 3));
      printf("%s\n", (const char *)((char *)hello + size / 3 * 2));
    }
  }

  // cleanup
  flagcxShmIpcClose(&myShmDesc);
  flagcxShmIpcClose(&peerShmDesc);
  free(allHandles);
  devHandle->ipcMemHandleClose(peerbuff);
  devHandle->ipcMemHandleFree(myIpcHandle);
  devHandle->ipcMemHandleFree(peerIpcHandle);

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
  // }
  free(hello);
  devHandle->streamDestroy(stream);
  flagcxHandleFree(handler);

  MPI_Finalize();
  return 0;
}