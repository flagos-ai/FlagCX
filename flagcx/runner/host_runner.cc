/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "runner.h"

static int hostRunnerGroupDepth = 0;
static std::vector<void *> recvHostBuffers;
static std::vector<void *> recvDeviceBuffers;
static std::vector<size_t> recvBufferSizes;

flagcxResult_t hostRunnerReduce(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                flagcxRedOp_t op, int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: reduce
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->reduce(
      buffIn, buffOut, count, datatype, op, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  if (comm->rank == root) {
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        recvbuff, buffOut, size, flagcxMemcpyHostToDevice, NULL, NULL));
  }
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s Reduce: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerGather(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: gather
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->gather(
      buffIn, buffOut, count, datatype, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, totalSize, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s gather: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerScatter(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxComm_t comm,
                                 flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, totalSize, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          totalSize, flagcxMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: scatter
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->scatter(
      buffIn, buffOut, count, datatype, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s Scatter: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   int root, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: broadcast
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->broadcast(
      buffIn, buffOut, count, datatype, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s Broadcast: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: allreduce
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->allReduce(
      buffIn, buffOut, count, datatype, op, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s AllReduce: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                       size_t recvcount,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxComm_t comm,
                                       flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t recvSize = recvcount * getFlagcxDataTypeSize(datatype);
  size_t sendSize = comm->nranks * recvSize;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, sendSize, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, recvSize, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          sendSize, flagcxMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: reducescatter
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->reduceScatter(
      buffIn, buffOut, recvcount, datatype, op, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, recvSize, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s ReduceScatter: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerAllGather(const void *sendbuff, void *recvbuff,
                                   size_t sendcount, flagcxDataType_t datatype,
                                   flagcxComm_t comm, flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = sendcount * getFlagcxDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, totalSize, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: allgather
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->allGather(
      buffIn, buffOut, sendcount, datatype, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, totalSize, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s AllGather: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = comm->nranks * count * getFlagcxDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: alltoall
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->alltoAll(
      buffIn, buffOut, count, datatype, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s AlltoAll: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                   size_t *sdispls, void *recvbuff,
                                   size_t *recvcounts, size_t *rdispls,
                                   flagcxDataType_t datatype, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  // Calculate max possible size needed for send and receive buffers
  size_t maxSendSize = 0, maxRecvSize = 0, sendSize = 0, recvSize = 0;
  for (int i = 0; i < comm->nranks; i++) {
    sendSize = (sendcounts[i] + sdispls[i]) * getFlagcxDataTypeSize(datatype);
    recvSize = (recvcounts[i] + rdispls[i]) * getFlagcxDataTypeSize(datatype);
    if (sendSize > maxSendSize)
      maxSendSize = sendSize;
    if (recvSize > maxRecvSize)
      maxRecvSize = recvSize;
  }

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, maxSendSize, 0));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, maxRecvSize, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          maxSendSize, flagcxMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: alltoallv
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->alltoAllv(
      buffIn, sendcounts, sdispls, buffOut, recvcounts, rdispls, datatype,
      comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, maxRecvSize, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s AlltoAllv: rank %d nranks %d total %.2fms "
       "(memory alloc %.2fms, memory free %.2fms, memory d2h %.2fms, "
       "memory h2d %.2fms, comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerSend(const void *sendbuff, size_t count,
                              flagcxDataType_t datatype, int peer,
                              flagcxComm_t comm, flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: send
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->send(
      buffIn, count, datatype, peer, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s Send: rank %d nranks %d total %.2fms (memory "
       "alloc "
       "%.2fms, memory d2h %.2fms, comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_MEM_D2H] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerRecv(void *recvbuff, size_t count,
                              flagcxDataType_t datatype, int peer,
                              flagcxComm_t comm, flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffOut;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: recv
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->recv(
      buffOut, count, datatype, peer, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 3: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  if (hostRunnerGroupDepth == 0) {
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        recvbuff, buffOut, size, flagcxMemcpyHostToDevice, NULL, NULL));
  } else {
    recvHostBuffers.push_back(buffOut);
    recvDeviceBuffers.push_back(recvbuff);
    recvBufferSizes.push_back(size);
  }
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 4: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "Flagcx timings - %s Recv: rank %d nranks %d total %.2fms (memory "
       "alloc "
       "%.2fms, memory free %.2fms, memory h2d %.2fms, comm %.2fms)",
       cclAdaptors[flagcxCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_H2D] / 1e6,
       timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

flagcxResult_t hostRunnerGroupStart() {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->groupStart());
  hostRunnerGroupDepth++;
  return flagcxSuccess;
}

flagcxResult_t hostRunnerGroupEnd() {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->groupEnd());
  hostRunnerGroupDepth--;
  if (hostRunnerGroupDepth == 0) {
    for (size_t i = 0; i < recvHostBuffers.size(); ++i) {
      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
          recvDeviceBuffers[i], recvHostBuffers[i], recvBufferSizes[i],
          flagcxMemcpyHostToDevice, NULL, NULL));
    }
    recvHostBuffers.clear();
    recvDeviceBuffers.clear();
    recvBufferSizes.clear();
  }
  return flagcxSuccess;
}

struct flagcxRunner hostRunner = {
    // Communication functions
    hostRunnerReduce, hostRunnerGather, hostRunnerScatter, hostRunnerBroadcast,
    hostRunnerAllReduce, hostRunnerReduceScatter, hostRunnerAllGather,
    hostRunnerAlltoAll, hostRunnerAlltoAllv, hostRunnerSend, hostRunnerRecv,
    // Group semantics
    hostRunnerGroupStart, hostRunnerGroupEnd};