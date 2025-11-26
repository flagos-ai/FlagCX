/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "runner.h"

flagcxResult_t hostRunnerReduce(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                flagcxRedOp_t op, int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buff_in;
  void *buff_out;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, size, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: reduce
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->reduce(
      buff_in, buff_out, count, datatype, op, root, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  if (comm->rank == root) {
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        recvbuff, buff_out, size, flagcxMemcpyHostToDevice, NULL, NULL));
  }
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, size, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, totalSize, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: gather
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->gather(
      buff_in, buff_out, count, datatype, root, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, totalSize, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_in, totalSize, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          totalSize, flagcxMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: scatter
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->scatter(
      buff_in, buff_out, count, datatype, root, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, size, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: broadcast
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->broadcast(
      buff_in, buff_out, count, datatype, root, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, size, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: allreduce
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->allReduce(
      buff_in, buff_out, count, datatype, op, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  size_t recv_size = recvcount * getFlagcxDataTypeSize(datatype);
  size_t send_size = comm->nranks * recv_size;

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_in, send_size, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, recv_size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          send_size, flagcxMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: reducescatter
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->reduceScatter(
      buff_in, buff_out, recvcount, datatype, op, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, recv_size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  size_t size = sendcount * getFlagcxDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, size, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, totalSize, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: allgather
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->allGather(
      buff_in, buff_out, sendcount, datatype, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, totalSize, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  size_t size = comm->nranks * count * getFlagcxDataTypeSize(datatype);

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, size, flagcxMemHost, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: alltoall
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->alltoAll(
      buff_in, buff_out, count, datatype, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  void *buff_out;
  // Calculate max possible size needed for send and receive buffers
  size_t max_send_size = 0, max_recv_size = 0, send_size = 0, recv_size = 0;
  for (int i = 0; i < comm->nranks; i++) {
    send_size = (sendcounts[i] + sdispls[i]) * getFlagcxDataTypeSize(datatype);
    recv_size = (recvcounts[i] + rdispls[i]) * getFlagcxDataTypeSize(datatype);
    if (send_size > max_send_size)
      max_send_size = send_size;
    if (recv_size > max_recv_size)
      max_recv_size = recv_size;
  }
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, max_send_size,
                                          flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_out, max_recv_size,
                                          flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      buff_in, const_cast<void *>(sendbuff), max_send_size,
      flagcxMemcpyDeviceToHost, NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->alltoAllv(
      buff_in, sendcounts, sdispls, buff_out, recvcounts, rdispls, datatype,
      comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, max_recv_size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  void *buff_in;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&buff_in, size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buff_in, const_cast<void *>(sendbuff),
                                          size, flagcxMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: send
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->send(
      buff_in, count, datatype, peer, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // buff_in will be freed in gloo adaptor send function?
  // TODO: check if buff_in should be freed here
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_in, flagcxMemHost, NULL));
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
  void *buff_out;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  // step 1: malloc host buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  FLAGCXCHECK(
      deviceAdaptor->deviceMalloc(&buff_out, size, flagcxMemHost, NULL));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: recv
  timers[TIMER_COLL_COMM] = clockNano();
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->recv(
      buff_out, count, datatype, peer, comm->host_comm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 3: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buff_out, size, flagcxMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 4: free host buffer
  timers[TIMER_COLL_FREE] = clockNano();
  FLAGCXCHECK(deviceAdaptor->deviceFree(buff_out, flagcxMemHost, NULL));
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
  return flagcxSuccess;
}

flagcxResult_t hostRunnerGroupEnd() {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->groupEnd());
  return flagcxSuccess;
}

struct flagcxRunner hostRunner = {
    // Communication functions
    hostRunnerReduce, hostRunnerGather, hostRunnerScatter, hostRunnerBroadcast,
    hostRunnerAllReduce, hostRunnerReduceScatter, hostRunnerAllGather,
    hostRunnerAlltoAll, hostRunnerAlltoAllv, hostRunnerSend, hostRunnerRecv,
    // Group semantics
    hostRunnerGroupStart, hostRunnerGroupEnd};