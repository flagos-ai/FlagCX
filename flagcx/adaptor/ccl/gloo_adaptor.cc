#include "gloo_adaptor.h"

#ifdef USE_GLOO_ADAPTOR

FLAGCX_PARAM(GlooIbDisable, "GLOO_IB_DISABLE", 0);

static int groupDepth = 0;
static constexpr std::chrono::milliseconds flagcxGlooDefaultTimeout =
    std::chrono::seconds(10000);
static std::vector<stagedBuffer_t> sendStagedBufferList;
static std::vector<stagedBuffer_t> recvStagedBufferList;
static std::vector<bufferPtr> unboundBufferStorage;

// key: peer, value: tag
static std::unordered_map<int, uint32_t> sendPeerTags;
static std::unordered_map<int, uint32_t> recvPeerTags;

// TODO: unsupported
flagcxResult_t glooAdaptorGetVersion(int *version) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t glooAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  stagedBuffer *sbuff = NULL;
  if (isRecv) {
    for (auto it = recvStagedBufferList.begin();
         it != recvStagedBufferList.end(); it++) {
      if ((*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  } else {
    for (auto it = sendStagedBufferList.begin();
         it != sendStagedBufferList.end(); it++) {
      if ((*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  }
  if (sbuff == NULL) {
    FLAGCXCHECK(flagcxCalloc(&sbuff, 1));
    sbuff->offset = 0;
    sbuff->cnt = 0;
    int newSize = GLOO_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
    while (newSize < size) {
      newSize *= 2;
    }
    sbuff->buffer = malloc(newSize);
    if (sbuff->buffer == NULL) {
      return flagcxSystemError;
    }
    sbuff->size = newSize;
    auto unboundBuffer = comm->base->createUnboundBuffer(
        const_cast<void *>(sbuff->buffer), sbuff->size);
    sbuff->unboundBuffer = unboundBuffer.get();
    unboundBufferStorage.push_back(std::move(unboundBuffer));
    if (isRecv) {
      recvStagedBufferList.push_back(sbuff);
    } else {
      sendStagedBufferList.push_back(sbuff);
    }
  }
  *buff = (void *)((char *)sbuff->buffer + sbuff->offset);
  return flagcxSuccess;
}

// TODO: unsupported
const char *glooAdaptorGetErrorString(flagcxResult_t result) {
  return "Not Implemented";
}

// TODO: unsupported
const char *glooAdaptorGetLastError(flagcxInnerComm_t comm) {
  return "Not Implemented";
}

flagcxResult_t glooAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t /*commId*/, int rank,
                                       bootstrapState *bootstrap) {
  // Create gloo transport device
  std::shared_ptr<::gloo::transport::Device> dev;
  flagcxNetProperties_t *properties =
      (flagcxNetProperties_t *)bootstrap->properties;
  if (flagcxParamGlooIbDisable()) {
    // Use transport tcp
    ::gloo::transport::tcp::attr attr;
    attr.iface = std::string(bootstrap->bootstrapNetIfName);
    dev = ::gloo::transport::tcp::CreateDevice(attr);
  } else {
    // Use transport ibverbs
    ::gloo::transport::ibverbs::attr attr;
    attr.name = properties->name;
    attr.port = properties->port;
    attr.index = 3; // default index
    const char *ibGidIndex = flagcxGetEnv("FLAGCX_IB_GID_INDEX");
    if (ibGidIndex != NULL) {
      attr.index = std::stoi(ibGidIndex);
    }
    dev = ::gloo::transport::ibverbs::CreateDevice(attr);
  }
  if (*comm == NULL) {
    FLAGCXCHECK(flagcxCalloc(comm, 1));
  }
  // Create gloo context
  (*comm)->base = std::make_shared<flagcxGlooContext>(rank, nranks, bootstrap);
  (*comm)->base->connectFullMesh(dev);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommFinalize(flagcxInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  unboundBufferStorage.clear();
  comm->base.reset();
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommDestroy(flagcxInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  unboundBufferStorage.clear();
  comm->base.reset();
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommAbort(flagcxInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  unboundBufferStorage.clear();
  comm->base.reset();
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t glooAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t glooAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  *count = comm->base->size;
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  device = NULL;
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  *rank = comm->base->rank;
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t glooAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t glooAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t glooAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t glooAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t glooAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t *win,
                                             int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t /*stream*/) {
  ::gloo::ReduceOptions opts(comm->base);
  opts.setRoot(root);
  opts.setReduceFunction(
      getFunction<::gloo::ReduceOptions::Func>(datatype, op));
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  ::gloo::reduce(opts);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t /*stream*/) {
  ::gloo::GatherOptions opts(comm->base);
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      count);
  // Set output pointer only when root
  if (root == comm->base->rank) {
    GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff,
                        comm->base->size * count);
  }
  opts.setRoot(root);
  ::gloo::gather(opts);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t /*stream*/) {
  ::gloo::ScatterOptions opts(comm->base);
  // one pointer per rank
  std::vector<void *> sendPtrs(comm->base->size);
  for (int i = 0; i < comm->base->size; ++i) {
    sendPtrs[i] = static_cast<void *>(
        (char *)sendbuff + i * count * getFlagcxDataTypeSize(datatype));
  }
  GENERATE_GLOO_TYPES(datatype, setInputs, opts,
                      const_cast<void **>(sendPtrs.data()), comm->base->size,
                      count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  opts.setRoot(root);
  ::gloo::scatter(opts);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  ::gloo::BroadcastOptions opts(comm->base);
  // Set input pointer only when root
  if (root == comm->base->rank) {
    GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                        count);
  }
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  opts.setRoot(root);
  ::gloo::broadcast(opts);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  ::gloo::AllreduceOptions opts(comm->base);
  opts.setReduceFunction(
      getFunction<::gloo::AllreduceOptions::Func>(datatype, op));
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  ::gloo::allreduce(opts);
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t
glooAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t /*stream*/) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  ::gloo::AllgatherOptions opts(comm->base);
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      sendcount);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff,
                      comm->base->size * sendcount);
  ::gloo::allgather(opts);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t /*stream*/) {
  ::gloo::AlltoallOptions opts(comm->base);
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      comm->base->size * count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff,
                      comm->base->size * count);
  ::gloo::alltoall(opts);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  // Note that sdispls and rdispls are not used in Gloo.
  ::gloo::AlltoallvOptions opts(comm->base);
  std::vector<int64_t> sendCnt(comm->base->size);
  std::vector<int64_t> recvCnt(comm->base->size);
  for (int i = 0; i < comm->base->size; ++i) {
    sendCnt[i] = sendcounts[i];
    recvCnt[i] = recvcounts[i];
  }

  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      sendCnt);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, recvCnt);
  ::gloo::alltoallv(opts);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm,
                               flagcxStream_t /*stream*/) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  stagedBuffer_t buff = sendStagedBufferList.back();
  uint32_t utag;
  if (sendPeerTags.find(peer) != sendPeerTags.end()) {
    utag = sendPeerTags[peer];
  } else {
    utag = 0;
    sendPeerTags[peer] = 0;
  }
  buff->unboundBuffer->send(peer, utag, buff->offset, size);
  buff->offset += size;
  sendPeerTags[peer] = utag + 1;
  if (groupDepth == 0) {
    buff->unboundBuffer->waitSend(flagcxGlooDefaultTimeout);
  } else {
    buff->cnt++;
  }
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm,
                               flagcxStream_t /*stream*/) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  stagedBuffer_t buff = recvStagedBufferList.back();
  uint32_t utag;
  if (recvPeerTags.find(peer) != recvPeerTags.end()) {
    utag = recvPeerTags[peer];
  } else {
    utag = 0;
    recvPeerTags[peer] = 0;
  }
  buff->unboundBuffer->recv(peer, utag, buff->offset, size);
  buff->offset += size;
  recvPeerTags[peer] = utag + 1;
  if (groupDepth == 0) {
    buff->unboundBuffer->waitRecv(flagcxGlooDefaultTimeout);
  } else {
    buff->cnt++;
  }
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorGroupStart() {
  groupDepth++;
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorGroupEnd() {
  groupDepth--;
  if (groupDepth == 0) {
    for (size_t i = 0; i < sendStagedBufferList.size(); ++i) {
      stagedBuffer *buff = sendStagedBufferList[i];
      while (buff->cnt > 0) {
        buff->unboundBuffer->waitSend(flagcxGlooDefaultTimeout);
        buff->cnt--;
      }
      buff->offset = 0;
    }
    for (size_t i = 0; i < recvStagedBufferList.size(); ++i) {
      stagedBuffer *buff = recvStagedBufferList[i];
      while (buff->cnt > 0) {
        buff->unboundBuffer->waitRecv(flagcxGlooDefaultTimeout);
        buff->cnt--;
      }
      buff->offset = 0;
    }
    sendPeerTags.clear();
    recvPeerTags.clear();
  }
  return flagcxSuccess;
}

struct flagcxCCLAdaptor glooAdaptor = {
    "GLOO",
    // Basic functions
    glooAdaptorGetVersion, glooAdaptorGetUniqueId, glooAdaptorGetErrorString,
    glooAdaptorGetLastError, glooAdaptorGetStagedBuffer,
    // Communicator functions
    glooAdaptorCommInitRank, glooAdaptorCommFinalize, glooAdaptorCommDestroy,
    glooAdaptorCommAbort, glooAdaptorCommResume, glooAdaptorCommSuspend,
    glooAdaptorCommCount, glooAdaptorCommCuDevice, glooAdaptorCommUserRank,
    glooAdaptorCommGetAsyncError, glooAdaptorMemAlloc, glooAdaptorMemFree,
    glooAdaptorCommRegister, glooAdaptorCommDeregister,
    // Symmetric functions
    glooAdaptorCommWindowRegister, glooAdaptorCommWindowDeregister,
    // Communication functions
    glooAdaptorReduce, glooAdaptorGather, glooAdaptorScatter,
    glooAdaptorBroadcast, glooAdaptorAllReduce, glooAdaptorReduceScatter,
    glooAdaptorAllGather, glooAdaptorAlltoAll, glooAdaptorAlltoAllv,
    glooAdaptorSend, glooAdaptorRecv,
    // Group semantics
    glooAdaptorGroupStart, glooAdaptorGroupEnd};

#endif // USE_GLOO_ADAPTOR