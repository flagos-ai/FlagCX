#include "gloo_adaptor.h"
#include <map>

#ifdef USE_GLOO_ADAPTOR

FLAGCX_PARAM(GlooIbDisable, "GLOO_IB_DISABLE", 0);

static int groupDepth = 0;
static constexpr std::chrono::milliseconds flagcxGlooDefaultTimeout =
    std::chrono::seconds(10000);
static std::vector<stagedBuffer_t> sendStagedBufferList;
static std::vector<stagedBuffer_t> recvStagedBufferList;
static std::vector<bufferPtr> unboundBufferStorage;

// key: owner and peer, value: tag
static std::map<std::pair<flagcxInnerComm_t, int>, uint32_t> sendPeerTags;
static std::map<std::pair<flagcxInnerComm_t, int>, uint32_t> recvPeerTags;

// TODO: unsupported
flagcxResult_t glooAdaptorGetVersion(int *version) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t glooAdaptorGetUniqueId(flagcxInnerUniqueId_t *uniqueId) {
  return flagcxNotSupported;
}

// Internal; requires the adaptor staging types and containers.
static void glooReleaseStagedBuffers(std::vector<stagedBuffer_t> *buffers,
                                     flagcxInnerComm_t owner) {
  for (auto it = buffers->begin(); it != buffers->end();) {
    stagedBuffer_t buffer = *it;
    if (buffer->owner != owner) {
      ++it;
      continue;
    }
    // The native buffer can retain a registration of its backing allocation.
    for (auto native = unboundBufferStorage.begin();
         native != unboundBufferStorage.end(); ++native) {
      if (native->get() == buffer->unboundBuffer) {
        unboundBufferStorage.erase(native);
        break;
      }
    }
    free(buffer->buffer);
    free(buffer);
    it = buffers->erase(it);
  }
}

static flagcxResult_t glooCleanupOwner(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  for (auto *buffer : sendStagedBufferList)
    if (buffer->owner == comm && buffer->cnt != 0)
      return flagcxInProgress;
  for (auto *buffer : recvStagedBufferList)
    if (buffer->owner == comm && buffer->cnt != 0)
      return flagcxInProgress;
  glooReleaseStagedBuffers(&sendStagedBufferList, comm);
  glooReleaseStagedBuffers(&recvStagedBufferList, comm);
  for (auto it = sendPeerTags.begin(); it != sendPeerTags.end();)
    if (it->first.first == comm)
      it = sendPeerTags.erase(it);
    else
      ++it;
  for (auto it = recvPeerTags.begin(); it != recvPeerTags.end();)
    if (it->first.first == comm)
      it = recvPeerTags.erase(it);
    else
      ++it;
  return flagcxSuccess;
}

static stagedBuffer_t glooFindStagedBuffer(flagcxInnerComm_t comm,
                                           const void *pointer, size_t size,
                                           int isRecv) {
  const auto &buffers = isRecv ? recvStagedBufferList : sendStagedBufferList;
  for (auto *buffer : buffers) {
    uintptr_t address = (uintptr_t)pointer;
    uintptr_t begin = (uintptr_t)buffer->buffer;
    if (buffer->owner == comm && address >= begin &&
        address - begin <= (size_t)buffer->size &&
        size <= (size_t)buffer->size - (address - begin))
      return buffer;
  }
  return NULL;
}

flagcxResult_t glooAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  if (comm == NULL || comm->base == nullptr || buff == NULL)
    return flagcxInvalidArgument;
  stagedBuffer *sbuff = NULL;
  if (isRecv) {
    for (auto it = recvStagedBufferList.begin();
         it != recvStagedBufferList.end(); it++) {
      if ((*it)->owner == comm && (*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  } else {
    for (auto it = sendStagedBufferList.begin();
         it != sendStagedBufferList.end(); it++) {
      if ((*it)->owner == comm && (*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  }
  if (sbuff == NULL) {
    FLAGCXCHECK(flagcxCalloc(&sbuff, 1));
    sbuff->owner = comm;
    sbuff->offset = 0;
    sbuff->cnt = 0;
    int newSize = GLOO_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
    while (newSize < size) {
      newSize *= 2;
    }
    sbuff->buffer = malloc(newSize);
    if (sbuff->buffer == NULL) {
      free(sbuff);
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

flagcxResult_t glooAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm,
                               flagcxStream_t /*stream*/) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  if (comm == NULL || comm->base == nullptr)
    return flagcxInvalidArgument;
  stagedBuffer_t buff = glooFindStagedBuffer(comm, sendbuff, size, 0);
  if (buff == NULL)
    return flagcxInvalidArgument;
  size_t offset = (uintptr_t)sendbuff - (uintptr_t)buff->buffer;
  auto key = std::make_pair(comm, peer);
  uint32_t utag;
  if (sendPeerTags.find(key) != sendPeerTags.end()) {
    utag = sendPeerTags[key];
  } else {
    utag = 0;
    sendPeerTags[key] = 0;
  }
  buff->unboundBuffer->send(peer, utag, offset, size);
  buff->offset = offset + size;
  sendPeerTags[key] = utag + 1;
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
  if (comm == NULL || comm->base == nullptr)
    return flagcxInvalidArgument;
  stagedBuffer_t buff = glooFindStagedBuffer(comm, recvbuff, size, 1);
  if (buff == NULL)
    return flagcxInvalidArgument;
  size_t offset = (uintptr_t)recvbuff - (uintptr_t)buff->buffer;
  auto key = std::make_pair(comm, peer);
  uint32_t utag;
  if (recvPeerTags.find(key) != recvPeerTags.end()) {
    utag = recvPeerTags[key];
  } else {
    utag = 0;
    recvPeerTags[key] = 0;
  }
  buff->unboundBuffer->recv(peer, utag, offset, size);
  buff->offset = offset + size;
  recvPeerTags[key] = utag + 1;
  if (groupDepth == 0) {
    buff->unboundBuffer->waitRecv(flagcxGlooDefaultTimeout);
  } else {
    buff->cnt++;
  }
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
                                       flagcxInnerUniqueId_t /*commId*/,
                                       int rank,
                                       struct bootstrapState *bootstrap) {
  // Create gloo transport device
  std::shared_ptr<::gloo::transport::Device> dev;
  flagcxNetProperties_t *properties = bootstrapGetNetProperties();
  if (flagcxParamGlooIbDisable() || flagcxParamTopoDetectionDisable()) {
    // Use transport tcp
    ::gloo::transport::tcp::attr attr;
    attr.iface = std::string(bootstrapGetNetIfName());
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
  flagcxResult_t result = glooCleanupOwner(comm);
  if (result != flagcxSuccess)
    return result;
  comm->base.reset();
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommDestroy(flagcxInnerComm_t comm) {
  flagcxResult_t result = glooAdaptorCommFinalize(comm);
  if (result != flagcxSuccess)
    return result;
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t glooAdaptorCommAbort(flagcxInnerComm_t comm) {
  return glooAdaptorCommFinalize(comm);
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
                                             size_t size,
                                             flagcxInnerWindow_t *win,
                                             int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxInnerWindow_t win) {
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

flagcxResult_t
glooAdaptorDevCommReqsInit(flagcxInnerComm_t /*comm*/,
                           flagcxDevCommRequirements * /*reqs*/) {
  return flagcxNotSupported;
}

flagcxResult_t
glooAdaptorDevCommCreate(flagcxInnerComm_t /*comm*/,
                         const flagcxDevCommRequirements * /*reqs*/,
                         flagcxInnerDevComm_t * /*devComm*/) {
  return flagcxNotSupported;
}

flagcxResult_t glooAdaptorDevCommDestroy(flagcxInnerComm_t /*comm*/,
                                         flagcxInnerDevComm_t /*devComm*/) {
  return flagcxNotSupported;
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
    glooAdaptorGroupStart, glooAdaptorGroupEnd,
    // Device API
    glooAdaptorDevCommReqsInit, glooAdaptorDevCommCreate,
    glooAdaptorDevCommDestroy};

#endif // USE_GLOO_ADAPTOR
