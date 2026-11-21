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
