// Internal; requires the adaptor staging types and containers.
static void mpiReleaseStagedBuffers(std::vector<stagedBuffer_t> *buffers,
                                    flagcxInnerComm_t owner) {
  for (auto it = buffers->begin(); it != buffers->end();) {
    stagedBuffer_t buffer = *it;
    if (buffer->owner != owner) {
      ++it;
      continue;
    }
    free(buffer->buffer);
    free(buffer);
    it = buffers->erase(it);
  }
}

static flagcxResult_t mpiCleanupOwner(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  mpiReleaseStagedBuffers(&sendStagedBufferList, comm);
  mpiReleaseStagedBuffers(&recvStagedBufferList, comm);
  return flagcxSuccess;
}

flagcxResult_t mpiAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                         void **buff, size_t size, int isRecv) {
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
    int newSize = MPI_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
    while (newSize < size) {
      newSize *= 2;
    }
    sbuff->buffer = malloc(newSize);
    if (sbuff->buffer == NULL) {
      free(sbuff);
      return flagcxSystemError;
    }
    sbuff->size = newSize;
    if (isRecv) {
      recvStagedBufferList.push_back(sbuff);
    } else {
      sendStagedBufferList.push_back(sbuff);
    }
  }
  *buff = (void *)((char *)sbuff->buffer + sbuff->offset);
  return flagcxSuccess;
}
