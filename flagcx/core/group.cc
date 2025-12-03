/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "adaptor.h"
#include "assert.h"
#include "collectives.h"
#include "debug.h"
#include "launch_kernel.h"
#include "net.h"
#include "p2p.h"
#include "transport.h"
#include "type.h"
#include <pthread.h>
#include <queue>
#include <stdio.h>
#include <vector>

__thread int flagcxGroupDepth = 0;
__thread bool flagcxGroupJobAbortFlag = false;
__thread struct flagcxHeteroComm *flagcxGroupCommHead = nullptr;
__thread struct flagcxHeteroComm *flagcxGroupCommPreconnectHead = nullptr;
__thread flagcxResult_t flagcxGroupError = flagcxSuccess;
__thread struct flagcxGroupJob *flagcxGroupJobMainPtr = NULL;
__thread struct flagcxGroupJob flagcxGroupJobMain;
__thread int flagcxGroupBlocking = 1; /* default mode */
__thread struct flagcxIntruQueue<struct flagcxAsyncJob, &flagcxAsyncJob::next>
    flagcxAsyncJobs;

flagcxResult_t flagcxHeteroGroupStart() {
  flagcxResult_t ret = flagcxSuccess;
  FLAGCXCHECK(flagcxGroupStartInternal());
  return ret;
}

flagcxResult_t flagcxHeteroGroupEnd() {
  flagcxResult_t ret = flagcxSuccess;
  FLAGCXCHECKGOTO(flagcxGroupEndInternal(), ret, exit);
exit:
  return ret;
}

struct flagcxPreconnectJob {
  struct flagcxAsyncJob base;
  struct flagcxHeteroComm *comm;
};

flagcxResult_t flagcxPreconnectFunc(struct flagcxAsyncJob *job_) {
  struct flagcxPreconnectJob *job = (struct flagcxPreconnectJob *)job_;
  struct flagcxHeteroComm *comm = job->comm;
  if (comm->proxyState->initialized == 0) {
    FLAGCXCHECK(flagcxProxyInit(comm));
  }
  FLAGCXCHECK(flagcxTransportP2pSetup(comm, NULL, 0));
  return flagcxSuccess;
}

/**
 * TODO: add proxy block to make sure the connect is complete
 **/

void *flagcxAsyncJobMain(void *arg) {
  struct flagcxAsyncJob *job = (struct flagcxAsyncJob *)arg;
  // flagcxSetDevice(job->comm->cudaDev);
  deviceAdaptor->setDevice(job->comm->cudaDev);
  job->result = job->func(job);
  if (job->result != flagcxSuccess) {
    INFO(FLAGCX_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__,
         job->result);
  }
  __atomic_store_n(&job->state, flagcxGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}

static flagcxResult_t groupLaunch(struct flagcxAsyncJob *job_) {
  flagcxResult_t ret = flagcxSuccess;
  // bool errorJobAbortFlag = false;
  struct flagcxGroupJob *gjob = (struct flagcxGroupJob *)job_;
  struct flagcxHeteroComm *groupCommHeadMain = *gjob->groupCommHeadPtr;

  struct flagcxHeteroComm *groupCommPreconnectHeadMain =
      *gjob->groupCommPreconnectHeadPtr;

  struct flagcxIntruQueue<struct flagcxAsyncJob, &flagcxAsyncJob::next>
      *asyncJobsMain = gjob->asyncJobsPtr;
  // volatile bool *groupAbortFlag = gjob->abortFlagPtr;

  // Each groupLaunch we create a semaphore to track the p2p ops
  // and a stream to launch host or device func
  std::shared_ptr<flagcxSemaphore> semaphore;
  if (deviceAsyncKernel) {
    semaphore = std::make_shared<flagcxDeviceSemaphore>();
  } else {
    semaphore = std::make_shared<flagcxHostSemaphore>();
  }
  flagcxStream_t launchStream = nullptr;
  flagcxEvent_t launchEvent = nullptr;

  if (groupCommPreconnectHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommPreconnectHeadMain;
    do {
      struct flagcxPreconnectJob *job;
      FLAGCXCHECKGOTO(flagcxCalloc(&job, 1), ret, fail);
      job->base.func = flagcxPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = flagcxGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->comm = job->base.comm = comm;
      flagcxIntruQueueEnqueue(asyncJobsMain, &job->base);

      struct flagcxHeteroComm *next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct flagcxHeteroComm *>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  if (!flagcxIntruQueueEmpty(asyncJobsMain)) {
    struct flagcxAsyncJob *job = flagcxIntruQueueHead(asyncJobsMain);
    do {
      SYSCHECKGOTO(
          pthread_create(&job->thread, nullptr, flagcxAsyncJobMain, job), ret,
          fail);
      job = job->next;
    } while (job != nullptr);

    job = flagcxIntruQueueHead(asyncJobsMain);
    do {
      pthread_join(job->thread, nullptr);
      job = job->next;
    } while (job != nullptr);

    if (ret != flagcxSuccess)
      goto fail;
  }

  if (groupCommHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommHeadMain;
    // post all send/recv tasks
    do {
      flagcxTasks *tasks = &comm->tasks;
      for (int i = 0; i < tasks->p2pOrderSteps; i++) {
        int peer = tasks->p2pOrder[i];
        if (peer != comm->rank) {
          // Handle cross-process send/recv: use proxy
          while (!flagcxIntruQueueEmpty(&tasks->peers[peer].sendQueue)) {
            flagcxTaskP2p *p2p =
                flagcxIntruQueueDequeue(&tasks->peers[peer].sendQueue);
            flagcxProxyOp *op;
            FLAGCXCHECK(flagcxCalloc(&op, 1));
            op->pattern = flagcxPatternSend;
            op->nbytes = p2p->bytes;
            op->recvbuff = (uint8_t *)p2p->buff;
            op->channelId = 0;
            op->root = peer;
            op->connection = comm->channels[op->channelId]
                                 .peers[peer]
                                 ->send[0]
                                 .proxyConn.connection;
            op->args.chunkSize = CHUNKSIZE;
            op->args.chunkSteps = (p2p->bytes + CHUNKSIZE - 1) / (CHUNKSIZE);
            op->args.sendStepMask = MAXSTEPS - 1;
            op->stream = p2p->stream;
            if (op->connection->transport == TRANSPORT_P2P) {
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Sender: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%d)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL(
                  "Sender: [peerRank(%d), rank(%d)] -> [peerSlotIdx(%ld), "
                  "peerOpHash(%d)]",
                  peer, comm->rank, op->args.p2pPeerSlotIdx,
                  op->args.p2pPeerOpHash);
            }
            // launch proxyRegister op if not yet registered
            if (op->connection->transport == TRANSPORT_NET) {
              flagcxConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->send};
              FLAGCXCHECK(flagcxNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->event = semaphore->getEvent();
            semaphore->addCounter(1);
            FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (semaphore->getCounter() == 1) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              FLAGCXCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            FLAGCXCHECK(flagcxProxySaveOp(comm, op));
            free(p2p);
          }
          while (!flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue)) {
            flagcxTaskP2p *p2p =
                flagcxIntruQueueDequeue(&tasks->peers[peer].recvQueue);
            flagcxProxyOp *op;
            FLAGCXCHECK(flagcxCalloc(&op, 1));
            op->pattern = flagcxPatternRecv;
            op->nbytes = p2p->bytes;
            op->recvbuff = (uint8_t *)p2p->buff;
            op->channelId = 0;
            op->root = peer;
            op->connection = comm->channels[op->channelId]
                                 .peers[peer]
                                 ->recv[0]
                                 .proxyConn.connection;
            op->args.chunkSize = CHUNKSIZE;
            op->args.chunkSteps = (p2p->bytes + CHUNKSIZE - 1) / (CHUNKSIZE);
            op->args.sendStepMask = MAXSTEPS - 1;
            op->stream = p2p->stream;
            if (op->connection->transport == TRANSPORT_P2P) {
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Receiver: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%d)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL("Receiver: [peerRank(%d), rank(%d)] -> "
                         "[peerSlotIdx(%ld), peerOpHash(%d)]",
                         peer, comm->rank, op->args.p2pPeerSlotIdx,
                         op->args.p2pPeerOpHash);
            }
            // launch proxyRegister op if not yet registered
            if (op->connection->transport == TRANSPORT_NET) {
              flagcxConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->recv};
              FLAGCXCHECK(flagcxNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->event = semaphore->getEvent();
            semaphore->addCounter(1);
            FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (semaphore->getCounter() == 1) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              FLAGCXCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            FLAGCXCHECK(flagcxProxySaveOp(comm, op));
            free(p2p);
          }
        } else {
          std::vector<flagcxTaskP2p *> sendTasks;
          std::vector<flagcxTaskP2p *> recvTasks;
          while (!flagcxIntruQueueEmpty(&tasks->peers[peer].sendQueue))
            sendTasks.push_back(
                flagcxIntruQueueDequeue(&tasks->peers[peer].sendQueue));
          while (!flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue))
            recvTasks.push_back(
                flagcxIntruQueueDequeue(&tasks->peers[peer].recvQueue));

          for (size_t i = 0; i < sendTasks.size();) {
            bool matched = false;
            for (size_t j = 0; j < recvTasks.size(); j++) {
              if (sendTasks[i]->bytes == recvTasks[j]->bytes &&
                  sendTasks[i]->dtype == recvTasks[j]->dtype) {
                if (sendTasks[i]->buff != recvTasks[j]->buff) {
                  flagcxProxyOp *op;
                  FLAGCXCHECK(flagcxCalloc(&op, 1));
                  op->pattern = flagcxPatternSend;
                  op->nbytes = sendTasks[i]->bytes;
                  op->sendbuff = (uint8_t *)sendTasks[i]->buff;
                  op->recvbuff = (uint8_t *)recvTasks[j]->buff;
                  op->channelId = 0;
                  op->root = peer;
                  op->connection = comm->channels[op->channelId]
                                       .peers[peer]
                                       ->send[0]
                                       .proxyConn.connection;
                  op->stream = sendTasks[i]->stream;
                  op->event = semaphore->getEvent();
                  op->args.chunkSteps = 1; // single step
                  op->args.semaphore = semaphore;
                  semaphore->addCounter(1);
                  FLAGCXCHECK(
                      deviceAdaptor->eventRecord(op->event, op->stream));
                  if (semaphore->getCounter() == 1) {
                    launchStream = op->stream;
                    launchEvent = op->event;
                  } else {
                    FLAGCXCHECK(deviceAdaptor->streamWaitEvent(launchStream,
                                                               op->event));
                  }
                  FLAGCXCHECK(flagcxProxySaveOp(comm, op));
                }
                free(sendTasks[i]);
                free(recvTasks[j]);
                sendTasks.erase(sendTasks.begin() + i);
                recvTasks.erase(recvTasks.begin() + j);
                matched = true;
                break;
              }
            }
            if (!matched)
              i++;
          }
          for (auto *task : sendTasks)
            flagcxIntruQueueEnqueue(&tasks->peers[peer].sendQueue, task);
          for (auto *task : recvTasks)
            flagcxIntruQueueEnqueue(&tasks->peers[peer].recvQueue, task);
        }
      }
      // Clean up p2pOrder: remove peers with empty queues, keep peers with
      // pending operations
      int newOrderSteps = 0;
      for (int i = 0; i < tasks->p2pOrderSteps; i++) {
        int peer = tasks->p2pOrder[i];
        // Keep peer in order if it still has pending send or recv operations
        if (!flagcxIntruQueueEmpty(&tasks->peers[peer].sendQueue) ||
            !flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue)) {
          tasks->p2pOrder[newOrderSteps++] = peer;
        }
      }
      tasks->p2pOrderSteps = newOrderSteps;
      comm = comm->groupNext;
    } while (comm != nullptr);
  }

  if (launchStream != nullptr && launchEvent != nullptr) {
    if (deviceAsyncKernel) {
      FLAGCXCHECK(deviceAdaptor->launchDeviceFunc(
          launchStream, deviceAsyncKernel, (void *)semaphore->getSignals()));
    } else {
      FLAGCXCHECK(deviceAdaptor->launchHostFunc(launchStream, cpuAsyncKernel,
                                                (void *)semaphore.get()));
    }
    // device semaphore need this event to signal completion
    FLAGCXCHECK(deviceAdaptor->eventRecord(launchEvent, launchStream));
  }

  while (!flagcxIntruQueueEmpty(asyncJobsMain)) {
    struct flagcxAsyncJob *job = flagcxIntruQueueDequeue(asyncJobsMain);
    free(job);
  }

  while (groupCommHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommHeadMain;
    struct flagcxHeteroComm *next = comm->groupNext;
    (void)flagcxGroupCommLeave(comm);
    groupCommHeadMain = next;
  }
exit:
  return ret;
fail:
  goto exit;
}

static flagcxResult_t groupCleanup(struct flagcxAsyncJob *job_) {
  struct flagcxGroupJob *gjob = (struct flagcxGroupJob *)job_;
  struct flagcxHeteroComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  struct flagcxHeteroComm *groupCommPreconnectHeadMain =
      *gjob->groupCommPreconnectHeadPtr;
  struct flagcxIntruQueue<struct flagcxAsyncJob, &flagcxAsyncJob::next>
      *asyncJobsMain = gjob->asyncJobsPtr;

  // clean up preconnect comms
  while (groupCommPreconnectHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommPreconnectHeadMain;
    struct flagcxHeteroComm *next = comm->preconnectNext;
    comm->preconnectNext = reinterpret_cast<struct flagcxHeteroComm *>(0x1);
    groupCommPreconnectHeadMain = next;
  }

  // clean up async jobs
  while (!flagcxIntruQueueEmpty(asyncJobsMain)) {
    struct flagcxAsyncJob *job = flagcxIntruQueueDequeue(asyncJobsMain);
    free(job);
  }

  // clean up comms
  while (groupCommHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommHeadMain;
    struct flagcxHeteroComm *next = comm->groupNext;
    (void)flagcxGroupCommLeave(comm);
    groupCommHeadMain = next;
  }

  return flagcxSuccess;
}

static inline void groupResetJobState() {
  flagcxGroupBlocking = 0;
  flagcxGroupJobMainPtr = NULL;
  flagcxGroupCommPreconnectHead = nullptr;
  flagcxGroupCommHead = nullptr;
  memset(&flagcxGroupJobMain, 0, sizeof(struct flagcxGroupJob));
}

flagcxResult_t flagcxGroupEndInternal() {
  flagcxResult_t ret = flagcxSuccess;
  flagcxGroupDepth--;
  if (flagcxGroupDepth < 0)
    return flagcxSystemError;
  if (flagcxGroupDepth == 0) {
    if (flagcxGroupCommPreconnectHead || flagcxGroupCommHead) {
      flagcxGroupJobMain.groupCommHeadPtr = &flagcxGroupCommHead;
      flagcxGroupJobMain.groupCommPreconnectHeadPtr =
          &flagcxGroupCommPreconnectHead;
      flagcxGroupJobMain.asyncJobsPtr = &flagcxAsyncJobs;
      flagcxGroupJobMain.initialized = true;
      flagcxGroupJobMainPtr = &flagcxGroupJobMain;
      FLAGCXCHECKGOTO(groupLaunch(&flagcxGroupJobMainPtr->base), ret, fail);
      groupResetJobState();
    }
  }

exit:
  return ret;
fail:
  groupCleanup(&flagcxGroupJobMainPtr->base);
  groupResetJobState();
  goto exit;
}
