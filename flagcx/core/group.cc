/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "group.h"
#include "adaptor.h"
#include "assert.h"
#include "debug.h"
#include "flagcx_hetero.h"
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

FLAGCX_PARAM(P2pScheduleDisable, "P2P_SCHEDULE_DISABLE", 0);

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

static int64_t p2pScheduleDisable = flagcxParamP2pScheduleDisable();

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

  // CustomizedSchedule has the highest priority, followed by P2PSchedule,
  // with DefaultSchedule as the fallback.
  // CustomizedSchedule: |op0{s0,s1,...,sN}|...|opN{s0,s1,...,sN}|
  // P2PSchedule: |recvOps{s0,s1,...,sN}|selfCopyOps{s0}|sendOps{s0,s1,...,sN}|
  // DefaultSchedule: |op0{s0}|op1{s0}|...|opN{s0}|
  int defaultOpId = 0;
  int defaultStep = 0;
  // Each groupLaunch we create a semaphore to track the
  // p2p ops and a stream to launch host or device func
  std::shared_ptr<flagcxSemaphore> semaphore;
  if (deviceAsyncKernel) {
    semaphore = std::make_shared<flagcxDeviceSemaphore>();
  } else {
    semaphore = std::make_shared<flagcxHostSemaphore>();
  }
  flagcxStream_t launchStream = nullptr;
  flagcxEvent_t launchEvent = nullptr;
  // temporary stored proxy ops in step order
  std::map<int, std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>>
      proxyOps;

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
      int nRanks = comm->nRanks;
      int localRanks = comm->localRanks;

      // Round 0: handle self send/recv (local copy)
      {
        int peer = comm->rank;
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
                op->args.opId = sendTasks[i]->opId == INT_MAX
                                    ? (p2pScheduleDisable ? defaultOpId : 0)
                                    : sendTasks[i]->opId;
                op->args.step = sendTasks[i]->step == -1
                                    ? (p2pScheduleDisable ? defaultStep : 0)
                                    : sendTasks[i]->step;
                semaphore->addCounter(op->args.opId);
                defaultOpId++;
                FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
                if (launchStream == nullptr) {
                  launchStream = op->stream;
                  launchEvent = op->event;
                } else {
                  FLAGCXCHECK(
                      deviceAdaptor->streamWaitEvent(launchStream, op->event));
                }
                if (proxyOps.find(op->args.step) == proxyOps.end()) {
                  proxyOps[op->args.step] = std::vector<
                      std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
                }
                proxyOps[op->args.step].push_back({comm, op});
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

      // Round 1..nRanks-1: use p2pSchedule to pair recv/send with different
      // peers
      int roundSendStep = 0;
      int roundRecvStep = 0;
      int roundOpId = 1;
      for (int round = 1; round < nRanks; round++) {
        int tmpRoundOpId = round / localRanks + 1;
        if (roundOpId != tmpRoundOpId) {
          roundSendStep = 0;
          roundRecvStep = 0;
          roundOpId = tmpRoundOpId;
        }
        int recvPeer = comm->p2pSchedule[round].recvRank;
        int sendPeer = comm->p2pSchedule[round].sendRank;
        while (!flagcxIntruQueueEmpty(&tasks->peers[recvPeer].recvQueue) ||
               !flagcxIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
          // Process one recv task (for IPC register)
          if (!flagcxIntruQueueEmpty(&tasks->peers[recvPeer].recvQueue)) {
            flagcxTaskP2p *p2p =
                flagcxIntruQueueDequeue(&tasks->peers[recvPeer].recvQueue);
            int peer = recvPeer;
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
            op->stream = p2p->stream;
            if (op->connection->transport == TRANSPORT_P2P) {
              op->args.chunkSize = computeP2pChunkSize(p2p->bytes);
              op->args.chunkSteps =
                  (p2p->bytes + op->args.chunkSize - 1) / (op->args.chunkSize);
              op->args.sendStepMask = flagcxP2pChunks - 1;
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Receiver: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%ld)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL("Receiver: [peerRank(%d), rank(%d)] -> "
                         "[peerSlotIdx(%ld), peerOpHash(%ld)]",
                         peer, comm->rank, op->args.p2pPeerSlotIdx,
                         op->args.p2pPeerOpHash);

              flagcxConnector *recvConn =
                  &comm->channels[op->channelId].peers[peer]->recv[0];
              flagcxConnector *peerConns[] = {recvConn};
              int peerRanks[] = {peer};
              uintptr_t regOffset = 0;
              uintptr_t *peerRmtAddr = NULL;
              op->args.regBufFlag = 0;
              FLAGCXCHECK(flagcxP2pRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, peerRanks, 1,
                  flagcxP2pRegisterModeRegister, &op->args.regBufFlag,
                  &regOffset, &peerRmtAddr));
              if (op->args.regBufFlag) {
                INFO(FLAGCX_REG,
                     "flagcxGroup P2P recv reg rank %d <- %d buff %p size %zu "
                     "offset %zu remote %p",
                     comm->rank, peer, p2p->buff, p2p->bytes, (size_t)regOffset,
                     peerRmtAddr ? (void *)(*peerRmtAddr) : NULL);
              }
            } else if (op->connection->transport == TRANSPORT_NET) {
              op->args.chunkSize = flagcxNetChunkSize;
              op->args.chunkSteps =
                  (p2p->bytes + flagcxNetChunkSize - 1) / (flagcxNetChunkSize);
              op->args.sendStepMask = flagcxNetChunks - 1;
              flagcxConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->recv};
              FLAGCXCHECK(flagcxNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->args.opId =
                p2p->opId == INT_MAX
                    ? (p2pScheduleDisable ? defaultOpId : -roundOpId)
                    : p2p->opId;
            op->args.step =
                p2p->step == -1
                    ? (p2pScheduleDisable ? defaultStep : roundRecvStep)
                    : p2p->step;
            op->event = semaphore->getEvent();
            semaphore->addCounter(op->args.opId);
            defaultOpId++;
            roundRecvStep++;
            FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (launchStream == nullptr) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              FLAGCXCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            if (proxyOps.find(op->args.step) == proxyOps.end()) {
              proxyOps[op->args.step] =
                  std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
            }
            proxyOps[op->args.step].push_back({comm, op});
            free(p2p);
          }
          // Process one send task (for IPC lookup - after recv's register)
          if (!flagcxIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
            flagcxTaskP2p *p2p =
                flagcxIntruQueueDequeue(&tasks->peers[sendPeer].sendQueue);
            int peer = sendPeer;
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
            op->stream = p2p->stream;
            if (op->connection->transport == TRANSPORT_P2P) {
              op->args.chunkSize = computeP2pChunkSize(p2p->bytes);
              op->args.chunkSteps =
                  (p2p->bytes + op->args.chunkSize - 1) / (op->args.chunkSize);
              op->args.sendStepMask = flagcxP2pChunks - 1;
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Sender: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%ld)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL(
                  "Sender: [peerRank(%d), rank(%d)] -> [peerSlotIdx(%ld), "
                  "peerOpHash(%ld)]",
                  peer, comm->rank, op->args.p2pPeerSlotIdx,
                  op->args.p2pPeerOpHash);
              flagcxConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->send};
              int peerRanks[] = {peer};
              uintptr_t regOffset = 0;
              uintptr_t *peerRmtAddr = NULL;
              FLAGCXCHECK(flagcxP2pRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, peerRanks, 1,
                  flagcxP2pRegisterModeLookup, &op->args.regBufFlag, &regOffset,
                  &peerRmtAddr));
              // Pass the remote address to sender for zero-copy
              // peerRmtAddr is the remote address itself (cast as uintptr_t*)
              if (op->args.regBufFlag && peerRmtAddr) {
                op->args.p2pRmtAddr =
                    (void *)((uintptr_t)peerRmtAddr + regOffset);
              }
            } else if (op->connection->transport == TRANSPORT_NET) {
              op->args.chunkSize = flagcxNetChunkSize;
              op->args.chunkSteps =
                  (p2p->bytes + flagcxNetChunkSize - 1) / (flagcxNetChunkSize);
              op->args.sendStepMask = flagcxNetChunks - 1;
              flagcxConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->send};
              FLAGCXCHECK(flagcxNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->args.opId = p2p->opId == INT_MAX
                                ? (p2pScheduleDisable ? defaultOpId : roundOpId)
                                : p2p->opId;
            op->args.step =
                p2p->step == -1
                    ? (p2pScheduleDisable ? defaultStep : roundSendStep)
                    : p2p->step;
            op->event = semaphore->getEvent();
            semaphore->addCounter(op->args.opId);
            defaultOpId++;
            roundSendStep++;
            FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (launchStream == nullptr) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              FLAGCXCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            if (proxyOps.find(op->args.step) == proxyOps.end()) {
              proxyOps[op->args.step] =
                  std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
            }
            proxyOps[op->args.step].push_back({comm, op});
            free(p2p);
          }
        }
      }
      tasks->p2pOrderSteps = 0;
      comm = comm->groupNext;
    } while (comm != nullptr);
  }

  // Save all proxy ops in step order
  for (auto it = proxyOps.begin(); it != proxyOps.end(); ++it) {
    for (auto pair : it->second) {
      FLAGCXCHECK(flagcxProxySaveOp(pair.first, pair.second));
    }
  }

  if (launchStream != nullptr && launchEvent != nullptr) {
    if (deviceAsyncKernel) {
      FLAGCXCHECK(deviceAdaptor->launchDeviceFunc(
          launchStream, deviceAsyncKernel, (void *)semaphore->getSignals()));
      // device semaphore need this event to signal completion
      FLAGCXCHECK(deviceAdaptor->eventRecord(launchEvent, launchStream));
    } else {
      FLAGCXCHECK(deviceAdaptor->launchHostFunc(launchStream, cpuAsyncKernel,
                                                (void *)semaphore.get()));
    }
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
