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
#include <new>
#include <pthread.h>
#include <queue>
#include <sched.h>
#include <stdio.h>
#include <string.h>
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

int flagcxGetGroupDepth() { return flagcxGroupDepth; }

FLAGCX_PARAM(P2pScheduleDisable, "P2P_SCHEDULE_DISABLE", 0);

static flagcxResult_t flagcxNewProxyOp(flagcxProxyOp **op) {
  *op = new (std::nothrow) flagcxProxyOp{};
  return *op == nullptr ? flagcxSystemError : flagcxSuccess;
}

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

typedef struct flagcxIntruQueue<struct flagcxTaskP2p, &flagcxTaskP2p::next>
    flagcxGroupTaskQueue;
static struct flagcxTaskP2p *groupTakeTask(flagcxGroupTaskQueue *queue,
                                           struct flagcxTaskP2p **owned) {
  struct flagcxTaskP2p *task = flagcxIntruQueueDequeue(queue);
  task->next = *owned;
  *owned = task;
  return task;
}
static void groupForgetTask(struct flagcxTaskP2p **owned,
                            struct flagcxTaskP2p *task) {
  while (*owned != task)
    owned = &(*owned)->next;
  *owned = task->next;
  task->next = NULL;
}

static void groupRetainRejected(flagcxHeteroComm_t comm, flagcxProxyOp *op) {
  if (op == NULL)
    return;
  struct flagcxProxyOp *head =
      __atomic_load_n(&comm->rejectedGroupOps, __ATOMIC_ACQUIRE);
  do {
    op->next = head;
  } while (!__atomic_compare_exchange_n(&comm->rejectedGroupOps, &head, op,
                                        false, __ATOMIC_RELEASE,
                                        __ATOMIC_ACQUIRE));
}

flagcxResult_t flagcxGroupDrainRejected(flagcxHeteroComm_t comm) {
  if (comm == NULL)
    return flagcxSuccess;
  flagcxResult_t first = flagcxSuccess;
  struct flagcxProxyOp *owned = __atomic_exchange_n(
      &comm->rejectedGroupOps, (struct flagcxProxyOp *)NULL, __ATOMIC_ACQ_REL);
  while (owned != NULL) {
    struct flagcxProxyOp *op = owned;
    owned = op->next;
    flagcxResult_t result = flagcxNetCleanupRelaySendOp(op);
    if (result == flagcxSuccess)
      result = flagcxReleaseSemaphore(&op->args.semaphore);
    if (result == flagcxSuccess)
      delete op;
    else {
      groupRetainRejected(comm, op);
      if (first == flagcxSuccess)
        first = result;
    }
  }
  return first;
}

static flagcxResult_t groupReadyStream(flagcxSemaphore *semaphore,
                                       flagcxStream_t *operationStream,
                                       flagcxEvent_t readyEvent,
                                       flagcxStream_t *launchStream,
                                       flagcxEvent_t *launchEvent) {
  if ((*operationStream) == NULL) {
    if (semaphore->ownedDefaultStream == NULL) {
      if (deviceAdaptor->streamCopy == NULL ||
          deviceAdaptor->streamFree == NULL)
        return flagcxNotSupported;
      FLAGCXCHECKSTRICT(
          deviceAdaptor->streamCopy(&semaphore->ownedDefaultStream, NULL));
      if (semaphore->ownedDefaultStream == NULL)
        return flagcxSystemError;
    }
    (*operationStream) = semaphore->ownedDefaultStream;
  }
  bool found = false;
  for (size_t i = 0; i < semaphore->notificationStreams.size(); ++i)
    found |= semaphore->notificationStreams[i] == (*operationStream);
  if (!found)
    semaphore->notificationStreams.push_back((*operationStream));
  FLAGCXCHECKSTRICT(deviceAdaptor->eventRecord(readyEvent, (*operationStream)));
  if (*launchEvent == NULL) {
    *launchStream = (*operationStream);
    *launchEvent = readyEvent;
  } else {
    FLAGCXCHECKSTRICT(
        deviceAdaptor->streamWaitEvent(*launchStream, readyEvent));
  }
  return flagcxSuccess;
}
static flagcxResult_t groupReadyOp(flagcxSemaphore *semaphore,
                                   flagcxProxyOp *op,
                                   flagcxStream_t *launchStream,
                                   flagcxEvent_t *launchEvent) {
  return groupReadyStream(semaphore, &op->stream, op->event, launchStream,
                          launchEvent);
}

struct flagcxGroupDomain {
  struct flagcxGroupDomain *next;
  struct flagcxHeteroComm *comm;
  int original, mapped;
};
static flagcxResult_t groupAddCounter(flagcxHeteroComm_t comm,
                                      flagcxProxyOp *op,
                                      flagcxSemaphore *semaphore, bool device,
                                      struct flagcxGroupDomain **domains,
                                      int *count) {
  struct flagcxGroupDomain *domain = *domains;
  while (domain != NULL &&
         (domain->comm != comm || domain->original != op->args.opId))
    domain = domain->next;
  if (domain == NULL) {
    if (*count == INT_MAX || (device && *count >= FLAGCX_OPS_PER_SEMAPHORE))
      return flagcxInvalidUsage;
    FLAGCXCHECKSTRICT(flagcxCalloc(&domain, 1));
    domain->comm = comm;
    domain->original = op->args.opId;
    domain->mapped = (*count)++;
    domain->next = *domains;
    *domains = domain;
  }
  op->args.opId = domain->mapped;
  semaphore->addCounter(op->args.opId);
  return flagcxSuccess;
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

  // CustomizedSchedule has the highest priority, followed by P2PSchedule,
  // with DefaultSchedule as the fallback.
  // CustomizedSchedule: |op0{s0,s1,...,sN}|...|opN{s0,s1,...,sN}|
  // P2PSchedule: |recvOps{s0,s1,...,sN}|selfCopyOps{s0}|sendOps{s0,s1,...,sN}|
  // DefaultSchedule: |op0{s0}|op1{s0}|...|opN{s0}|
  int defaultOpId = 0;
  int defaultStep = 0;
  // Preserve the default host/device semaphore path for ordinary groups.
  bool useStreamValueGroup = false;
  if (gjob->useStreamValue) {
    flagcxResult_t probeResult =
        flagcxStreamValueSupported(&useStreamValueGroup);
    if (probeResult != flagcxSuccess)
      return probeResult;
  }
  std::shared_ptr<flagcxSemaphore> semaphore;
  std::shared_ptr<flagcxStreamValueSemaphore> streamValueSemaphore;
  try {
    if (useStreamValueGroup) {
      streamValueSemaphore = std::make_shared<flagcxStreamValueSemaphore>();
      if (streamValueSemaphore->initResult != flagcxSuccess)
        return streamValueSemaphore->initResult;
      semaphore = streamValueSemaphore;
    } else if (deviceAsyncKernel) {
      semaphore = std::make_shared<flagcxDeviceSemaphore>();
    } else {
      semaphore = std::make_shared<flagcxHostSemaphore>();
    }
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  if (semaphore->initResult != flagcxSuccess)
    return semaphore->initResult;
  flagcxStream_t launchStream = nullptr;
  flagcxEvent_t launchEvent = nullptr;
  struct flagcxProxyOp *preparingOp = NULL;
  struct flagcxHeteroComm *preparingOwner = NULL;
  struct flagcxTaskP2p *ownedTasks = NULL;
  flagcxSemaphore *notification = NULL;
  bool published = false, retained = false;
  struct flagcxGroupDomain *domains = NULL;
  int domainCount = 0;
  semaphore->prepareNotifications();
  // temporary stored proxy ops in step order
  std::map<int, std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>>
      proxyOps;

  try {
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
        comm = comm->preconnectNext;
      } while (comm != nullptr);

      // Job allocation is the only fallible part of materializing the
      // preconnect list. Transfer the head only after every job exists, so an
      // allocation failure leaves the original list intact for groupCleanup().
      *gjob->groupCommPreconnectHeadPtr = nullptr;
      comm = groupCommPreconnectHeadMain;
      do {
        struct flagcxHeteroComm *next = comm->preconnectNext;
        comm->preconnectNext = reinterpret_cast<struct flagcxHeteroComm *>(0x1);
        comm = next;
      } while (comm != nullptr);
    }

    if (!flagcxIntruQueueEmpty(asyncJobsMain)) {
      struct flagcxAsyncJob *job = flagcxIntruQueueHead(asyncJobsMain);
      size_t startedJobs = 0;
      while (job != nullptr) {
        int error =
            pthread_create(&job->thread, nullptr, flagcxAsyncJobMain, job);
        if (error != 0) {
          WARN("Failed to create async group thread: %s", strerror(error));
          ret = flagcxSystemError;
          break;
        }
        startedJobs++;
        job = job->next;
      }

      job = flagcxIntruQueueHead(asyncJobsMain);
      for (size_t i = 0; i < startedJobs; i++) {
        int error = pthread_join(job->thread, nullptr);
        if (error != 0) {
          WARN("Failed to join async group thread: %s", strerror(error));
          if (ret == flagcxSuccess)
            ret = flagcxSystemError;
          // The job object cannot be released while its worker may still use
          // it. A successfully created worker always publishes Done before
          // returning, so wait for that publication if pthread_join itself
          // unexpectedly fails.
          while (__atomic_load_n(&job->state, __ATOMIC_ACQUIRE) ==
                 flagcxGroupJobRunning)
            sched_yield();
        }
        if (job->result != flagcxSuccess && ret == flagcxSuccess) {
          WARN("Async job failed with result %d", job->result);
          ret = job->result;
        }
        job = job->next;
      }

      if (ret != flagcxSuccess)
        goto fail;
    }

    TRACE(FLAGCX_COLL, "group launch prepare head %p", groupCommHeadMain);
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
                groupTakeTask(&tasks->peers[peer].sendQueue, &ownedTasks));
          while (!flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue))
            recvTasks.push_back(
                groupTakeTask(&tasks->peers[peer].recvQueue, &ownedTasks));

          for (size_t i = 0; i < sendTasks.size();) {
            bool matched = false;
            for (size_t j = 0; j < recvTasks.size(); j++) {
              if (sendTasks[i]->bytes == recvTasks[j]->bytes &&
                  sendTasks[i]->dtype == recvTasks[j]->dtype &&
                  sendTasks[i]->channelId == recvTasks[j]->channelId) {
                if (sendTasks[i]->buff != recvTasks[j]->buff) {
                  flagcxProxyOp *op;
                  FLAGCXCHECKGOTO(flagcxNewProxyOp(&op), ret, fail);
                  if (ret != flagcxSuccess)
                    goto fail;
                  preparingOp = op;
                  preparingOwner = comm;
                  op->args.semaphore = semaphore;
                  op->pattern = flagcxPatternSend;
                  op->nbytes = sendTasks[i]->bytes;
                  op->sendbuff = (uint8_t *)sendTasks[i]->buff;
                  op->recvbuff = (uint8_t *)recvTasks[j]->buff;
                  op->channelId = sendTasks[i]->channelId;
                  op->root = peer;
                  op->connection = comm->channels[op->channelId]
                                       .peers[peer]
                                       ->send[0]
                                       .proxyConn.connection;
                  op->stream = sendTasks[i]->stream;
                  op->event = semaphore->getEvent();
                  if (op->event == NULL) {
                    ret = semaphore->initResult == flagcxSuccess
                              ? flagcxSystemError
                              : semaphore->initResult;
                    goto fail;
                  }
                  op->args.chunkSteps = 1; // single step
                  op->args.semaphore = semaphore;
                  op->args.opId = sendTasks[i]->opId == INT_MAX
                                      ? (p2pScheduleDisable ? defaultOpId : 0)
                                      : sendTasks[i]->opId;
                  op->args.step = sendTasks[i]->step == -1
                                      ? (p2pScheduleDisable ? defaultStep : 0)
                                      : sendTasks[i]->step;
                  FLAGCXCHECKGOTO(groupAddCounter(comm, op, semaphore.get(),
                                                  !useStreamValueGroup &&
                                                      deviceAsyncKernel != NULL,
                                                  &domains, &domainCount),
                                  ret, fail);
                  if (ret != flagcxSuccess)
                    goto fail;
                  defaultOpId++;
                  FLAGCXCHECKGOTO(groupReadyOp(semaphore.get(), op,
                                               &launchStream, &launchEvent),
                                  ret, fail);
                  if (ret != flagcxSuccess)
                    goto fail;
                  if (proxyOps.find(op->args.step) == proxyOps.end()) {
                    proxyOps[op->args.step] = std::vector<
                        std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
                  }
                  proxyOps[op->args.step].push_back({comm, op});
                  preparingOp = NULL;
                  if (sendTasks[i]->stream != recvTasks[j]->stream) {
                    flagcxStream_t recvStream = recvTasks[j]->stream;
                    flagcxEvent_t recvReady = semaphore->getEvent();
                    if (recvReady == NULL) {
                      ret = semaphore->initResult == flagcxSuccess
                                ? flagcxSystemError
                                : semaphore->initResult;
                      goto fail;
                    }
                    FLAGCXCHECKGOTO(groupReadyStream(
                                        semaphore.get(), &recvStream, recvReady,
                                        &launchStream, &launchEvent),
                                    ret, fail);
                    if (ret != flagcxSuccess)
                      goto fail;
                  }
                }
                groupForgetTask(&ownedTasks, sendTasks[i]);
                free(sendTasks[i]);
                groupForgetTask(&ownedTasks, recvTasks[j]);
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
          for (auto *task : sendTasks) {
            groupForgetTask(&ownedTasks, task);
            flagcxIntruQueueEnqueue(&tasks->peers[peer].sendQueue, task);
          }
          for (auto *task : recvTasks) {
            groupForgetTask(&ownedTasks, task);
            flagcxIntruQueueEnqueue(&tasks->peers[peer].recvQueue, task);
          }
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
                  groupTakeTask(&tasks->peers[recvPeer].recvQueue, &ownedTasks);
              int peer = recvPeer;
              flagcxProxyOp *op;
              FLAGCXCHECKGOTO(flagcxNewProxyOp(&op), ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
              preparingOp = op;
              preparingOwner = comm;
              op->args.semaphore = semaphore;
              op->pattern = flagcxPatternRecv;
              op->nbytes = p2p->bytes;
              op->recvbuff = (uint8_t *)p2p->buff;
              op->channelId = p2p->channelId;
              op->root = peer;
              op->connection = comm->channels[op->channelId]
                                   .peers[peer]
                                   ->recv[0]
                                   .proxyConn.connection;
              op->stream = p2p->stream;
              if (op->connection == NULL) {
                WARN("groupLaunch: recv proxyConn.connection is NULL for rank "
                     "%d "
                     "peer %d channel %d",
                     comm->rank, peer, op->channelId);
                return flagcxInternalError;
              }
              FLAGCXCHECKGOTO(flagcxTransportPrepareProxyOp(comm, op, p2p->buff,
                                                            p2p->bytes, peer,
                                                            p2p->dtype),
                              ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
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
              if (op->event == NULL) {
                ret = semaphore->initResult == flagcxSuccess
                          ? flagcxSystemError
                          : semaphore->initResult;
                delete op;
                goto fail;
              }
              FLAGCXCHECKGOTO(groupAddCounter(comm, op, semaphore.get(),
                                              !useStreamValueGroup &&
                                                  deviceAsyncKernel != NULL,
                                              &domains, &domainCount),
                              ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
              defaultOpId++;
              roundRecvStep++;
              FLAGCXCHECKGOTO(groupReadyOp(semaphore.get(), op, &launchStream,
                                           &launchEvent),
                              ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
              if (proxyOps.find(op->args.step) == proxyOps.end()) {
                proxyOps[op->args.step] = std::vector<
                    std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
              }
              proxyOps[op->args.step].push_back({comm, op});
              preparingOp = NULL;
              groupForgetTask(&ownedTasks, p2p);
              free(p2p);
            }
            // Process one send task (for IPC lookup - after recv's register)
            if (!flagcxIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
              flagcxTaskP2p *p2p =
                  groupTakeTask(&tasks->peers[sendPeer].sendQueue, &ownedTasks);
              int peer = sendPeer;
              flagcxProxyOp *op;
              FLAGCXCHECKGOTO(flagcxNewProxyOp(&op), ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
              preparingOp = op;
              preparingOwner = comm;
              op->args.semaphore = semaphore;
              op->pattern = flagcxPatternSend;
              op->nbytes = p2p->bytes;
              op->recvbuff = (uint8_t *)p2p->buff;
              op->channelId = p2p->channelId;
              op->root = peer;
              op->connection = comm->channels[op->channelId]
                                   .peers[peer]
                                   ->send[0]
                                   .proxyConn.connection;
              op->comm = comm;
              op->stream = p2p->stream;
              if (op->connection == NULL) {
                WARN("groupLaunch: send proxyConn.connection is NULL for rank "
                     "%d "
                     "peer %d channel %d",
                     comm->rank, peer, op->channelId);
                return flagcxInternalError;
              }
              FLAGCXCHECKGOTO(flagcxTransportPrepareProxyOp(comm, op, p2p->buff,
                                                            p2p->bytes, peer,
                                                            p2p->dtype),
                              ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
              op->args.semaphore = semaphore;
              op->args.opId =
                  p2p->opId == INT_MAX
                      ? (p2pScheduleDisable ? defaultOpId : roundOpId)
                      : p2p->opId;
              op->args.step =
                  p2p->step == -1
                      ? (p2pScheduleDisable ? defaultStep : roundSendStep)
                      : p2p->step;
              op->event = semaphore->getEvent();
              if (op->event == NULL) {
                ret = semaphore->initResult == flagcxSuccess
                          ? flagcxSystemError
                          : semaphore->initResult;
                delete op;
                goto fail;
              }
              FLAGCXCHECKGOTO(groupAddCounter(comm, op, semaphore.get(),
                                              !useStreamValueGroup &&
                                                  deviceAsyncKernel != NULL,
                                              &domains, &domainCount),
                              ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
              defaultOpId++;
              roundSendStep++;
              FLAGCXCHECKGOTO(groupReadyOp(semaphore.get(), op, &launchStream,
                                           &launchEvent),
                              ret, fail);
              if (ret != flagcxSuccess)
                goto fail;
              if (proxyOps.find(op->args.step) == proxyOps.end()) {
                proxyOps[op->args.step] = std::vector<
                    std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
              }
              proxyOps[op->args.step].push_back({comm, op});
              preparingOp = NULL;
              groupForgetTask(&ownedTasks, p2p);
              free(p2p);
            }
          }
        }
        tasks->p2pOrderSteps = 0;
        comm = comm->groupNext;
      } while (comm != nullptr);
    }

    if (launchEvent != NULL) {
      // This event precedes every host callback. It merges only input
      // readiness.
      FLAGCXCHECKGOTO(deviceAdaptor->eventRecord(launchEvent, launchStream),
                      ret, fail);
      if (ret != flagcxSuccess)
        goto fail;
      for (size_t i = 0; i < semaphore->notificationStreams.size(); ++i) {
        flagcxStream_t stream = semaphore->notificationStreams[i];
        if (stream != launchStream)
          FLAGCXCHECKGOTO(deviceAdaptor->streamWaitEvent(stream, launchEvent),
                          ret, fail);
        if (ret != flagcxSuccess)
          goto fail;
        if (!useStreamValueGroup && !deviceAsyncKernel) {
          struct flagcxGroupCallback *context = NULL;
          FLAGCXCHECKGOTO(flagcxCalloc(&context, 1), ret, fail);
          if (ret != flagcxSuccess)
            goto fail;
          context->semaphore = (flagcxHostSemaphore *)semaphore.get();
          context->next = context->semaphore->callbackContexts;
          context->semaphore->callbackContexts = context;
          __atomic_fetch_add(&context->semaphore->callbacks, 1,
                             __ATOMIC_RELEASE);
        }
      }
    }
    // Publish every Comm's operations before submitting any blocking callback.
    for (auto &step : proxyOps) {
      for (auto &entry : step.second) {
        FLAGCXCHECKGOTO(flagcxProxySaveOp(entry.first, entry.second), ret,
                        fail);
        if (ret != flagcxSuccess)
          goto fail;
        published = true;
        entry.second = NULL;
      }
    }
    if (launchEvent != NULL) {
      if (useStreamValueGroup) {
        FLAGCXCHECKGOTO(streamValueSemaphore->enqueueCompletion(launchStream),
                        ret, fail);
        if (ret != flagcxSuccess)
          goto fail;
        for (size_t i = 0; i < semaphore->notificationStreams.size(); ++i)
          if (semaphore->notificationStreams[i] != launchStream)
            FLAGCXCHECKGOTO(deviceAdaptor->streamWaitEvent(
                                semaphore->notificationStreams[i],
                                streamValueSemaphore->completionEvent),
                            ret, fail);
        if (ret != flagcxSuccess)
          goto fail;
      } else if (deviceAsyncKernel) {
        FLAGCXCHECKGOTO(
            deviceAdaptor->launchDeviceFunc(launchStream, deviceAsyncKernel,
                                            semaphore->getSignals()),
            ret, fail);
        if (ret != flagcxSuccess)
          goto fail;
        FLAGCXCHECKGOTO(deviceAdaptor->eventRecord(launchEvent, launchStream),
                        ret, fail);
        if (ret != flagcxSuccess)
          goto fail;
        __atomic_store_n(
            &((flagcxDeviceSemaphore *)semaphore.get())->completionArmed, true,
            __ATOMIC_RELEASE);
        for (size_t i = 0; i < semaphore->notificationStreams.size(); ++i)
          if (semaphore->notificationStreams[i] != launchStream)
            FLAGCXCHECKGOTO(deviceAdaptor->streamWaitEvent(
                                semaphore->notificationStreams[i], launchEvent),
                            ret, fail);
        if (ret != flagcxSuccess)
          goto fail;
      } else {
        struct flagcxGroupCallback *context =
            ((flagcxHostSemaphore *)semaphore.get())->callbackContexts;
        for (size_t i = 0; i < semaphore->notificationStreams.size();
             ++i, context = context->next) {
          context->submitted = 1;
          FLAGCXCHECKGOTO(
              deviceAdaptor->launchHostFunc(semaphore->notificationStreams[i],
                                            cpuAsyncKernel, context),
              ret, fail);
          if (ret != flagcxSuccess)
            goto fail;
        }
      }
    }
  } catch (const std::bad_alloc &) {
    ret = flagcxSystemError;
    goto fail;
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
  while (domains != NULL) {
    struct flagcxGroupDomain *domain = domains;
    domains = domain->next;
    free(domain);
  }
  notification = semaphore.get();
  streamValueSemaphore.reset();
  if (published || retained) {
    semaphore.reset();
    notification
        ->sealNotifications(); // Last builder access; proxy owns it now.
  } else {
    notification->sealNotifications();
    flagcxResult_t cleanup = flagcxReleaseSemaphore(&semaphore);
    if (ret == flagcxSuccess)
      ret = cleanup;
  }
  return ret;
fail:
  semaphore->cancelNotifications(ret);
  if (published) {
    for (struct flagcxHeteroComm *comm = groupCommHeadMain; comm != NULL;
         comm = comm->groupNext)
      flagcxProxyRecordAsyncError(comm->proxyState, ret);
  }
  while (ownedTasks != NULL) {
    struct flagcxTaskP2p *task = ownedTasks;
    ownedTasks = task->next;
    free(task);
  }
  if (preparingOp != NULL) {
    groupRetainRejected(preparingOwner, preparingOp);
    retained = true;
  }
  for (auto &step : proxyOps)
    for (auto &entry : step.second)
      if (entry.second != NULL) {
        groupRetainRejected(entry.first, entry.second);
        retained = true;
      }
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
    if (comm->tasks.peers != NULL) {
      for (int peer = 0; peer < comm->nRanks; ++peer) {
        while (!flagcxIntruQueueEmpty(&comm->tasks.peers[peer].sendQueue))
          free(flagcxIntruQueueDequeue(&comm->tasks.peers[peer].sendQueue));
        while (!flagcxIntruQueueEmpty(&comm->tasks.peers[peer].recvQueue))
          free(flagcxIntruQueueDequeue(&comm->tasks.peers[peer].recvQueue));
      }
    }
    comm->tasks.nTasksP2p = 0;
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

flagcxResult_t flagcxGroupEndInternal(bool useStreamValue) {
  flagcxResult_t ret = flagcxSuccess;
  if (flagcxGroupDepth <= 0)
    return flagcxInvalidUsage;
  flagcxGroupDepth--;
  if (flagcxGroupDepth == 0) {
    if (flagcxGroupCommPreconnectHead || flagcxGroupCommHead) {
      flagcxGroupJobMain.groupCommHeadPtr = &flagcxGroupCommHead;
      flagcxGroupJobMain.groupCommPreconnectHeadPtr =
          &flagcxGroupCommPreconnectHead;
      flagcxGroupJobMain.asyncJobsPtr = &flagcxAsyncJobs;
      flagcxGroupJobMain.useStreamValue = useStreamValue;
      flagcxGroupJobMain.initialized = true;
      flagcxGroupJobMainPtr = &flagcxGroupJobMain;
      FLAGCXCHECKGOTO(groupLaunch(&flagcxGroupJobMainPtr->base), ret, fail);
      if (ret != flagcxSuccess)
        goto fail;
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
