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
#include "work.h"
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

static __thread flagcxWorkTracker *proxyGroupOwner;

static flagcxResult_t flagcxNewProxyOp(flagcxProxyOp **op) {
  *op = new (std::nothrow) flagcxProxyOp{};
  return *op == nullptr ? flagcxSystemError : flagcxSuccess;
}

flagcxResult_t flagcxHeteroGroupStart() {
  flagcxResult_t ret = flagcxSuccess;
  FLAGCXCHECK(flagcxGroupStartInternal());
  return ret;
}

flagcxResult_t flagcxHeteroGroupEnd() { return flagcxGroupEndInternal(); }

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

enum flagcxGroupCompletionState {
  flagcxGroupCompletionNotSubmitted = 0,
  flagcxGroupCompletionAccepted,
  flagcxGroupCompletionRejected,
  flagcxGroupCompletionUncertain,
  flagcxGroupCompletionCancelled
};

struct flagcxGroupPendingWork {
  flagcxGroupPendingWork *next;
  std::shared_ptr<flagcxSemaphore> semaphore;
  flagcxWorkRecord *work;
  flagcxStream_t launchStream;
  flagcxProxyOp *unpublished;
  flagcxStreamValueSemaphore *failedCompletion;
  enum flagcxGroupCompletionState completionState;
};

static void groupKeepUnpublished(flagcxGroupPendingWork *pending,
                                 flagcxProxyOp *op) {
  if (op == NULL)
    return;
  if (op->relaySendState != NULL) {
    assert(pending != NULL);
    op->next = pending->unpublished;
    pending->unpublished = op;
  } else {
    delete op;
  }
}

static void groupPublishWork(flagcxHeteroComm_t comm,
                             flagcxGroupPendingWork *node) {
  auto *head = __atomic_load_n(&comm->pendingGroupWork, __ATOMIC_ACQUIRE);
  do {
    node->next = head;
  } while (!__atomic_compare_exchange_n(&comm->pendingGroupWork, &head, node,
                                        false, __ATOMIC_RELEASE,
                                        __ATOMIC_ACQUIRE));
}

flagcxResult_t flagcxGroupDrainWork(flagcxHeteroComm_t comm, int wait) {
  flagcxResult_t first = flagcxSuccess;
  // Each drainer owns a detached finite list. FIFO workers may publish a new
  // group while Finalize drains an earlier one; neither can free the other's
  // node.
  auto *owned =
      __atomic_exchange_n(&comm->pendingGroupWork,
                          (flagcxGroupPendingWork *)NULL, __ATOMIC_ACQ_REL);
  flagcxGroupPendingWork *retained = NULL;
  auto **link = &owned;
  while (*link != NULL) {
    auto *node = *link;
    flagcxResult_t result = wait ? flagcxWorkDrainRecord(node->work, comm)
                                 : flagcxWorkPollRecord(node->work);
    if (result == flagcxSuccess && node->failedCompletion != NULL) {
      // The saved fallback fence, not the operation error, proves that every
      // queued write/wait is finished. Let deferred proxy users retire now.
      node->failedCompletion->markCompletionDrained();
      node->failedCompletion = NULL;
    }
    if (result == flagcxSuccess && node->semaphore) {
      if (node->completionState == flagcxGroupCompletionNotSubmitted ||
          node->completionState == flagcxGroupCompletionRejected) {
        // The fallback fence covers the queued prefix. No callback/kernel was
        // accepted to finish this semaphore; proxy request retirement remains
        // independent of this permission.
        node->semaphore->markCompletionCancelled();
        node->completionState = flagcxGroupCompletionCancelled;
      } else if (node->completionState == flagcxGroupCompletionUncertain) {
        result = flagcxInProgress;
      }
    }
    while (result == flagcxSuccess && node->unpublished != NULL) {
      auto *op = node->unpublished;
      result = flagcxNetCleanupRelaySendOp(op);
      if (result == flagcxSuccess) {
        node->unpublished = op->next;
        delete op;
      }
    }
    while (result == flagcxSuccess && node->semaphore &&
           node->semaphore.use_count() != 1) {
      if (comm->proxyState != NULL) {
        flagcxResult_t error =
            __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
        if (error == flagcxSuccess)
          error = __atomic_load_n(&comm->proxyState->workerCleanupResult,
                                  __ATOMIC_ACQUIRE);
        if (error != flagcxSuccess && error != flagcxInProgress) {
          result = error;
          break;
        }
      }
      if (!wait) {
        result = flagcxInProgress;
        break;
      }
      sched_yield();
    }
    if (result == flagcxInProgress && comm->proxyState != NULL) {
      flagcxResult_t cleanup = __atomic_load_n(
          &comm->proxyState->workerCleanupResult, __ATOMIC_ACQUIRE);
      flagcxResult_t execution =
          __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
      if (cleanup != flagcxSuccess)
        result = cleanup;
      else if (execution != flagcxSuccess && execution != flagcxInProgress)
        result = execution;
    }
    if (result == flagcxSuccess && node->semaphore)
      result = node->semaphore->release();
    if (result == flagcxSuccess && node->launchStream != NULL) {
      result = deviceAdaptor->streamDestroy(node->launchStream);
      if (result == flagcxSuccess)
        node->launchStream = NULL;
    }
    if (result != flagcxSuccess) {
      if (first == flagcxSuccess && (wait || result != flagcxInProgress))
        first = result;
      *link = node->next;
      node->next = retained;
      retained = node;
      continue;
    }
    *link = node->next;
    flagcxWorkDrop(node->work);
    delete node;
  }
  while (retained != NULL) {
    auto *node = retained;
    retained = retained->next;
    groupPublishWork(comm, node);
  }
  return first;
}

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

struct flagcxGroupLaunch {
  flagcxHeteroComm_t comm;
  flagcxGroupPendingWork *pending;
  flagcxStream_t stream;
  flagcxEvent_t event;
  bool streamValue, sawDefaultStream;
  int submitted;
  std::vector<flagcxStream_t> operationStreams;
};

static flagcxResult_t groupPrepareOne(struct flagcxAsyncJob *job_,
                                      flagcxGroupLaunch *launch,
                                      bool privateCompletion) {
  flagcxResult_t ret = flagcxSuccess;
  // A published proxy prefix needs a transport-completion boundary.
  struct flagcxGroupJob *gjob = (struct flagcxGroupJob *)job_;
  struct flagcxHeteroComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  TRACE(FLAGCX_COLL,
        "Group comm=%p rank=%d stage=prepare-begin private-completion=%d",
        groupCommHeadMain, groupCommHeadMain ? groupCommHeadMain->rank : -1,
        (int)privateCompletion);

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
  flagcxStream_t launchStream = nullptr;
  flagcxEvent_t launchEvent = nullptr;
  // temporary stored proxy ops in step order
  std::map<int, std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>>
      proxyOps;
  struct flagcxProxyOp *preparingOp = NULL;
  struct flagcxTaskP2p *ownedTasks = NULL;
  struct flagcxGroupPendingWork *pending = NULL;
  flagcxHeteroComm_t pendingOwner = groupCommHeadMain;
  bool launchSet = false, sawDefaultStream = false;
  std::vector<flagcxStream_t> operationStreams;
  try {
    if (groupCommHeadMain != NULL) {
      ret = flagcxGroupDrainWork(groupCommHeadMain, 0);
      if (ret != flagcxSuccess)
        goto fail;
      pending = new (std::nothrow) flagcxGroupPendingWork{};
      if (pending == NULL) {
        ret = flagcxSystemError;
        goto fail;
      }
      auto *tracker = groupCommHeadMain->workOwner != NULL
                          ? groupCommHeadMain->workOwner
                          : &groupCommHeadMain->submittedWork;
      if (proxyGroupOwner == &groupCommHeadMain->proxyWork)
        tracker = proxyGroupOwner;
      ret = flagcxWorkPrepareAccepted(tracker, NULL, &pending->work);
      if (ret != flagcxSuccess) {
        delete pending;
        pending = NULL;
        goto fail;
      }
      flagcxWorkRetain(pending->work);
      // Publish only after the builder has stopped accessing the node.
    }
    if (useStreamValueGroup) {
      streamValueSemaphore = std::make_shared<flagcxStreamValueSemaphore>();
      semaphore = streamValueSemaphore;
    } else if (deviceAsyncKernel) {
      semaphore = std::make_shared<flagcxDeviceSemaphore>();
    } else {
      semaphore = std::make_shared<flagcxHostSemaphore>();
    }
    if (pending != NULL)
      pending->semaphore = semaphore;
    if (semaphore->initResult != flagcxSuccess) {
      ret = semaphore->initResult;
      goto fail;
    }

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
                  preparingOp = op;
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
                    ret = semaphore->initResult != flagcxSuccess
                              ? semaphore->initResult
                              : flagcxInternalError;
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
                  semaphore->addCounter(op->args.opId);
                  defaultOpId++;
                  operationStreams.push_back(op->stream);
                  sawDefaultStream |= op->stream == NULL;
                  FLAGCXCHECKGOTO(
                      deviceAdaptor->eventRecord(op->event, op->stream), ret,
                      fail);
                  if (!launchSet) {
                    launchSet = true;
                    launchStream = op->stream;
                    if (launchStream == NULL || privateCompletion) {
                      FLAGCXCHECKGOTO(
                          deviceAdaptor->streamCreate(&pending->launchStream),
                          ret, fail);
                      launchStream = pending->launchStream;
                      FLAGCXCHECKGOTO(deviceAdaptor->streamWaitEvent(
                                          launchStream, op->event),
                                      ret, fail);
                    }
                    launchEvent = op->event;
                  } else {
                    FLAGCXCHECKGOTO(
                        deviceAdaptor->streamWaitEvent(launchStream, op->event),
                        ret, fail);
                  }
                  if (proxyOps.find(op->args.step) == proxyOps.end()) {
                    proxyOps[op->args.step] = std::vector<
                        std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
                  }
                  proxyOps[op->args.step].push_back({comm, op});
                  preparingOp = NULL;
                }
                groupForgetTask(&ownedTasks, sendTasks[i]);
                groupForgetTask(&ownedTasks, recvTasks[j]);
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
              preparingOp = op;
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
                ret = flagcxInternalError;
                goto fail;
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
                ret = flagcxSystemError;
                goto fail;
              }
              semaphore->addCounter(op->args.opId);
              defaultOpId++;
              roundRecvStep++;
              operationStreams.push_back(op->stream);
              sawDefaultStream |= op->stream == NULL;
              FLAGCXCHECKGOTO(deviceAdaptor->eventRecord(op->event, op->stream),
                              ret, fail);
              if (!launchSet) {
                launchSet = true;
                launchStream = op->stream;
                if (launchStream == NULL || privateCompletion) {
                  FLAGCXCHECKGOTO(
                      deviceAdaptor->streamCreate(&pending->launchStream), ret,
                      fail);
                  launchStream = pending->launchStream;
                  FLAGCXCHECKGOTO(
                      deviceAdaptor->streamWaitEvent(launchStream, op->event),
                      ret, fail);
                }
                launchEvent = op->event;
              } else {
                FLAGCXCHECKGOTO(
                    deviceAdaptor->streamWaitEvent(launchStream, op->event),
                    ret, fail);
              }
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
              preparingOp = op;
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
                ret = flagcxInternalError;
                goto fail;
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
                ret = flagcxSystemError;
                goto fail;
              }
              semaphore->addCounter(op->args.opId);
              defaultOpId++;
              roundSendStep++;
              operationStreams.push_back(op->stream);
              sawDefaultStream |= op->stream == NULL;
              FLAGCXCHECKGOTO(deviceAdaptor->eventRecord(op->event, op->stream),
                              ret, fail);
              if (!launchSet) {
                launchSet = true;
                launchStream = op->stream;
                if (launchStream == NULL || privateCompletion) {
                  FLAGCXCHECKGOTO(
                      deviceAdaptor->streamCreate(&pending->launchStream), ret,
                      fail);
                  launchStream = pending->launchStream;
                  FLAGCXCHECKGOTO(
                      deviceAdaptor->streamWaitEvent(launchStream, op->event),
                      ret, fail);
                }
                launchEvent = op->event;
              } else {
                FLAGCXCHECKGOTO(
                    deviceAdaptor->streamWaitEvent(launchStream, op->event),
                    ret, fail);
              }
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

    // Save all proxy ops in step order
    for (auto it = proxyOps.begin(); it != proxyOps.end(); ++it) {
      for (auto &pair : it->second) {
        gjob->completion.submitted = 1;
        FLAGCXCHECKGOTO(flagcxProxySaveOp(pair.first, pair.second), ret, fail);
        pair.second = NULL; // Ownership transferred to the proxy queue.
      }
    }

    launch->comm = pendingOwner;
    launch->pending = pending;
    launch->stream = launchStream;
    launch->event = launchEvent;
    launch->streamValue = useStreamValueGroup;
    launch->sawDefaultStream = sawDefaultStream;
    launch->submitted = gjob->completion.submitted;
    launch->operationStreams.swap(operationStreams);
    TRACE(FLAGCX_COLL,
          "Group comm=%p rank=%d stage=prepare-end launch-stream=%p "
          "private-completion=%d operations=%zu stream-value=%d",
          pendingOwner, pendingOwner ? pendingOwner->rank : -1, launchStream,
          (int)privateCompletion, launch->operationStreams.size(),
          (int)useStreamValueGroup);
    for (size_t i = 0; i < launch->operationStreams.size(); ++i)
      TRACE(FLAGCX_COLL,
            "Group comm=%p rank=%d stage=operation-stream index=%zu "
            "operation-stream=%p launch-stream=%p",
            pendingOwner, pendingOwner ? pendingOwner->rank : -1, i,
            launch->operationStreams[i], launchStream);
    pending = NULL; // Completion phase owns the unpublished node.

    while (!flagcxIntruQueueEmpty(asyncJobsMain)) {
      struct flagcxAsyncJob *job = flagcxIntruQueueDequeue(asyncJobsMain);
      free(job);
    }

  } catch (const std::bad_alloc &) {
    ret = flagcxSystemError;
    goto fail;
  }
exit:
  if (pending != NULL) {
    // Seal every borrowed marker before publishing a node whose private
    // completion stream may be reclaimed concurrently by another drainer.
    flagcxResult_t completed = flagcxWorkGroupCompleteAtBoundary(
        flagcxWorkHeteroGroup, ret, &gjob->completion);
    if (ret == flagcxSuccess)
      ret = completed;
    groupPublishWork(pendingOwner, pending);
  }
  return ret;
fail:
  if (gjob->completion.submitted && groupCommHeadMain != NULL)
    flagcxProxyRecordAsyncError(groupCommHeadMain->proxyState, ret);
  if (pending != NULL && pending->work->state == flagcxWorkPreparing) {
    TRACE(FLAGCX_COLL,
          "Group comm=%p rank=%d stage=work-finish-begin path=prepare-rollback "
          "event=%p record-stream=%p record=%d result=%d",
          pendingOwner, pendingOwner ? pendingOwner->rank : -1,
          pending->work->event, launchStream, (int)launchSet, (int)ret);
    flagcxResult_t finished =
        launchSet ? flagcxWorkFinish(pending->work, launchStream, ret,
                                     flagcxWorkHeteroGroup, 0, 1)
                  : flagcxWorkCancel(pending->work);
    TRACE(FLAGCX_COLL,
          "Group comm=%p rank=%d stage=work-finish-end path=prepare-rollback "
          "event=%p record-stream=%p result=%d",
          pendingOwner, pendingOwner ? pendingOwner->rank : -1,
          pending->work->event, launchStream, (int)finished);
    if (finished == flagcxSuccess ||
        pending->work->state == flagcxWorkRecorded) {
      gjob->completion.stream = launchStream;
      gjob->completion.ready = 1;
    }
    gjob->completion.owner = pending->work->owner;
  }
  while (ownedTasks != NULL) {
    struct flagcxTaskP2p *task = ownedTasks;
    ownedTasks = task->next;
    free(task);
  }
  groupKeepUnpublished(pending, preparingOp);
  for (auto &step : proxyOps)
    for (auto &entry : step.second)
      groupKeepUnpublished(pending, entry.second);
  goto exit;
}

// Finish only after every Comm has recorded readiness and posted its proxy
// operations. Multiple Comm callbacks/kernels use independent streams so their
// start signals cannot sit behind another Comm's blocking completion wait.
static flagcxResult_t groupFinishOne(flagcxGroupJob *gjob,
                                     flagcxGroupLaunch *launch,
                                     flagcxResult_t ret) {
  auto *pending = launch->pending;
  if (pending == NULL)
    return ret;
  auto &semaphore = pending->semaphore;
  auto *comm = launch->comm;
  flagcxStream_t stream = launch->stream;
  const char *completionMethod = launch->streamValue ? "stream-value"
                                 : deviceAsyncKernel ? "device-kernel"
                                                     : "host-callback";
  gjob->completion = {};
  gjob->completion.submitted = launch->submitted;
  gjob->completion.owner = pending->work->owner;
  if (ret != flagcxSuccess)
    goto fail;
  if (launch->event != NULL) {
    TRACE(FLAGCX_COLL,
          "Group comm=%p rank=%d stage=notify-begin method=%s stream=%p", comm,
          comm->rank, completionMethod, stream);
    if (launch->streamValue) {
      pending->completionState = flagcxGroupCompletionAccepted;
      auto *completion =
          static_cast<flagcxStreamValueSemaphore *>(semaphore.get());
      ret = completion->enqueueCompletion(stream);
      TRACE(FLAGCX_COLL,
            "Group comm=%p rank=%d stage=notify-end method=%s stream=%p "
            "result=%d",
            comm, comm->rank, completionMethod, stream, (int)ret);
      if (ret != flagcxSuccess) {
        completion->markCompletionUnsafe();
        pending->failedCompletion = completion;
        goto fail;
      }
    } else if (deviceAsyncKernel) {
      ret = deviceAdaptor->launchDeviceFunc(stream, deviceAsyncKernel,
                                            semaphore->getSignals());
      TRACE(FLAGCX_COLL,
            "Group comm=%p rank=%d stage=notify-end method=%s stream=%p "
            "result=%d",
            comm, comm->rank, completionMethod, stream, (int)ret);
      pending->completionState =
          ret == flagcxSuccess      ? flagcxGroupCompletionAccepted
          : ret == flagcxInProgress ? flagcxGroupCompletionUncertain
                                    : flagcxGroupCompletionRejected;
      if (ret != flagcxSuccess)
        goto fail;
      TRACE(FLAGCX_COLL,
            "Group comm=%p rank=%d stage=notify-event-record-begin event=%p "
            "record-stream=%p",
            comm, comm->rank, launch->event, stream);
      FLAGCXCHECKGOTO(deviceAdaptor->eventRecord(launch->event, stream), ret,
                      fail);
      TRACE(FLAGCX_COLL,
            "Group comm=%p rank=%d stage=notify-event-record-end event=%p "
            "record-stream=%p result=%d",
            comm, comm->rank, launch->event, stream, (int)ret);
      if (ret != flagcxSuccess)
        goto fail;
    } else {
      ret = deviceAdaptor->launchHostFunc(stream, cpuAsyncKernel,
                                          semaphore.get());
      TRACE(FLAGCX_COLL,
            "Group comm=%p rank=%d stage=notify-end method=%s stream=%p "
            "result=%d",
            comm, comm->rank, completionMethod, stream, (int)ret);
      pending->completionState =
          ret == flagcxSuccess      ? flagcxGroupCompletionAccepted
          : ret == flagcxInProgress ? flagcxGroupCompletionUncertain
                                    : flagcxGroupCompletionRejected;
      if (ret != flagcxSuccess)
        goto fail;
    }
  }
  TRACE(FLAGCX_COLL,
        "Group comm=%p rank=%d stage=work-finish-begin path=normal event=%p "
        "record-stream=%p record=%d",
        comm, comm->rank, pending->work->event, stream,
        (int)(launch->event != NULL));
  ret = launch->event != NULL
            ? flagcxWorkFinish(pending->work, stream, flagcxSuccess,
                               flagcxWorkHeteroGroup, 0, 1)
            : flagcxWorkCancel(pending->work);
  TRACE(FLAGCX_COLL,
        "Group comm=%p rank=%d stage=work-finish-end path=normal event=%p "
        "record-stream=%p result=%d",
        comm, comm->rank, pending->work->event, stream, (int)ret);
  if (ret != flagcxSuccess)
    goto fail;
  gjob->completion.stream = stream;
  gjob->completion.ready = 1;
  if (launch->event != NULL) {
    for (auto opStream : launch->operationStreams)
      if (opStream != NULL && opStream != stream) {
        TRACE(FLAGCX_COLL,
              "Group comm=%p rank=%d stage=completion-wait-begin "
              "wait-stream=%p event=%p record-stream=%p",
              comm, comm->rank, opStream, pending->work->event, stream);
        flagcxResult_t waited =
            deviceAdaptor->streamWaitEvent(opStream, pending->work->event);
        TRACE(FLAGCX_COLL,
              "Group comm=%p rank=%d stage=completion-wait-end "
              "wait-stream=%p event=%p record-stream=%p result=%d",
              comm, comm->rank, opStream, pending->work->event, stream,
              (int)waited);
        FLAGCXCHECKGOTO(waited, ret, fail);
      }
  }
  goto exit;
fail:
  if (launch->submitted && comm->proxyState != NULL)
    flagcxProxyRecordAsyncError(comm->proxyState, ret);
  if (pending->work->state == flagcxWorkPreparing) {
    TRACE(FLAGCX_COLL,
          "Group comm=%p rank=%d stage=work-finish-begin path=finish-rollback "
          "event=%p record-stream=%p record=%d result=%d",
          comm, comm->rank, pending->work->event, stream,
          (int)(launch->event != NULL), (int)ret);
    flagcxResult_t finished =
        launch->event != NULL ? flagcxWorkFinish(pending->work, stream, ret,
                                                 flagcxWorkHeteroGroup, 0, 1)
                              : flagcxWorkCancel(pending->work);
    TRACE(FLAGCX_COLL,
          "Group comm=%p rank=%d stage=work-finish-end path=finish-rollback "
          "event=%p record-stream=%p result=%d",
          comm, comm->rank, pending->work->event, stream, (int)finished);
    if (finished == flagcxSuccess ||
        pending->work->state == flagcxWorkRecorded) {
      gjob->completion.stream = stream;
      gjob->completion.ready = 1;
    }
  }
exit:
  flagcxResult_t completed = flagcxWorkGroupCompleteAtBoundary(
      flagcxWorkHeteroGroup, ret, &gjob->completion);
  if (ret == flagcxSuccess)
    ret = completed;
  groupPublishWork(comm, pending);
  launch->pending = NULL;
  return ret;
}

static flagcxResult_t groupLaunch(struct flagcxAsyncJob *job_) {
  auto *job = (flagcxGroupJob *)job_;
  auto *head = *job->groupCommHeadPtr;
  if (head == NULL) {
    flagcxGroupLaunch launch = {};
    return groupPrepareOne(job_, &launch, false);
  }
  size_t count = 0;
  for (auto *comm = head; comm != NULL; comm = comm->groupNext)
    ++count;
  std::vector<flagcxGroupLaunch> launches;
  try {
    launches.resize(count);
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  flagcxResult_t result = flagcxSuccess;
  size_t prepared = 0;
  for (auto *comm = head; comm != NULL;) {
    auto *next = comm->groupNext;
    comm->groupNext = NULL;
    *job->groupCommHeadPtr = comm;
    job->completion = {};
    result = groupPrepareOne(job_, &launches[prepared], count > 1);
    comm->groupNext = next;
    *job->groupCommHeadPtr = head;
    if (result != flagcxSuccess)
      break;
    ++prepared;
    comm = next;
  }
  for (size_t i = 0; i < prepared; ++i) {
    // A later preparation failure must not strand the already posted prefix
    // without its start callback/kernel and transport-completion boundary.
    flagcxResult_t finished = groupFinishOne(job, &launches[i], flagcxSuccess);
    if (result == flagcxSuccess)
      result = finished;
  }
  // NULL-stream wrappers cannot enqueue a completion wait on every platform.
  // Wait only after every Comm's completion has been launched.
  for (size_t i = 0; result == flagcxSuccess && i < prepared; ++i)
    if (launches[i].sawDefaultStream)
      result = flagcxGroupDrainWork(launches[i].comm, 1);
  if (result == flagcxSuccess) {
    while (head != NULL) {
      auto *next = head->groupNext;
      (void)flagcxGroupCommLeave(head);
      head = next;
    }
    *job->groupCommHeadPtr = NULL;
  }
  return result;
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
      for (int peer = 0; peer < comm->nRanks; peer++) {
        while (!flagcxIntruQueueEmpty(&comm->tasks.peers[peer].sendQueue))
          free(flagcxIntruQueueDequeue(&comm->tasks.peers[peer].sendQueue));
        while (!flagcxIntruQueueEmpty(&comm->tasks.peers[peer].recvQueue))
          free(flagcxIntruQueueDequeue(&comm->tasks.peers[peer].recvQueue));
      }
    }
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
      ret = groupLaunch(&flagcxGroupJobMainPtr->base);
      // Preserve the borrowed completion stream before resetting the job.
      if (ret != flagcxSuccess)
        groupCleanup(&flagcxGroupJobMainPtr->base);
      groupResetJobState();
    }
    // The group wait lives on one stream, after all operation readiness
    // events. Every communicator owns its own marker on that actual boundary.
    ret = flagcxWorkGroupComplete(flagcxWorkHeteroGroup, ret);
  }
  return ret;
}

flagcxResult_t flagcxHeteroProxyGroupEnd(struct flagcxHeteroComm *comm) {
  flagcxWorkTracker *previous = proxyGroupOwner;
  proxyGroupOwner = &comm->proxyWork;
  flagcxResult_t result = flagcxHeteroGroupEnd();
  proxyGroupOwner = previous;
  return result;
}
