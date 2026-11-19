#include "comm_lifecycle.h"
#include "adaptor.h"
#include "alloc.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include <sched.h>
#include <stdlib.h>

flagcxResult_t flagcxCommRequireActive(flagcxComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  return __atomic_load_n(&comm->finalizeState.lifecycle, __ATOMIC_ACQUIRE) ==
                 flagcxCommActive
             ? flagcxSuccess
             : flagcxInvalidUsage;
}

struct flagcxLifecycleBatch {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int start;
};

struct flagcxLifecycleJob {
  struct flagcxLifecycleBatch *batch;
  flagcxComm_t comm;
  flagcxLifecycleFn fn;
  pthread_t thread;
  flagcxResult_t result;
};

static void *flagcxLifecycleWorker(void *argument) {
  struct flagcxLifecycleJob *job = (struct flagcxLifecycleJob *)argument;
  pthread_mutex_lock(&job->batch->mutex);
  while (job->batch->start == 0)
    pthread_cond_wait(&job->batch->cond, &job->batch->mutex);
  int start = job->batch->start;
  pthread_mutex_unlock(&job->batch->mutex);
  job->result = start > 0 ? job->fn(job->comm) : flagcxSystemError;
  return NULL;
}

flagcxResult_t flagcxCommRunLifecycleJobs(flagcxComm_t *members, int count,
                                          flagcxLifecycleFn fn,
                                          flagcxLifecycleThreadCreate create) {
  if (members == NULL || count <= 0 || fn == NULL)
    return flagcxInvalidArgument;
  if (count == 1)
    return fn(members[0]);
  if (create == NULL)
    create = pthread_create;
  struct flagcxLifecycleJob *jobs =
      (struct flagcxLifecycleJob *)calloc(count, sizeof(*jobs));
  if (jobs == NULL)
    return flagcxSystemError;
  struct flagcxLifecycleBatch batch;
  batch.start = 0;
  if (pthread_mutex_init(&batch.mutex, NULL) != 0) {
    free(jobs);
    return flagcxSystemError;
  }
  if (pthread_cond_init(&batch.cond, NULL) != 0) {
    pthread_mutex_destroy(&batch.mutex);
    free(jobs);
    return flagcxSystemError;
  }
  int started = 0;
  for (int i = 0; i < count; i++) {
    jobs[i].batch = &batch;
    jobs[i].comm = members[i];
    jobs[i].fn = fn;
    jobs[i].result = flagcxSystemError;
  }
  for (; started < count - 1; started++)
    if (create(&jobs[started].thread, NULL, flagcxLifecycleWorker,
               &jobs[started]) != 0)
      break;
  // No worker enters a collective until *all* worker creation succeeded.
  pthread_mutex_lock(&batch.mutex);
  batch.start = started == count - 1 ? 1 : -1;
  pthread_cond_broadcast(&batch.cond);
  pthread_mutex_unlock(&batch.mutex);
  if (batch.start > 0)
    flagcxLifecycleWorker(&jobs[count - 1]);
  for (int i = 0; i < started; i++)
    pthread_join(jobs[i].thread, NULL);
  flagcxResult_t result = batch.start > 0 ? flagcxSuccess : flagcxSystemError;
  for (int i = 0; i < count; i++)
    if (result == flagcxSuccess && jobs[i].result != flagcxSuccess)
      result = jobs[i].result;
  pthread_cond_destroy(&batch.cond);
  pthread_mutex_destroy(&batch.mutex);
  free(jobs);
  return result;
}

flagcxResult_t flagcxCommMakeIntraProcGroup(flagcxComm_t *members, int count) {
  if (members == NULL || count <= 0)
    return flagcxInvalidArgument;
  for (int i = 0; i < count; i++) {
    if (members[i] == NULL || members[i]->intraLifecycle != NULL)
      return flagcxInvalidUsage;
    for (int j = 0; j < i; j++)
      if (members[j] == members[i])
        return flagcxInvalidArgument;
  }
  if (count == 1)
    return flagcxSuccess;
  struct flagcxIntraProcLifecycle *group =
      (struct flagcxIntraProcLifecycle *)calloc(1, sizeof(*group));
  if (group == NULL)
    return flagcxSystemError;
  if (pthread_mutex_init(&group->mutex, NULL) != 0) {
    free(group);
    return flagcxSystemError;
  }
  group->memberCount = (unsigned int)count;
  group->members = members[0];
  for (int i = 0; i < count; i++)
    members[i]->intraNext = i + 1 < count ? members[i + 1] : NULL;
  for (int i = 0; i < count; i++)
    __atomic_store_n(&members[i]->intraLifecycle, group, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

flagcxResult_t flagcxCommInitIntraProcGroup(flagcxComm_t comm) {
  struct Peer {
    uint64_t pidHash;
    flagcxComm_t comm;
  };
  struct Peer *peers = (struct Peer *)calloc(comm->nranks, sizeof(*peers));
  flagcxComm_t *members =
      (flagcxComm_t *)calloc(comm->localRanks, sizeof(*members));
  flagcxResult_t common;
  flagcxResult_t result = flagcxConvergeBootstrapStatus(
      comm,
      peers != NULL && members != NULL ? flagcxSuccess : flagcxSystemError,
      &common);
  if (result == flagcxSuccess)
    result = common;
  if (result != flagcxSuccess) {
    free(peers);
    free(members);
    return result;
  }
  uint64_t pidHash = getPidHash();
  peers[comm->rank].pidHash = pidHash;
  peers[comm->rank].comm = comm;
  result = bootstrapCollAllGather(comm->bootstrap, peers, sizeof(*peers));
  int count = 0;
  if (result == flagcxSuccess) {
    for (int i = 0; i < comm->localRanks; i++) {
      int rank = comm->localRankToRank[i];
      if (peers[rank].pidHash == pidHash)
        members[count++] = peers[rank].comm;
    }
    flagcxResult_t local = count > 0 ? flagcxSuccess : flagcxInternalError;
    if (count > 1 && members[0] == comm)
      local = flagcxCommMakeIntraProcGroup(members, count);
    result = flagcxConvergeBootstrapStatus(comm, local, &common);
    if (result == flagcxSuccess)
      result = common;
    if (result != flagcxSuccess && count > 1 && members[0] == comm &&
        comm->intraLifecycle != NULL) {
      struct flagcxIntraProcLifecycle *group = comm->intraLifecycle;
      for (int i = 0; i < count; i++) {
        members[i]->intraLifecycle = NULL;
        members[i]->intraNext = NULL;
      }
      pthread_mutex_destroy(&group->mutex);
      free(group);
    }
    if (result == flagcxSuccess && count > 1) {
      // The leader publishes only fully initialized process-local metadata.
      while (__atomic_load_n(&comm->intraLifecycle, __ATOMIC_ACQUIRE) == NULL)
        sched_yield();
    }
  }
  free(peers);
  free(members);
  return result;
}

flagcxResult_t flagcxCommReclaim(flagcxComm_t comm) {
  if (flagcxCommLifecycleGroupDepth() != 0)
    return flagcxInvalidUsage;
  struct flagcxIntraProcLifecycle *group = comm->intraLifecycle;
  if (group == NULL) {
    if (comm->finalizeState.commonPhase != flagcxFinalizeComplete)
      FLAGCXCHECK(flagcxCommFinalizeInternal(comm));
    return flagcxCommDestroyLocal(comm);
  }
  pthread_mutex_lock(&group->mutex);
  if (group->reclaiming ||
      (comm->destroyRequested && group->retryOwner != comm)) {
    pthread_mutex_unlock(&group->mutex);
    return flagcxInvalidUsage;
  }
  if (!comm->destroyRequested) {
    comm->destroyRequested = 1;
    group->destroyRequested++;
    __atomic_store_n(&comm->finalizeState.lifecycle, flagcxCommDestroying,
                     __ATOMIC_RELEASE);
  }
  if (group->destroyRequested < group->memberCount) {
    pthread_mutex_unlock(&group->mutex);
    return flagcxSuccess;
  }
  group->retryOwner = comm;
  group->reclaiming = 1;
  int count = 0;
  for (flagcxComm_t member = group->members; member != NULL;
       member = member->intraNext)
    count++;
  flagcxComm_t *members = (flagcxComm_t *)calloc(count, sizeof(*members));
  if (members == NULL) {
    group->reclaiming = 0;
    pthread_mutex_unlock(&group->mutex);
    return flagcxSystemError;
  }
  int index = 0;
  for (flagcxComm_t member = group->members; member != NULL;
       member = member->intraNext)
    members[index++] = member;
  pthread_mutex_unlock(&group->mutex);
  flagcxResult_t result = flagcxCommRunLifecycleJobs(
      members, count, flagcxCommFinalizeInternal, NULL);
  free(members);
  if (result == flagcxSuccess) {
    flagcxComm_t *link = &group->members;
    while (*link != NULL) {
      flagcxComm_t member = *link;
      if (member == comm) {
        link = &member->intraNext;
        continue;
      }
      flagcxComm_t next = member->intraNext;
      result = flagcxCommDestroyLocal(member);
      if (result != flagcxSuccess)
        break;
      *link = next;
    }
  }
  if (result == flagcxSuccess)
    result = flagcxCommDestroyLocal(comm);
  if (result == flagcxSuccess) {
    pthread_mutex_destroy(&group->mutex);
    free(group);
  } else {
    pthread_mutex_lock(&group->mutex);
    group->reclaiming = 0;
    pthread_mutex_unlock(&group->mutex);
  }
  return result;
}

struct flagcxFinalizeRequest {
  flagcxComm_t comm;
  struct flagcxFinalizeRequest *next;
};
static __thread int lifecycleGroupDepth;
static __thread flagcxResult_t lifecycleGroupError;
static __thread struct flagcxFinalizeRequest *finalizeRequests;

int flagcxCommLifecycleGroupDepth() { return lifecycleGroupDepth; }
void flagcxCommLifecycleGroupStart() { lifecycleGroupDepth++; }

flagcxResult_t flagcxCommFinalizeEnqueue(flagcxComm_t comm) {
  if (lifecycleGroupDepth == 0)
    return flagcxCommFinalizeInternal(comm);
  for (struct flagcxFinalizeRequest *p = finalizeRequests; p != NULL;
       p = p->next)
    if (p->comm == comm)
      return flagcxSuccess;
  struct flagcxFinalizeRequest *request =
      (struct flagcxFinalizeRequest *)calloc(1, sizeof(*request));
  if (request == NULL)
    return lifecycleGroupError = flagcxSystemError;
  request->comm = comm;
  request->next = finalizeRequests;
  finalizeRequests = request;
  if (comm->finalizeState.commonPhase != flagcxFinalizeComplete) {
    __atomic_store_n(&comm->finalizeState.lifecycle, flagcxCommFinalizing,
                     __ATOMIC_RELEASE);
    __atomic_store_n(&comm->finalizeState.lastResult, flagcxInProgress,
                     __ATOMIC_RELEASE);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommLifecycleGroupEnd(flagcxResult_t incoming) {
  if (lifecycleGroupDepth <= 0)
    return flagcxInvalidUsage;
  if (incoming != flagcxSuccess && lifecycleGroupError == flagcxSuccess)
    lifecycleGroupError = incoming;
  if (--lifecycleGroupDepth != 0)
    return lifecycleGroupError;
  flagcxResult_t result = lifecycleGroupError;
  lifecycleGroupError = flagcxSuccess;
  int count = 0;
  for (struct flagcxFinalizeRequest *p = finalizeRequests; p != NULL;
       p = p->next)
    count++;
  flagcxComm_t *members =
      count == 0 ? NULL : (flagcxComm_t *)calloc(count, sizeof(*members));
  if (count != 0 && members == NULL)
    result = flagcxSystemError;
  if (result == flagcxSuccess && count != 0) {
    int index = 0;
    for (struct flagcxFinalizeRequest *p = finalizeRequests; p != NULL;
         p = p->next)
      members[index++] = p->comm;
    result = flagcxCommRunLifecycleJobs(members, count,
                                        flagcxCommFinalizeInternal, NULL);
  }
  free(members);
  while (finalizeRequests != NULL) {
    struct flagcxFinalizeRequest *request = finalizeRequests;
    finalizeRequests = request->next;
    if (result != flagcxSuccess)
      __atomic_store_n(&request->comm->finalizeState.lastResult, result,
                       __ATOMIC_RELEASE);
    free(request);
  }
  return result;
}
