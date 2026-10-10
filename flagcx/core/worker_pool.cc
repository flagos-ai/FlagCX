/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "worker_pool.h"

#include <algorithm>
#include <sched.h>

void *flagcxWorkerPool::run(void *arg) {
  Worker *worker = static_cast<Worker *>(arg);
  flagcxResult_t initResult = flagcxSuccess;
  try {
    if (worker->ops_.initOnThread != nullptr)
      initResult = worker->ops_.initOnThread(worker->state_);
  } catch (...) {
    initResult = flagcxInternalError;
  }

  {
    std::lock_guard<std::mutex> lock(worker->initMutex_);
    worker->initResult_ = initResult;
    worker->initDone_ = true;
  }
  (void)pthread_cond_signal(&worker->initCond_);

  flagcxResult_t result = initResult;
  if (initResult == flagcxSuccess) {
    try {
      while (true) {
        bool madeProgress = false;
        bool outstanding = false;
        const bool stopping = worker->stop_.load(std::memory_order_acquire);
        result = worker->ops_.progressOnce(worker->state_, stopping,
                                           &madeProgress, &outstanding);
        if (result != flagcxSuccess && result != flagcxInProgress)
          break;
        if (stopping && !outstanding) {
          result = flagcxSuccess;
          break;
        }
        if (!madeProgress)
          sched_yield();
      }
    } catch (...) {
      result = flagcxInternalError;
    }
  }

  try {
    if (worker->ops_.finishOnThread != nullptr)
      worker->ops_.finishOnThread(worker->state_);
  } catch (...) {
    if (result == flagcxSuccess || result == flagcxInProgress)
      result = flagcxInternalError;
  }
  worker->result_.store(result, std::memory_order_release);
  return nullptr;
}

flagcxResult_t flagcxWorkerPool::start(const flagcxWorkerOps *ops, void *state,
                                       Worker **worker) {
  if (worker == nullptr)
    return flagcxInvalidArgument;
  *worker = nullptr;
  if (ops == nullptr || ops->progressOnce == nullptr)
    return flagcxInvalidArgument;

  std::shared_ptr<Worker> candidate;
  try {
    candidate.reset(new Worker(*ops, state));
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
      return flagcxInvalidUsage;
    workers_.push_back(candidate);
    if (pthread_create(&candidate->thread_, nullptr, &flagcxWorkerPool::run,
                       candidate.get()) != 0) {
      workers_.pop_back();
      return flagcxSystemError;
    }
    candidate->threadStarted_ = true;
  } catch (...) {
    return flagcxSystemError;
  }
  Worker *entry = candidate.get();

  flagcxResult_t initResult = flagcxSystemError;
  try {
    std::unique_lock<std::mutex> initLock(entry->initMutex_);
    while (!entry->initDone_)
      (void)pthread_cond_wait(&entry->initCond_,
                              entry->initMutex_.native_handle());
    initResult = entry->initResult_;
  } catch (...) {
    // The candidate remains registered until its thread has exited.
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initResult == flagcxSuccess && stopping_)
      initResult = flagcxInvalidUsage;
    if (initResult == flagcxSuccess) {
      *worker = entry;
      return flagcxSuccess;
    }
  }

  requestStop(entry);
  flagcxResult_t joinResult = join(entry);
  {
    std::lock_guard<std::mutex> joinLock(entry->joinMutex_);
    if (!entry->threadJoined_) {
      // Keep the registration and its state alive if pthread_join failed.
      *worker = entry;
      return joinResult;
    }
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(workers_.begin(), workers_.end(),
                           [entry](const std::shared_ptr<Worker> &registered) {
                             return registered.get() == entry;
                           });
    if (it != workers_.end())
      workers_.erase(it);
  }
  return initResult;
}

void flagcxWorkerPool::requestStop(Worker *worker) {
  if (worker != nullptr)
    worker->stop_.store(true, std::memory_order_release);
}

flagcxResult_t flagcxWorkerPool::join(Worker *worker) {
  if (worker == nullptr)
    return flagcxInvalidArgument;
  std::lock_guard<std::mutex> lock(worker->joinMutex_);
  if (worker->threadStarted_ && !worker->threadJoined_) {
    if (pthread_equal(worker->thread_, pthread_self()))
      return flagcxInvalidUsage;
    if (pthread_join(worker->thread_, nullptr) != 0)
      return flagcxSystemError;
    worker->threadJoined_ = true;
  }
  return worker->result_.load(std::memory_order_acquire);
}

flagcxResult_t flagcxWorkerPool::stopAndJoin(Worker *worker) {
  requestStop(worker);
  return join(worker);
}

flagcxResult_t flagcxWorkerPool::status(const Worker *worker) const {
  if (worker == nullptr)
    return flagcxInvalidArgument;
  return worker->result_.load(std::memory_order_acquire);
}

flagcxResult_t flagcxWorkerPool::retire(Worker **worker) {
  if (worker == nullptr || *worker == nullptr)
    return flagcxInvalidArgument;
  Worker *target = *worker;
  std::shared_ptr<Worker> entry;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(workers_.begin(), workers_.end(),
                           [target](const std::shared_ptr<Worker> &registered) {
                             return registered.get() == target;
                           });
    if (it == workers_.end())
      return flagcxInvalidArgument;
    if ((*it)->retiring_)
      return flagcxInvalidUsage;
    (*it)->retiring_ = true;
    entry = *it;
  }

  flagcxResult_t result = stopAndJoin(entry.get());
  bool joined;
  {
    std::lock_guard<std::mutex> joinLock(entry->joinMutex_);
    joined = entry->threadJoined_;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(workers_.begin(), workers_.end(),
                           [target](const std::shared_ptr<Worker> &registered) {
                             return registered.get() == target;
                           });
    if (it != workers_.end()) {
      if (!joined)
        (*it)->retiring_ = false;
      else
        workers_.erase(it);
    }
  }
  if (!joined)
    return result;
  *worker = nullptr;
  return result;
}

void flagcxWorkerPool::requestStopAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopping_ = true;
  for (const auto &worker : workers_)
    requestStop(worker.get());
}

flagcxResult_t flagcxWorkerPool::joinAll() {
  requestStopAll();
  std::vector<std::shared_ptr<Worker>> snapshot;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot = workers_;
  }
  flagcxResult_t firstError = flagcxSuccess;
  for (const auto &worker : snapshot) {
    flagcxResult_t result = join(worker.get());
    if (firstError == flagcxSuccess && result != flagcxSuccess)
      firstError = result;
  }
  return firstError;
}

size_t flagcxWorkerPool::workerCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return workers_.size();
}

flagcxWorkerPool::~flagcxWorkerPool() { (void)joinAll(); }
