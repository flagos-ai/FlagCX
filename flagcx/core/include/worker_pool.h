/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_WORKER_POOL_H_
#define FLAGCX_WORKER_POOL_H_

#include "flagcx.h"

#include <pthread.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

// Each registration owns one thread. The callback only progresses its own
// business state; the pool neither queues operations nor schedules other
// worker types on that thread. `state` must outlive the worker thread. Control
// methods must be called outside worker callbacks; do not destroy the pool on
// one of its own worker threads. Destruction must not race with pool methods.
struct flagcxWorkerOps {
  // Called on the worker thread before progressOnce(). On failure,
  // finishOnThread() is still called to release partially initialized state.
  flagcxResult_t (*initOnThread)(void *state);

  // A single bounded progress pass. On stop, keep reporting outstanding=true
  // until accepted work is retired or the business has safely failed it.
  // flagcxInProgress is a nonfatal result, like flagcxSuccess.
  flagcxResult_t (*progressOnce)(void *state, bool stopping, bool *madeProgress,
                                 bool *outstanding);

  // Called on the same thread after progress ends, including init failure.
  void (*finishOnThread)(void *state);
};

class flagcxWorkerPool {
public:
  class Worker {
  public:
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;
    ~Worker() { (void)pthread_cond_destroy(&initCond_); }

  private:
    friend class flagcxWorkerPool;
    Worker(const flagcxWorkerOps &ops, void *state)
        : ops_(ops), state_(state) {}

    const flagcxWorkerOps ops_;
    void *state_;
    pthread_t thread_{};
    bool threadStarted_ = false;
    // Accessed under joinMutex_ after the thread has been started.
    bool threadJoined_ = false;
    std::atomic<bool> stop_{false};
    std::atomic<flagcxResult_t> result_{flagcxInProgress};
    std::mutex initMutex_;
    pthread_cond_t initCond_ = PTHREAD_COND_INITIALIZER;
    bool initDone_ = false;
    flagcxResult_t initResult_ = flagcxSuccess;
    std::mutex joinMutex_;
    // Protected by the pool mutex. Prevents two callers from retiring the
    // same registration while its thread is draining.
    bool retiring_ = false;
  };

  flagcxWorkerPool() = default;
  ~flagcxWorkerPool();
  flagcxWorkerPool(const flagcxWorkerPool &) = delete;
  flagcxWorkerPool &operator=(const flagcxWorkerPool &) = delete;

  // Blocks until initOnThread() finishes. A failed initialization removes
  // the worker and sets *worker to nullptr after a successful pthread_join.
  // If pthread_join itself fails, the handle remains registered so the caller
  // can retry cleanup without invalidating the callback state.
  flagcxResult_t start(const flagcxWorkerOps *ops, void *state,
                       Worker **worker);

  void requestStop(Worker *worker);
  flagcxResult_t join(Worker *worker);
  flagcxResult_t stopAndJoin(Worker *worker);
  // Returns flagcxInProgress while the thread is running, then its terminal
  // result. The handle must still be registered in this pool.
  flagcxResult_t status(const Worker *worker) const;
  // Stops, joins, and removes one registration. Sets *worker to nullptr; the
  // handle must not be used again. Callers must synchronize use of a handle
  // with its retirement. Use this for independently destroyed comms.
  flagcxResult_t retire(Worker **worker);
  void requestStopAll();
  flagcxResult_t joinAll();
  // Number of registered handles, including workers already joined but not
  // yet retired.
  size_t workerCount() const;

private:
  static void *run(void *arg);

  mutable std::mutex mutex_;
  bool stopping_ = false;
  // A local reference keeps a worker alive while a control method waits
  // without holding the pool mutex.
  std::vector<std::shared_ptr<Worker>> workers_;
};

#endif // FLAGCX_WORKER_POOL_H_
