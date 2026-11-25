#pragma once
#include "flagcx.h"
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <utility>

using flagcxPairComms = std::map<std::pair<int, int>, flagcxComm_t>;

// Retained callbacks own their remaining handles and dependencies. Retrying is
// explicit and runs on the caller's thread, with the original participants.
class flagcxPendingBackendCleanup {
public:
  static flagcxPendingBackendCleanup &instance() {
    // Pending tensor dependencies must not be destroyed by static teardown.
    // The registry has process lifetime; successful retries consume all owners.
    static auto *registry = new flagcxPendingBackendCleanup;
    return *registry;
  }
  void retain(std::function<flagcxResult_t()> cleanup) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(std::move(cleanup));
  }
  flagcxResult_t retry() {
    std::list<std::function<flagcxResult_t()>> work;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      work.splice(work.end(), pending_);
    }
    flagcxResult_t first = flagcxSuccess;
    while (!work.empty()) {
      flagcxResult_t result;
      try {
        result = work.front()();
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.splice(pending_.begin(), work);
        throw;
      }
      if (result != flagcxSuccess) {
        first = result;
        break;
      }
      work.pop_front();
    }
    if (!work.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.splice(pending_.begin(), work);
    }
    return first;
  }

private:
  std::mutex mutex_;
  std::list<std::function<flagcxResult_t()>> pending_;
};
