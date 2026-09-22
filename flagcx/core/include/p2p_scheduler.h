/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Internal helpers shared by the P2P worker and ACCL schedulers.
 ************************************************************************/

#ifndef FLAGCX_P2P_SCHEDULER_H_
#define FLAGCX_P2P_SCHEDULER_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace flagcxP2pScheduling {

inline int workerForAddress(uintptr_t address, int numWorkers) {
  if (numWorkers <= 0 || address == 0)
    return -1;
  /* Communication objects are at least 16-byte aligned.  Drop those constant
     bits before mixing so power-of-two worker counts use entropy from the
     whole address rather than only the zero alignment bits. */
  uint64_t value = static_cast<uint64_t>(address >> 4);
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<int>(value % static_cast<uint64_t>(numWorkers));
}

inline size_t channelForTicket(uint64_t ticket, size_t ordinal,
                               size_t channelCount) {
  if (channelCount == 0)
    return 0;
  return static_cast<size_t>((ticket + ordinal) % channelCount);
}

struct CompletionTracker {
  std::atomic<int> pending{0};
  std::atomic<int> failed{0};
  std::atomic<int> hardFailed{0};
  std::mutex waitMu;
  std::condition_variable waitCv;

  void complete(int count = 1, int failedCount = 0) {
    if (failedCount > 0)
      failed.fetch_add(failedCount, std::memory_order_release);
    if (pending.fetch_sub(count, std::memory_order_acq_rel) == count) {
      std::lock_guard<std::mutex> lk(waitMu);
      waitCv.notify_all();
    }
  }

  void wait() {
    std::unique_lock<std::mutex> lk(waitMu);
    waitCv.wait(
        lk, [this] { return pending.load(std::memory_order_acquire) == 0; });
  }
};

} // namespace flagcxP2pScheduling

#endif
