/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * DefaultBackend compile-time completion storage and operations.
 ************************************************************************/

#ifndef FLAGCX_DEFAULT_COMPLETION_H_
#define FLAGCX_DEFAULT_COMPLETION_H_

#include "flagcx_device_enums.h"
#include "flagcx_kernel_core.h"

template <typename Word, bool = (sizeof(Word) == sizeof(uint32_t))>
struct DefaultCompletionStorage {
  template <typename Comm>
  FLAGCX_DEVICE_INLINE_DECORATOR Word *
  getDirectSignalBuffer(const Comm &comm) const {
    return reinterpret_cast<Word *>(comm.signalBuffer);
  }

  template <typename Comm>
  FLAGCX_DEVICE_INLINE_DECORATOR Word *
  getDirectCounterBuffer(const Comm &comm) const {
    return reinterpret_cast<Word *>(comm.counterBuffer);
  }

  template <typename Comm>
  FLAGCX_DEVICE_INLINE_DECORATOR Word *getDirectSignalPeerPtr(const Comm &comm,
                                                              int peer) const {
    return reinterpret_cast<Word *>(comm.getSignalPeerPtr(peer));
  }

  template <typename DI>
  FLAGCX_HOST_DEVICE_INLINE void populateCompletion(const DI &) {}

  template <typename Atomic>
  FLAGCX_DEVICE_INLINE_DECORATOR static Word
  loadCompletion(Word *direct, uint64_t *, int index,
                 flagcxDeviceMemoryOrder_t order) {
    return Atomic::load(&direct[index], order);
  }

  template <typename Atomic>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  resetProxy(uint64_t *, int, flagcxDeviceMemoryOrder_t) {}

  FLAGCX_DEVICE_INLINE_DECORATOR static void validate(uint64_t) {}

  // Existing 64-bit backends historically accept both 32/56/64-bit callers;
  // preserve that behavior. The width argument remains a backend concern.
  FLAGCX_DEVICE_INLINE_DECORATOR static void validateBits(int) {}

  FLAGCX_DEVICE_INLINE_DECORATOR static Word completionValue(uint64_t value) {
    return static_cast<Word>(value);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static Word advance(uint64_t current,
                                                     uint64_t delta) {
    return static_cast<Word>(current + delta);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool waitBefore(Word current,
                                                        Word target) {
    return current < target;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  useDirectSignal(bool hasPeerPointer, bool communicatorUsesDirect) {
    return hasPeerPointer && communicatorUsesDirect;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool before(Word lhs, Word rhs) {
    return (int64_t)(lhs - rhs) < 0;
  }
};

template <typename Word>
struct DefaultCompletionStorage<Word, true> {
  Word *directSignalBuffer;
  Word *directCounterBuffer;
  Word **directSignalPeerPtrs;

  template <typename Comm>
  FLAGCX_DEVICE_INLINE_DECORATOR Word *
  getDirectSignalBuffer(const Comm &) const {
    return directSignalBuffer;
  }

  template <typename Comm>
  FLAGCX_DEVICE_INLINE_DECORATOR Word *
  getDirectCounterBuffer(const Comm &) const {
    return directCounterBuffer;
  }

  template <typename Comm>
  FLAGCX_DEVICE_INLINE_DECORATOR Word *getDirectSignalPeerPtr(const Comm &,
                                                              int peer) const {
    return directSignalPeerPtrs ? directSignalPeerPtrs[peer] : nullptr;
  }

  template <typename DI>
  FLAGCX_HOST_DEVICE_INLINE void populateCompletion(const DI &di) {
    directSignalBuffer = static_cast<Word *>(di.completionSignalBuffer);
    directCounterBuffer = static_cast<Word *>(di.completionCounterBuffer);
    directSignalPeerPtrs =
        reinterpret_cast<Word **>(di.completionSignalPeerPtrs);
  }

  template <typename Atomic>
  FLAGCX_DEVICE_INLINE_DECORATOR static Word
  loadCompletion(Word *direct, uint64_t *proxy, int index,
                 flagcxDeviceMemoryOrder_t order) {
    Word directValue = Atomic::load(&direct[index], order);
    uint64_t proxyValue = Atomic::load(&proxy[index], order);
    return static_cast<Word>(directValue + static_cast<Word>(proxyValue));
  }

  template <typename Atomic>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  resetProxy(uint64_t *proxy, int index, flagcxDeviceMemoryOrder_t order) {
    Atomic::store(&proxy[index], uint64_t{0}, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static void validate(uint64_t value) {
    if (static_cast<uint64_t>(static_cast<Word>(value)) != value)
      __builtin_trap();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static void validateBits(int bits) {
    if (bits != 32)
      __builtin_trap();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static Word completionValue(uint64_t value) {
    validate(value);
    return static_cast<Word>(value);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static Word advance(uint64_t current,
                                                     uint64_t delta) {
    validate(delta);
    return static_cast<Word>(static_cast<Word>(current) +
                             static_cast<Word>(delta));
  }

  // 32-bit completion values are circular sequence numbers. Comparisons are
  // well-defined while the outstanding distance stays below 2^31.
  FLAGCX_DEVICE_INLINE_DECORATOR static bool waitBefore(Word current,
                                                        Word target) {
    return static_cast<int32_t>(current - target) < 0;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  useDirectSignal(bool hasPeerPointer, bool) {
    return hasPeerPointer;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool before(Word lhs, Word rhs) {
    return static_cast<int32_t>(lhs - rhs) < 0;
  }
};

#endif // FLAGCX_DEFAULT_COMPLETION_H_
