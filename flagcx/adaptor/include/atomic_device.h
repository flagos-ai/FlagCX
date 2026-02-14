#ifndef FLAGCX_ADAPTOR_ATOMIC_DEVICE_H_
#define FLAGCX_ADAPTOR_ATOMIC_DEVICE_H_

#include "device_utils.h"

// Unified enums for memory order and scope
typedef enum {
  flagcxDeviceMemoryOrderRelaxed = 0,
  flagcxDeviceMemoryOrderAcquire = 1,
  flagcxDeviceMemoryOrderRelease = 2,
  flagcxDeviceMemoryOrderAcqRel = 3,
  flagcxDeviceMemoryOrderSeqCst = 4
} flagcxDeviceMemoryOrder_t;

typedef enum {
  flagcxDeviceScopeSystem = 0,
  flagcxDeviceScopeDevice = 1,
  flagcxDeviceScopeBlock = 2
} flagcxDeviceScope_t;

#if defined(USE_NVIDIA_ADAPTOR)
#include <cuda/atomic>

// Mapping arrays from flagcx enums to CUDA types
static FLAGCX_DEVICE_CONSTANT_DECORATOR cuda::memory_order
    flagcxDeviceMemoryOrderMap[] = {
        cuda::memory_order_relaxed, cuda::memory_order_acquire,
        cuda::memory_order_release, cuda::memory_order_acq_rel,
        cuda::memory_order_seq_cst};

// Helper to dispatch based on scope at compile time
template <typename T, flagcxDeviceScope_t Scope>
struct flagcxAtomicHelper;

template <typename T>
struct flagcxAtomicHelper<T, flagcxDeviceScopeSystem> {
  using atomic_ref_t = cuda::atomic_ref<T, cuda::thread_scope_system>;
};

template <typename T>
struct flagcxAtomicHelper<T, flagcxDeviceScopeDevice> {
  using atomic_ref_t = cuda::atomic_ref<T, cuda::thread_scope_device>;
};

template <typename T>
struct flagcxAtomicHelper<T, flagcxDeviceScopeBlock> {
  using atomic_ref_t = cuda::atomic_ref<T, cuda::thread_scope_block>;
};

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T
flagcxDeviceAtomicLoad(T *ptr, flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  return ref_t{*ptr}.load(flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDeviceAtomicStore(T *ptr, const T &val,
                        flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  ref_t{*ptr}.store(val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchAdd(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  return ref_t{*ptr}.fetch_add(val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchSub(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  return ref_t{*ptr}.fetch_sub(val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchOr(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  return ref_t{*ptr}.fetch_or(val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchAnd(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  return ref_t{*ptr}.fetch_and(val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicExchange(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  return ref_t{*ptr}.exchange(val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR bool
flagcxDeviceAtomicCompareExchange(T *ptr, T &expected, const T &desired,
                                  flagcxDeviceMemoryOrder_t memoryOrder) {
  using ref_t = typename flagcxAtomicHelper<T, Scope>::atomic_ref_t;
  return ref_t{*ptr}.compare_exchange_strong(
      expected, desired, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

#else
// Fallback for other platforms using GCC-style atomics

static const int flagcxDeviceMemoryOrderMap[] = {
    __ATOMIC_RELAXED, __ATOMIC_ACQUIRE, __ATOMIC_RELEASE, __ATOMIC_ACQ_REL,
    __ATOMIC_SEQ_CST};

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T
flagcxDeviceAtomicLoad(T *ptr, flagcxDeviceMemoryOrder_t memoryOrder) {
  return __atomic_load_n(ptr, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDeviceAtomicStore(T *ptr, const T &val,
                        flagcxDeviceMemoryOrder_t memoryOrder) {
  __atomic_store_n(ptr, val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchAdd(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  return __atomic_fetch_add(ptr, val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchSub(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  return __atomic_fetch_sub(ptr, val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchOr(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  return __atomic_fetch_or(ptr, val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicFetchAnd(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  return __atomic_fetch_and(ptr, val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR T flagcxDeviceAtomicExchange(
    T *ptr, const T &val, flagcxDeviceMemoryOrder_t memoryOrder) {
  return __atomic_exchange_n(ptr, val, flagcxDeviceMemoryOrderMap[memoryOrder]);
}

template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
FLAGCX_DEVICE_INLINE_DECORATOR bool
flagcxDeviceAtomicCompareExchange(T *ptr, T &expected, const T &desired,
                                  flagcxDeviceMemoryOrder_t memoryOrder) {
  return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                     flagcxDeviceMemoryOrderMap[memoryOrder],
                                     flagcxDeviceMemoryOrderMap[memoryOrder]);
}

#endif

#endif // FLAGCX_ADAPTOR_ATOMIC_DEVICE_H_
