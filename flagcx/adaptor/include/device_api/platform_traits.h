/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Platform Traits - Compile-time dispatch for platform-level capabilities.
 *
 * PlatformTraits<P> provides:
 *   - Intrin: SIMT intrinsics (lane, activemask, syncwarp, popc, ...)
 *   - Atomic: Scoped atomic operations (load, store, fetchAdd, ...)
 *
 * Each platform (NVIDIA, Cambricon, ...) provides a specialization.
 * DeviceTraits<D> pulls in platform capabilities via using-aliases.
 ************************************************************************/

#ifndef FLAGCX_PLATFORM_TRAITS_H_
#define FLAGCX_PLATFORM_TRAITS_H_

#include "device_utils.h"

// Common enum types used as parameters to PlatformTraits::Atomic methods
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
  flagcxDeviceScopeBlock = 2,
  flagcxDeviceScopeThread = 3
} flagcxDeviceScope_t;

// Primary template — each platform provides a specialization
template <typename Platform>
struct PlatformTraits;

// Common CoopAny — vtable-based type erasure, platform-independent.
// Each PlatformTraits<P> aliases CoopAny = PlatformCoop.
struct PlatformCoop {
  struct Storage {
    alignas(alignof(void *)) char space[16];
  };
  struct VTable {
    int (*threadRank)(void const *);
    int (*size)(void const *);
    void (*sync)(void *);
  };

  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static int threadRank_fn(void const *o) {
    return static_cast<Impl const *>(o)->threadRank();
  }
  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static int size_fn(void const *o) {
    return static_cast<Impl const *>(o)->size();
  }
  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static void sync_fn(void *o) {
    static_cast<Impl *>(o)->sync();
  }

  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static VTable const *get_vtable() {
    static_assert(sizeof(Impl) <= sizeof(Storage), "Coop type too large");
    static_assert(alignof(Impl) <= alignof(Storage),
                  "Coop alignment too large");
    static constexpr VTable v = {&threadRank_fn<Impl>, &size_fn<Impl>,
                                 &sync_fn<Impl>};
    return &v;
  }

  Storage storage;
  VTable const *vtable;

  // Default ctor: single-thread no-op
  FLAGCX_DEVICE_INLINE_DECORATOR PlatformCoop()
      : storage{}, vtable(noop_vtable()) {}
  PlatformCoop(PlatformCoop const &) = default;

  // Convert from any Coop type with threadRank()/size()/sync()
  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR PlatformCoop(Impl impl) {
    char const *src = reinterpret_cast<char const *>(&impl);
    for (unsigned i = 0; i < sizeof(Impl); ++i)
      storage.space[i] = src[i];
    vtable = get_vtable<Impl>();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return vtable->threadRank(&storage);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
    return vtable->size(&storage);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { vtable->sync(&storage); }

private:
  static FLAGCX_DEVICE_INLINE_DECORATOR int noop_rank(void const *) {
    return 0;
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR int noop_size(void const *) {
    return 1;
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void noop_sync(void *) {}
  static FLAGCX_DEVICE_INLINE_DECORATOR VTable const *noop_vtable() {
    static constexpr VTable v = {&noop_rank, &noop_size, &noop_sync};
    return &v;
  }
};

// Include platform specializations
#ifdef USE_NVIDIA_ADAPTOR
#include "nvidia_platform_traits.h"
#endif

// Future:
// #ifdef USE_CAMBRICON_ADAPTOR
// #include "cambricon_platform_traits.h"
// #endif

#endif // FLAGCX_PLATFORM_TRAITS_H_
