/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Comm Traits — Unified compile-time dispatch for device APIs.
 *
 * Architecture:
 *   PlatformTraits<P>         — platform-level: Intrin, Atomic, CoopOps, Clock
 *   CommTraits<D>             — backend-level:  Window, Comm, Team, ...
 *   DefaultBackend<PlatformTag> — common IPC fallback (partial specialization)
 *
 * CommTraits pulls in platform capabilities via using-aliases (not
 * inheritance). Vendor specializations wrap vendor types with member
 * functions. The DefaultBackend partial specialization provides IPC-based
 * types that work with any platform.
 *
 * Selection:
 *   NVIDIA + NCCL > 2.28:    DeviceAPI = CommTraits<NcclBackend>
 *   NVIDIA + fallback:       DeviceAPI =
 *CommTraits<DefaultBackend<NvidiaPlatform>>
 *
 * Kernel code uses DeviceAPI::* exclusively, no #ifdef branches.
 ************************************************************************/

#ifndef FLAGCX_COMM_TRAITS_H_
#define FLAGCX_COMM_TRAITS_H_

#include "flagcx_device_enums.h"
#include "platform_traits.h"
#include <cstddef>
#include <cstdint>

// Primary template — each backend provides a specialization.  Unified IR uses
// only this backend-neutral contract:
//   Comm: getRank(), getIntraRank(), getIntraSize(),
//         isOneSidedTransportReady(), supportsDirectCounterAccess().
//   Net:  signal/wait/read/reset completion semantics plus the transport
//         operations. Internal completion word width belongs to the backend.
// The IR layer must not inspect backend storage fields or derive slot offsets.
template <typename Impl>
struct CommTraits;

// Backend-neutral defaults for the public C++ wrapper. Vendor backends keep
// the established 64-bit signal / 56-bit counter defaults. A backend may
// expose compile-time constants to narrow those defaults without changing the
// public ABI or adding a runtime capability branch.
template <typename...>
using flagcxCommVoidT = void;

template <typename API, typename = void>
struct flagcxBackendCompletionBits {
  static constexpr int value = 64;
};

template <typename API>
struct flagcxBackendCompletionBits<
    API, flagcxCommVoidT<decltype(API::completionBits)>> {
  static constexpr int value = API::completionBits;
};

template <typename API, typename = void>
struct flagcxBackendDefaultCounterBits {
  static constexpr int value = 56;
};

template <typename API>
struct flagcxBackendDefaultCounterBits<
    API, flagcxCommVoidT<decltype(API::defaultCounterBits)>> {
  static constexpr int value = API::defaultCounterBits;
};

// DefaultBackend tag — parameterized by platform for the partial specialization
template <typename PlatformTag>
struct DefaultBackend {};

// ============================================================
// Action types for one-sided operations (needed by traits Net types).
// Pure POD structs with no device builtins.
// ============================================================

struct flagcxDevNet_None {};
struct flagcxDevNet_SignalInc {
  flagcxDevSignal_t signal;
};
struct flagcxDevNet_SignalAdd {
  flagcxDevSignal_t signal;
  uint64_t value;
};
struct flagcxDevNet_CounterInc {
  flagcxDevCounter_t counter;
};

// Shared memory descriptor for NIC descriptor optimization.
// Uses void* on all paths; vendor Net casts to native type in toNccl().
struct flagcxDescriptorSmem {
  void *_impl = nullptr;
};

struct flagcxDevNet_DescriptorSmem {
  flagcxDescriptorSmem smem;
};

// Put/Get are supported by Inter/World sync(), not their legacy split
// arrive()/wait() calls. Intra barriers have no Net fence parameter.
// The fence covers one Net context supplied to the barrier. Team tags select
// the participating ranks; they do not expand the fence to other contexts.
// Put makes prior puts from participating ranks visible at this rank, while
// Get completes this rank's prior gets into local memory. Both apply only to
// requests submitted through that context before the barrier.
// TODO: Introduce an explicit all-context barrier if callers need one fence
// for requests issued through multiple Net contexts.
enum class flagcxDevNetFenceLevel : uint32_t {
  Relaxed = 0,
  Put = 1u << 0,
  Get = 1u << 1,
};

FLAGCX_HOST_DEVICE_INLINE constexpr flagcxDevNetFenceLevel
operator|(flagcxDevNetFenceLevel a, flagcxDevNetFenceLevel b) {
  return static_cast<flagcxDevNetFenceLevel>(static_cast<uint32_t>(a) |
                                             static_cast<uint32_t>(b));
}

FLAGCX_HOST_DEVICE_INLINE constexpr flagcxDevNetFenceLevel
operator&(flagcxDevNetFenceLevel a, flagcxDevNetFenceLevel b) {
  return static_cast<flagcxDevNetFenceLevel>(static_cast<uint32_t>(a) &
                                             static_cast<uint32_t>(b));
}

FLAGCX_HOST_DEVICE_INLINE constexpr bool
flagcxDevNetFenceHas(flagcxDevNetFenceLevel fence, flagcxDevNetFenceLevel bit) {
  return static_cast<uint32_t>(fence & bit) != 0;
}

FLAGCX_HOST_DEVICE_INLINE constexpr bool
flagcxDevNetFenceValid(flagcxDevNetFenceLevel fence) {
  return (static_cast<uint32_t>(fence) & ~uint32_t(3)) == 0;
}

// ============================================================
// Unified team/barrier tag types.
// Used as both Barrier<Backend, Tag> template parameter
// and as ctor dispatch tags — eliminating the old two-tag redundancy.
// ============================================================
struct flagcxTeamTagIntra {};
struct flagcxTeamTagInter {};
struct flagcxTeamTagWorld {};

// Primary template — each backend provides specializations
template <typename Backend, typename Tag, typename Coop>
struct Barrier;

// Vendor specializations + DeviceAPI selection
#if defined(USE_NVIDIA_ADAPTOR)
#include "nvidia_comm_traits.h"
#elif defined(USE_DU_ADAPTOR)
#include "du_comm_traits.h"
#elif defined(USE_SUNRISE_ADAPTOR)
#include "sunrise_comm_traits.h"
#elif defined(USE_KUNLUNXIN_ADAPTOR)
#include "kunlunxin_comm_traits.h"
#elif defined(USE_ILUVATAR_ADAPTOR)
#include "iluvatar_comm_traits.h"
#else
#include "default_comm_traits.h"
using DeviceAPI = CommTraits<DefaultBackend<DefaultPlatform>>;
#endif

#endif // FLAGCX_COMM_TRAITS_H_
