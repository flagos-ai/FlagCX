/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Ascend (CANN) Platform Traits.
 *
 * One header, two selections keyed on __CCE_IS_AICORE__ (bisheng defines it
 * on the device pass only; measured on Ascend 910C, CANN 9.1.0-beta.3,
 * bisheng = clang 15.0.5, __NPU_ARCH__ 2201):
 *
 *   host pass   - the Default backend's control path: GCC __atomic_* builtins,
 *                 assert-loud stubs where the platform has no capability.
 *   device pass - the CCE/AICore path: volatile load/store on the unqualified
 *                 pointers the Default backend passes down, plus
 *                 __builtin_cce_atom_* as the intended atomic channel - which
 *                 today the back end cannot select (see CceAtomic).
 *
 * Why simtWidth is 1 and not 32 (three independent measurements, which is what
 * adaptation doc §5.1 asks for before a platform states its lane width):
 *   1. -cce-enable-puresimt rejects --npu-soc=Ascend910_9382;
 *   2. __simt_vf__ and __simd_vf__ are each reported "not compatible" with the
 *      cce_kernel attribute that __global__ expands to
 *      (__clang_cce_defines.h:31);
 *   3. laneid() is undeclared in this toolchain, and the one header that does
 *      define warpSize = 32 plus the lane primitives is
 *      __clang_cce_simt_dav3510_intrinsic.h - a different architecture
 *      (dav3510), and not even self-contained here (unknown type 'int4', and
 *      it collides with the 2201 dispatcher: "redefinition of '__ldg'").
 * The execution model therefore is one logical thread per AI core; the
 * intended cross-core channel is global-memory atomics, which this toolchain
 * cannot emit yet (see CceAtomic). The Coop types below
 * are shared by both passes for exactly that reason - they are not no-ops
 * standing in for hardware that exists.
 *
 * Used by CommTraits<DefaultBackend<AscendPlatform>> when USE_ASCEND_ADAPTOR
 * is defined (see platform_traits.h / comm_traits.h).
 ************************************************************************/

#ifndef FLAGCX_ASCEND_PLATFORM_TRAITS_H_
#define FLAGCX_ASCEND_PLATFORM_TRAITS_H_

#if defined(__CCE_IS_AICORE__)
// Verified includable on this soc (a probe calling pipe_barrier(PIPE_ALL) and
// __cce_pipe_barrier(PIPE_ALL) through these two headers returned rc=0). They
// also provide the integer types used below, which is why <cassert>/<cstdint>
// are host-side only in this header.
#include "__clang_cce_aicore_functions.h"
#include "cce_aicore_intrinsics.h"
#else
#include <cassert>
#include <cstdint>
#endif
#include <stddef.h>

struct AscendPlatform {};

#ifndef __CCE_IS_AICORE__
// Host pass: an assert is a declared, debug-visible failure.
#define FLAGCX_ASCEND_UNSUPPORTED(name) \
  do {                                  \
    assert(false && (name));            \
  } while (0)
#else
// Device pass: a trap. The doc (§13.3.2) permits declaring a capability
// absent; what it forbids is quietly doing nothing in a path the caller
// depends on. Unreachable-after-trap returns are marked to keep compilers
// from warning about them.
#define FLAGCX_ASCEND_UNSUPPORTED(name)    \
  do {                                     \
    (void)(name);                          \
    FLAGCX_DEVICE_TRAP();                  \
  } while (0)
#endif  // __CCE_IS_AICORE__

template <>
struct PlatformTraits<AscendPlatform> {
  struct CoopOps {};
  struct Clock {};

  // ==============================================================
  // Intrin
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = 1;
    static FLAGCX_HOST_DEVICE_INLINE constexpr flagcxLaneMask_t fullMask() { return 1ull; }

#if defined(__CCE_IS_AICORE__)
    // Single-lane model: these are definitions, not stubs. Lane 0 is the only
    // lane that exists, so a mask never has more than its low bit set.
    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() { return 0; }
    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t lanemaskLt() { return 0; }
    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t activemask() { return 1ull; }
    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(flagcxLaneMask_t mask) { return (int)(mask & 1ull); }
    // A convergence point of a one-lane group has nothing to converge on.
    static FLAGCX_DEVICE_INLINE_DECORATOR void syncwarp(flagcxLaneMask_t = fullMask()) {}
    static FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int cycles) {
      // No backoff/sleep hint instruction is exposed for this soc, so honour
      // the caller's intent with a bounded busy loop.
      for (volatile int i = 0; i < cycles; ++i) {
      }
    }
    // namedBarrierSync / the two fences are NOT defined yet: the 2201 spellings
    // for a cross-core barrier (set_flag/wait_flag event ids) and for GM cache
    // control (dcci) still need their argument lists measured. __builtin_cce_
    // mem_bar_* are dav3510-gated ("does not support the given target feature")
    // and calling hset_flag()/hwait_flag() with a wrong arity crashes bisheng
    // outright, so guessing is not an option here. They trap until then.
    static FLAGCX_DEVICE_INLINE_DECORATOR void namedBarrierSync(int, int) {
      FLAGCX_ASCEND_UNSUPPORTED("namedBarrierSync() on AscendPlatform");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceDevice() {
      FLAGCX_ASCEND_UNSUPPORTED("threadfenceDevice() on AscendPlatform");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceSystem() {
      FLAGCX_ASCEND_UNSUPPORTED("threadfenceSystem() on AscendPlatform");
    }
    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void coopCopyBytes(DstPtr dst, SrcPtr src, size_t bytes, int,
                                     int) {
      // One lane in the group: this thread walks the whole range. Byte-wide
      // because the address spaces of caller pointers are not known here.
      volatile uint8_t *d = reinterpret_cast<volatile uint8_t *>(dst);
      const uint8_t *s = reinterpret_cast<const uint8_t *>(src);
      for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
    }
#else
    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() {
      FLAGCX_ASCEND_UNSUPPORTED("lane() on AscendPlatform");
      return 0;
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t lanemaskLt() {
      FLAGCX_ASCEND_UNSUPPORTED("lanemaskLt() on AscendPlatform");
      return 0;
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t activemask() {
      FLAGCX_ASCEND_UNSUPPORTED("activemask() on AscendPlatform");
      return 1;
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void syncwarp(flagcxLaneMask_t = fullMask()) {
      FLAGCX_ASCEND_UNSUPPORTED("syncwarp() on AscendPlatform");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(flagcxLaneMask_t x) {
      (void)x;
      FLAGCX_ASCEND_UNSUPPORTED("popc() on AscendPlatform");
      return 0;
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void namedBarrierSync(int, int) {
      FLAGCX_ASCEND_UNSUPPORTED("namedBarrierSync() on AscendPlatform");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int) {
      FLAGCX_ASCEND_UNSUPPORTED("spinBackoff() on AscendPlatform");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceSystem() {
      FLAGCX_ASCEND_UNSUPPORTED("threadfenceSystem() on AscendPlatform");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceDevice() {
      FLAGCX_ASCEND_UNSUPPORTED("threadfenceDevice() on AscendPlatform");
    }
    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void coopCopyBytes(DstPtr, SrcPtr, size_t, int, int) {
      FLAGCX_ASCEND_UNSUPPORTED("coopCopyBytes() on AscendPlatform");
    }
#endif  // __CCE_IS_AICORE__
  };

#if defined(__CCE_IS_AICORE__)
  // ==============================================================
  // CceAtomic - the only place in FlagCX that spells a vendor atomic name.
  // ==============================================================
  //
  // Type coverage is expressed as an overload set rather than explicit
  // specializations (specializing a member function template inside its class
  // is not legal C++), with a trapping template as the catch-all: a type with
  // no CCE spelling fails loudly instead of being reinterpreted.
  struct CceAtomic {
    // ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV == 0, measured at
    // __clang_cce_simt_dav3510_intrinsic.h:45. The intrinsics accept the hint
    // as a plain uint32_t, so no bisheng header is needed for the value -
    // which matters, because the header that declares the enum cannot be
    // included on this soc.
    static constexpr uint32_t kHint = 0u;

    // Plain accesses, unqualified pointer in, volatile load/store out.
    // No address space is spelled here, deliberately. bisheng refuses every
    // explicit conversion between address spaces inside an aicore function -
    //   "casting 'const uint32_t *' to type 'volatile __gm__ uint32_t *' is
    //    not allowed in aicore function"
    // (gate run of df5a92c; the reverse direction is refused as well:
    //  "reinterpret_cast from '__gm__ uint32_t *' to 'uint32_t *' is not
    //   allowed"). The Default backend never hands __gm__ pointers down to
    // this layer anyway: its device structs keep pointer members unqualified,
    // because FLAGCX_DEVICE_GLOBAL_PTR expands to __global_ptr__ only on
    // Kunlunxin (device_utils.h:46-52), and an unqualified pointer does
    // dereference on this pass - DeviceAPI::Net's own constructor body reads
    // dc.fifoBuffers[_contextId] in the same run with no diagnostic. So only
    // cv-qualification is added here, with static_cast, which does not change
    // address space. du_platform_traits.h does the same thing for the HIP
    // atomics: plain T* straight into __hip_atomic_load.
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t load(const uint32_t *p) {
      return *static_cast<const volatile uint32_t *>(p);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t load(const int32_t *p) {
      return *static_cast<const volatile int32_t *>(p);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t load(const uint64_t *p) {
      return *static_cast<const volatile uint64_t *>(p);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t load(const int64_t *p) {
      return *static_cast<const volatile int64_t *>(p);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(uint32_t *p, uint32_t v) {
      *static_cast<volatile uint32_t *>(p) = v;
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(int32_t *p, int32_t v) {
      *static_cast<volatile int32_t *>(p) = v;
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(uint64_t *p, uint64_t v) {
      *static_cast<volatile uint64_t *>(p) = v;
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(int64_t *p, int64_t v) {
      *static_cast<volatile int64_t *>(p) = v;
    }

    // __builtin_cce_atom_{add,sub,or,and,exch,cas}_{u32,s32,u64,s64} - each
    // name resolved by bisheng's front end on 2201 (no "undeclared builtin"
    // diagnostic for any of them). The (uint32_t) hint is the third trailing
    // argument, and cas takes compare then value before it.
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t add(uint32_t *p, uint32_t v) {
      return __builtin_cce_atom_add_u32(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t add(int32_t *p, int32_t v) {
      return __builtin_cce_atom_add_s32(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t add(uint64_t *p, uint64_t v) {
      return __builtin_cce_atom_add_u64(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t add(int64_t *p, int64_t v) {
      return __builtin_cce_atom_add_s64(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t sub(uint32_t *p, uint32_t v) {
      return __builtin_cce_atom_sub_u32(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t sub(uint64_t *p, uint64_t v) {
      return __builtin_cce_atom_sub_u64(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t bitOr(uint32_t *p, uint32_t v) {
      return __builtin_cce_atom_or_u32(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t bitOr(uint64_t *p, uint64_t v) {
      return __builtin_cce_atom_or_u64(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t bitAnd(uint32_t *p, uint32_t v) {
      return __builtin_cce_atom_and_u32(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t bitAnd(uint64_t *p, uint64_t v) {
      return __builtin_cce_atom_and_u64(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t exch(uint32_t *p, uint32_t v) {
      return __builtin_cce_atom_exch_u32(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t exch(uint64_t *p, uint64_t v) {
      return __builtin_cce_atom_exch_u64(p, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t cas(uint32_t *p, uint32_t cmp, uint32_t v) {
      return __builtin_cce_atom_cas_u32(p, cmp, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t cas(int32_t *p, int32_t cmp, int32_t v) {
      return __builtin_cce_atom_cas_s32(p, cmp, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t cas(uint64_t *p, uint64_t cmp, uint64_t v) {
      return __builtin_cce_atom_cas_u64(p, cmp, v, kHint);
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t cas(int64_t *p, int64_t cmp, int64_t v) {
      return __builtin_cce_atom_cas_s64(p, cmp, v, kHint);
    }

    // Catch-alls. Any type outside the four integer widths reaches one of
    // these, which is a declared failure rather than a silent reinterpretation.
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T load(const T *p) {
      (void)p;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::load: no CCE spelling for type");
      return T();
    }
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(T *p, const T &) {
      (void)p;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::store: no CCE spelling for type");
    }
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T add(T *p, T v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::add: no CCE spelling for type");
      return T();
    }
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T sub(T *p, T v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::sub: no CCE spelling for type");
      return T();
    }
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T bitOr(T *p, T v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::bitOr: no CCE spelling for type");
      return T();
    }
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T bitAnd(T *p, T v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::bitAnd: no CCE spelling for type");
      return T();
    }
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T exch(T *p, T v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::exch: no CCE spelling for type");
      return T();
    }
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T cas(T *p, T cmp, T v) {
      (void)p;
      (void)cmp;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED("CceAtomic::cas: no CCE spelling for type");
      return T();
    }
  };
#endif  // __CCE_IS_AICORE__

  // ==============================================================
  // Atomic - the contract the Default backend's device API calls through.
  // ==============================================================
  struct Atomic {
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T load(T *ptr, flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      return CceAtomic::load(ptr);
#else
      return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
#endif
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      CceAtomic::store(ptr, val);
#else
      __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST);
#endif
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T fetchAdd(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      return CceAtomic::add(ptr, val);
#else
      return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
#endif
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T fetchSub(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      return CceAtomic::sub(ptr, val);
#else
      return __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST);
#endif
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T fetchOr(T *ptr, const T &val,
                            flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      return CceAtomic::bitOr(ptr, val);
#else
      return __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST);
#endif
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T fetchAnd(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      return CceAtomic::bitAnd(ptr, val);
#else
      return __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST);
#endif
    }
    // No CCE atomic intrinsic is selectable on this soc in the spelling the
    // intrinsics force, and the scope of that is measured, not assumed: the
    // memory operand reaches the back end with align 1 - the intrinsics
    // declare their address operand as void*, so the DAG knows no alignment -
    // and the pattern refuses it. Measured on __NPU_ARCH__ 2201 on the
    // read-modify-write ops too, not only on CAS/EXCH:
    //   "Cannot select: HiIPUISD::ATOM_SCALAR_ADD_u64<(store (s64) into
    //       %ir.0, align 1, addrspace 1)>"             (add, __gm__ operand)
    //   "Cannot select: HiIPUISD::ATOM_SCALAR_ADD_u32<...>"  (add, full link)
    //   "Cannot select: HiIPUISD::ATOM_SCALAR_CAS_u32/u64/EXCH_u64<...>"
    // Address space is not the differentiator: an __gm__ operand fails the
    // same way. With an *unqualified* operand the same builtin instead
    // crashes the selector in 'HiIPU Non VF DAG->DAG Pattern Instruction
    // Selection' (probes A2/B7), which is a compiler bug rather than a
    // spelling the traits could adopt.
    // The vendor wrappers cannot dodge this either: the header that declares
    // them (__clang_cce_simt_atomic.h) is not self-contained on this soc, and
    // the header carrying its cache-hint enum belongs to dav3510 and collides
    // with the 2201 dispatcher.
    //
    // CceAtomic::add/sub/or/and/exch/cas above stay in place so that settling
    // the alignment question remains a small edit here rather than a rewrite.
    // exchange()/compareExchange() below declare the gap loudly; whether
    // fetchAdd/fetchSub/fetchOr/fetchAnd must do the same is exactly what the
    // open alignment probes (assume_aligned, constant __gm__ address, hint
    // variants) are for. Emulating CAS from the other ops is not possible (no
    // op can express a conditional store), and emulating it with a
    // non-atomic test-and-set would be a correctness bug dressed as a
    // feature.
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T exchange(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      (void)ptr;
      (void)val;
      FLAGCX_ASCEND_UNSUPPORTED("Atomic::exchange() not selectable on 2201");
      return T();
#else
      return __atomic_exchange_n(ptr, val, __ATOMIC_SEQ_CST);
#endif
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR bool compareExchange(T *ptr, T &expected, const T &desired,
                                       flagcxDeviceMemoryOrder_t) {
      (void)Scope;
#if defined(__CCE_IS_AICORE__)
      (void)ptr;
      (void)expected;
      (void)desired;
      FLAGCX_ASCEND_UNSUPPORTED("Atomic::compareExchange() not selectable on "
                                "2201");
      return false;
#else
      return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
#endif
    }
  };

  // ==============================================================
  // Coop types - one logical thread per core, identical in both passes.
  // ==============================================================
  struct CoopBlock {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return 1; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {}
  };
  template <int N>
  struct CoopTile {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t laneMask() const {
      return Intrin::fullMask();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {}
  };
  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<1>;
  struct CoopTileSpan {
    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int, int, int) {}
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return 1; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {}
  };
  struct CoopLanes {
    FLAGCX_DEVICE_INLINE_DECORATOR
    CoopLanes(flagcxLaneMask_t mask = Intrin::fullMask()) { (void)mask; }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return 1; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {}
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t getLmask() const {
      return Intrin::fullMask();
    }
  };
  using CoopAny = PlatformCoop;
};

#endif // FLAGCX_ASCEND_PLATFORM_TRAITS_H_
