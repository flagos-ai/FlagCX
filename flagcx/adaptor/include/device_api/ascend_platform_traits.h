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

    // Read-modify-write is a declared-absent capability on this toolchain,
    // not an implementation. Three independent measurements on Ascend 910C /
    // __NPU_ARCH__ 2201 / CANN 9.1.0-beta.3 say the device atomic channel is
    // not reachable:
    //   1. the generic builtins parse - bisheng's front end resolves every
    //      __builtin_cce_atom_{add,sub,or,and,cas,exch}_{u32,s32,u64,s64}
    //      name, the (uint32_t) hint being the third trailing argument and
    //      cas taking compare then value before it - but nothing selects:
    //        "Cannot select: HiIPUISD::ATOM_SCALAR_ADD_u64<(store (s64) into
    //         %ir.0, align 1, addrspace 1)>"   (probes B1/RA, __gm__ operand)
    //      and with an unqualified operand the same builtin SEGFAULTS the
    //      selector in 'HiIPU Non VF DAG->DAG Pattern Instruction Selection'
    //      (probes A2/B7). The align-1 is inherent to the builtin's void*
    //      address operand, not a property of the target: a constant __gm__
    //      global (probe D0), __attribute__((assume_aligned)) (D2) and hint
    //      values 1 and 2 (D3/D4) all fail identically.
    //   2. the global-memory family does exist in this compiler -
    //      __builtin_cce_atom_{add,sub,cas,and,or,nand,xor}_G_u{32,64} /
    //      _s{32,64} and __builtin_cce_red_add_G_u{32,}s32/f32 appear in its
    //      builtin table with a __gm__ operand - but the target gate refuses
    //      it: "function type 'unsigned long (__gm__ void *, unsigned long,
    //      unsigned int) noexcept' of '__builtin_cce_atom_add_G_u64' does not
    //      support the given target feature" (probe F0). That is also why the
    //      reduce variants are u32/s32/f32 only, with no u64.
    //   3. every vendor surface above the builtins is arch-gated away from
    //      2201: simt_api/device_atomic_functions.h:21 and
    //      basic_api/kernel_operator_atomic_intf.h:30 both key on
    //      (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102), so atomicAdd here
    //      is "use of undeclared identifier" (probe F5) and
    //      __clang_cce_simt_atomic.h, which is not gated, routes only to the
    //      _G_ names of measurement 2.
    //
    // There is also no way to reach the _G_ family from the pointer types the
    // Default backend hands this layer. bisheng refuses conversion between
    // __gm__ and unqualified pointers inside an aicore function in every
    // spelling: implicit (probes B2/B3), reinterpret_cast and static_cast,
    // and the C-style form that Kunlunxin's bridge relies on
    // (kunlunxin_platform_traits.h:120-128 generates
    // ((__global_ptr__ uint32_t *)(p)), whose Ascend equivalent
    // ((__gm__ uint32_t *)(p)) is refused too - probes F1/F2/F3, both with
    // and without a space before the star). __builtin_astype, which is how
    // one opts out of address spaces in clang, is absent from this compiler
    // (probe B4). Within ONE address space casts are legal - the vendor's own
    // st_atomic (__clang_cce_aicore_functions.h:5852) does
    // reinterpret_cast<__gm__ int32_t *>(addr) - which is why load/store
    // above need no cast at all: they stay on the unqualified pointers the
    // by-value kernel parameters carry, the same shape
    // DeviceAPI::Net's own constructor body uses.
    //
    // So each member below traps loudly through FLAGCX_ASCEND_UNSUPPORTED
    // instead of naming a builtin that would break the compiler. That keeps
    // the whole Atomic contract instantiable (a device translation unit can
    // name every member, which is what the kernel smoke gate does) while
    // making the gap impossible to mistake for support. Restoring it is a
    // small edit here once a CANN release gates 2201 in: the spellings to
    // reinstate are the generic builtins listed at measurement 1 for plain
    // pointers and the _G_ family at measurement 2 for __gm__ ones.
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T noAtomic(const char *op) {
      (void)op;  // call sites name the op; the trap carries the reason
      FLAGCX_ASCEND_UNSUPPORTED(
          "CceAtomic: no selectable device read-modify-write on 2201");
      return T();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t add(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>("add");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t add(int32_t *p, int32_t v) {
      return noAtomic<int32_t>("add");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t add(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>("add");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t add(int64_t *p, int64_t v) {
      return noAtomic<int64_t>("add");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t sub(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>("sub");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t sub(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>("sub");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t bitOr(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>("bitOr");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t bitOr(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>("bitOr");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t bitAnd(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>("bitAnd");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t bitAnd(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>("bitAnd");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t exch(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>("exchange");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t exch(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>("exchange");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t cas(uint32_t *p, uint32_t cmp, uint32_t v) {
      return noAtomic<uint32_t>("compareExchange");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t cas(int32_t *p, int32_t cmp, int32_t v) {
      return noAtomic<int32_t>("compareExchange");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t cas(uint64_t *p, uint64_t cmp, uint64_t v) {
      return noAtomic<uint64_t>("compareExchange");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t cas(int64_t *p, int64_t cmp, int64_t v) {
      return noAtomic<int64_t>("compareExchange");
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

    // exchange() and compareExchange() share the measured reason documented at
    // the top of CceAtomic: no device-side read-modify-write is reachable on
    // 2201 in this toolchain. They are declared absent rather than emulated,
    // because add cannot express a conditional store and a non-atomic
    // test-and-set would be a correctness bug dressed as a feature.
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
