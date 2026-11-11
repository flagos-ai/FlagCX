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
 *   device pass - the CCE/AICore path. It claims no memory order at all: this
 *                 toolchain has no selectable read-modify-write, and no fence
 *                 whose cross-core effect has been verified on hardware, so
 *                 every load, store and RMW here is a declared-absent
 *                 capability that fails the build if a kernel reaches it (see
 *                 CceAtomic, and the review decision recorded on PR #705).
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
// Device pass: a call to a function that is declared and never defined. The
// doc (§13.3.2) permits declaring a capability absent; what it forbids is
// quietly doing nothing in a path the caller depends on. A trap is not
// available here - bisheng lowers __builtin_trap() to a call to abort and the
// CCE verifier refuses it ("cannot call to symbol: abort", probes T0/T1) - and
// because bisheng links the device image during -c (probes T2/T3), a call to a
// never-defined aicore function is refused at build time instead: louder than a
// runtime trap, and not paperable. This is deliberately separate from
// FLAGCX_DEVICE_TRAP(), which the shared headers use as a noreturn marker on
// paths no platform may reach at all; those sites are instantiated for every
// platform through a runtime provider switch, so giving them a never-defined
// symbol would make even a correct library unbuildable - the split is
// documented in device_utils.h.
//
// The argument is deliberately unused on this pass: inside an aicore function a
// string literal is a const __gm__ char[N], and wrapping it in anything named
// const char * is a type error (gate run of 9366a1f), so the operation name
// reaches only the host assert.
#define FLAGCX_ASCEND_UNSUPPORTED(name)       \
  do {                                        \
    flagcxAscendDeviceOpUnsupported();        \
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

    // Load and store are declared ABSENT on this pass, not implemented.
    //
    // What used to be here was a volatile access through the unqualified
    // pointer the Default backend hands down, which is not what the callers
    // asked for: default_comm_traits.h requests AcqRel on the FIFO slot
    // reservation (:574), Acquire on the consumer's reload of the consumed
    // counter (:580) and Release on completion-counter updates (:858). A
    // volatile qualifier constrains compiler folding only; on a machine whose
    // AI cores reach global memory through their own caches it orders nothing
    // for a peer core. Saying so was the review objection on PR #705, the
    // objection is right, and the maintainer's decision (2026-11-10) is to mark
    // the Ascend device load/store unsupported here and take the ordered
    // version in a follow-up once it has been verified.
    //
    // The channel for that follow-up is already measured - all probes of
    // 2026-11-10 on bisheng -c, RC=0:
    //   instr_impl/npu_arch_2201/cache_ctrl_impl.h: asc_dcci_entire_ub,
    //     asc_dcci_entire_out, asc_dcci_entire_all, asc_dcci_entire_atomic,
    //     asc_dcci_single(__gm__ void*) (per-address, and therefore only
    //     reachable from __gm__ code, since plain -> __gm__ is refused);
    //   instr_impl/npu_arch_2201/sync_impl.h: asc_sync_data_barrier(arg) ->
    //     dsb(arg), asc_sync_block_arrive(pipe, flag) ->
    //     ffts_cross_core_sync(pipe, GetfftsConfig(flag)),
    //     asc_sync_notify / asc_sync_wait(pipe, tpipe, event_t).
    // Both directions also compile in the shape this layer actually uses:
    //   plain-pointer store, then asc_dcci_entire_out()   (release shape)
    //   asc_dcci_entire_all(), then plain-pointer load    (acquire shape)
    // RC=0 through -c means the instruction was selected and the device image
    // linked - NOT that a peer core observes the ordering. That needs a
    // two-core producer/consumer run on hardware, which is why this PR does not
    // ship it, and why shipping a volatile access as an acquire would be worse
    // than shipping nothing.
    //
    // The absence is enforced the same way as the read-modify-write members
    // below: a call to a function that is declared and defined nowhere, which
    // bisheng refuses at BUILD time because it links the device image inside
    // -c. The consequence is the one the reviewer asked for - no
    // ordering-dependent path, in this header or in the Default backend, can be
    // instantiated on Ascend until the ordered version lands.
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T noOrderedLoad() {
      FLAGCX_ASCEND_UNSUPPORTED(
          "CceAtomic: device load has no hardware-verified acquire on 2201");
      return T();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t load(const uint32_t *p) {
      (void)p;
      return noOrderedLoad<uint32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t load(const int32_t *p) {
      (void)p;
      return noOrderedLoad<int32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t load(const uint64_t *p) {
      (void)p;
      return noOrderedLoad<uint64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t load(const int64_t *p) {
      (void)p;
      return noOrderedLoad<int64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(uint32_t *p, uint32_t v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED(
          "CceAtomic: device store has no hardware-verified release on 2201");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(int32_t *p, int32_t v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED(
          "CceAtomic: device store has no hardware-verified release on 2201");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(uint64_t *p, uint64_t v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED(
          "CceAtomic: device store has no hardware-verified release on 2201");
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(int64_t *p, int64_t v) {
      (void)p;
      (void)v;
      FLAGCX_ASCEND_UNSUPPORTED(
          "CceAtomic: device store has no hardware-verified release on 2201");
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
    // reinterpret_cast<__gm__ int32_t *>(addr) - so the blockage is purely the
    // cross-space step, and it applies to this whole layer: the unqualified
    // pointers that by-value kernel parameters carry (the shape
    // DeviceAPI::Net's own constructor body uses) can never reach the _G_
    // forms, and the load/store above are absent for the ordering reason
    // recorded there instead.
    //
    // So each member below names FLAGCX_ASCEND_UNSUPPORTED instead of a
    // builtin the compiler cannot emit. On this pass that macro expands to a
    // call to a function that is declared and never defined anywhere: a trap
    // is not available - the CCE back end refuses __builtin_trap() outright
    // (probes T0/T1) - and bisheng links the device image inside the -c step,
    // so the undefined symbol is reported there ("ld.lld: error: undefined
    // symbol: nf.cube", probes T2/T3). Reaching a read-modify-write therefore
    // fails the BUILD of any kernel that asks for one: louder than a runtime
    // trap, and certainly not a silent no-op. The members still parse in every
    // translation unit - CceAtomic is not a template, so these bodies are
    // type-checked even in a host helper that never calls them, which is how
    // the stage-2 gate keeps them honest. Restoring the capability is a small
    // edit here once a CANN release gates 2201 in: the spellings to reinstate
    // are the generic builtins listed at measurement 1 for plain pointers and
    // the _G_ family at measurement 2 for __gm__ ones.
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T noAtomic() {
      FLAGCX_ASCEND_UNSUPPORTED(
          "CceAtomic: no selectable device read-modify-write on 2201");
      return T();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t add(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t add(int32_t *p, int32_t v) {
      return noAtomic<int32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t add(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t add(int64_t *p, int64_t v) {
      return noAtomic<int64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t sub(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t sub(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t bitOr(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t bitOr(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t bitAnd(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t bitAnd(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t exch(uint32_t *p, uint32_t v) {
      return noAtomic<uint32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t exch(uint64_t *p, uint64_t v) {
      return noAtomic<uint64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t cas(uint32_t *p, uint32_t cmp, uint32_t v) {
      return noAtomic<uint32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int32_t cas(int32_t *p, int32_t cmp, int32_t v) {
      return noAtomic<int32_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t cas(uint64_t *p, uint64_t cmp, uint64_t v) {
      return noAtomic<uint64_t>();
    }
    static FLAGCX_DEVICE_INLINE_DECORATOR int64_t cas(int64_t *p, int64_t cmp, int64_t v) {
      return noAtomic<int64_t>();
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
    // One logical lane per core (simtWidth = 1, threadRank() is always 0),
    // so a tile of N>1 reports a size it cannot populate: a loop striped by
    // `i += size()` from rank 0 silently omits every other item even if
    // nobody calls sync(), and sync() on such a group would be a no-op
    // barrier. That combination is a bug, not a configuration, so
    // instantiating it fails the build on both passes. (Review finding,
    // 11-11; rejected at instantiation rather than only in sync() because
    // the rank/size pair itself is what lies.)
    static_assert(N == 1,
                  "Ascend 2201 exposes a single logical lane: CoopTile<N> "
                  "with N>1 has no real rank and no working sync");
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
    // Whether a span covers more than one tile is a runtime argument, so the
    // synchronization itself is the declared-absent capability: a device
    // build that reaches it fails on the never-defined symbol instead of
    // passing on a no-op barrier. Single-lane semantics (size 1, rank 0) are
    // kept honest; only the barrier is refused. (Review finding, 11-11.)
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      FLAGCX_ASCEND_UNSUPPORTED("CoopTileSpan::sync() on AscendPlatform");
    }
  };
  struct CoopLanes {
    FLAGCX_DEVICE_INLINE_DECORATOR
    CoopLanes(flagcxLaneMask_t mask = Intrin::fullMask()) { (void)mask; }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return 1; }
    // Same split as CoopTileSpan: a mask with more than its low bit set
    // selects lanes that do not exist on 2201, and the constructor cannot
    // reject it at compile time, so the barrier is the declared-absent
    // capability. (Review finding, 11-11.)
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      FLAGCX_ASCEND_UNSUPPORTED("CoopLanes::sync() on AscendPlatform");
    }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t getLmask() const {
      return Intrin::fullMask();
    }
  };
  using CoopAny = PlatformCoop;
};

#endif // FLAGCX_ASCEND_PLATFORM_TRAITS_H_
