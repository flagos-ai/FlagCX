"""Compile PPU traits without NVIDIA headers; SDK shims check API wiring only.

GPU intrinsics and memory visibility are exercised by device_api_traits.cu on
PPU CI. These host shims cannot validate HGGC lowering or device execution.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CXX = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")

RUNTIME = r"""
#pragma once
#include <cstdint>
#define __host__
#define __device__
#define __global__
#define __shared__
#define __forceinline__ inline
#define __constant__
struct Dim { unsigned x; Dim(unsigned value = 0) : x(value) {} };
static Dim threadIdx, blockIdx, blockDim{128}, gridDim{1};
struct alignas(16) int4 { int x, y, z, w; };
inline unsigned __ppu_read_laneid() { return threadIdx.x % 32; }
inline unsigned __ppu_read_lanemask_lt() { return (1u << __ppu_read_laneid()) - 1; }
inline unsigned __activemask() { return 0xffffffffu; }
inline unsigned __popc(unsigned x) { return __builtin_popcount(x); }
static unsigned warpSyncCalls, blockSyncCalls, namedSyncCalls, lastWarpMask;
static int lastBarrierId, lastBarrierCount;
inline void __syncwarp(unsigned mask = 0xffffffffu) {
  ++warpSyncCalls; lastWarpMask = mask;
}
inline void __barrier_sync_count(int id, int count) {
  ++namedSyncCalls; lastBarrierId = id; lastBarrierCount = count;
}
inline void __syncthreads() { ++blockSyncCalls; }
inline void __threadfence_system() {}
inline void __threadfence() {}
inline void __threadfence_block() {}
inline uint64_t clock64() { static uint64_t ticks = 0; return ++ticks; }
inline unsigned __shfl_sync(unsigned, unsigned value, int) { return value; }
"""

ATOMIC = r"""
#pragma once
#include <atomic>
namespace hggc {
using std::memory_order;
using std::memory_order_relaxed;
using std::memory_order_acquire;
using std::memory_order_release;
using std::memory_order_acq_rel;
using std::memory_order_seq_cst;
enum thread_scope { thread_scope_system, thread_scope_device,
                    thread_scope_block, thread_scope_thread };
static thread_scope last_scope;
static int weak_failures = 0;
static uint64_t concurrent_value = 0;
inline int orderCode(memory_order order) {
  switch (order) {
    case memory_order_relaxed: return __ATOMIC_RELAXED;
    case memory_order_acquire: return __ATOMIC_ACQUIRE;
    case memory_order_release: return __ATOMIC_RELEASE;
    case memory_order_acq_rel: return __ATOMIC_ACQ_REL;
    default: return __ATOMIC_SEQ_CST;
  }
}
template <class T, thread_scope S> struct atomic_ref {
  T *p;
  explicit atomic_ref(T &value): p(&value) { last_scope = S; }
  T load(memory_order o) { return __atomic_load_n(p, orderCode(o)); }
  void store(T v, memory_order o) { __atomic_store_n(p, v, orderCode(o)); }
  T fetch_add(T v, memory_order o) { return __atomic_fetch_add(p, v, orderCode(o)); }
  T fetch_sub(T v, memory_order o) { return __atomic_fetch_sub(p, v, orderCode(o)); }
  T fetch_or(T v, memory_order o) {
    if (S == thread_scope_system) __builtin_trap();
    return __atomic_fetch_or(p, v, orderCode(o));
  }
  T fetch_and(T v, memory_order o) {
    if (S == thread_scope_system) __builtin_trap();
    return __atomic_fetch_and(p, v, orderCode(o));
  }
  T exchange(T v, memory_order o) { return __atomic_exchange_n(p, v, orderCode(o)); }
  bool compare_exchange_strong(T &e, T d, memory_order o) {
    memory_order f = o == memory_order_release ? memory_order_relaxed :
                    o == memory_order_acq_rel ? memory_order_acquire : o;
    return __atomic_compare_exchange_n(p, &e, d, false, orderCode(o), orderCode(f));
  }
  bool compare_exchange_weak(T &e, T d, memory_order s, memory_order f) {
    if (weak_failures > 0) {
      --weak_failures;
      __atomic_store_n(p, static_cast<T>(concurrent_value), __ATOMIC_RELAXED);
      e = __atomic_load_n(p, __ATOMIC_RELAXED);
      return false;
    }
    return __atomic_compare_exchange_n(p, &e, d, true, orderCode(s), orderCode(f));
  }
};
}
"""

PRELUDE = r"""
#include "device_api/platform_traits.h"
#include <cassert>
#include <type_traits>
#ifdef FLAGCX_NVIDIA_PLATFORM_TRAITS_H_
#error PPU must not include NVIDIA platform traits
#endif
#ifdef FLAGCX_NVIDIA_PTX_MEMORY_H_
#error PPU must not include NVIDIA PTX memory helpers
#endif
using P = PlatformTraits<PpuPlatform>;
static_assert(!std::is_same<P::CoopAny, PlatformCoop>::value, "PPU direct-dispatch CoopAny");
static_assert(std::is_standard_layout<P::CoopAny>::value &&
              std::is_trivially_copyable<P::CoopAny>::value, "CoopAny value semantics");
static_assert(sizeof(P::CoopAny) == 24 && alignof(P::CoopAny) == 8, "CoopAny ABI");
static_assert(sizeof(PlatformCompletionWord<PpuPlatform>::type) == 8, "completion width");
"""


@unittest.skipUnless(CXX, "C++ compiler required")
class PpuPlatformTraitsTest(unittest.TestCase):
    def compile_source(self, source, *, native=False, run=True, expect_success=True):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "hggc.h").write_text("")
            (tmp / "hggc_runtime.h").write_text("")
            if native:
                (tmp / "hggc_runtime.h").write_text(RUNTIME)
                (tmp / "hggc").mkdir()
                (tmp / "hggc/atomic").write_text(ATOMIC)
            cpp = tmp / "test.cpp"
            cpp.write_text(PRELUDE + source)
            cmd = [CXX, "-std=c++11", "-Wall", "-Wextra",
                   "-DUSE_PPU_ADAPTOR", f"-I{tmp}",
                   f"-I{ROOT / 'flagcx/adaptor/include'}",
                   f"-I{ROOT / 'flagcx/include'}",
                   f"-I{ROOT / 'flagcx/core/include'}",
                   str(cpp), "-o", str(tmp / "test")]
            if native:
                # Native HGGC marker without __CUDACC__: test detection of the
                # independent PPU branch, not NVIDIA-compatible preprocessing.
                cmd.insert(1, "-D__HGGCCC__")
            result = subprocess.run(cmd, text=True, capture_output=True)
            if not expect_success:
                self.assertNotEqual(result.returncode, 0, "unsupported contract compiled")
                return result.stderr
            self.assertEqual(result.returncode, 0, result.stderr)
            if run:
                executed = subprocess.run([str(tmp / "test")], text=True, capture_output=True)
                self.assertEqual(executed.returncode, 0, executed.stderr)
            return result.stderr

    def test_host_header_and_portable_operations(self):
        self.compile_source(r"""
int main() {
  P::CoopAny any(P::CoopThread{});
  assert(any.threadRank() == 0 && any.size() == 1);
  any.sync();
  P::CoopAny noop;
  assert(noop.threadRank() == 0 && noop.size() == 1);
  noop.sync();
  assert(P::Intrin::fullMask() == 0xffffffffull);
  assert(P::Intrin::byteSwap(uint32_t{0x12345678}) == 0x78563412u);
  assert(P::Intrin::funnelShiftRight(uint32_t{0x11223344}, uint32_t{0x55667788}, 8) == 0x88112233u);
  assert((P::Intrin::funnelShiftRight<uint64_t, flagcxDeviceFunnelShiftClamp>(1, 2, 64) == 2));
  assert(P::Clock::elapsedNs(100, 1600, 1500000) == 1000);
  assert(P::Clock::elapsedTicks(UINT64_MAX - 2, 4) == 7);
  assert(P::Clock::elapsedNs(0, 2000000, 1000000) == 2000000);
  uint32_t a = 42, b = 0;
  P::Intrin::copy<uint32_t, Scalar, Scalar, flagcxDeviceMemoryLocal, flagcxDeviceMemoryLocal>(&b, &a);
  assert(b == a);
}
""")

    def test_hggc_marker_and_atomic_contract(self):
        self.compile_source(r"""
#ifndef FLAGCX_PPU_DEVICE_COMPILE
#error Native HGGC must select PPU device definitions
#endif
template <typename T, flagcxDeviceScope_t Scope>
void checkAtomics() {
  T word = 0;
  P::Atomic::store<T, Scope>(&word, T{7}, flagcxDeviceMemoryOrderRelease);
  assert((P::Atomic::load<T, Scope>(&word, flagcxDeviceMemoryOrderAcquire) == 7));
  assert(hggc::last_scope == P::Atomic::toNativeScope(Scope));
  assert((P::Atomic::fetchAdd<T, Scope>(&word, T{2}, flagcxDeviceMemoryOrderRelaxed) == 7));
  assert((P::Atomic::fetchSub<T, Scope>(&word, T{1}, flagcxDeviceMemoryOrderSeqCst) == 9));
  assert((P::Atomic::fetchOr<T, Scope>(&word, T{1}, flagcxDeviceMemoryOrderSeqCst) == 8));
  assert((P::Atomic::fetchAnd<T, Scope>(&word, T{7}, flagcxDeviceMemoryOrderSeqCst) == 9));
  assert((P::Atomic::exchange<T, Scope>(&word, T{3}, flagcxDeviceMemoryOrderAcqRel) == 1));
  T expected = 0;
  assert((!P::Atomic::compareExchange<T, Scope>(&word, expected, T{5}, flagcxDeviceMemoryOrderAcqRel)));
  assert(expected == 3);
  assert((P::Atomic::compareExchange<T, Scope>(&word, expected, T{5}, flagcxDeviceMemoryOrderAcqRel)));
  assert((P::Atomic::fetchMax<T, Scope>(&word, T{11}, flagcxDeviceMemoryOrderAcqRel) == 5));
  assert((P::Atomic::fetchMax<T, Scope>(&word, T{2}, flagcxDeviceMemoryOrderRelaxed) == 11));
  assert(word == 11);
}
int main() {
  checkAtomics<uint32_t, flagcxDeviceScopeThread>();
  checkAtomics<uint32_t, flagcxDeviceScopeBlock>();
  checkAtomics<uint32_t, flagcxDeviceScopeDevice>();
  checkAtomics<uint32_t, flagcxDeviceScopeSystem>();
  checkAtomics<uint64_t, flagcxDeviceScopeThread>();
  checkAtomics<uint64_t, flagcxDeviceScopeBlock>();
  checkAtomics<uint64_t, flagcxDeviceScopeDevice>();
  checkAtomics<uint64_t, flagcxDeviceScopeSystem>();
  assert(P::Atomic::toNativeScope(flagcxDeviceScopeThread) == hggc::thread_scope_thread);
  assert(P::Atomic::toNativeScope(flagcxDeviceScopeBlock) == hggc::thread_scope_block);
  assert(P::Atomic::toNativeScope(flagcxDeviceScopeDevice) == hggc::thread_scope_device);
  assert(P::Atomic::toNativeScope(flagcxDeviceScopeSystem) == hggc::thread_scope_system);
  assert(P::Atomic::toNativeOrder(flagcxDeviceMemoryOrderAcquire) == hggc::memory_order_acquire);
  assert(P::Atomic::toNativeOrder(flagcxDeviceMemoryOrderRelease) == hggc::memory_order_release);
  assert(P::Atomic::toNativeOrder(flagcxDeviceMemoryOrderAcqRel) == hggc::memory_order_acq_rel);
  assert(P::Atomic::toNativeOrder(flagcxDeviceMemoryOrderRelaxed) == hggc::memory_order_relaxed);
  assert(P::Atomic::toNativeOrder(flagcxDeviceMemoryOrderSeqCst) == hggc::memory_order_seq_cst);
  for (unsigned lane = 0; lane < 32; ++lane) {
    threadIdx.x = lane;
    assert(P::Intrin::lane() == int(lane));
    assert(P::Intrin::lanemaskLt() == ((uint64_t{1} << lane) - 1));
    P::CoopTile<8> tile;
    assert(tile.threadRank() == int(lane % 8));
    assert(tile.laneMask() == (uint64_t{0xff} << (lane & ~7)));
  }
  alignas(16) unsigned char src[129], dst[129] = {};
  for (int i = 0; i < 129; ++i) src[i] = static_cast<unsigned char>(i + 1);
  for (int rank = 0; rank < 32; ++rank)
    P::Intrin::copyBytesStridedImpl(dst + 1, src + 1, 127, rank, 32);
  for (int i = 1; i < 128; ++i) assert(src[i] == dst[i]);
  assert(dst[0] == 0 && dst[128] == 0);
}
""", native=True)

    def test_coop_any_tiles_and_block_preserve_calling_thread_semantics(self):
        self.compile_source(r"""
void resetSync() { warpSyncCalls = blockSyncCalls = namedSyncCalls = 0; }
template <int N> void checkTile() {
  threadIdx.x = 0;
  P::CoopTile<N> group;
  P::CoopAny original(group), copied(original), assigned;
  assigned = original;
  for (unsigned tid = 0; tid < 128; ++tid) {
    threadIdx.x = tid;
    assert(copied.threadRank() == int(tid % N) && copied.size() == N);
    assert(assigned.threadRank() == group.threadRank() && assigned.size() == group.size());
    resetSync();
    copied.sync();
    assert(blockSyncCalls == 0 && namedSyncCalls == 0);
    assert(warpSyncCalls == (N > 1 ? 1u : 0u));
    if (N > 1) {
      unsigned base = (tid % 32) / N * N;
      assert(lastWarpMask == (((uint64_t{1} << N) - 1) << base));
    }
  }
}
int main() {
  checkTile<1>(); checkTile<2>(); checkTile<4>();
  checkTile<8>(); checkTile<16>(); checkTile<32>();
  threadIdx.x = 0;
  P::CoopAny original(P::CoopBlock{}), copied(original), assigned;
  assigned = original;
  for (unsigned tid = 0; tid < 128; ++tid) {
    threadIdx.x = tid;
    assert(copied.threadRank() == int(tid) && assigned.threadRank() == int(tid));
    assert(copied.size() == 128 && assigned.size() == 128);
    resetSync(); copied.sync();
    assert(blockSyncCalls == 1 && warpSyncCalls == 0 && namedSyncCalls == 0);
  }
  blockDim.x = 256; threadIdx.x = 145;
  assert(copied.threadRank() == 145 && copied.size() == 256);
  P::CoopAny noop;
  assigned = noop;
  resetSync(); assigned.sync();
  assert(assigned.threadRank() == 0 && assigned.size() == 1);
  assert(blockSyncCalls == 0 && warpSyncCalls == 0 && namedSyncCalls == 0);
}
""", native=True)

    def test_coop_any_lane_masks_and_named_barrier_parameters(self):
        self.compile_source(r"""
void resetSync() { warpSyncCalls = blockSyncCalls = namedSyncCalls = 0; }
int main() {
  const uint64_t masks[] = {0, 1, 0xffff, 0x55555555, 0x80000001, 0xffffffff};
  for (uint64_t mask : masks) {
    threadIdx.x = 0;
    P::CoopLanes group(mask);
    P::CoopAny original(group), copied(original), assigned;
    assigned = original;
    for (unsigned tid = 0; tid < 128; ++tid) {
      threadIdx.x = tid;
      int rank = __builtin_popcountll(mask & ((uint64_t{1} << (tid % 32)) - 1));
      assert(copied.threadRank() == rank && assigned.threadRank() == group.threadRank());
      assert(copied.size() == __builtin_popcountll(mask));
      resetSync(); assigned.sync();
      assert(warpSyncCalls == 1 && lastWarpMask == mask);
      assert(blockSyncCalls == 0 && namedSyncCalls == 0);
    }
  }
  for (int first = 0; first < 3; ++first) {
    for (int tiles = 1; tiles <= 4 - first; ++tiles) {
      for (int id = 0; id < 3; ++id) {
        threadIdx.x = 0;
        P::CoopTileSpan group(first, tiles, id);
        P::CoopAny original(group), copied(original), assigned;
        assigned = original;
        for (int tid = first * 32; tid < (first + tiles) * 32; ++tid) {
          threadIdx.x = tid;
          assert(copied.threadRank() == tid - first * 32 && copied.size() == tiles * 32);
          assert(assigned.threadRank() == group.threadRank() && assigned.size() == group.size());
          resetSync(); assigned.sync();
          assert(namedSyncCalls == 1 && lastBarrierId == 1 + id && lastBarrierCount == tiles * 32);
          assert(blockSyncCalls == 0 && warpSyncCalls == 0);
        }
      }
    }
  }
}
""", native=True)

    def test_sdk_probe_and_default_backend_compile_with_native_api_shims(self):
        # Compile all device test bodies and the actual PPU default backend;
        # launch syntax/runtime execution still require the real PPU SDK.
        source = (ROOT / "test/kernel/ppu/device_api_traits.cu").read_text()
        source = source.split("#define SDK_CHECK", 1)[0]
        self.compile_source(source + "\nint main() {}\n", native=True, run=False)

    def test_system_bitwise_atomics_retry_after_competing_update(self):
        self.compile_source(r"""
template <typename T> void checkRetry() {
  T word = 1;
  hggc::weak_failures = 1;
  hggc::concurrent_value = 16;
  assert(P::Atomic::fetchOr(&word, T{8}, flagcxDeviceMemoryOrderSeqCst) == 16);
  assert(word == 24);
  word = 15;
  hggc::weak_failures = 1;
  hggc::concurrent_value = 7;
  assert(P::Atomic::fetchAnd(&word, T{6}, flagcxDeviceMemoryOrderAcqRel) == 7);
  assert(word == 6);
  assert(P::Atomic::fetchOr(&word, T{0}, flagcxDeviceMemoryOrderRelease) == 6);
  assert(P::Atomic::fetchAnd(&word, T{6}, flagcxDeviceMemoryOrderAcquire) == 6);
}
int main() { checkRetry<uint32_t>(); checkRetry<uint64_t>(); }
""", native=True)

    def test_unsupported_contracts_fail_to_compile(self):
        cases = [
            ("int main() { return P::Clock::elapsedNs(0, 10); }", "deleted"),
            ("int main() { unsigned x = 0; P::Intrin::load<unsigned, Scalar, flagcxDeviceMemoryGlobal, flagcxDeviceMemoryOrderWeak, flagcxDeviceScopeSystem, flagcxDeviceMemoryAccessKindMmio>(&x); }", "MMIO"),
            ("int main() { P::CoopOps::AsyncCopyBatch b; P::CoopOps::copyAsync(P::CoopThread{}, b, (int*)0, (int*)0, 1); }", "not supported"),
        ]
        for source, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                error = self.compile_source(source, run=False, expect_success=False)
                self.assertIn(diagnostic, error)

    def test_traits_probe_is_built_by_ppu_ci(self):
        result = subprocess.run(
            ["make", "-n", "-B", "-C", str(ROOT / "test/kernel/ppu"),
             "device_traits", "USE_PPU=1", "COMPILE_KERNEL=1"],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("device_api_traits.cu", result.stdout)
        self.assertIn("-DUSE_PPU_ADAPTOR", result.stdout)
        self.assertIn("/bin/test_device_api_traits", result.stdout)
        script = (ROOT / ".github/scripts/set_env/ppu.sh").read_text()
        self.assertIn('"$PROJECT_ROOT/test/kernel/ppu"', script)
        self.assertIn('device_traits "${args[@]}"', script)


if __name__ == "__main__":
    unittest.main()
