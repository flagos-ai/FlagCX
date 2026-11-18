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
struct Dim { unsigned x = 0; };
static Dim threadIdx, blockIdx, blockDim{128}, gridDim{1};
struct alignas(16) int4 { int x, y, z, w; };
inline unsigned __ppu_read_laneid() { return threadIdx.x % 32; }
inline unsigned __ppu_read_lanemask_lt() { return (1u << __ppu_read_laneid()) - 1; }
inline unsigned __activemask() { return 0xffffffffu; }
inline unsigned __popc(unsigned x) { return __builtin_popcount(x); }
inline void __syncwarp(unsigned = 0xffffffffu) {}
inline void __barrier_sync_count(int, int) {}
inline void __syncthreads() {}
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
  T fetch_or(T v, memory_order o) { return __atomic_fetch_or(p, v, orderCode(o)); }
  T fetch_and(T v, memory_order o) { return __atomic_fetch_and(p, v, orderCode(o)); }
  T exchange(T v, memory_order o) { return __atomic_exchange_n(p, v, orderCode(o)); }
  bool compare_exchange_strong(T &e, T d, memory_order o) {
    memory_order f = o == memory_order_release ? memory_order_relaxed :
                    o == memory_order_acq_rel ? memory_order_acquire : o;
    return __atomic_compare_exchange_n(p, &e, d, false, orderCode(o), orderCode(f));
  }
  bool compare_exchange_weak(T &e, T d, memory_order s, memory_order f) {
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
static_assert(std::is_same<P::CoopAny, PlatformCoop>::value, "common CoopAny");
static_assert(sizeof(P::CoopAny) == 24 && alignof(P::CoopAny) == 8, "CoopAny ABI");
static_assert(sizeof(PlatformCompletionWord<PpuPlatform>::type) == 8, "completion width");
"""


@unittest.skipUnless(CXX, "C++ compiler required")
class PpuPlatformTraitsTest(unittest.TestCase):
    def compile_source(self, source, *, native=False, run=True, expect_success=True):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "cuda.h").write_text("")
            (tmp / "cuda_runtime.h").write_text("")
            if native:
                (tmp / "hggc_runtime.h").write_text(RUNTIME)
                (tmp / "hggc").mkdir()
                (tmp / "hggc/atomic").write_text(ATOMIC)
            cpp = tmp / "test.cpp"
            cpp.write_text(PRELUDE + source)
            cmd = [CXX, "-std=c++17", "-Wall", "-Wextra",
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
struct Group {
  int threadRank() const { return 3; }
  int size() const { return 8; }
  void sync() {}
};
int main() {
  P::CoopAny any(Group{});
  assert(any.threadRank() == 3 && any.size() == 8);
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

    def test_sdk_probe_and_default_backend_compile_with_native_api_shims(self):
        # Compile all device test bodies and the actual PPU default backend;
        # launch syntax/runtime execution still require the real PPU SDK.
        source = (ROOT / "test/kernel/ppu/device_api_traits.cu").read_text()
        source = source.split("#define SDK_CHECK", 1)[0]
        self.compile_source(source + "\nint main() {}\n", native=True, run=False)

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
