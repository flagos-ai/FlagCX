#ifndef FLAGCX_NVIDIA_PTX_MEMORY_H_
#define FLAGCX_NVIDIA_PTX_MEMORY_H_

#include "device_utils.h"
#include "flagcx_device_enums.h"

#include <cassert>
#include <cstdint>
#include <cuda/atomic>
#include <type_traits>

// All selection is C++11 template dispatch; assembly mnemonics are
// compile-time literals.
namespace flagcxNvidiaMemory {

template <typename Shape>
struct ShapeLanes {
  enum { value = 0 };
};
template <>
struct ShapeLanes<Scalar> {
  enum { value = 1 };
};
template <>
struct ShapeLanes<Vector2> {
  enum { value = 2 };
};
template <>
struct ShapeLanes<Vector4> {
  enum { value = 4 };
};
template <>
struct ShapeLanes<Vector8> {
  enum { value = 8 };
};

template <int Bytes>
struct Register;
template <>
struct Register<1> {
  typedef uint32_t type;
};
template <>
struct Register<2> {
  typedef uint16_t type;
};
template <>
struct Register<4> {
  typedef uint32_t type;
};
template <>
struct Register<8> {
  typedef uint64_t type;
};
template <>
struct Register<16> {
  typedef uint32_t type;
};

template <typename T, int Lanes, int Bytes>
static FLAGCX_DEVICE_INLINE_DECORATOR T
unpack(const typename Register<Bytes>::type *regs) {
  T result;
  unsigned char *out = reinterpret_cast<unsigned char *>(&result);
  for (int lane = 0; lane < Lanes; ++lane)
    for (int byte = 0; byte < Bytes; ++byte)
      out[lane * Bytes + byte] =
          static_cast<unsigned char>(regs[lane] >> (byte * 8));
  return result;
}

template <typename T, int Lanes, int Bytes>
static FLAGCX_DEVICE_INLINE_DECORATOR void
pack(T value, typename Register<Bytes>::type *regs) {
  const unsigned char *in = reinterpret_cast<const unsigned char *>(&value);
  for (int lane = 0; lane < Lanes; ++lane) {
    regs[lane] = 0;
    for (int byte = 0; byte < Bytes; ++byte)
      regs[lane] |= static_cast<typename Register<Bytes>::type>(
          static_cast<uint64_t>(in[lane * Bytes + byte]) << (byte * 8));
  }
}

template <typename T>
static FLAGCX_DEVICE_INLINE_DECORATOR T unpack128(const uint32_t *words) {
  T result;
  unsigned char *out = reinterpret_cast<unsigned char *>(&result);
  for (int i = 0; i < 16; ++i)
    out[i] = static_cast<unsigned char>(words[i / 4] >> ((i % 4) * 8));
  return result;
}

template <typename T>
static FLAGCX_DEVICE_INLINE_DECORATOR void pack128(T value, uint32_t *words) {
  const unsigned char *in = reinterpret_cast<const unsigned char *>(&value);
  for (int i = 0; i < 4; ++i) {
    words[i] = 0;
    for (int byte = 0; byte < 4; ++byte)
      words[i] |= static_cast<uint32_t>(in[i * 4 + byte]) << (byte * 8);
  }
}

template <int Mode, int Lanes, int Bytes>
struct PtxLoad {
  template <typename Address, typename Reg>
  static FLAGCX_DEVICE_INLINE_DECORATOR void run(Address, Reg *) {
    static_assert(Mode < 0, "PTX load format requires a newer target/toolkit");
  }
};
template <int Mode, int Lanes, int Bytes>
struct PtxStore {
  template <typename Address, typename Reg>
  static FLAGCX_DEVICE_INLINE_DECORATOR void run(Address, const Reg *) {
    static_assert(Mode < 0, "PTX store format requires a newer target/toolkit");
  }
};

#if defined(__CUDA_ARCH__)
// Mode numbers are private to this header. 0/1/2 are weak global/shared/local;
// 3..14 are relaxed/acquire or relaxed/release with sys/gpu/cta scope.
#define FCX_LD_SC(M, P, A, AC, W, B, R, RC)                                    \
  template <>                                                                  \
  struct PtxLoad<M, 1, W> {                                                    \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, R *r) {                \
      asm volatile(P ".b" #B " %0, [%1];" : RC(r[0]) : AC(a) : "memory");      \
    }                                                                          \
  };
#define FCX_ST_SC(M, P, A, AC, W, B, R, RC)                                    \
  template <>                                                                  \
  struct PtxStore<M, 1, W> {                                                   \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, const R *r) {          \
      asm volatile(P ".b" #B " [%0], %1;" : : AC(a), RC(r[0]) : "memory");     \
    }                                                                          \
  };
#define FCX_LD_V2(M, P, A, AC, W, B, R, RC)                                    \
  template <>                                                                  \
  struct PtxLoad<M, 2, W> {                                                    \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, R *r) {                \
      asm volatile(P ".v2.b" #B " {%0,%1}, [%2];"                              \
                   : RC(r[0]), RC(r[1])                                        \
                   : AC(a)                                                     \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_ST_V2(M, P, A, AC, W, B, R, RC)                                    \
  template <>                                                                  \
  struct PtxStore<M, 2, W> {                                                   \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, const R *r) {          \
      asm volatile(P ".v2.b" #B " [%0], {%1,%2};"                              \
                   :                                                           \
                   : AC(a), RC(r[0]), RC(r[1])                                 \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_LD_V4(M, P, A, AC, W, B, R, RC)                                    \
  template <>                                                                  \
  struct PtxLoad<M, 4, W> {                                                    \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, R *r) {                \
      asm volatile(P ".v4.b" #B " {%0,%1,%2,%3}, [%4];"                        \
                   : RC(r[0]), RC(r[1]), RC(r[2]), RC(r[3])                    \
                   : AC(a)                                                     \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_ST_V4(M, P, A, AC, W, B, R, RC)                                    \
  template <>                                                                  \
  struct PtxStore<M, 4, W> {                                                   \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, const R *r) {          \
      asm volatile(P ".v4.b" #B " [%0], {%1,%2,%3,%4};"                        \
                   :                                                           \
                   : AC(a), RC(r[0]), RC(r[1]), RC(r[2]), RC(r[3])             \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_LD_V8_B32(M, P, A, AC)                                             \
  template <>                                                                  \
  struct PtxLoad<M, 8, 4> {                                                    \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, uint32_t *r) {         \
      asm volatile(P ".v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"                \
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),           \
                     "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])            \
                   : AC(a)                                                     \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_ST_V8_B32(M, P, A, AC)                                             \
  template <>                                                                  \
  struct PtxStore<M, 8, 4> {                                                   \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, const uint32_t *r) {   \
      asm volatile(P ".v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"                \
                   :                                                           \
                   : AC(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]),        \
                     "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])                \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_LD_B128(M, P, A, AC)                                               \
  template <>                                                                  \
  struct PtxLoad<M, 1, 16> {                                                   \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, uint32_t *r) {         \
      asm volatile("{ .reg .b128 q; " P ".b128 q, [%4]; "                      \
                   "mov.b128 {%0,%1,%2,%3}, q; }"                              \
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])            \
                   : AC(a)                                                     \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_ST_B128(M, P, A, AC)                                               \
  template <>                                                                  \
  struct PtxStore<M, 1, 16> {                                                  \
    static FLAGCX_DEVICE_INLINE_DECORATOR void run(A a, const uint32_t *r) {   \
      asm volatile("{ .reg .b128 q; mov.b128 q, {%1,%2,%3,%4}; " P             \
                   ".b128 [%0], q; }"                                          \
                   :                                                           \
                   : AC(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])         \
                   : "memory");                                                \
    }                                                                          \
  };
#define FCX_LD_BASE(M, P, A, AC)                                               \
  FCX_LD_SC(M, P, A, AC, 1, 8, uint32_t, "=r")                                 \
  FCX_LD_SC(M, P, A, AC, 2, 16, uint16_t, "=h")                                \
  FCX_LD_SC(M, P, A, AC, 4, 32, uint32_t, "=r")                                \
  FCX_LD_SC(M, P, A, AC, 8, 64, uint64_t, "=l")                                \
  FCX_LD_V2(M, P, A, AC, 1, 8, uint32_t, "=r")                                 \
  FCX_LD_V2(M, P, A, AC, 2, 16, uint16_t, "=h")                                \
  FCX_LD_V2(M, P, A, AC, 4, 32, uint32_t, "=r")                                \
  FCX_LD_V2(M, P, A, AC, 8, 64, uint64_t, "=l")                                \
  FCX_LD_V4(M, P, A, AC, 1, 8, uint32_t, "=r")                                 \
  FCX_LD_V4(M, P, A, AC, 2, 16, uint16_t, "=h")                                \
  FCX_LD_V4(M, P, A, AC, 4, 32, uint32_t, "=r")
#define FCX_ST_BASE(M, P, A, AC)                                               \
  FCX_ST_SC(M, P, A, AC, 1, 8, uint32_t, "r")                                  \
  FCX_ST_SC(M, P, A, AC, 2, 16, uint16_t, "h")                                 \
  FCX_ST_SC(M, P, A, AC, 4, 32, uint32_t, "r")                                 \
  FCX_ST_SC(M, P, A, AC, 8, 64, uint64_t, "l")                                 \
  FCX_ST_V2(M, P, A, AC, 1, 8, uint32_t, "r")                                  \
  FCX_ST_V2(M, P, A, AC, 2, 16, uint16_t, "h")                                 \
  FCX_ST_V2(M, P, A, AC, 4, 32, uint32_t, "r")                                 \
  FCX_ST_V2(M, P, A, AC, 8, 64, uint64_t, "l")                                 \
  FCX_ST_V4(M, P, A, AC, 1, 8, uint32_t, "r")                                  \
  FCX_ST_V4(M, P, A, AC, 2, 16, uint16_t, "h")                                 \
  FCX_ST_V4(M, P, A, AC, 4, 32, uint32_t, "r")

FCX_LD_BASE(0, "ld.global", uint64_t, "l")
FCX_LD_BASE(1, "ld.shared", uint32_t, "r")
FCX_LD_BASE(2, "ld.local", uint32_t, "r")
FCX_ST_BASE(0, "st.global", uint64_t, "l")
FCX_ST_BASE(1, "st.shared", uint32_t, "r")
FCX_ST_BASE(2, "st.local", uint32_t, "r")

#if __CUDA_ARCH__ >= 700
FCX_LD_BASE(3, "ld.global.relaxed.sys", uint64_t, "l")
FCX_LD_BASE(4, "ld.global.relaxed.gpu", uint64_t, "l")
FCX_LD_BASE(5, "ld.global.relaxed.cta", uint64_t, "l")
FCX_LD_BASE(6, "ld.shared.relaxed.sys", uint32_t, "r")
FCX_LD_BASE(7, "ld.shared.relaxed.gpu", uint32_t, "r")
FCX_LD_BASE(8, "ld.shared.relaxed.cta", uint32_t, "r")
FCX_LD_BASE(9, "ld.global.acquire.sys", uint64_t, "l")
FCX_LD_BASE(10, "ld.global.acquire.gpu", uint64_t, "l")
FCX_LD_BASE(11, "ld.global.acquire.cta", uint64_t, "l")
FCX_LD_BASE(12, "ld.shared.acquire.sys", uint32_t, "r")
FCX_LD_BASE(13, "ld.shared.acquire.gpu", uint32_t, "r")
FCX_LD_BASE(14, "ld.shared.acquire.cta", uint32_t, "r")
FCX_ST_BASE(3, "st.global.relaxed.sys", uint64_t, "l")
FCX_ST_BASE(4, "st.global.relaxed.gpu", uint64_t, "l")
FCX_ST_BASE(5, "st.global.relaxed.cta", uint64_t, "l")
FCX_ST_BASE(6, "st.shared.relaxed.sys", uint32_t, "r")
FCX_ST_BASE(7, "st.shared.relaxed.gpu", uint32_t, "r")
FCX_ST_BASE(8, "st.shared.relaxed.cta", uint32_t, "r")
FCX_ST_BASE(9, "st.global.release.sys", uint64_t, "l")
FCX_ST_BASE(10, "st.global.release.gpu", uint64_t, "l")
FCX_ST_BASE(11, "st.global.release.cta", uint64_t, "l")
FCX_ST_BASE(12, "st.shared.release.sys", uint32_t, "r")
FCX_ST_BASE(13, "st.shared.release.gpu", uint32_t, "r")
FCX_ST_BASE(14, "st.shared.release.cta", uint32_t, "r")
#endif

#if __CUDA_ARCH__ >= 320
FCX_LD_BASE(15, "ld.global.nc", uint64_t, "l")
#endif

// L1 eviction hints are advisory and do not change the memory order.
#if __CUDA_ARCH__ >= 700 &&                                                    \
    (__CUDACC_VER_MAJOR__ > 11 ||                                              \
     (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4))
FCX_LD_BASE(19, "ld.global.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(20, "ld.global.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(21, "ld.global.L1::evict_last", uint64_t, "l")
FCX_LD_BASE(22, "ld.global.nc.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(23, "ld.global.nc.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(24, "ld.global.nc.L1::evict_last", uint64_t, "l")
FCX_ST_BASE(18, "st.global.L1::no_allocate", uint64_t, "l")
FCX_ST_BASE(19, "st.global.L1::evict_first", uint64_t, "l")
FCX_ST_BASE(20, "st.global.L1::evict_last", uint64_t, "l")
FCX_LD_BASE(25, "ld.global.relaxed.sys.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(26, "ld.global.relaxed.gpu.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(27, "ld.global.relaxed.cta.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(28, "ld.global.relaxed.sys.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(29, "ld.global.relaxed.gpu.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(30, "ld.global.relaxed.cta.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(31, "ld.global.relaxed.sys.L1::evict_last", uint64_t, "l")
FCX_LD_BASE(32, "ld.global.relaxed.gpu.L1::evict_last", uint64_t, "l")
FCX_LD_BASE(33, "ld.global.relaxed.cta.L1::evict_last", uint64_t, "l")
FCX_LD_BASE(34, "ld.global.acquire.sys.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(35, "ld.global.acquire.gpu.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(36, "ld.global.acquire.cta.L1::no_allocate", uint64_t, "l")
FCX_LD_BASE(37, "ld.global.acquire.sys.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(38, "ld.global.acquire.gpu.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(39, "ld.global.acquire.cta.L1::evict_first", uint64_t, "l")
FCX_LD_BASE(40, "ld.global.acquire.sys.L1::evict_last", uint64_t, "l")
FCX_LD_BASE(41, "ld.global.acquire.gpu.L1::evict_last", uint64_t, "l")
FCX_LD_BASE(42, "ld.global.acquire.cta.L1::evict_last", uint64_t, "l")
FCX_ST_BASE(21, "st.global.relaxed.sys.L1::no_allocate", uint64_t, "l")
FCX_ST_BASE(22, "st.global.relaxed.gpu.L1::no_allocate", uint64_t, "l")
FCX_ST_BASE(23, "st.global.relaxed.cta.L1::no_allocate", uint64_t, "l")
FCX_ST_BASE(24, "st.global.relaxed.sys.L1::evict_first", uint64_t, "l")
FCX_ST_BASE(25, "st.global.relaxed.gpu.L1::evict_first", uint64_t, "l")
FCX_ST_BASE(26, "st.global.relaxed.cta.L1::evict_first", uint64_t, "l")
FCX_ST_BASE(27, "st.global.relaxed.sys.L1::evict_last", uint64_t, "l")
FCX_ST_BASE(28, "st.global.relaxed.gpu.L1::evict_last", uint64_t, "l")
FCX_ST_BASE(29, "st.global.relaxed.cta.L1::evict_last", uint64_t, "l")
FCX_ST_BASE(30, "st.global.release.sys.L1::no_allocate", uint64_t, "l")
FCX_ST_BASE(31, "st.global.release.gpu.L1::no_allocate", uint64_t, "l")
FCX_ST_BASE(32, "st.global.release.cta.L1::no_allocate", uint64_t, "l")
FCX_ST_BASE(33, "st.global.release.sys.L1::evict_first", uint64_t, "l")
FCX_ST_BASE(34, "st.global.release.gpu.L1::evict_first", uint64_t, "l")
FCX_ST_BASE(35, "st.global.release.cta.L1::evict_first", uint64_t, "l")
FCX_ST_BASE(36, "st.global.release.sys.L1::evict_last", uint64_t, "l")
FCX_ST_BASE(37, "st.global.release.gpu.L1::evict_last", uint64_t, "l")
FCX_ST_BASE(38, "st.global.release.cta.L1::evict_last", uint64_t, "l")
#endif

// PTX 8.3 adds b128; system-scoped b128 needs PTX 8.4.
#if __CUDA_ARCH__ >= 700 &&                                                    \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
FCX_LD_B128(0, "ld.global", uint64_t, "l")
FCX_LD_B128(1, "ld.shared", uint32_t, "r")
FCX_LD_B128(2, "ld.local", uint32_t, "r")
FCX_ST_B128(0, "st.global", uint64_t, "l")
FCX_ST_B128(1, "st.shared", uint32_t, "r")
FCX_ST_B128(2, "st.local", uint32_t, "r")
FCX_LD_B128(15, "ld.global.nc", uint64_t, "l")
FCX_LD_B128(19, "ld.global.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(20, "ld.global.L1::evict_first", uint64_t, "l")
FCX_LD_B128(21, "ld.global.L1::evict_last", uint64_t, "l")
FCX_LD_B128(22, "ld.global.nc.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(23, "ld.global.nc.L1::evict_first", uint64_t, "l")
FCX_LD_B128(24, "ld.global.nc.L1::evict_last", uint64_t, "l")
FCX_ST_B128(18, "st.global.L1::no_allocate", uint64_t, "l")
FCX_ST_B128(19, "st.global.L1::evict_first", uint64_t, "l")
FCX_ST_B128(20, "st.global.L1::evict_last", uint64_t, "l")
#endif
#if __CUDA_ARCH__ >= 700 &&                                                    \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
FCX_LD_B128(3, "ld.global.relaxed.sys", uint64_t, "l")
FCX_LD_B128(4, "ld.global.relaxed.gpu", uint64_t, "l")
FCX_LD_B128(5, "ld.global.relaxed.cta", uint64_t, "l")
FCX_LD_B128(6, "ld.shared.relaxed.sys", uint32_t, "r")
FCX_LD_B128(7, "ld.shared.relaxed.gpu", uint32_t, "r")
FCX_LD_B128(8, "ld.shared.relaxed.cta", uint32_t, "r")
FCX_LD_B128(9, "ld.global.acquire.sys", uint64_t, "l")
FCX_LD_B128(10, "ld.global.acquire.gpu", uint64_t, "l")
FCX_LD_B128(11, "ld.global.acquire.cta", uint64_t, "l")
FCX_LD_B128(12, "ld.shared.acquire.sys", uint32_t, "r")
FCX_LD_B128(13, "ld.shared.acquire.gpu", uint32_t, "r")
FCX_LD_B128(14, "ld.shared.acquire.cta", uint32_t, "r")
FCX_ST_B128(3, "st.global.relaxed.sys", uint64_t, "l")
FCX_ST_B128(4, "st.global.relaxed.gpu", uint64_t, "l")
FCX_ST_B128(5, "st.global.relaxed.cta", uint64_t, "l")
FCX_ST_B128(6, "st.shared.relaxed.sys", uint32_t, "r")
FCX_ST_B128(7, "st.shared.relaxed.gpu", uint32_t, "r")
FCX_ST_B128(8, "st.shared.relaxed.cta", uint32_t, "r")
FCX_ST_B128(9, "st.global.release.sys", uint64_t, "l")
FCX_ST_B128(10, "st.global.release.gpu", uint64_t, "l")
FCX_ST_B128(11, "st.global.release.cta", uint64_t, "l")
FCX_ST_B128(12, "st.shared.release.sys", uint32_t, "r")
FCX_ST_B128(13, "st.shared.release.gpu", uint32_t, "r")
FCX_ST_B128(14, "st.shared.release.cta", uint32_t, "r")
FCX_LD_B128(25, "ld.global.relaxed.sys.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(26, "ld.global.relaxed.gpu.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(27, "ld.global.relaxed.cta.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(28, "ld.global.relaxed.sys.L1::evict_first", uint64_t, "l")
FCX_LD_B128(29, "ld.global.relaxed.gpu.L1::evict_first", uint64_t, "l")
FCX_LD_B128(30, "ld.global.relaxed.cta.L1::evict_first", uint64_t, "l")
FCX_LD_B128(31, "ld.global.relaxed.sys.L1::evict_last", uint64_t, "l")
FCX_LD_B128(32, "ld.global.relaxed.gpu.L1::evict_last", uint64_t, "l")
FCX_LD_B128(33, "ld.global.relaxed.cta.L1::evict_last", uint64_t, "l")
FCX_LD_B128(34, "ld.global.acquire.sys.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(35, "ld.global.acquire.gpu.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(36, "ld.global.acquire.cta.L1::no_allocate", uint64_t, "l")
FCX_LD_B128(37, "ld.global.acquire.sys.L1::evict_first", uint64_t, "l")
FCX_LD_B128(38, "ld.global.acquire.gpu.L1::evict_first", uint64_t, "l")
FCX_LD_B128(39, "ld.global.acquire.cta.L1::evict_first", uint64_t, "l")
FCX_LD_B128(40, "ld.global.acquire.sys.L1::evict_last", uint64_t, "l")
FCX_LD_B128(41, "ld.global.acquire.gpu.L1::evict_last", uint64_t, "l")
FCX_LD_B128(42, "ld.global.acquire.cta.L1::evict_last", uint64_t, "l")
FCX_ST_B128(21, "st.global.relaxed.sys.L1::no_allocate", uint64_t, "l")
FCX_ST_B128(22, "st.global.relaxed.gpu.L1::no_allocate", uint64_t, "l")
FCX_ST_B128(23, "st.global.relaxed.cta.L1::no_allocate", uint64_t, "l")
FCX_ST_B128(24, "st.global.relaxed.sys.L1::evict_first", uint64_t, "l")
FCX_ST_B128(25, "st.global.relaxed.gpu.L1::evict_first", uint64_t, "l")
FCX_ST_B128(26, "st.global.relaxed.cta.L1::evict_first", uint64_t, "l")
FCX_ST_B128(27, "st.global.relaxed.sys.L1::evict_last", uint64_t, "l")
FCX_ST_B128(28, "st.global.relaxed.gpu.L1::evict_last", uint64_t, "l")
FCX_ST_B128(29, "st.global.relaxed.cta.L1::evict_last", uint64_t, "l")
FCX_ST_B128(30, "st.global.release.sys.L1::no_allocate", uint64_t, "l")
FCX_ST_B128(31, "st.global.release.gpu.L1::no_allocate", uint64_t, "l")
FCX_ST_B128(32, "st.global.release.cta.L1::no_allocate", uint64_t, "l")
FCX_ST_B128(33, "st.global.release.sys.L1::evict_first", uint64_t, "l")
FCX_ST_B128(34, "st.global.release.gpu.L1::evict_first", uint64_t, "l")
FCX_ST_B128(35, "st.global.release.cta.L1::evict_first", uint64_t, "l")
FCX_ST_B128(36, "st.global.release.sys.L1::evict_last", uint64_t, "l")
FCX_ST_B128(37, "st.global.release.gpu.L1::evict_last", uint64_t, "l")
FCX_ST_B128(38, "st.global.release.cta.L1::evict_last", uint64_t, "l")
#endif

// PTX 8.8 / sm_100 adds 32-byte global vectors.
#if __CUDA_ARCH__ >= 1000 &&                                                   \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
#define FCX_LD_256(M, P)                                                       \
  FCX_LD_V4(M, P, uint64_t, "l", 8, 64, uint64_t, "=l")                        \
  FCX_LD_V8_B32(M, P, uint64_t, "l")
#define FCX_ST_256(M, P)                                                       \
  FCX_ST_V4(M, P, uint64_t, "l", 8, 64, uint64_t, "l")                         \
  FCX_ST_V8_B32(M, P, uint64_t, "l")
FCX_LD_256(0, "ld.global")
FCX_ST_256(0, "st.global")
#if __CUDA_ARCH__ >= 700
FCX_LD_256(3, "ld.global.relaxed.sys")
FCX_LD_256(4, "ld.global.relaxed.gpu")
FCX_LD_256(5, "ld.global.relaxed.cta")
FCX_LD_256(9, "ld.global.acquire.sys")
FCX_LD_256(10, "ld.global.acquire.gpu")
FCX_LD_256(11, "ld.global.acquire.cta")
FCX_ST_256(3, "st.global.relaxed.sys")
FCX_ST_256(4, "st.global.relaxed.gpu")
FCX_ST_256(5, "st.global.relaxed.cta")
FCX_ST_256(9, "st.global.release.sys")
FCX_ST_256(10, "st.global.release.gpu")
FCX_ST_256(11, "st.global.release.cta")
#endif
FCX_LD_256(15, "ld.global.nc")
FCX_LD_256(19, "ld.global.L1::no_allocate")
FCX_LD_256(20, "ld.global.L1::evict_first")
FCX_LD_256(21, "ld.global.L1::evict_last")
FCX_LD_256(22, "ld.global.nc.L1::no_allocate")
FCX_LD_256(23, "ld.global.nc.L1::evict_first")
FCX_LD_256(24, "ld.global.nc.L1::evict_last")
FCX_ST_256(18, "st.global.L1::no_allocate")
FCX_ST_256(19, "st.global.L1::evict_first")
FCX_ST_256(20, "st.global.L1::evict_last")
FCX_LD_256(25, "ld.global.relaxed.sys.L1::no_allocate")
FCX_LD_256(26, "ld.global.relaxed.gpu.L1::no_allocate")
FCX_LD_256(27, "ld.global.relaxed.cta.L1::no_allocate")
FCX_LD_256(28, "ld.global.relaxed.sys.L1::evict_first")
FCX_LD_256(29, "ld.global.relaxed.gpu.L1::evict_first")
FCX_LD_256(30, "ld.global.relaxed.cta.L1::evict_first")
FCX_LD_256(31, "ld.global.relaxed.sys.L1::evict_last")
FCX_LD_256(32, "ld.global.relaxed.gpu.L1::evict_last")
FCX_LD_256(33, "ld.global.relaxed.cta.L1::evict_last")
FCX_LD_256(34, "ld.global.acquire.sys.L1::no_allocate")
FCX_LD_256(35, "ld.global.acquire.gpu.L1::no_allocate")
FCX_LD_256(36, "ld.global.acquire.cta.L1::no_allocate")
FCX_LD_256(37, "ld.global.acquire.sys.L1::evict_first")
FCX_LD_256(38, "ld.global.acquire.gpu.L1::evict_first")
FCX_LD_256(39, "ld.global.acquire.cta.L1::evict_first")
FCX_LD_256(40, "ld.global.acquire.sys.L1::evict_last")
FCX_LD_256(41, "ld.global.acquire.gpu.L1::evict_last")
FCX_LD_256(42, "ld.global.acquire.cta.L1::evict_last")
FCX_ST_256(21, "st.global.relaxed.sys.L1::no_allocate")
FCX_ST_256(22, "st.global.relaxed.gpu.L1::no_allocate")
FCX_ST_256(23, "st.global.relaxed.cta.L1::no_allocate")
FCX_ST_256(24, "st.global.relaxed.sys.L1::evict_first")
FCX_ST_256(25, "st.global.relaxed.gpu.L1::evict_first")
FCX_ST_256(26, "st.global.relaxed.cta.L1::evict_first")
FCX_ST_256(27, "st.global.relaxed.sys.L1::evict_last")
FCX_ST_256(28, "st.global.relaxed.gpu.L1::evict_last")
FCX_ST_256(29, "st.global.relaxed.cta.L1::evict_last")
FCX_ST_256(30, "st.global.release.sys.L1::no_allocate")
FCX_ST_256(31, "st.global.release.gpu.L1::no_allocate")
FCX_ST_256(32, "st.global.release.cta.L1::no_allocate")
FCX_ST_256(33, "st.global.release.sys.L1::evict_first")
FCX_ST_256(34, "st.global.release.gpu.L1::evict_first")
FCX_ST_256(35, "st.global.release.cta.L1::evict_first")
FCX_ST_256(36, "st.global.release.sys.L1::evict_last")
FCX_ST_256(37, "st.global.release.gpu.L1::evict_last")
FCX_ST_256(38, "st.global.release.cta.L1::evict_last")
#undef FCX_LD_256
#undef FCX_ST_256
#endif

// MMIO is scalar-only and must preserve each requested device transaction.
#if __CUDA_ARCH__ >= 700 &&                                                    \
    (__CUDACC_VER_MAJOR__ > 12 ||                                              \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2))
FCX_LD_SC(16, "ld.global.mmio.relaxed.sys", uint64_t, "l", 4, 32, uint32_t,
          "=r")
FCX_LD_SC(16, "ld.global.mmio.relaxed.sys", uint64_t, "l", 8, 64, uint64_t,
          "=l")
FCX_ST_SC(15, "st.global.mmio.relaxed.sys", uint64_t, "l", 4, 32, uint32_t, "r")
FCX_ST_SC(15, "st.global.mmio.relaxed.sys", uint64_t, "l", 8, 64, uint64_t, "l")
#endif
#if __CUDA_ARCH__ >= 750 && __CUDACC_VER_MAJOR__ >= 13 &&                      \
    (__CUDACC_VER_MAJOR__ > 13 || __CUDACC_VER_MINOR__ >= 3)
FCX_LD_SC(17, "ld.global.acquire.mmio.sys", uint64_t, "l", 4, 32, uint32_t,
          "=r")
FCX_LD_SC(17, "ld.global.acquire.mmio.sys", uint64_t, "l", 8, 64, uint64_t,
          "=l")
FCX_ST_SC(16, "st.global.release.mmio.sys", uint64_t, "l", 4, 32, uint32_t, "r")
FCX_ST_SC(16, "st.global.release.mmio.sys", uint64_t, "l", 8, 64, uint64_t, "l")
#endif

#undef FCX_LD_SC
#undef FCX_ST_SC
#undef FCX_LD_V2
#undef FCX_ST_V2
#undef FCX_LD_V4
#undef FCX_ST_V4
#undef FCX_LD_V8_B32
#undef FCX_ST_V8_B32
#undef FCX_LD_B128
#undef FCX_ST_B128
#undef FCX_LD_BASE
#undef FCX_ST_BASE
#endif // __CUDA_ARCH__

template <flagcxDeviceMemorySpace_t Space>
struct Address;
template <>
struct Address<flagcxDeviceMemoryGlobal> {
  typedef uint64_t type;
  static FLAGCX_DEVICE_INLINE_DECORATOR type get(const void *p) {
#if defined(__CUDA_ARCH__)
    if (!__isGlobal(p))
      __trap();
#endif
    return static_cast<type>(reinterpret_cast<uintptr_t>(p));
  }
};
template <>
struct Address<flagcxDeviceMemoryScratchpad> {
  typedef uint32_t type;
  static FLAGCX_DEVICE_INLINE_DECORATOR type get(const void *p) {
#if defined(__CUDA_ARCH__)
    if (!__isShared(p))
      __trap();
    return static_cast<type>(__cvta_generic_to_shared(p));
#else
    return static_cast<type>(reinterpret_cast<uintptr_t>(p));
#endif
  }
};
template <>
struct Address<flagcxDeviceMemoryLocal> {
  typedef uint32_t type;
  static FLAGCX_DEVICE_INLINE_DECORATOR type get(const void *p) {
#if defined(__CUDA_ARCH__)
    if (!__isLocal(p))
      __trap();
    return static_cast<type>(__cvta_generic_to_local(p));
#else
    return static_cast<type>(reinterpret_cast<uintptr_t>(p));
#endif
  }
};

template <typename T, flagcxDeviceScope_t Scope>
struct AtomicSeq;
#define FCX_ATOMIC_SEQ(SCOPE, CUDA_SCOPE)                                      \
  template <typename T>                                                        \
  struct AtomicSeq<T, SCOPE> {                                                 \
    static FLAGCX_DEVICE_INLINE_DECORATOR T load(const T *p) {                 \
      return cuda::atomic_ref<T, CUDA_SCOPE>(*const_cast<T *>(p))              \
          .load(cuda::memory_order_seq_cst);                                   \
    }                                                                          \
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(T *p, T value) {          \
      cuda::atomic_ref<T, CUDA_SCOPE>(*p).store(value,                         \
                                                cuda::memory_order_seq_cst);   \
    }                                                                          \
  };
FCX_ATOMIC_SEQ(flagcxDeviceScopeSystem, cuda::thread_scope_system)
FCX_ATOMIC_SEQ(flagcxDeviceScopeDevice, cuda::thread_scope_device)
FCX_ATOMIC_SEQ(flagcxDeviceScopeBlock, cuda::thread_scope_block)
FCX_ATOMIC_SEQ(flagcxDeviceScopeThread, cuda::thread_scope_thread)
#undef FCX_ATOMIC_SEQ

template <typename T, typename Shape, flagcxDeviceMemorySpace_t Space,
          flagcxDeviceMemoryOrder_t Order, flagcxDeviceScope_t Scope,
          flagcxDeviceMemoryAccessKind_t Kind, flagcxDeviceLoadIntent_t Intent,
          flagcxDeviceCacheHint_t CacheHint>
struct Load {
  enum {
    lanes = ShapeLanes<Shape>::value,
    bytes = lanes ? sizeof(T) / lanes : 0
  };
  static_assert(lanes != 0 && sizeof(T) % lanes == 0, "invalid load shape");
  static_assert(std::is_trivially_copyable<T>::value &&
                    !std::is_same<T, bool>::value && !std::is_const<T>::value &&
                    !std::is_volatile<T>::value,
                "load requires a non-cv trivially copyable non-bool value");
  static_assert(Space == flagcxDeviceMemoryGlobal ||
                    Space == flagcxDeviceMemoryScratchpad ||
                    Space == flagcxDeviceMemoryLocal,
                "invalid load space");
  static_assert(Scope >= flagcxDeviceScopeSystem &&
                    Scope <= flagcxDeviceScopeThread,
                "invalid load scope");
  static_assert(CacheHint >= flagcxDeviceCacheHintNone &&
                    CacheHint <= flagcxDeviceCacheHintEvictLast &&
                    (CacheHint == flagcxDeviceCacheHintNone ||
                     (Space == flagcxDeviceMemoryGlobal &&
                      Kind == flagcxDeviceMemoryAccessKindNormal &&
                      (Order == flagcxDeviceMemoryOrderWeak ||
                       (Intent == flagcxDeviceLoadIntentOrdinary &&
                        (Order == flagcxDeviceMemoryOrderRelaxed ||
                         Order == flagcxDeviceMemoryOrderAcquire))))),
                "cache hint requires a supported normal Global load");
  static_assert(
      (lanes == 1 &&
       (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16)) ||
          (lanes == 2 &&
           (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8)) ||
          (lanes == 4 && (bytes == 1 || bytes == 2 || bytes == 4 ||
                          (bytes == 8 && Space == flagcxDeviceMemoryGlobal))) ||
          (lanes == 8 && bytes == 4 && Space == flagcxDeviceMemoryGlobal),
      "PTX load shape has no matching instruction");
  // Immutable+Weak only says the caller requests no ordering. ld.global.nc
  // uses a non-coherent read-only cache and is outside the PTX memory model.
  static_assert((Kind == flagcxDeviceMemoryAccessKindNormal &&
                 Intent == flagcxDeviceLoadIntentOrdinary &&
                 (Order == flagcxDeviceMemoryOrderWeak ||
                  ((Space != flagcxDeviceMemoryLocal &&
                    Scope != flagcxDeviceScopeThread) &&
                   (Order == flagcxDeviceMemoryOrderRelaxed ||
                    Order == flagcxDeviceMemoryOrderAcquire)) ||
                  (Space == flagcxDeviceMemoryGlobal &&
                   Order == flagcxDeviceMemoryOrderSeqCst && lanes == 1 &&
                   bytes <= 8))) ||
                    (Kind == flagcxDeviceMemoryAccessKindNormal &&
                     Intent == flagcxDeviceLoadIntentImmutable &&
                     Space == flagcxDeviceMemoryGlobal &&
                     Order == flagcxDeviceMemoryOrderWeak) ||
                    (Kind == flagcxDeviceMemoryAccessKindMmio &&
                     Intent == flagcxDeviceLoadIntentOrdinary &&
                     Space == flagcxDeviceMemoryGlobal &&
                     Scope == flagcxDeviceScopeSystem && lanes == 1 &&
                     (std::is_same<T, uint32_t>::value ||
                      std::is_same<T, uint64_t>::value) &&
                     (Order == flagcxDeviceMemoryOrderRelaxed ||
                      Order == flagcxDeviceMemoryOrderAcquire)),
                "invalid load order/space/kind/intent");
  enum {
    mode =
        Kind == flagcxDeviceMemoryAccessKindMmio
            ? (Order == flagcxDeviceMemoryOrderAcquire ? 17 : 16)
        : Intent == flagcxDeviceLoadIntentImmutable
            ? (CacheHint == flagcxDeviceCacheHintNone ? 15 : 21 + CacheHint)
        : Order == flagcxDeviceMemoryOrderSeqCst ? 18
        : Order == flagcxDeviceMemoryOrderWeak
            ? (Space == flagcxDeviceMemoryGlobal
                   ? (CacheHint == flagcxDeviceCacheHintNone ? 0
                                                             : 18 + CacheHint)
               : Space == flagcxDeviceMemoryScratchpad ? 1
                                                       : 2)
        : (Space == flagcxDeviceMemoryGlobal &&
           CacheHint != flagcxDeviceCacheHintNone)
            ? (Order == flagcxDeviceMemoryOrderRelaxed ? 25 : 34) +
                  (CacheHint - 1) * 3 + Scope
            : (Order == flagcxDeviceMemoryOrderRelaxed ? 3 : 9) +
                  (Space == flagcxDeviceMemoryScratchpad ? 3 : 0) + Scope
  };

  static FLAGCX_DEVICE_INLINE_DECORATOR T run(const T *src) {
#if defined(__CUDA_ARCH__)
    if (reinterpret_cast<uintptr_t>(src) % sizeof(T) != 0)
      __trap();
    return runImpl(
        src,
        std::integral_constant<bool, Order == flagcxDeviceMemoryOrderSeqCst>());
#else
    (void)src;
    assert(false && "NVIDIA load called on host");
    return T{};
#endif
  }

private:
  static FLAGCX_DEVICE_INLINE_DECORATOR T runImpl(const T *src,
                                                  std::true_type) {
    return AtomicSeq<T, Scope>::load(src);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR T runImpl(const T *src,
                                                  std::false_type) {
    return runPtx(src, std::integral_constant<bool, bytes == 16>());
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR T runPtx(const T *src,
                                                 std::false_type) {
    typename Register<bytes>::type regs[lanes];
    PtxLoad<mode, lanes, bytes>::run(Address<Space>::get(src), regs);
    return unpack<T, lanes, bytes>(regs);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR T runPtx(const T *src, std::true_type) {
    uint32_t words[4];
    PtxLoad<mode, 1, 16>::run(Address<Space>::get(src), words);
    return unpack128<T>(words);
  }
};

template <typename T, typename Shape, flagcxDeviceMemorySpace_t Space,
          flagcxDeviceMemoryOrder_t Order, flagcxDeviceScope_t Scope,
          flagcxDeviceMemoryAccessKind_t Kind,
          flagcxDeviceCacheHint_t CacheHint>
struct Store {
  enum {
    lanes = ShapeLanes<Shape>::value,
    bytes = lanes ? sizeof(T) / lanes : 0
  };
  static_assert(lanes != 0 && sizeof(T) % lanes == 0, "invalid store shape");
  static_assert(std::is_trivially_copyable<T>::value &&
                    !std::is_same<T, bool>::value && !std::is_const<T>::value &&
                    !std::is_volatile<T>::value,
                "store requires a non-cv trivially copyable non-bool value");
  static_assert(Space == flagcxDeviceMemoryGlobal ||
                    Space == flagcxDeviceMemoryScratchpad ||
                    Space == flagcxDeviceMemoryLocal,
                "invalid store space");
  static_assert(Scope >= flagcxDeviceScopeSystem &&
                    Scope <= flagcxDeviceScopeThread,
                "invalid store scope");
  static_assert(CacheHint >= flagcxDeviceCacheHintNone &&
                    CacheHint <= flagcxDeviceCacheHintEvictLast &&
                    (CacheHint == flagcxDeviceCacheHintNone ||
                     (Space == flagcxDeviceMemoryGlobal &&
                      Kind == flagcxDeviceMemoryAccessKindNormal &&
                      (Order == flagcxDeviceMemoryOrderWeak ||
                       Order == flagcxDeviceMemoryOrderRelaxed ||
                       Order == flagcxDeviceMemoryOrderRelease))),
                "cache hint requires a supported normal Global store");
  static_assert(
      (lanes == 1 &&
       (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16)) ||
          (lanes == 2 &&
           (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8)) ||
          (lanes == 4 && (bytes == 1 || bytes == 2 || bytes == 4 ||
                          (bytes == 8 && Space == flagcxDeviceMemoryGlobal))) ||
          (lanes == 8 && bytes == 4 && Space == flagcxDeviceMemoryGlobal),
      "PTX store shape has no matching instruction");
  static_assert((Kind == flagcxDeviceMemoryAccessKindNormal &&
                 (Order == flagcxDeviceMemoryOrderWeak ||
                  ((Space != flagcxDeviceMemoryLocal &&
                    Scope != flagcxDeviceScopeThread) &&
                   (Order == flagcxDeviceMemoryOrderRelaxed ||
                    Order == flagcxDeviceMemoryOrderRelease)) ||
                  (Space == flagcxDeviceMemoryGlobal &&
                   Order == flagcxDeviceMemoryOrderSeqCst && lanes == 1 &&
                   bytes <= 8))) ||
                    (Kind == flagcxDeviceMemoryAccessKindMmio &&
                     Space == flagcxDeviceMemoryGlobal &&
                     Scope == flagcxDeviceScopeSystem && lanes == 1 &&
                     (std::is_same<T, uint32_t>::value ||
                      std::is_same<T, uint64_t>::value) &&
                     (Order == flagcxDeviceMemoryOrderRelaxed ||
                      Order == flagcxDeviceMemoryOrderRelease)),
                "invalid store order/space/kind");
  enum {
    mode =
        Kind == flagcxDeviceMemoryAccessKindMmio
            ? (Order == flagcxDeviceMemoryOrderRelease ? 16 : 15)
        : Order == flagcxDeviceMemoryOrderSeqCst ? 17
        : Order == flagcxDeviceMemoryOrderWeak
            ? (Space == flagcxDeviceMemoryGlobal
                   ? (CacheHint == flagcxDeviceCacheHintNone ? 0
                                                             : 17 + CacheHint)
               : Space == flagcxDeviceMemoryScratchpad ? 1
                                                       : 2)
        : (Space == flagcxDeviceMemoryGlobal &&
           CacheHint != flagcxDeviceCacheHintNone)
            ? (Order == flagcxDeviceMemoryOrderRelaxed ? 21 : 30) +
                  (CacheHint - 1) * 3 + Scope
            : (Order == flagcxDeviceMemoryOrderRelaxed ? 3 : 9) +
                  (Space == flagcxDeviceMemoryScratchpad ? 3 : 0) + Scope
  };

  static FLAGCX_DEVICE_INLINE_DECORATOR void run(T *dst, T value) {
#if defined(__CUDA_ARCH__)
    if (reinterpret_cast<uintptr_t>(dst) % sizeof(T) != 0)
      __trap();
    runImpl(
        dst, value,
        std::integral_constant<bool, Order == flagcxDeviceMemoryOrderSeqCst>());
#else
    (void)dst;
    (void)value;
    assert(false && "NVIDIA store called on host");
#endif
  }

private:
  static FLAGCX_DEVICE_INLINE_DECORATOR void runImpl(T *dst, T value,
                                                     std::true_type) {
    AtomicSeq<T, Scope>::store(dst, value);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void runImpl(T *dst, T value,
                                                     std::false_type) {
    runMmioRelease(dst, value, std::integral_constant < bool,
                   Kind == flagcxDeviceMemoryAccessKindMmio &&
                       Order == flagcxDeviceMemoryOrderRelease > ());
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void runMmioRelease(T *dst, T value,
                                                            std::false_type) {
    runPtx(dst, value, std::integral_constant<bool, bytes == 16>());
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void runMmioRelease(T *dst, T value,
                                                            std::true_type) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 &&                          \
    (__CUDACC_VER_MAJOR__ > 13 ||                                              \
     (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 3))
    runPtx(dst, value, std::false_type());
#else
    // The existing Release behavior on pre-PTX-9.3 toolchains.
#if defined(__CUDA_ARCH__)
    __threadfence_system();
#endif
    typename Register<bytes>::type regs[lanes];
    pack<T, lanes, bytes>(value, regs);
    PtxStore<15, lanes, bytes>::run(Address<Space>::get(dst), regs);
#endif
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void runPtx(T *dst, T value,
                                                    std::false_type) {
    typename Register<bytes>::type regs[lanes];
    pack<T, lanes, bytes>(value, regs);
    PtxStore<mode, lanes, bytes>::run(Address<Space>::get(dst), regs);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void runPtx(T *dst, T value,
                                                    std::true_type) {
    uint32_t words[4];
    pack128(value, words);
    PtxStore<mode, 1, 16>::run(Address<Space>::get(dst), words);
  }
};

} // namespace flagcxNvidiaMemory

#endif // FLAGCX_NVIDIA_PTX_MEMORY_H_
