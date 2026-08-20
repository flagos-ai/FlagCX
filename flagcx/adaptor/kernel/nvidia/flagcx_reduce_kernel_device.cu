#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/flagcx_device.h"

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include <math.h>
#include <limits>
#include <type_traits>

#define SLOT_IDX 4
#define FST_IDX 5
#define SND_IDX 6
#define OUT_IDX 7
#define COUNT_IDX 8
#define NTHREADS_IDX 9
#define DATATYPE_IDX 10
#define REDOP_IDX 11
#define FLAG_IN_IDX 12
#define FLAG_OUT_IDX 13
#define STATIC_ABORT_IDX 14
#define STATIC_IPC_ABORT_IDX 0
#define STATIC_IPC_STATE_IDX 1

static constexpr uint32_t kStaticReduceAbortPollInterval = 8;

FLAGCX_DEVICE_INLINE_DECORATOR flagcxStreamFlagState
loadStreamFlagState(uint64_t flagAddr) {
  return static_cast<flagcxStreamFlagState>(DeviceAPI::Atomic::load(
      reinterpret_cast<uint64_t *>(flagAddr), flagcxDeviceMemoryOrderAcquire));
}

FLAGCX_DEVICE_INLINE_DECORATOR bool
isStreamFlagStatePending(flagcxStreamFlagState state) {
  return state == flagcxStreamFlagIdle || state == flagcxStreamFlagPend;
}

FLAGCX_DEVICE_INLINE_DECORATOR bool
isStreamFlagStateDone(flagcxStreamFlagState state) {
  return state == flagcxStreamFlagDone;
}

FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getInput1() {
  return value[0];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getInput2() {
  return value[1];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getOutput() {
  return value[2];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getCount() {
  return value[3] >> flagcxReduceTriggerOffCount &
         flagcxTriggerMask(flagcxReduceTriggerBitsCount);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getNThreads() {
  return value[3] >> flagcxReduceTriggerOffNThreads &
         flagcxTriggerMask(flagcxReduceTriggerBitsNThreads);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getDatatype() {
  return value[3] >> flagcxReduceTriggerOffDatatype &
         flagcxTriggerMask(flagcxReduceTriggerBitsDatatype);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getRedop() {
  return value[3] >> flagcxReduceTriggerOffRedop &
         flagcxTriggerMask(flagcxReduceTriggerBitsRedop);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getState() {
  return value[3] >> flagcxReduceTriggerOffState &
         flagcxTriggerMask(flagcxReduceTriggerBitsState);
}
FLAGCX_DEVICE_INLINE_DECORATOR void flagcxReduceTrigger::setComplete() {
  uint64_t flagOut = getFlagOut();
  if (flagOut != 0) {
    flagcxStreamFlagState flagState = loadStreamFlagState(flagOut);
    if (isStreamFlagStatePending(flagState)) {
      DeviceAPI::Atomic::store(reinterpret_cast<uint64_t *>(flagOut),
                               (uint64_t)flagcxStreamFlagDone,
                               flagcxDeviceMemoryOrderRelease);
    }
  }
  // Recycle the FIFO slot only after the output flag is visible as DONE, so a
  // host-side re-enqueue cannot overwrite flagOut before dependent streams
  // observe completion.
  uint64_t currVal =
      DeviceAPI::Atomic::load(value + 3, flagcxDeviceMemoryOrderAcquire);
  currVal &= ~(flagcxTriggerMask(flagcxReduceTriggerBitsState)
               << flagcxReduceTriggerOffState);
  currVal |= (flagcxReduceTriggerAvailable &
              flagcxTriggerMask(flagcxReduceTriggerBitsState))
             << flagcxReduceTriggerOffState;
  DeviceAPI::Atomic::store(value + 3, currVal, flagcxDeviceMemoryOrderRelease);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getFlagIn() {
  return value[4];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getFlagOut() {
  return value[5];
}

FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(uint64_t *buffer,
                                                      int *idx) {
  while (true) {
    uint64_t oldConsumed = DeviceAPI::Atomic::load(
        buffer + flagcxFifoIdxConsumed, flagcxDeviceMemoryOrderAcquire);
    uint64_t curProduced = DeviceAPI::Atomic::load(
        buffer + flagcxFifoIdxProduced, flagcxDeviceMemoryOrderAcquire);
    if (oldConsumed >= curProduced) {
      // no-op, task dequeued by other consumers
      *idx = -1;
      break;
    }
    // set consumed from `oldConsumed` to `oldConsumed+1`
    uint64_t expected = oldConsumed;
    if (DeviceAPI::Atomic::compareExchange(buffer + flagcxFifoIdxConsumed,
                                           expected, oldConsumed + 1,
                                           flagcxDeviceMemoryOrderAcqRel)) {
      *idx = oldConsumed;
      break;
    }
  }
  return flagcxSuccess;
}

template <typename T, typename Enable = void> struct uniRunnerReduceTraits {
  FLAGCX_DEVICE_INLINE_DECORATOR static T sum(T a, T b) {
    return a + b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T prod(T a, T b) {
    return a * b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T max(T a, T b) {
    return a > b ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T min(T a, T b) {
    return a < b ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T avg(T a, T b,
                                               uint64_t divisor) {
    return static_cast<T>((a + b) / static_cast<T>(divisor));
  }
};

template <typename T>
struct uniRunnerReduceTraits<
    T, typename std::enable_if<std::is_integral<T>::value>::type> {
  using U = typename std::make_unsigned<T>::type;

  FLAGCX_DEVICE_INLINE_DECORATOR static T sum(T a, T b) {
    return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T prod(T a, T b) {
    return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T max(T a, T b) {
    return a > b ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T min(T a, T b) {
    return a < b ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static T avg(T a, T b,
                                               uint64_t divisor) {
    T value = sum(a, b);
    if constexpr (std::is_signed<T>::value) {
      return static_cast<T>(static_cast<int64_t>(value) /
                            static_cast<int64_t>(divisor));
    } else {
      return static_cast<T>(static_cast<uint64_t>(value) / divisor);
    }
  }
};

template <> struct uniRunnerReduceTraits<float> {
  FLAGCX_DEVICE_INLINE_DECORATOR static float sum(float a, float b) {
    return a + b;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static float prod(float a, float b) {
    return a * b;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static float max(float a, float b) {
    return fmaxf(a, b);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static float min(float a, float b) {
    return fminf(a, b);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static float avg(float a, float b,
                                                   uint64_t divisor) {
    return (a + b) / static_cast<float>(divisor);
  }
};

template <> struct uniRunnerReduceTraits<double> {
  FLAGCX_DEVICE_INLINE_DECORATOR static double sum(double a, double b) {
    return a + b;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static double prod(double a, double b) {
    return a * b;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static double max(double a, double b) {
    return fmax(a, b);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static double min(double a, double b) {
    return fmin(a, b);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static double avg(double a, double b,
                                                    uint64_t divisor) {
    return (a + b) / static_cast<double>(divisor);
  }
};

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
template <> struct uniRunnerReduceTraits<__half> {
  FLAGCX_DEVICE_INLINE_DECORATOR static __half sum(__half a, __half b) {
    return __hadd(a, b);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __half prod(__half a, __half b) {
    return __hmul(a, b);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __half max(__half a, __half b) {
    return __half2float(a) > __half2float(b) ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __half min(__half a, __half b) {
    return __half2float(a) < __half2float(b) ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __half avg(__half a, __half b,
                                                    uint64_t divisor) {
    return __float2half((__half2float(__hadd(a, b))) /
                        static_cast<float>(divisor));
  }
};

template <> struct uniRunnerReduceTraits<__nv_bfloat16> {
  FLAGCX_DEVICE_INLINE_DECORATOR static __nv_bfloat16 sum(
      __nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __nv_bfloat16 prod(
      __nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __nv_bfloat16 max(
      __nv_bfloat16 a, __nv_bfloat16 b) {
    return __bfloat162float(a) > __bfloat162float(b) ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __nv_bfloat16 min(
      __nv_bfloat16 a, __nv_bfloat16 b) {
    return __bfloat162float(a) < __bfloat162float(b) ? a : b;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static __nv_bfloat16 avg(
      __nv_bfloat16 a, __nv_bfloat16 b, uint64_t divisor) {
    return __float2bfloat16((__bfloat162float(a) + __bfloat162float(b)) /
                            static_cast<float>(divisor));
  }
};
#endif

template <typename T, flagcxRedOp_t Op>
FLAGCX_DEVICE_INLINE_DECORATOR T uniRunnerApplyReduce(T a, T b,
                                                       uint64_t divisor) {
  if constexpr (Op == flagcxSum) {
    return uniRunnerReduceTraits<T>::sum(a, b);
  } else if constexpr (Op == flagcxProd) {
    return uniRunnerReduceTraits<T>::prod(a, b);
  } else if constexpr (Op == flagcxMax) {
    return uniRunnerReduceTraits<T>::max(a, b);
  } else if constexpr (Op == flagcxMin) {
    return uniRunnerReduceTraits<T>::min(a, b);
  } else {
    return uniRunnerReduceTraits<T>::avg(a, b, divisor);
  }
}

template <typename T, flagcxRedOp_t Op>
FLAGCX_DEVICE_INLINE_DECORATOR void uniRunnerReduceTyped(
    uint64_t fst, uint64_t snd, uint64_t out, uint64_t count,
    uint64_t nthreads, uint64_t avgDivisor) {
  const T *fstPtr = reinterpret_cast<const T *>(fst);
  const T *sndPtr = reinterpret_cast<const T *>(snd);
  T *outPtr = reinterpret_cast<T *>(out);
  const uint64_t tid = static_cast<uint64_t>(FLAGCX_THREAD_IDX_X);
  (void)nthreads;
  const uint64_t stride = static_cast<uint64_t>(FLAGCX_BLOCK_DIM_X);
  for (uint64_t i = tid; i < count; i += stride) {
    T a = fstPtr[i];
    T b = sndPtr[i];
    outPtr[i] = uniRunnerApplyReduce<T, Op>(a, b, avgDivisor);
  }
}

template <typename T>
FLAGCX_DEVICE_INLINE_DECORATOR void uniRunnerDispatchReduceOp(
    uint64_t fst, uint64_t snd, uint64_t out, uint64_t count,
    uint64_t nthreads, uint64_t redOp, uint64_t avgDivisor) {
  switch (static_cast<flagcxRedOp_t>(redOp)) {
    case flagcxSum:
      uniRunnerReduceTyped<T, flagcxSum>(fst, snd, out, count, nthreads,
                                         avgDivisor);
      break;
    case flagcxProd:
      uniRunnerReduceTyped<T, flagcxProd>(fst, snd, out, count, nthreads,
                                          avgDivisor);
      break;
    case flagcxMax:
      uniRunnerReduceTyped<T, flagcxMax>(fst, snd, out, count, nthreads,
                                         avgDivisor);
      break;
    case flagcxMin:
      uniRunnerReduceTyped<T, flagcxMin>(fst, snd, out, count, nthreads,
                                         avgDivisor);
      break;
    case flagcxAvg:
      uniRunnerReduceTyped<T, flagcxAvg>(fst, snd, out, count, nthreads,
                                         avgDivisor);
      break;
    default:
      break;
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void flagcxReduceKernel(
    uint64_t fst, uint64_t snd, uint64_t out, uint64_t count,
    uint64_t nthreads, uint64_t datatype, uint64_t redOp,
    uint64_t avgDivisor) {
  if (nthreads == 0 || avgDivisor == 0)
    return;

  switch (static_cast<flagcxDataType_t>(datatype)) {
    case flagcxInt8:
      uniRunnerDispatchReduceOp<int8_t>(fst, snd, out, count, nthreads, redOp,
                                        avgDivisor);
      break;
    case flagcxUint8:
      uniRunnerDispatchReduceOp<uint8_t>(fst, snd, out, count, nthreads,
                                         redOp, avgDivisor);
      break;
    case flagcxInt32:
      uniRunnerDispatchReduceOp<int32_t>(fst, snd, out, count, nthreads, redOp,
                                         avgDivisor);
      break;
    case flagcxUint32:
      uniRunnerDispatchReduceOp<uint32_t>(fst, snd, out, count, nthreads, redOp,
                                          avgDivisor);
      break;
    case flagcxInt64:
      uniRunnerDispatchReduceOp<int64_t>(fst, snd, out, count, nthreads, redOp,
                                         avgDivisor);
      break;
    case flagcxUint64:
      uniRunnerDispatchReduceOp<uint64_t>(fst, snd, out, count, nthreads, redOp,
                                          avgDivisor);
      break;
#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
    case flagcxFloat16:
      uniRunnerDispatchReduceOp<__half>(fst, snd, out, count, nthreads, redOp,
                                        avgDivisor);
      break;
#endif
    case flagcxFloat32:
      uniRunnerDispatchReduceOp<float>(fst, snd, out, count, nthreads, redOp,
                                       avgDivisor);
      break;
    case flagcxFloat64:
      uniRunnerDispatchReduceOp<double>(fst, snd, out, count, nthreads, redOp,
                                        avgDivisor);
      break;
#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
    case flagcxBfloat16:
      uniRunnerDispatchReduceOp<__nv_bfloat16>(fst, snd, out, count, nthreads,
                                               redOp, avgDivisor);
      break;
#endif
    default:
      break;
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void runReduceExecutor(void *fifoBuffer,
                                                       uint64_t avgDivisor) {
  FLAGCX_SHARED uint64_t shm[16];
  uint64_t *vBuf = (uint64_t *)fifoBuffer;
  int emptyIter = 0; // backoff counter
  int cap = -1;
  int c = -1;
  int p = -1;
  int term = -1;
  int slot = -1;
  int tid = FLAGCX_THREAD_IDX_X;
  if (tid == 0) {
    shm[flagcxFifoIdxCapacity] = vBuf[flagcxFifoIdxCapacity];
  }
  FLAGCX_DEVICE_SYNC_THREADS();
  cap = shm[flagcxFifoIdxCapacity];

  while (true) {
    // (1) dequeue
    if (tid == 0) {
      shm[flagcxFifoIdxConsumed] = DeviceAPI::Atomic::load(
          &vBuf[flagcxFifoIdxConsumed], flagcxDeviceMemoryOrderAcquire);
      shm[flagcxFifoIdxProduced] = DeviceAPI::Atomic::load(
          &vBuf[flagcxFifoIdxProduced], flagcxDeviceMemoryOrderAcquire);
      shm[flagcxFifoIdxTerminate] = DeviceAPI::Atomic::load(
          &vBuf[flagcxFifoIdxTerminate], flagcxDeviceMemoryOrderAcquire);
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    c = shm[flagcxFifoIdxConsumed];
    p = shm[flagcxFifoIdxProduced];
    term = shm[flagcxFifoIdxTerminate];

    // (2) backoff if queue empty
    if (c >= p) {
      // check terminate
      if (term == 1)
        break;
      emptyIter++;
      DeviceAPI::Intrin::spinBackoff(emptyIter);
      continue;
    }

    // (3) dequeue task (lane 0 in a warp)
    if (tid == 0) {
      int myIdx = -1;
      dequeue(vBuf, &myIdx);
      slot = myIdx & (cap - 1);
      shm[SLOT_IDX] = myIdx < 0 ? cap : slot;
      if (myIdx >= 0) {
        flagcxReduceTrigger *t =
            (flagcxReduceTrigger *)(vBuf + flagcxFifoIdxData) + slot;
        shm[FST_IDX] = t->getInput1();
        shm[SND_IDX] = t->getInput2();
        shm[OUT_IDX] = t->getOutput();
        shm[COUNT_IDX] = t->getCount();
        shm[NTHREADS_IDX] = t->getNThreads();
        shm[DATATYPE_IDX] = t->getDatatype();
        shm[REDOP_IDX] = t->getRedop();
        shm[FLAG_IN_IDX] = t->getFlagIn();
        shm[FLAG_OUT_IDX] = t->getFlagOut();
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    // sync slot to warp
    slot = shm[SLOT_IDX];
    if (slot == cap) {
      if (term == 1)
        break;
      emptyIter++;
      DeviceAPI::Intrin::spinBackoff(emptyIter);
      continue;
    }

    // RED nodes are submitted from the host before they are executable, so the
    // kernel marks the output flag as pending once it has claimed the FIFO
    // slot.
    if (tid == 0 && shm[FLAG_OUT_IDX] != 0) {
      uint64_t flagOut = shm[FLAG_OUT_IDX];
      flagcxStreamFlagState flagState = loadStreamFlagState(flagOut);
      if (flagState == flagcxStreamFlagIdle) {
        DeviceAPI::Atomic::store(reinterpret_cast<uint64_t *>(flagOut),
                                 (uint64_t)flagcxStreamFlagPend,
                                 flagcxDeviceMemoryOrderRelease);
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();

    uint64_t flagIn = shm[FLAG_IN_IDX];
    while (flagIn != 0) {
      flagcxStreamFlagState flagState = loadStreamFlagState(flagIn);
      if (isStreamFlagStateDone(flagState)) {
        break;
      }
      if (isStreamFlagStatePending(flagState)) {
        emptyIter++;
        DeviceAPI::Intrin::spinBackoff(emptyIter);
        continue;
      }
      emptyIter++;
      DeviceAPI::Intrin::spinBackoff(emptyIter);
    }

    // (4) perform reduce task
    emptyIter = 0;
    uint64_t fst = shm[FST_IDX];
    uint64_t snd = shm[SND_IDX];
    uint64_t out = shm[OUT_IDX];
    uint64_t count = shm[COUNT_IDX];
    uint64_t nthreads = shm[NTHREADS_IDX];
    uint64_t datatype = shm[DATATYPE_IDX];
    uint64_t redop = shm[REDOP_IDX];
    flagcxReduceKernel(fst, snd, out, count, nthreads, datatype, redop,
                       avgDivisor);
    FLAGCX_DEVICE_SYNC_THREADS();
    FLAGCX_DEVICE_THREAD_FENCE();

    // (5) signal completion and recycle the FIFO slot
    if (tid == 0) {
      flagcxReduceTrigger *t =
          (flagcxReduceTrigger *)(vBuf + flagcxFifoIdxData) + slot;
      t->setComplete();
    }
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR bool
isStaticAbortRequested(const uint64_t *abortFlag) {
  return DeviceAPI::Atomic::load(
             const_cast<uint64_t *>(abortFlag),
             flagcxDeviceMemoryOrderAcquire) != 0;
}

FLAGCX_DEVICE_INLINE_DECORATOR void
requestStaticAbort(const uint64_t *abortFlag) {
  DeviceAPI::Atomic::store(const_cast<uint64_t *>(abortFlag), uint64_t(1),
                           flagcxDeviceMemoryOrderRelease);
}

FLAGCX_DEVICE_INLINE_DECORATOR void publishStaticReduceAbort(
    flagcxReduceTrigger *triggers, uint64_t firstOrdinal,
    uint64_t numTriggers, uint64_t stride) {
  if (FLAGCX_THREAD_IDX_X == 0) {
    for (uint64_t ordinal = firstOrdinal; ordinal < numTriggers;
         ordinal += stride) {
      // No reduction result is valid on this path. Publishing DONE is solely
      // an error-unwind mechanism that releases already-submitted stream
      // waits before the original host error is returned. Static triggers are
      // immutable invocation entries, so their mapped FIFO state is not
      // recycled.
      const uint64_t flagOut = triggers[ordinal].getFlagOut();
      if (flagOut != 0) {
        DeviceAPI::Atomic::store(
            reinterpret_cast<uint64_t *>(flagOut),
            static_cast<uint64_t>(flagcxStreamFlagDone),
            flagcxDeviceMemoryOrderRelease);
      }
    }
  }
  FLAGCX_DEVICE_SYNC_THREADS();
}

FLAGCX_DEVICE_INLINE_DECORATOR void runStaticReduceExecutor(
    flagcxReduceTrigger *triggers, uint64_t numTriggers,
    const uint64_t *abortFlag, uint64_t avgDivisor, uint64_t firstOrdinal,
    uint64_t stride, uint64_t *shm) {
  const int tid = FLAGCX_THREAD_IDX_X;

  for (uint64_t ordinal = firstOrdinal; ordinal < numTriggers;
       ordinal += stride) {
    if (tid == 0) {
      shm[STATIC_ABORT_IDX] = isStaticAbortRequested(abortFlag) ? 1 : 0;
      if (shm[STATIC_ABORT_IDX] == 0) {
        flagcxReduceTrigger *trigger = &triggers[ordinal];
        shm[FST_IDX] = trigger->getInput1();
        shm[SND_IDX] = trigger->getInput2();
        shm[OUT_IDX] = trigger->getOutput();
        shm[COUNT_IDX] = trigger->getCount();
        shm[NTHREADS_IDX] = trigger->getNThreads();
        shm[DATATYPE_IDX] = trigger->getDatatype();
        shm[REDOP_IDX] = trigger->getRedop();
        shm[FLAG_IN_IDX] = trigger->getFlagIn();
        shm[FLAG_OUT_IDX] = trigger->getFlagOut();
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    if (shm[STATIC_ABORT_IDX] != 0) {
      publishStaticReduceAbort(triggers, ordinal, numTriggers, stride);
      return;
    }

    if (tid == 0) {
      shm[STATIC_ABORT_IDX] = 0;
      const uint64_t flagIn = shm[FLAG_IN_IDX];
      int backoff = 0;
      uint32_t abortPollIterations = 0;
      while (flagIn != 0 &&
             !isStreamFlagStateDone(loadStreamFlagState(flagIn))) {
        ++abortPollIterations;
        if ((abortPollIterations & (kStaticReduceAbortPollInterval - 1)) == 0 &&
            isStaticAbortRequested(abortFlag)) {
          shm[STATIC_ABORT_IDX] = 1;
          break;
        }
        DeviceAPI::Intrin::spinBackoff(backoff++);
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    if (shm[STATIC_ABORT_IDX] != 0) {
      publishStaticReduceAbort(triggers, ordinal, numTriggers, stride);
      return;
    }

    flagcxReduceKernel(shm[FST_IDX], shm[SND_IDX], shm[OUT_IDX],
                       shm[COUNT_IDX], shm[NTHREADS_IDX], shm[DATATYPE_IDX],
                       shm[REDOP_IDX], avgDivisor);
    FLAGCX_DEVICE_SYNC_THREADS();
    FLAGCX_DEVICE_THREAD_FENCE();
    // Every reduction thread must finish its system fence before thread 0
    // publishes DONE to a consumer on another stream. Unlike the dynamic
    // FIFO, this exact static array is never re-enqueued, so completion does
    // not need a mapped trigger-state read/modify/write.
    FLAGCX_DEVICE_SYNC_THREADS();
    if (tid == 0 && shm[FLAG_OUT_IDX] != 0) {
      DeviceAPI::Atomic::store(
          reinterpret_cast<uint64_t *>(shm[FLAG_OUT_IDX]),
          static_cast<uint64_t>(flagcxStreamFlagDone),
          flagcxDeviceMemoryOrderRelease);
    }
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxStaticReduceKernel(
    flagcxReduceTrigger *triggers, uint64_t numTriggers,
    const uint64_t *abortFlag, uint64_t avgDivisor) {
  FLAGCX_SHARED uint64_t shm[16];
  runStaticReduceExecutor(
      triggers, numTriggers, abortFlag, avgDivisor,
      static_cast<uint64_t>(FLAGCX_BLOCK_IDX_X),
      static_cast<uint64_t>(FLAGCX_GRID_DIM_X), shm);
}

struct alignas(16) flagcxIpcVector128 {
  uint64_t x;
  uint64_t y;
};

FLAGCX_DEVICE_INLINE_DECORATOR void
waitIpcParents(const flagcxIpcTrigger &trigger,
               const uint64_t *parentFlags) {
  if (FLAGCX_THREAD_IDX_X != 0)
    return;
  for (uint32_t i = 0; i < trigger.numParentFlags; ++i) {
    uint64_t flagAddr = parentFlags[trigger.parentFlagsOffset + i];
    int iter = 0;
    while (!isStreamFlagStateDone(loadStreamFlagState(flagAddr))) {
      DeviceAPI::Intrin::spinBackoff(iter++);
    }
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void
copyIpcBytes(void *dst, const void *src, size_t bytes) {
  uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst);
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);
  size_t vectorBytes = 0;
  if (((dstAddr | srcAddr) & (alignof(flagcxIpcVector128) - 1)) == 0) {
    vectorBytes = bytes & ~(sizeof(flagcxIpcVector128) - 1);
    flagcxIpcVector128 *dstVec =
        reinterpret_cast<flagcxIpcVector128 *>(dst);
    const flagcxIpcVector128 *srcVec =
        reinterpret_cast<const flagcxIpcVector128 *>(src);
    size_t nvec = vectorBytes / sizeof(flagcxIpcVector128);
    for (size_t i = FLAGCX_THREAD_IDX_X; i < nvec;
         i += FLAGCX_BLOCK_DIM_X) {
      dstVec[i] = srcVec[i];
    }
  }

  char *dstBytes = static_cast<char *>(dst);
  const char *srcBytes = static_cast<const char *>(src);
  for (size_t i = vectorBytes + FLAGCX_THREAD_IDX_X; i < bytes;
       i += FLAGCX_BLOCK_DIM_X) {
    dstBytes[i] = srcBytes[i];
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *getStaticIpcReadyPointer(
    const flagcxDevMem &readyMem, uint64_t slot) {
  return static_cast<uint64_t *>(flagcxGetLocalPointer(
      readyMem, slot * sizeof(uint64_t)));
}

FLAGCX_DEVICE_INLINE_DECORATOR void publishStaticIpcControlValue(
    const flagcxDevMem &readyMem, uint64_t slot, uint64_t value,
    uint32_t localRanks) {
  for (uint32_t peer = 0; peer < localRanks; ++peer) {
    uint64_t *remote = static_cast<uint64_t *>(flagcxGetIntraPointer(
        readyMem, slot * sizeof(uint64_t), static_cast<int>(peer)));
    DeviceAPI::Atomic::store(remote, value, flagcxDeviceMemoryOrderRelease);
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR bool isStaticIpcAbortRequested(
    const uint64_t *abortFlag, const flagcxDevMem &readyMem,
    const flagcxStaticIpcControl &control) {
  if (isStaticAbortRequested(abortFlag)) {
    return true;
  }
  const uint64_t observed = DeviceAPI::Atomic::load(
      getStaticIpcReadyPointer(readyMem, control.abortSlot),
      flagcxDeviceMemoryOrderAcquire);
  if (flagcxIpcEpochReached(observed, control.epoch)) {
    requestStaticAbort(abortFlag);
    return true;
  }
  return false;
}

FLAGCX_DEVICE_INLINE_DECORATOR void publishStaticIpcRemoteAbort(
    const flagcxDevMem &readyMem,
    const flagcxStaticIpcControl &control) {
  publishStaticIpcControlValue(readyMem, control.abortSlot, control.epoch,
                               control.localRanks);
}

FLAGCX_DEVICE_INLINE_DECORATOR void publishStaticIpcAbortCompletion(
    const flagcxDevMem &readyMem, const uint64_t *abortFlag,
    const flagcxStaticIpcControl &control) {
  // All stores below are system-scope atomics through DeviceAPI::Atomic. This
  // transition is deliberately idempotent: a rank may learn about the same
  // abort through its local flag, the shared abort slot, and a peer done slot.
  requestStaticAbort(abortFlag);
  publishStaticIpcRemoteAbort(readyMem, control);
  publishStaticIpcControlValue(
      readyMem, control.doneBase + control.localRank,
      control.epoch | flagcxIpcControlAbortBit, control.localRanks);
}

FLAGCX_DEVICE_INLINE_DECORATOR void publishStaticIpcAbort(
    flagcxIpcTrigger *triggers, uint64_t numTriggers,
    uint64_t firstOrdinal, uint64_t stride) {
  if (FLAGCX_THREAD_IDX_X == 0) {
    for (uint64_t ordinal = firstOrdinal; ordinal < numTriggers;
         ordinal += stride) {
      flagcxIpcTrigger *trigger = &triggers[ordinal];
      if (trigger->flagOut != 0) {
        // This cleanup is deliberately idempotent. Future triggers can still
        // be Idle when abort is observed, but their submitted stream waits
        // must be released just like Pending triggers.
        DeviceAPI::Atomic::store(
            reinterpret_cast<uint64_t *>(trigger->flagOut),
            static_cast<uint64_t>(flagcxStreamFlagDone),
            flagcxDeviceMemoryOrderRelease);
      }
      DeviceAPI::Atomic::store(
          &trigger->state,
          static_cast<uint32_t>(flagcxReduceTriggerAvailable),
          flagcxDeviceMemoryOrderRelease);
    }
  }
  FLAGCX_DEVICE_SYNC_THREADS();
}

FLAGCX_DEVICE_INLINE_DECORATOR void abortStaticIpcExecutor(
    flagcxIpcTrigger *triggers, uint64_t numTriggers,
    const flagcxDevMem &readyMem,
    const flagcxStaticIpcControl &control, uint64_t firstOrdinal,
    uint64_t stride) {
  if (FLAGCX_THREAD_IDX_X == 0) {
    publishStaticIpcRemoteAbort(readyMem, control);
  }
  FLAGCX_DEVICE_SYNC_THREADS();
  publishStaticIpcAbort(triggers, numTriggers, firstOrdinal, stride);
}

// Static IPC keeps a flat logical-trigger array in topological order. Every
// IPC block advances that array in lockstep, while chunk c is permanently
// assigned to local IPC block c % numIpcBlocks. The per-trigger state and
// completedChunks fields form software phase gates; no grid barrier or
// dynamic chunk claim is needed.
FLAGCX_DEVICE_INLINE_DECORATOR void runStaticIpcExecutor(
    flagcxIpcTrigger *triggers, uint64_t numTriggers,
    const flagcxDevMem &inputMem, const flagcxDevMem &outputMem,
    const flagcxDevMem &readyMem, const uint64_t *parentFlags,
    uint64_t numParentFlags, uint64_t localBlockIdx,
    uint64_t numIpcBlocks, const uint64_t *abortFlag,
    const flagcxStaticIpcControl &control, uint64_t *shm) {
  for (uint64_t ordinal = 0; ordinal < numTriggers; ++ordinal) {
    flagcxIpcTrigger *trigger = &triggers[ordinal];
    const bool leader = ordinal % numIpcBlocks == localBlockIdx;

    if (FLAGCX_THREAD_IDX_X == 0) {
      shm[STATIC_IPC_ABORT_IDX] =
          isStaticIpcAbortRequested(abortFlag, readyMem, control) ? 1 : 0;
      shm[STATIC_IPC_STATE_IDX] = DeviceAPI::Atomic::load(
          &trigger->state, flagcxDeviceMemoryOrderAcquire);
      if (shm[STATIC_IPC_ABORT_IDX] == 0 && leader &&
          shm[STATIC_IPC_STATE_IDX] == flagcxReduceTriggerEnqueued) {
        const uint64_t parentOffset = trigger->parentFlagsOffset;
        const uint64_t triggerParents = trigger->numParentFlags;
        if ((triggerParents != 0 && parentFlags == nullptr) ||
            parentOffset > numParentFlags ||
            triggerParents > numParentFlags - parentOffset) {
          requestStaticAbort(abortFlag);
          shm[STATIC_IPC_ABORT_IDX] = 1;
        }
      }
      if (shm[STATIC_IPC_ABORT_IDX] == 0 && leader &&
          shm[STATIC_IPC_STATE_IDX] == flagcxReduceTriggerEnqueued) {
        if (trigger->flagOut != 0 &&
            loadStreamFlagState(trigger->flagOut) == flagcxStreamFlagIdle) {
          DeviceAPI::Atomic::store(
              reinterpret_cast<uint64_t *>(trigger->flagOut),
              static_cast<uint64_t>(flagcxStreamFlagPend),
              flagcxDeviceMemoryOrderRelease);
        }

        for (uint32_t parent = 0; parent < trigger->numParentFlags;
             ++parent) {
          const uint64_t flagAddr =
              parentFlags[trigger->parentFlagsOffset + parent];
          int backoff = 0;
          while (!isStreamFlagStateDone(loadStreamFlagState(flagAddr))) {
            if (isStaticIpcAbortRequested(abortFlag, readyMem, control)) {
              shm[STATIC_IPC_ABORT_IDX] = 1;
              break;
            }
            DeviceAPI::Intrin::spinBackoff(backoff++);
          }
          if (shm[STATIC_IPC_ABORT_IDX] != 0) {
            break;
          }
        }
        if (shm[STATIC_IPC_ABORT_IDX] == 0) {
          DeviceAPI::Atomic::store(
              &trigger->state,
              static_cast<uint32_t>(flagcxReduceTriggerInprogress),
              flagcxDeviceMemoryOrderRelease);
          shm[STATIC_IPC_STATE_IDX] = flagcxReduceTriggerInprogress;
        }
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    if (shm[STATIC_IPC_ABORT_IDX] != 0) {
      abortStaticIpcExecutor(triggers, numTriggers, readyMem, control,
                             localBlockIdx, numIpcBlocks);
      return;
    }

    if (!leader) {
      if (FLAGCX_THREAD_IDX_X == 0) {
        int backoff = 0;
        while (shm[STATIC_IPC_STATE_IDX] ==
               flagcxReduceTriggerEnqueued) {
          if (isStaticIpcAbortRequested(abortFlag, readyMem, control)) {
            shm[STATIC_IPC_ABORT_IDX] = 1;
            break;
          }
          shm[STATIC_IPC_STATE_IDX] = DeviceAPI::Atomic::load(
              &trigger->state, flagcxDeviceMemoryOrderAcquire);
          DeviceAPI::Intrin::spinBackoff(backoff++);
        }
      }
      FLAGCX_DEVICE_SYNC_THREADS();
      if (shm[STATIC_IPC_ABORT_IDX] != 0) {
        abortStaticIpcExecutor(triggers, numTriggers, readyMem, control,
                               localBlockIdx, numIpcBlocks);
        return;
      }
    }

    // A zero-work block can observe Available if the leader completes this
    // trigger before it observes the intermediate Inprogress state.
    if (shm[STATIC_IPC_STATE_IDX] == flagcxReduceTriggerAvailable) {
      continue;
    }

    uint32_t localCompletedChunks = 0;
    for (uint64_t chunk = localBlockIdx; chunk < trigger->numChunks;
         chunk += numIpcBlocks) {
      if (FLAGCX_THREAD_IDX_X == 0) {
        shm[STATIC_IPC_ABORT_IDX] =
            isStaticIpcAbortRequested(abortFlag, readyMem, control) ? 1 : 0;
      }
      FLAGCX_DEVICE_SYNC_THREADS();
      if (shm[STATIC_IPC_ABORT_IDX] != 0) {
        break;
      }

      const uint64_t chunkOffset = chunk * trigger->chunkSize;
      uint64_t chunkBytes = trigger->bytes - chunkOffset;
      if (chunkBytes > trigger->chunkSize) {
        chunkBytes = trigger->chunkSize;
      }
      const flagcxDevMem &srcMem =
          trigger->srcBufferType == flagcxIpcBufferInput ? inputMem
                                                          : outputMem;
      const void *src = flagcxGetLocalPointer(
          srcMem, trigger->srcOffsetBytes + chunkOffset);
      void *remoteDst = flagcxGetIntraPointer(
          outputMem, trigger->dstOffsetBytes + chunkOffset,
          static_cast<int>(trigger->peerLocalRank));
      copyIpcBytes(remoteDst, src, static_cast<size_t>(chunkBytes));
      FLAGCX_DEVICE_SYNC_THREADS();
      ++localCompletedChunks;
    }

    if (shm[STATIC_IPC_ABORT_IDX] != 0) {
      abortStaticIpcExecutor(triggers, numTriggers, readyMem, control,
                             localBlockIdx, numIpcBlocks);
      return;
    }
    if (localCompletedChunks != 0) {
      // Aggregate this block's statically owned chunks with one atomic after
      // every copy thread has made all of its peer stores system-visible.
      DeviceAPI::Intrin::threadfenceSystem();
      FLAGCX_DEVICE_SYNC_THREADS();
      if (FLAGCX_THREAD_IDX_X == 0) {
        DeviceAPI::Atomic::fetchAdd(
            &trigger->completedChunks, localCompletedChunks,
            flagcxDeviceMemoryOrderAcqRel);
      }
      FLAGCX_DEVICE_SYNC_THREADS();
    }

    if (leader) {
      if (FLAGCX_THREAD_IDX_X == 0) {
        int backoff = 0;
        while (DeviceAPI::Atomic::load(
                   &trigger->completedChunks,
                   flagcxDeviceMemoryOrderAcquire) != trigger->numChunks) {
          if (isStaticIpcAbortRequested(abortFlag, readyMem, control)) {
            shm[STATIC_IPC_ABORT_IDX] = 1;
            break;
          }
          DeviceAPI::Intrin::spinBackoff(backoff++);
        }
        if (shm[STATIC_IPC_ABORT_IDX] == 0) {
          uint64_t *remoteReady =
              static_cast<uint64_t *>(flagcxGetIntraPointer(
                  readyMem,
                  (control.readyDataOffset + trigger->readySlot) *
                      sizeof(uint64_t),
                  static_cast<int>(trigger->peerLocalRank)));
          DeviceAPI::Atomic::store(remoteReady, trigger->epoch,
                                   flagcxDeviceMemoryOrderRelease);

          uint64_t *localReady =
              static_cast<uint64_t *>(flagcxGetLocalPointer(
                  readyMem,
                  (control.readyDataOffset + trigger->readySlot) *
                      sizeof(uint64_t)));
          backoff = 0;
          while (!flagcxIpcEpochReached(
              DeviceAPI::Atomic::load(localReady,
                                      flagcxDeviceMemoryOrderAcquire),
              trigger->epoch)) {
            if (isStaticIpcAbortRequested(abortFlag, readyMem, control)) {
              shm[STATIC_IPC_ABORT_IDX] = 1;
              break;
            }
            DeviceAPI::Intrin::spinBackoff(backoff++);
          }
        }
        if (shm[STATIC_IPC_ABORT_IDX] == 0) {
          if (trigger->flagOut != 0) {
            DeviceAPI::Atomic::store(
                reinterpret_cast<uint64_t *>(trigger->flagOut),
                static_cast<uint64_t>(flagcxStreamFlagDone),
                flagcxDeviceMemoryOrderRelease);
          }
          DeviceAPI::Atomic::store(
              &trigger->state,
              static_cast<uint32_t>(flagcxReduceTriggerAvailable),
              flagcxDeviceMemoryOrderRelease);
        }
      }
      FLAGCX_DEVICE_SYNC_THREADS();
      if (shm[STATIC_IPC_ABORT_IDX] != 0) {
        abortStaticIpcExecutor(triggers, numTriggers, readyMem, control,
                               localBlockIdx, numIpcBlocks);
        return;
      }
    } else {
      if (FLAGCX_THREAD_IDX_X == 0) {
        int backoff = 0;
        while (DeviceAPI::Atomic::load(
                   &trigger->state,
                   flagcxDeviceMemoryOrderAcquire) !=
               flagcxReduceTriggerAvailable) {
          if (isStaticIpcAbortRequested(abortFlag, readyMem, control)) {
            shm[STATIC_IPC_ABORT_IDX] = 1;
            break;
          }
          DeviceAPI::Intrin::spinBackoff(backoff++);
        }
      }
      FLAGCX_DEVICE_SYNC_THREADS();
      if (shm[STATIC_IPC_ABORT_IDX] != 0) {
        abortStaticIpcExecutor(triggers, numTriggers, readyMem, control,
                               localBlockIdx, numIpcBlocks);
        return;
      }
    }
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void finishStaticCollectiveExecution(
    const flagcxDevMem &readyMem, uint64_t *blocksDone,
    uint64_t numExecutorBlocks, const uint64_t *abortFlag,
    const flagcxStaticIpcControl &control, uint64_t *shm) {
  // Every executor block reaches this epilogue only after it has stopped all
  // peer writes. The last block therefore represents rank-local quiescence.
  DeviceAPI::Intrin::threadfenceSystem();
  FLAGCX_DEVICE_SYNC_THREADS();
  if (FLAGCX_THREAD_IDX_X == 0) {
    const uint64_t completedBefore = DeviceAPI::Atomic::fetchAdd(
        blocksDone, uint64_t(1), flagcxDeviceMemoryOrderAcqRel);
    shm[STATIC_IPC_STATE_IDX] =
        completedBefore + 1 == numExecutorBlocks ? 1 : 0;
  }
  FLAGCX_DEVICE_SYNC_THREADS();
  if (shm[STATIC_IPC_STATE_IDX] == 0) {
    return;
  }

  if (FLAGCX_THREAD_IDX_X == 0) {
    bool aborted =
        isStaticIpcAbortRequested(abortFlag, readyMem, control);
    if (aborted) {
      publishStaticIpcAbortCompletion(readyMem, abortFlag, control);
    } else {
      publishStaticIpcControlValue(
          readyMem, control.doneBase + control.localRank, control.epoch,
          control.localRanks);
    }

    for (uint32_t peer = 0; peer < control.localRanks; ++peer) {
      uint64_t *localDone = getStaticIpcReadyPointer(
          readyMem, control.doneBase + static_cast<uint64_t>(peer));
      int backoff = 0;
      while (true) {
        const uint64_t observed = DeviceAPI::Atomic::load(
            localDone, flagcxDeviceMemoryOrderAcquire);
        const bool epochReached =
            flagcxIpcControlEpochReached(observed, control.epoch);
        const bool peerAborted =
            epochReached && (observed & flagcxIpcControlAbortBit) != 0;
        if (!aborted &&
            (peerAborted ||
             isStaticIpcAbortRequested(abortFlag, readyMem, control))) {
          aborted = true;
          // Do not wait for the remaining peers before propagating a late
          // abort. One of them may still be executing (or waiting on a DAG
          // dependency) and needs this rank's abort publication to quiesce.
          publishStaticIpcAbortCompletion(readyMem, abortFlag, control);
        }
        if (epochReached) {
          break;
        }
        DeviceAPI::Intrin::spinBackoff(backoff++);
      }
    }
  }
  FLAGCX_DEVICE_SYNC_THREADS();
}

FLAGCX_DEVICE_INLINE_DECORATOR void runIpcExecutor(
    void *fifoBuffer, const flagcxDevMem &inputMem,
    const flagcxDevMem &outputMem, const flagcxDevMem &readyMem,
    const uint64_t *parentFlags) {
  FLAGCX_SHARED uint64_t activeIdx;
  FLAGCX_SHARED uint64_t claimedChunk;
  FLAGCX_SHARED int finalChunk;
  uint64_t *buffer = static_cast<uint64_t *>(fifoBuffer);
  int capacity = static_cast<int>(buffer[flagcxFifoIdxCapacity]);
  int emptyIter = 0;

  while (true) {
    if (FLAGCX_THREAD_IDX_X == 0) {
      uint64_t consumed = DeviceAPI::Atomic::load(
          buffer + flagcxFifoIdxConsumed, flagcxDeviceMemoryOrderAcquire);
      uint64_t produced = DeviceAPI::Atomic::load(
          buffer + flagcxFifoIdxProduced, flagcxDeviceMemoryOrderAcquire);
      activeIdx = consumed < produced ? consumed : ~uint64_t(0);
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    uint64_t absoluteIdx = activeIdx;
    if (absoluteIdx == ~uint64_t(0)) {
      if (DeviceAPI::Atomic::load(buffer + flagcxFifoIdxTerminate,
                                  flagcxDeviceMemoryOrderAcquire) == 1) {
        break;
      }
      DeviceAPI::Intrin::spinBackoff(emptyIter++);
      continue;
    }

    emptyIter = 0;
    int slot = static_cast<int>(absoluteIdx % capacity);
    flagcxIpcTrigger *trigger =
        reinterpret_cast<flagcxIpcTrigger *>(buffer + flagcxFifoIdxData) +
        slot;

    if (FLAGCX_THREAD_IDX_X == 0) {
      uint32_t expected = flagcxReduceTriggerEnqueued;
      bool firstBlock = DeviceAPI::Atomic::compareExchange(
          &trigger->state, expected,
          static_cast<uint32_t>(flagcxReduceTriggerInprogress),
          flagcxDeviceMemoryOrderAcqRel);
      if (firstBlock && trigger->flagOut != 0) {
        uint64_t *flagOut =
            reinterpret_cast<uint64_t *>(trigger->flagOut);
        if (loadStreamFlagState(trigger->flagOut) == flagcxStreamFlagIdle) {
          DeviceAPI::Atomic::store(flagOut,
                                   static_cast<uint64_t>(flagcxStreamFlagPend),
                                   flagcxDeviceMemoryOrderRelease);
        }
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();

    waitIpcParents(*trigger, parentFlags);
    FLAGCX_DEVICE_SYNC_THREADS();

    while (true) {
      if (FLAGCX_THREAD_IDX_X == 0) {
        claimedChunk = DeviceAPI::Atomic::fetchAdd(
            &trigger->nextChunk, uint64_t(1),
            flagcxDeviceMemoryOrderAcqRel);
        finalChunk = 0;
      }
      FLAGCX_DEVICE_SYNC_THREADS();

      uint64_t chunk = claimedChunk;
      if (chunk >= trigger->numChunks)
        break;

      uint64_t chunkOffset = chunk * trigger->chunkSize;
      uint64_t chunkBytes = trigger->bytes - chunkOffset;
      if (chunkBytes > trigger->chunkSize)
        chunkBytes = trigger->chunkSize;

      const flagcxDevMem &srcMem =
          trigger->srcBufferType == flagcxIpcBufferInput ? inputMem : outputMem;
      const void *src = flagcxGetLocalPointer(
          srcMem, trigger->srcOffsetBytes + chunkOffset);
      void *remoteDst = flagcxGetIntraPointer(
          outputMem, trigger->dstOffsetBytes + chunkOffset,
          static_cast<int>(trigger->peerLocalRank));
      copyIpcBytes(remoteDst, src, static_cast<size_t>(chunkBytes));
      FLAGCX_DEVICE_SYNC_THREADS();

      // Every copy thread publishes its peer stores before this chunk becomes
      // visible in completedChunks. The last completed chunk may then safely
      // publish the logical IPC node's ready epoch.
      DeviceAPI::Intrin::threadfenceSystem();
      FLAGCX_DEVICE_SYNC_THREADS();

      if (FLAGCX_THREAD_IDX_X == 0) {
        uint32_t completed = DeviceAPI::Atomic::fetchAdd(
            &trigger->completedChunks, uint32_t(1),
            flagcxDeviceMemoryOrderAcqRel);
        finalChunk = completed + 1 == trigger->numChunks;
      }
      FLAGCX_DEVICE_SYNC_THREADS();

      if (finalChunk) {
        if (FLAGCX_THREAD_IDX_X == 0) {
          uint64_t *remoteReady =
              static_cast<uint64_t *>(flagcxGetIntraPointer(
                  readyMem, trigger->readySlot * sizeof(uint64_t),
                  static_cast<int>(trigger->peerLocalRank)));
          DeviceAPI::Atomic::store(remoteReady, trigger->epoch,
                                   flagcxDeviceMemoryOrderRelease);

          uint64_t *localReady =
              static_cast<uint64_t *>(flagcxGetLocalPointer(
                  readyMem, trigger->readySlot * sizeof(uint64_t)));
          int iter = 0;
          while (!flagcxIpcEpochReached(
              DeviceAPI::Atomic::load(localReady,
                                      flagcxDeviceMemoryOrderAcquire),
              trigger->epoch)) {
            DeviceAPI::Intrin::spinBackoff(iter++);
          }
          if (trigger->flagOut != 0) {
            DeviceAPI::Atomic::store(
                reinterpret_cast<uint64_t *>(trigger->flagOut),
                static_cast<uint64_t>(flagcxStreamFlagDone),
                flagcxDeviceMemoryOrderRelease);
          }
          DeviceAPI::Atomic::store(
              &trigger->state,
              static_cast<uint32_t>(flagcxReduceTriggerAvailable),
              flagcxDeviceMemoryOrderRelease);
          DeviceAPI::Atomic::fetchAdd(buffer + flagcxFifoIdxConsumed,
                                      uint64_t(1),
                                      flagcxDeviceMemoryOrderRelease);
        }
        FLAGCX_DEVICE_SYNC_THREADS();
        break;
      }
    }

    // A block that found no remaining chunk waits for the unique finalizer to
    // advance the logical FIFO entry before looking for more work.
    if (FLAGCX_THREAD_IDX_X == 0) {
      int iter = 0;
      while (DeviceAPI::Atomic::load(buffer + flagcxFifoIdxConsumed,
                                     flagcxDeviceMemoryOrderAcquire) <=
             absoluteIdx) {
        DeviceAPI::Intrin::spinBackoff(iter++);
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(
    void *redFifoBuffer, void *ipcFifoBuffer, flagcxDevMem inputMem,
    flagcxDevMem outputMem, flagcxDevMem readyMem,
    const uint64_t *ipcParentFlags, int nRedBlocks, int nIpcBlocks,
    uint64_t avgDivisor) {
  if (FLAGCX_BLOCK_IDX_X < nRedBlocks) {
    runReduceExecutor(redFifoBuffer, avgDivisor);
  } else if (FLAGCX_BLOCK_IDX_X < nRedBlocks + nIpcBlocks) {
    runIpcExecutor(ipcFifoBuffer, inputMem, outputMem, readyMem,
                   ipcParentFlags);
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxStaticCollectiveKernel(
    flagcxReduceTrigger *redTriggers, uint64_t numRedTriggers,
    flagcxIpcTrigger *ipcTriggers, uint64_t numIpcTriggers,
    flagcxDevMem inputMem, flagcxDevMem outputMem, flagcxDevMem readyMem,
    const uint64_t *ipcParentFlags, uint64_t numIpcParentFlags,
    uint64_t numRedBlocks, uint64_t numIpcBlocks,
    uint64_t *blocksDone, const uint64_t *abortFlag,
    flagcxStaticIpcControl ipcControl, uint64_t avgDivisor) {
  FLAGCX_SHARED uint64_t shm[16];
  const uint64_t globalBlock = static_cast<uint64_t>(FLAGCX_BLOCK_IDX_X);
  if (globalBlock < numRedBlocks) {
    runStaticReduceExecutor(redTriggers, numRedTriggers, abortFlag,
                            avgDivisor, globalBlock, numRedBlocks, shm);
  } else if (globalBlock < numRedBlocks + numIpcBlocks) {
    runStaticIpcExecutor(ipcTriggers, numIpcTriggers, inputMem, outputMem,
                         readyMem, ipcParentFlags, numIpcParentFlags,
                         globalBlock - numRedBlocks, numIpcBlocks, abortFlag,
                         ipcControl, shm);
  }
  finishStaticCollectiveExecution(
      readyMem, blocksDone, numRedBlocks + numIpcBlocks, abortFlag,
      ipcControl, shm);
}

// Error-only launch used when the cooperative collective kernel did not start
// on this rank. It propagates the failed epoch and joins the same cross-rank
// quiescence barrier as successfully launched peers.
FLAGCX_GLOBAL_DECORATOR void flagcxStaticIpcRecoveryKernel(
    flagcxDevMem readyMem, flagcxStaticIpcControl ipcControl) {
  if (FLAGCX_THREAD_IDX_X != 0) {
    return;
  }
  publishStaticIpcRemoteAbort(readyMem, ipcControl);
  DeviceAPI::Intrin::threadfenceSystem();
  publishStaticIpcControlValue(
      readyMem, ipcControl.doneBase + ipcControl.localRank,
      ipcControl.epoch | flagcxIpcControlAbortBit, ipcControl.localRanks);
  for (uint32_t peer = 0; peer < ipcControl.localRanks; ++peer) {
    uint64_t *localDone = getStaticIpcReadyPointer(
        readyMem, ipcControl.doneBase + static_cast<uint64_t>(peer));
    int backoff = 0;
    while (!flagcxIpcControlEpochReached(
        DeviceAPI::Atomic::load(localDone, flagcxDeviceMemoryOrderAcquire),
        ipcControl.epoch)) {
      DeviceAPI::Intrin::spinBackoff(backoff++);
    }
  }
}

flagcxResult_t flagcxLaunchCollectiveKernel(
    void *redFifoBuffer, void *ipcFifoBuffer, flagcxDevMem_t inputMem,
    flagcxDevMem_t outputMem, flagcxDevMem_t readyMem,
    const uint64_t *ipcParentFlags, size_t nthreads, size_t nRedBlocks,
    size_t nIpcBlocks, uint64_t avgDivisor, flagcxStream_t stream) {
  if (nRedBlocks == 0 && nIpcBlocks == 0) {
    return flagcxSuccess;
  }
  const size_t maxBlocks =
      static_cast<size_t>(std::numeric_limits<int>::max());
  if (nRedBlocks > maxBlocks || nIpcBlocks > maxBlocks ||
      nRedBlocks > maxBlocks - nIpcBlocks) {
    return flagcxInvalidArgument;
  }
  if (nthreads == 0) {
    return flagcxInvalidArgument;
  }
  const size_t nblocks = nRedBlocks + nIpcBlocks;

  flagcxDevMem input;
  flagcxDevMem output;
  flagcxDevMem ready;
  if (inputMem != nullptr)
    input = flagcxDevMem(*inputMem);
  if (outputMem != nullptr)
    output = flagcxDevMem(*outputMem);
  if (readyMem != nullptr)
    ready = flagcxDevMem(*readyMem);
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      redFifoBuffer, ipcFifoBuffer, input, output, ready, ipcParentFlags,
      static_cast<int>(nRedBlocks), static_cast<int>(nIpcBlocks), avgDivisor);
  return flagcxSuccess;
}

static flagcxResult_t getStaticKernelMaxExecutorBlocks(
    const void *kernel, const char *kernelName, size_t nthreads,
    size_t *maxExecutorBlocks) {
  if (maxExecutorBlocks == nullptr) {
    return flagcxInvalidArgument;
  }
  *maxExecutorBlocks = 0;
  if (nthreads == 0 ||
      nthreads > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
  int device = -1;
  int cooperativeLaunch = 0;
  int concurrentKernels = 0;
  int smCount = 0;
  int maxThreadsPerBlock = 0;
  int activeBlocksPerSm = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&cooperativeLaunch,
                             cudaDevAttrCooperativeLaunch, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&concurrentKernels,
                             cudaDevAttrConcurrentKernels, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&maxThreadsPerBlock,
                             cudaDevAttrMaxThreadsPerBlock, device) !=
          cudaSuccess) {
    return flagcxUnhandledDeviceError;
  }
  if (smCount < 0 || maxThreadsPerBlock < 0) {
    return flagcxInternalError;
  }
  if (nthreads > static_cast<size_t>(maxThreadsPerBlock)) {
    return flagcxInvalidArgument;
  }
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &activeBlocksPerSm, kernel, static_cast<int>(nthreads), 0) !=
      cudaSuccess) {
    return flagcxUnhandledDeviceError;
  }
  if (activeBlocksPerSm < 0) {
    return flagcxInternalError;
  }

  flagcxResult_t result = resolveUniRunnerStaticExecutorResidencyBudget(
      cooperativeLaunch != 0, concurrentKernels != 0,
      static_cast<size_t>(smCount), static_cast<size_t>(activeBlocksPerSm),
      static_cast<size_t>(maxThreadsPerBlock), nthreads, maxExecutorBlocks);
  TRACE(FLAGCX_UNIRUNNER,
        "static %s residency: device=%d cooperative=%d concurrent=%d "
        "sms=%d active_per_sm=%d nthreads=%zu budget=%zu result=%d",
        kernelName, device, cooperativeLaunch, concurrentKernels, smCount,
        activeBlocksPerSm, nthreads, *maxExecutorBlocks, result);
  return result;
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t flagcxGetStaticReduceKernelMaxExecutorBlocks(
    size_t nthreads, size_t *maxExecutorBlocks) {
  return getStaticKernelMaxExecutorBlocks(
      reinterpret_cast<const void *>(flagcxStaticReduceKernel), "RED",
      nthreads, maxExecutorBlocks);
}

flagcxResult_t flagcxGetStaticCollectiveKernelMaxExecutorBlocks(
    size_t nthreads, size_t *maxExecutorBlocks) {
  return getStaticKernelMaxExecutorBlocks(
      reinterpret_cast<const void *>(flagcxStaticCollectiveKernel),
      "RED+IPC", nthreads, maxExecutorBlocks);
}

flagcxResult_t flagcxLaunchStaticReduceKernel(
    void *redFifoDeviceBuffer, size_t numTriggers, size_t nthreads,
    size_t nRedBlocks, size_t maxExecutorBlocks, uint64_t avgDivisor,
    flagcxStream_t stream) {
  if (numTriggers == 0) {
    return nRedBlocks == 0 ? flagcxSuccess : flagcxInvalidArgument;
  }
  const size_t maxInt =
      static_cast<size_t>(std::numeric_limits<int>::max());
  if (redFifoDeviceBuffer == nullptr || stream == nullptr || nthreads == 0 ||
      nRedBlocks == 0 || nRedBlocks > numTriggers || avgDivisor == 0 ||
      numTriggers > maxInt || nthreads > maxInt || nRedBlocks > maxInt ||
      maxExecutorBlocks == 0 || maxExecutorBlocks > maxInt ||
      nRedBlocks > maxExecutorBlocks) {
    return flagcxInvalidArgument;
  }

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
  uint64_t *words = static_cast<uint64_t *>(redFifoDeviceBuffer);
  flagcxReduceTrigger *triggers = reinterpret_cast<flagcxReduceTrigger *>(
      words + flagcxFifoIdxData);
  uint64_t *abortFlag = words + flagcxFifoIdxTerminate;
  uint64_t numTriggers64 = static_cast<uint64_t>(numTriggers);
  void *args[] = {&triggers, &numTriggers64, &abortFlag, &avgDivisor};
  cudaError_t launchResult = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void *>(flagcxStaticReduceKernel),
      dim3(static_cast<unsigned int>(nRedBlocks)),
      dim3(static_cast<unsigned int>(nthreads)), args, 0,
      *(FLAGCX_DEVICE_STREAM_PTR)stream);
  return launchResult == cudaSuccess ? flagcxSuccess
                                     : flagcxUnhandledDeviceError;
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t flagcxLaunchStaticCollectiveKernel(
    void *redFifoDeviceBuffer, size_t numRedTriggers,
    void *ipcFifoDeviceBuffer, size_t numIpcTriggers,
    flagcxDevMem_t inputMem, flagcxDevMem_t outputMem,
    flagcxDevMem_t readyMem, const uint64_t *ipcParentFlags,
    size_t numIpcParentFlags, size_t nthreads, size_t nRedBlocks,
    size_t nIpcBlocks, size_t maxExecutorBlocks, uint64_t avgDivisor,
    const flagcxStaticIpcControl *ipcControl, flagcxStream_t stream) {
  const bool hasRed = numRedTriggers != 0;
  const bool hasIpc = numIpcTriggers != 0;
  if (!hasIpc) {
    // RED-only static execution uses flagcxLaunchStaticReduceKernel. The
    // combined kernel requires IPC control storage for its rank barrier.
    return flagcxInvalidArgument;
  }

  const size_t maxInt =
      static_cast<size_t>(std::numeric_limits<int>::max());
  if (stream == nullptr || nthreads == 0 || avgDivisor == 0 ||
      (hasRed != (nRedBlocks != 0)) || (hasIpc != (nIpcBlocks != 0)) ||
      (hasRed && (redFifoDeviceBuffer == nullptr ||
                  nRedBlocks > numRedTriggers)) ||
      (hasIpc && (ipcFifoDeviceBuffer == nullptr || inputMem == nullptr ||
                  outputMem == nullptr || readyMem == nullptr ||
                  ipcControl == nullptr ||
                  (numIpcParentFlags != 0 && ipcParentFlags == nullptr))) ||
      numRedTriggers > maxInt || numIpcTriggers > maxInt ||
      nthreads > maxInt || nRedBlocks > maxInt || nIpcBlocks > maxInt ||
      nRedBlocks > maxInt - nIpcBlocks || maxExecutorBlocks == 0 ||
      maxExecutorBlocks > maxInt ||
      nRedBlocks + nIpcBlocks > maxExecutorBlocks) {
    return flagcxInvalidArgument;
  }
  const uint64_t controlSlots =
      uint64_t(1) + static_cast<uint64_t>(ipcControl->localRanks);
  if (!flagcxIpcControlEpochValid(ipcControl->epoch) ||
      ipcControl->localRanks == 0 ||
      ipcControl->localRank >= ipcControl->localRanks ||
      ipcControl->abortSlot != 0 || ipcControl->doneBase != 1 ||
      ipcControl->readyDataOffset != controlSlots ||
      ipcControl->readyDataOffset >
          std::numeric_limits<uint64_t>::max() - numIpcTriggers ||
      ipcControl->readyDataOffset + numIpcTriggers >
          std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    return flagcxInvalidArgument;
  }

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
  uint64_t *redWords = static_cast<uint64_t *>(redFifoDeviceBuffer);
  uint64_t *ipcWords = static_cast<uint64_t *>(ipcFifoDeviceBuffer);
  flagcxReduceTrigger *redTriggers =
      hasRed ? reinterpret_cast<flagcxReduceTrigger *>(
                   redWords + flagcxFifoIdxData)
             : nullptr;
  flagcxIpcTrigger *ipcTriggers =
      hasIpc ? reinterpret_cast<flagcxIpcTrigger *>(
                   ipcWords + flagcxFifoIdxData)
             : nullptr;
  uint64_t *abortFlag =
      (hasIpc ? ipcWords : redWords) + flagcxFifoIdxTerminate;
  uint64_t *blocksDone =
      (hasIpc ? ipcWords : redWords) + flagcxFifoIdxConsumed;

  flagcxDevMem input;
  flagcxDevMem output;
  flagcxDevMem ready;
  if (hasIpc) {
    input = flagcxDevMem(*inputMem);
    output = flagcxDevMem(*outputMem);
    ready = flagcxDevMem(*readyMem);
  }

  uint64_t numRedTriggers64 = static_cast<uint64_t>(numRedTriggers);
  uint64_t numIpcTriggers64 = static_cast<uint64_t>(numIpcTriggers);
  uint64_t numIpcParentFlags64 =
      static_cast<uint64_t>(numIpcParentFlags);
  uint64_t numRedBlocks64 = static_cast<uint64_t>(nRedBlocks);
  uint64_t numIpcBlocks64 = static_cast<uint64_t>(nIpcBlocks);
  flagcxStaticIpcControl control = *ipcControl;
  void *args[] = {&redTriggers,
                  &numRedTriggers64,
                  &ipcTriggers,
                  &numIpcTriggers64,
                  &input,
                  &output,
                  &ready,
                  &ipcParentFlags,
                  &numIpcParentFlags64,
                  &numRedBlocks64,
                  &numIpcBlocks64,
                  &blocksDone,
                  &abortFlag,
                  &control,
                  &avgDivisor};
  const size_t nblocks = nRedBlocks + nIpcBlocks;
  cudaError_t launchResult = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void *>(flagcxStaticCollectiveKernel),
      dim3(static_cast<unsigned int>(nblocks)),
      dim3(static_cast<unsigned int>(nthreads)), args, 0,
      *(FLAGCX_DEVICE_STREAM_PTR)stream);
  return launchResult == cudaSuccess ? flagcxSuccess
                                     : flagcxUnhandledDeviceError;
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t flagcxLaunchStaticIpcRecoveryKernel(
    flagcxDevMem_t readyMem, const flagcxStaticIpcControl *ipcControl,
    flagcxStream_t stream) {
  if (readyMem == nullptr || ipcControl == nullptr || stream == nullptr ||
      !flagcxIpcControlEpochValid(ipcControl->epoch) ||
      ipcControl->localRanks == 0 ||
      ipcControl->localRank >= ipcControl->localRanks ||
      ipcControl->abortSlot != 0 || ipcControl->doneBase != 1 ||
      ipcControl->readyDataOffset !=
          uint64_t(1) + static_cast<uint64_t>(ipcControl->localRanks)) {
    return flagcxInvalidArgument;
  }

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
  flagcxDevMem ready(*readyMem);
  flagcxStaticIpcControl control = *ipcControl;
  void *args[] = {&ready, &control};
  cudaError_t launchResult = cudaLaunchKernel(
      reinterpret_cast<const void *>(flagcxStaticIpcRecoveryKernel), dim3(1),
      dim3(1), args, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream);
  return launchResult == cudaSuccess ? flagcxSuccess
                                     : flagcxUnhandledDeviceError;
#else
  return flagcxNotSupported;
#endif
}
