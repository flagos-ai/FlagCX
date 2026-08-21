/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Kernel Core — Device-side types, enums, and constants.
 *
 * This header contains device-visible content safe for LLVM bitcode
 * compilation. It includes:
 *   - Device primitive enums (flagcxDevicePrim, flagcxFifoIndex, etc.)
 *   - Trigger bit layout constants (flagcxDeviceTrigger*)
 *   - Struct definitions (flagcxDeviceTrigger, flagcxReduceTrigger)
 *   - Kernel launch configuration macros
 *
 * For host-side methods and lifecycle functions, see flagcx_kernel_internal.h.
 * For normal builds, include flagcx_kernel.h (umbrella header).
 ************************************************************************/

#ifndef FLAGCX_KERNEL_CORE_H_
#define FLAGCX_KERNEL_CORE_H_

#include "device_utils.h"
#include "flagcx.h"

struct flagcxHeteroComm;

#define FLAGCX_FIFO_CAPACITY 128
#define flagcxTriggerMask(w) ((w == 64) ? ~0ull : ((1ull << w) - 1))

typedef enum {
  flagcxDevicePrimSend = 0,
  flagcxDevicePrimRecv = 1,
  flagcxDevicePrimTerm = 2,
  flagcxDevicePrimWait = 3,
  flagcxDevicePrimPut = 4,
  flagcxDevicePrimSignal = 5,
  flagcxDevicePrimBarrierSignal =
      6, // Legacy: no-op in proxy (barriers now use PrimSignal)
  flagcxDevicePrimWaitSignal =
      7, // Legacy: no-op in proxy (GPU polls signal buffer directly)
  flagcxDevicePrimPutValue = 8,
  flagcxDevicePrimPutSignal = 9,
  flagcxDevicePrimGet = 10
} flagcxDevicePrim;

// Unified buffer index enumeration for fifo
// Layout: [capacity][consumed][produced][terminate|completed][data...]
//
// Slot 3 is shared between two FIFO types with mutually exclusive semantics:
//   - flagcxReduceTrigger FIFO: uses slot 3 as "terminate" (host signals GPU to
//     stop). GPU reads, host writes.
//   - flagcxDeviceTrigger FIFO: uses slot 3 as "completed" (proxy reports how
//     many IB ops have finished). GPU reads, single proxy thread writes.
// A given FIFO instance is one type or the other — never both.
typedef enum {
  flagcxFifoIdxCapacity = 0,
  flagcxFifoIdxConsumed = 1,
  flagcxFifoIdxProduced = 2,
  flagcxFifoIdxTerminate = 3, // ReduceTrigger FIFO only
  flagcxFifoIdxCompleted = 3, // DeviceTrigger FIFO only (IB completion count)
  flagcxFifoIdxData = 4
} flagcxFifoIndex;

typedef enum {
  flagcxReduceTriggerAvailable = 0,
  flagcxReduceTriggerEnqueued = 1,
  flagcxReduceTriggerInprogress = 2,
  flagcxReduceTriggerComplete = 3
} flagcxReduceTriggerState;

// ==========================================================================
// flagcxDeviceTrigger bit layout (24 bytes = 3 × uint64_t: fst, snd, trd)
//
// trd (word2, control header — written last with valid bit):
//   [63]    valid
//   [62:59] prim (4 bits)
//   [58:39] peerRank (20 bits)
//   [38:36] slotIdx (3 bits, reserved for future multi-FIFO)
//   [35:0]  prim-specific (36 bits)
//
// fst (word0, payload — written first):
//   prim-specific (64 bits)
//
// snd (word1, payload — written second):
//   prim-specific (64 bits)
// ==========================================================================

// Valid bit (trd[63])
constexpr unsigned int flagcxDeviceTriggerOffValid = 63;
constexpr uint64_t flagcxDeviceTriggerValidMask = (1ULL << 63);

// Common header in trd
constexpr unsigned int flagcxDeviceTriggerOffPrim = 59;
constexpr unsigned int flagcxDeviceTriggerBitsPrim = 4;
constexpr unsigned int flagcxDeviceTriggerOffPeerRank = 39;
constexpr unsigned int flagcxDeviceTriggerBitsPeerRank = 20;
constexpr unsigned int flagcxDeviceTriggerOffSlotIdx = 36;
constexpr unsigned int flagcxDeviceTriggerBitsSlotIdx = 3;

// Two-sided Send/Recv: trd prim-specific
//   trd[35:32] = datatype(4), trd[31:0] = count(32)
//   fst = addr(64), snd = 0
constexpr unsigned int flagcxDeviceTriggerOffDatatype = 32;
constexpr unsigned int flagcxDeviceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxDeviceTriggerOffCount = 0;
constexpr unsigned int flagcxDeviceTriggerBitsCount = 32;

// One-sided Put/PutSignal: trd prim-specific
//   trd[35:29] = srcMrIdx(7), trd[28:22] = dstMrIdx(7)
//   PutSignal: trd[21:10] = signalIdx(12), trd[9:0] = unused
//   fst = srcOffset(32)|dstOffset(32), snd =
//   size(32)|signalValue(16)|reserved(16)
constexpr unsigned int flagcxDeviceTriggerOffSrcMrIdx = 29;
constexpr unsigned int flagcxDeviceTriggerBitsSrcMrIdx = 7;
constexpr unsigned int flagcxDeviceTriggerOffDstMrIdx = 22;
constexpr unsigned int flagcxDeviceTriggerBitsDstMrIdx = 7;
constexpr unsigned int flagcxDeviceTriggerOffSignalIdx = 10;
constexpr unsigned int flagcxDeviceTriggerBitsSignalIdx = 12;
// PutSignal signalValue in snd[15:0] (same max as PrimSignal: 16b)
constexpr unsigned int flagcxDeviceTriggerOffSignalValuePut = 0;
constexpr unsigned int flagcxDeviceTriggerBitsSignalValuePut = 16;
// fst offsets for srcOffset/dstOffset (shared with PutValue dstOffset accessor)
constexpr unsigned int flagcxDeviceTriggerOffSrcOffset = 32;
constexpr unsigned int flagcxDeviceTriggerBitsSrcOffset = 32;
constexpr unsigned int flagcxDeviceTriggerOffDstOffset = 0;
constexpr unsigned int flagcxDeviceTriggerBitsDstOffset = 32;
// snd offset for size
constexpr unsigned int flagcxDeviceTriggerOffSize = 32;
constexpr unsigned int flagcxDeviceTriggerBitsSize = 32;

// One-sided PutValue: trd prim-specific
//   trd[28:22] = dstMrIdx(7) (same position as Put/PutSignal dstMrIdx)
//   fst = 0|dstOffset(32) (fst[31:0], same position as Put/PutSignal)
//   snd = value(64)

// Signal/WaitSignal: all in trd prim-specific
//   trd[35:34] = bufferType(2), trd[33:22] = signalIdx(12),
//   trd[21:6] = signalValue/expectedValue(16), trd[5:0] = unused
//   fst = 0, snd = 0
constexpr unsigned int flagcxDeviceTriggerOffBufferType = 34;
constexpr unsigned int flagcxDeviceTriggerBitsBufferType = 2;
constexpr unsigned int flagcxDeviceTriggerOffSignalIdxSig = 22;
constexpr unsigned int flagcxDeviceTriggerBitsSignalIdxSig = 12;
constexpr unsigned int flagcxDeviceTriggerOffSignalValue = 6;
constexpr unsigned int flagcxDeviceTriggerBitsSignalValue =
    16; // max signal value: 2^16 (65535)

constexpr unsigned int flagcxReduceTriggerBitsAddr = 64;
constexpr unsigned int flagcxReduceTriggerOffCount = 0;
constexpr unsigned int flagcxReduceTriggerBitsCount = 32;
constexpr unsigned int flagcxReduceTriggerOffNThreads =
    flagcxReduceTriggerOffCount + flagcxReduceTriggerBitsCount;
constexpr unsigned int flagcxReduceTriggerBitsNThreads = 16;
constexpr unsigned int flagcxReduceTriggerOffDatatype =
    flagcxReduceTriggerOffNThreads + flagcxReduceTriggerBitsNThreads;
constexpr unsigned int flagcxReduceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxReduceTriggerOffRedop =
    flagcxReduceTriggerOffDatatype + flagcxReduceTriggerBitsDatatype;
constexpr unsigned int flagcxReduceTriggerBitsRedop = 4;
constexpr unsigned int flagcxReduceTriggerOffState =
    flagcxReduceTriggerOffRedop + flagcxReduceTriggerBitsRedop;
/* op state: 0 for available, 1 for enqueued, 2 for in-progress, 3 for done */
constexpr unsigned int flagcxReduceTriggerBitsState = 2;
constexpr unsigned int flagcxReduceTriggerBitsFifoReserved = 1;

// Kernel launch configuration constants.
// Also defined in device_api/flagcx_device.h (with same include guard).
#ifndef FLAGCX_DEVICE_CTA_COUNT
#define FLAGCX_DEVICE_CTA_COUNT 36
#endif
#ifndef FLAGCX_DEVICE_THREADS_PER_CTA
#define FLAGCX_DEVICE_THREADS_PER_CTA 512
#endif

struct flagcxDeviceTrigger {
  uint64_t fst; // word0 — payload, written first
  uint64_t snd; // word1 — payload, written second
  uint64_t trd; // word2 — control header (valid bit), written last

  // Common accessors (trd common header)
  FLAGCX_HOST_DECORATOR uint64_t getPrim();
  FLAGCX_HOST_DECORATOR uint64_t getPeerRank();
  FLAGCX_HOST_DECORATOR uint64_t getSlotIdx();

  // Two-sided accessors (Send/Recv)
  FLAGCX_HOST_DECORATOR uint64_t getAddr();     // fst
  FLAGCX_HOST_DECORATOR uint64_t getDatatype(); // trd[35:32]
  FLAGCX_HOST_DECORATOR uint64_t getCount();    // trd[31:0]

  // One-sided accessors (Put/PutSignal/PutValue)
  FLAGCX_HOST_DECORATOR uint64_t getSrcMrIdx();  // trd[35:29]
  FLAGCX_HOST_DECORATOR uint64_t getDstMrIdx();  // trd[28:22]
  FLAGCX_HOST_DECORATOR uint64_t getSize();      // snd[63:32]
  FLAGCX_HOST_DECORATOR uint64_t getSrcOffset(); // fst[63:32]
  FLAGCX_HOST_DECORATOR uint64_t getDstOffset(); // fst[31:0]
  FLAGCX_HOST_DECORATOR uint64_t getValue();     // snd (PutValue)
  FLAGCX_HOST_DECORATOR uint64_t
  getSignalIdx(); // trd (PutSignal/Signal/WaitSignal)
  FLAGCX_HOST_DECORATOR uint64_t getSignalValue();   // trd (Signal)
  FLAGCX_HOST_DECORATOR uint64_t getExpectedValue(); // trd (WaitSignal)
  FLAGCX_HOST_DECORATOR uint64_t getBufferType();    // trd (Signal/WaitSignal)

  // Term accessor
  FLAGCX_HOST_DECORATOR uint64_t getTotalCoops(); // fst (PrimTerm)

  // Backward compat alias
  FLAGCX_HOST_DECORATOR uint64_t getType(); // alias for getPrim()
};
typedef flagcxDeviceTrigger *flagcxDeviceTrigger_t;

struct alignas(16) flagcxReduceTrigger {
  uint64_t value[4];

#ifdef COMPILE_KERNEL
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getInput1();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getInput2();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getOutput();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getCount();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getNThreads();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getDatatype();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getRedop();
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getState();
  FLAGCX_DEVICE_INLINE_DECORATOR void setComplete();
#endif
  FLAGCX_HOST_DECORATOR void setValue(uint64_t fst, uint64_t snd, uint64_t out,
                                      size_t count, size_t nthreads,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t redOp,
                                      flagcxReduceTriggerState state);
  FLAGCX_HOST_DECORATOR uint64_t pollState();
  FLAGCX_HOST_DECORATOR void setState(int state);
};
typedef flagcxReduceTrigger *flagcxReduceTrigger_t;

#endif // FLAGCX_KERNEL_CORE_H_
