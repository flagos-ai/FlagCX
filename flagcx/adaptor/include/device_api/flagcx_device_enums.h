/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API Enums — Scalar enum types for IR/Triton integration.
 *
 * These enums encode cooperative group kind and team kind as plain
 * integers, enabling LLVM IR callers (e.g. Triton) to express these
 * concepts without instantiating C++ structs.
 *
 * Safe to include from: CUDA device code, host code, LLVM bitcode builds.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_ENUMS_H_
#define FLAGCX_DEVICE_ENUMS_H_

#include <stdint.h>

/* ================================================================
 * Device IR ABI
 *
 * Lane masks are part of the public C++ and LLVM IR interfaces. Keep a
 * fixed-width type so platforms with up to 64 hardware lanes share one ABI.
 * ================================================================ */
#define FLAGCX_DEVICE_IR_ABI_VERSION 2
typedef uint64_t flagcxLaneMask_t;

#if defined(__cplusplus)
static_assert(sizeof(flagcxLaneMask_t) == 8,
              "flagcxLaneMask_t must be a 64-bit ABI type");
#endif

/* ================================================================
 * Cooperative Group Kind
 *
 * Identifies the cooperation scope for collective device operations.
 * Used by the scalar IR API in place of struct-based cooperative groups.
 * ================================================================ */
typedef enum {
  FLAGCX_COOP_THREAD = 0,    /* Single thread (no cooperation) */
  FLAGCX_COOP_WARP = 1,      /* Full warp (FLAGCX_SIMT_WIDTH threads) */
  FLAGCX_COOP_BLOCK = 2,     /* Entire CTA */
  FLAGCX_COOP_TILE_SPAN = 3, /* Consecutive tile span (needs t0, nTiles, id) */
  FLAGCX_COOP_LANES = 4,     /* Arbitrary lane bitmask */
} flagcxCoopKind_t;

/* ================================================================
 * Team Kind
 *
 * Identifies the team scope within a communicator.
 * Used by the scalar IR API in place of struct-based teams.
 * ================================================================ */
typedef enum {
  FLAGCX_TEAM_INTRA = 0, /* Intra-node ranks */
  FLAGCX_TEAM_INTER = 1, /* Inter-node representatives */
  FLAGCX_TEAM_WORLD = 2, /* All ranks */
} flagcxTeamKind_t;

/* ================================================================
 * Memory Order
 *
 * Memory ordering semantics for synchronization operations.
 * Platform-agnostic classification based on C11/C++ memory model.
 * Platform implementations map these to hardware-specific fences.
 * ================================================================ */
typedef enum {
  FLAGCX_MEMORY_ORDER_RELAXED = 0, /* No ordering constraints */
  FLAGCX_MEMORY_ORDER_ACQUIRE = 1, /* Acquire semantics (load barrier) */
  FLAGCX_MEMORY_ORDER_RELEASE = 2, /* Release semantics (store barrier) */
  FLAGCX_MEMORY_ORDER_ACQ_REL = 3, /* Both acquire and release */
  FLAGCX_MEMORY_ORDER_SEQ_CST = 4, /* Sequential consistency */
} flagcxMemoryOrder_t;

/* ================================================================
 * Memory Scope
 *
 * Visibility scope for synchronization operations.
 * Platform-agnostic classification - all platforms must support
 * these scopes, though implementation details vary by hardware.
 * ================================================================ */
typedef enum {
  FLAGCX_MEMORY_SCOPE_SYSTEM = 0, /* Visible to all threads in the system */
  FLAGCX_MEMORY_SCOPE_DEVICE = 1, /* Visible to all threads on the device */
  FLAGCX_MEMORY_SCOPE_BLOCK = 2,  /* Visible to all threads in the block */
  FLAGCX_MEMORY_SCOPE_THREAD = 3, /* Visible to current thread only */
} flagcxMemoryScope_t;

// Backward-compatible enum constant names (used by platform implementation
// files)
#define flagcxDeviceMemoryOrderRelaxed FLAGCX_MEMORY_ORDER_RELAXED
#define flagcxDeviceMemoryOrderAcquire FLAGCX_MEMORY_ORDER_ACQUIRE
#define flagcxDeviceMemoryOrderRelease FLAGCX_MEMORY_ORDER_RELEASE
#define flagcxDeviceMemoryOrderAcqRel FLAGCX_MEMORY_ORDER_ACQ_REL
#define flagcxDeviceMemoryOrderSeqCst FLAGCX_MEMORY_ORDER_SEQ_CST

#define flagcxDeviceScopeSystem FLAGCX_MEMORY_SCOPE_SYSTEM
#define flagcxDeviceScopeDevice FLAGCX_MEMORY_SCOPE_DEVICE
#define flagcxDeviceScopeBlock FLAGCX_MEMORY_SCOPE_BLOCK
#define flagcxDeviceScopeThread FLAGCX_MEMORY_SCOPE_THREAD

// Type aliases for Unified IR naming convention. Cooperative and team values
// are kind selectors, not the internal flagcxCoopAny/flagcxTeam objects.
typedef flagcxCoopKind_t flagcxDevCoopKind_t;
typedef flagcxTeamKind_t flagcxDevTeamKind_t;
typedef flagcxMemoryOrder_t flagcxDevMemoryOrder_t;
typedef flagcxMemoryScope_t flagcxDevMemoryScope_t;

// Legacy type aliases for backward compatibility with platform implementations
typedef flagcxMemoryOrder_t flagcxDeviceMemoryOrder_t;
typedef flagcxMemoryScope_t flagcxDeviceScope_t;

/* ================================================================
 * Device API Slot Identifiers
 *
 * Opaque slot types for signal, counter, and context IDs.
 * These are simple uint32_t indices used across all device APIs.
 * ================================================================ */
typedef uint32_t flagcxDevSignal_t;
typedef uint32_t flagcxDevCounter_t;
typedef uint32_t flagcxDevContext_t;

// Legacy one-sided IR slot names. Keep these aliases for source compatibility
// while the deprecated flagcxDevNet* entry points remain exported.
typedef flagcxDevSignal_t flagcxDevNetSignal_t;
typedef flagcxDevCounter_t flagcxDevNetCounter_t;

/* ================================================================
 * Peer Pointer Access
 *
 * Specifies the operations the caller may perform through a translated peer
 * pointer. Read-write preserves the legacy behavior; read-only is for load
 * paths and write-only is for store paths. A backend must reject an access
 * mode that its peer-memory transport cannot support. Multiple supported modes
 * may use the same native mapping, but the caller must still obey the requested
 * access contract.
 * ================================================================ */
typedef enum {
  flagcxDevPeerAccessReadWrite = 0,
  flagcxDevPeerAccessWriteOnly = 1,
  flagcxDevPeerAccessReadOnly = 2,
} flagcxDevPeerAccess_t;

#endif /* FLAGCX_DEVICE_ENUMS_H_ */
